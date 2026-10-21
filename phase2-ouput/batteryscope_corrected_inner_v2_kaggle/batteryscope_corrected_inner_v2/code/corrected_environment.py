"""Reusable strict environment profile and offline reproducibility inventory.

No package installs, network, account actions or environment-variable dumps.
Core drift fails closed; a new profile/review is required for future changes.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys


def require(ok,message):
    if not bool(ok):
        raise RuntimeError(message)


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def verify_environment_profile(bundle,config):
    path=Path(bundle)/"provenance/environment_profile.frozen.json"
    profile=json.loads(path.read_text(encoding="utf-8"))
    require(profile["schema"]=="corrected-environment-profile-v1","ENV_PROFILE_SCHEMA")
    payload={k:v for k,v in profile.items() if k!="canonical_profile_sha256"}
    require(canonical_sha(payload)==profile["canonical_profile_sha256"],"ENV_PROFILE_CANONICAL_HASH_MISMATCH")
    require(hashlib.sha256(path.read_bytes()).hexdigest()==config["environment_profile_file_sha256"],"ENV_PROFILE_FILE_HASH_MISMATCH")
    require(config["environment_profile_id"]==profile["profile_id"],"ENV_PROFILE_ID_MISMATCH")
    require(config["environment_reference"]==profile["observed_environment"],"ACTIVE_ENV_NOT_BOUND_TO_APPROVED_PROFILE")
    require(profile["future_unknown_versions_auto_accepted"] is False and profile["automatic_pip_install"] is False,"UNAPPROVED_ENVIRONMENT_BYPASS")
    return profile


def runtime_environment():
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import sklearn
    try:
        import torch
    except ImportError:
        torch=None
    cuda=torch is not None and torch.cuda.is_available()
    return dict(packages=dict(python=sys.version.split()[0],numpy=np.__version__,pandas=pd.__version__,
                             pyarrow=pa.__version__,scikit_learn=sklearn.__version__,
                             torch=str(torch.__version__) if torch else None),
                cuda_available=bool(cuda),torch_cuda_version=torch.version.cuda if torch else None,
                cudnn_version=torch.backends.cudnn.version() if torch else None,
                gpu_name=torch.cuda.get_device_name(0) if cuda else None,
                gpu_count=torch.cuda.device_count() if cuda else 0)


def distribution_inventory():
    packages=[]
    for dist in metadata.distributions():
        name=dist.metadata.get("Name","")
        if name:
            packages.append(dict(name=name,version=dist.version))
    return sorted(packages,key=lambda r:(r["name"].lower(),r["version"]))


def inventory(*,actual,profile_id,config_sha256,phase):
    packages=distribution_inventory()
    result=dict(schema="corrected-runtime-inventory-v1",captured_at_utc=datetime.now(timezone.utc).isoformat(),
                profile_id=profile_id,config_sha256=config_sha256,phase=phase,core_environment=actual,
                os=dict(system=platform.system(),release=platform.release(),machine=platform.machine(),
                        python_implementation=platform.python_implementation()),
                installed_distributions=packages,installed_distributions_sha256=canonical_sha(packages),
                declared_runtime_controls={name:os.environ.get(name) for name in
                    ["CUBLAS_WORKSPACE_CONFIG","CUDA_VISIBLE_DEVICES","NVIDIA_VISIBLE_DEVICES","KAGGLE_DOCKER_IMAGE"]},
                image_digest_verified=False,full_environment_variables_dumped=False,
                package_installation_performed=False,network_used=False)
    try:
        import torch
        result["torch_build_configuration"]=torch.__config__.show()
        result["torch_backend_flags"]=dict(cudnn_deterministic=torch.backends.cudnn.deterministic,
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                num_threads=torch.get_num_threads())
        if torch.cuda.is_available():
            result["cuda_devices"]=[dict(index=i,name=torch.cuda.get_device_name(i),
                                        capability=list(torch.cuda.get_device_capability(i)),
                                        total_memory_bytes=torch.cuda.get_device_properties(i).total_memory)
                                    for i in range(torch.cuda.device_count())]
    except Exception as exc:
        result["torch_inventory_unavailable"]=f"{type(exc).__name__}: {exc}"
    try:
        proc=subprocess.run(["nvidia-smi","--query-gpu=name,driver_version,memory.total","--format=csv,noheader"],
                            capture_output=True,text=True,timeout=5,shell=False)
        result["nvidia_smi"]=dict(exit_code=proc.returncode,stdout=proc.stdout[:10000],stderr=proc.stderr[:2000])
    except (OSError,subprocess.TimeoutExpired) as exc:
        result["nvidia_smi_unavailable"]=f"{type(exc).__name__}: {exc}"
    return result


def write_reproducibility_inventory(run,config,actual,bundle,*,phase="parent_preflight"):
    verify_environment_profile(bundle,config)
    record=inventory(actual=actual,profile_id=config["environment_profile_id"],config_sha256=canonical_sha(config),phase=phase)
    target=Path(run)/"reproducibility_inventory.json"
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_suffix(".json.partial")
    temporary.write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    os.replace(temporary,target)
    # Diagnostic snapshot, not a script to install the entire Kaggle image.
    lines=[]
    for row in record["installed_distributions"]:
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*",row["name"]) and re.fullmatch(r"[A-Za-z0-9.+!_-]+",row["version"]):
            lines.append(f"{row['name']}=={row['version']}")
    (Path(run)/"installed_packages.snapshot.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return record
