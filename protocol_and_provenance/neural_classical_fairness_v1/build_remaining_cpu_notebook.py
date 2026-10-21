"""Build one self-contained Kaggle CPU notebook for the 19 unstarted local-v2 jobs."""
from __future__ import annotations

import ast
import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys
import zlib

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "notebook/neural-classical-fairness-v1/remaining19-cpu-v1"
LOCAL = ROOT / "paper/audits/neural_classical_fairness_2026-10-08_v1/local_cpu_dev_v2"
BUNDLE = ROOT / "phase2-ouput/batteryscope_corrected_inner_v2_kaggle/batteryscope_corrected_inner_v2"
NOTEBOOK = "BatteryScope_NFR1_Remaining19_CPU_v1.ipynb"
sys.path.insert(0, str(ROOT))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def binding(path):
    return dict(bytes=path.stat().st_size, sha256=sha(path))


def require(ok, reason):
    if not ok:
        raise RuntimeError("REMAINING_NOTEBOOK_BUILD:" + reason)


def payload():
    from protocol_and_provenance.neural_classical_fairness_v1.runtime import local_cpu_pilots as local
    manifest = json.loads((LOCAL / "launch_manifest.json").read_text())
    ledger = json.loads((LOCAL / "compute_ledger.json").read_text())
    summary = json.loads((LOCAL / "summary.json").read_text())
    require(manifest["jobs"] == local.jobs() and len(ledger) == 47 and
            len({j["id"] for j in ledger}) == 47 and
            {j["id"] for j in ledger} == {j["id"] for j in local.jobs()}, "LOCAL_QUEUE")
    require(summary["completed_jobs"] == 28 and summary["error"] is None and
            summary["protected_hashes_unchanged"] is True, "LOCAL_SUMMARY")
    for record in manifest["bindings"] + manifest["protected"]:
        require(binding(ROOT / record["path"]) == {k:record[k] for k in ("bytes", "sha256")}, "LOCAL_SEAL:" + record["path"])
    completed = [j["id"] for j in ledger if j["state"] == "COMPLETE"]
    require(len(completed) == 28 and all(j["state"] in
            ("COMPLETE", "DEFERRED_RESOURCE", "NOT_STARTED_AFTER_STOP") for j in ledger), "LOCAL_STATES")
    # Own sealed artifacts only: no unpickle or model fits while building.
    accepted_receipts = {}
    for job_id in completed:
        directory = LOCAL / "jobs" / job_id
        result = json.loads((directory / "result.json").read_text())
        replay = json.loads((directory / "replay.json").read_text())
        integrity = json.loads((directory / "model.integrity.json").read_text())
        require(result["status"] == "PILOT_TRAIN_COMPLETE_RELOAD_PENDING" and
                replay["status"] == "FRESH_PROCESS_FORWARD_FIXTURE_EXACT_PASS", "LOCAL_ACCEPTANCE:" + job_id)
        names = [("model.pt", "sha256"), ("replay_x.npy", "replay_x_sha256"), ("replay_expected.npz", "replay_expected_sha256")] if job_id.startswith("neural.") else [
            ("pipeline.joblib", "sha256"), ("replay_features.parquet", "feature_sha256"), ("replay_expected.npy", "expected_sha256")]
        for name, key in names:
            require(sha(directory / name) == integrity[key], "LOCAL_ARTIFACT:" + job_id)
        accepted_receipts[job_id] = {name:binding(directory / name) for name in
                                    ("result.json", "replay.json", "model.integrity.json", "roles.json")}
    files = {}
    sources = {
        "remaining_runtime.py": HERE / "runtime/kaggle_classical_remaining.py",
        "protocol_and_provenance/neural_classical_fairness_v1/runtime/local_cpu_pilots.py": HERE / "runtime/local_cpu_pilots.py",
        "protocol_and_provenance/neural_classical_fairness_v1/runtime/selection_controls.py": HERE / "runtime/selection_controls.py",
        "protocol_and_provenance/neural_classical_fairness_v1/runtime/environment_identity.py": HERE / "runtime/environment_identity.py",
        "phase2-ouput/batteryscope_stage7_programmatic_v6/stage8_outer_model_executor.py": local.OUTER,
        "kaggle_bootstrap.py": ROOT / "notebook/neural-classical-fairness-v1/diagnostic-v4/batteryscope_nfr1_diagnostic_v4/scripts/kaggle_bootstrap.py",
    }
    for name, path in sources.items():
        ast.parse(path.read_text(encoding="utf-8"))
        files[name] = base64.b64encode(path.read_bytes()).decode()
    original = json.loads((HERE / "source_bindings.json").read_text())
    prefix = "phase2-ouput/batteryscope_corrected_inner_v2_kaggle/batteryscope_corrected_inner_v2/"
    inputs = {r["path"][len(prefix):]:{k:r[k] for k in ("bytes", "sha256")} for r in original["inputs"]}
    for name in ("inner_config.frozen.json", "code/inner_recipe.json", "code/smoke_reference_runtime.py",
                 "code/corrected_environment.py", "phase2/tabular_feature_index.parquet"):
        inputs[name] = binding(BUNDLE / name)
    spec = dict(schema="nfr1-kaggle-remaining19-cpu-v1", author_request_date="2026-10-10",
        execution_role="POST_HOC_KAGGLE_CPU_CLASSICAL_CONTINUATION_NOT_FAIRNESS_CLOSURE",
        jobs=[j for j in local.jobs() if j["id"] not in completed], local_completed_job_ids=completed,
        local_attestation=dict(manifest=binding(LOCAL / "launch_manifest.json"),
            ledger=binding(LOCAL / "compute_ledger.json"), summary=binding(LOCAL / "summary.json"),
            receipts=accepted_receipts, model_or_checkpoint_content_included=False),
        input_files=inputs, source_bindings={k:original[k] for k in ("config", "inputs")},
        classical_grid=local.grid(), prediction_policies=local.CLASSICAL_PREDICTION_POLICIES,
        expected_python="3.13.15", expected_packages={"numpy":"2.1.3", "pandas":"2.3.3",
            "pyarrow":"23.0.1", "scikit-learn":"1.6.1"},
        other_imported_dependencies="record_and_exact_session_lock_not_windows_version_parity",
        budget=dict(total_wall_seconds=28800, reserve_seconds=1800, job_cap_seconds=3600,
            replay_cap_seconds=300, sequential_workers=1, synthetic_fits=2, real_classical_fits=13,
            reference_jobs=6, maximum_reference_executions=16),
        files={name:dict(bytes=len(base64.b64decode(data)), sha256=hashlib.sha256(base64.b64decode(data)).hexdigest())
               for name,data in files.items()}, outer_scoring=False, fairness_closed=False,
        automatic_retry=False, automatic_resume=False, cross_host_parity_claim=False)
    from protocol_and_provenance.neural_classical_fairness_v1.runtime.kaggle_classical_remaining import validate_jobs
    validate_jobs(spec)
    files["remaining_delivery.json"] = base64.b64encode(json.dumps(spec, sort_keys=True, ensure_ascii=False,
        allow_nan=False).encode()).decode()
    return files, spec


BOOTSTRAP = '''# Code is embedded in this notebook. Mount ONLY corrected inner v2.
import os,sys,json,base64,zlib,hashlib,subprocess
from pathlib import Path
os.environ.update(OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",
                  CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1",PYTHONUNBUFFERED="1")
WORKSPACE=Path('/kaggle/working/nfr1_remaining19_cpu_v1_workspace')
assert not WORKSPACE.exists(), 'Use a fresh Kaggle session; no automatic resume or overwrite.'
WORKSPACE.mkdir(parents=True)
CODE_BLOB=__BLOB__
EXPECTED_PAYLOAD_SHA256=__SHA__
raw=zlib.decompress(base64.b64decode(CODE_BLOB))
assert hashlib.sha256(raw).hexdigest()==EXPECTED_PAYLOAD_SHA256, 'Embedded code hash mismatch'
payload=json.loads(raw)
for relative,data in payload.items():
    target=(WORKSPACE/relative).resolve()
    assert target.is_relative_to(WORKSPACE.resolve()) and target!=WORKSPACE.resolve()
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('xb') as handle: handle.write(base64.b64decode(data))
sys.path.insert(0,str(WORKSPACE))
import kaggle_bootstrap
try:
    bundle=kaggle_bootstrap.resolve_bundle('/kaggle/input',WORKSPACE)
except Exception:
    import traceback,zipfile
    from IPython.display import FileLink,display
    failure=WORKSPACE/'bootstrap_failure.json'
    failure.write_text(json.dumps(dict(status='BOOTSTRAP_FAILED_BEFORE_FIT',
        traceback=traceback.format_exc(),outer_scoring=False,fairness_closed=False),indent=2))
    diagnostic=Path('/kaggle/working/batteryscope_nfr1_remaining19_cpu_v1_results.zip')
    with zipfile.ZipFile(diagnostic,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(failure,'bootstrap_failure.json')
        archive.write(WORKSPACE/'remaining_delivery.json','delivery_manifest.snapshot.json')
    display(FileLink(str(diagnostic)))
    raise
print('INNER BUNDLE:',bundle)
manifest=json.loads((WORKSPACE/'remaining_delivery.json').read_text())
print('LOCKED QUEUE:',len(manifest['jobs']),'jobs: 13 fits + 6 reference jobs')
print('No raw data, neural jobs, outer scoring, Windows/Kaggle parity claim, or automatic resume.')
'''

RUN = '''# CPU only. Synthetic gate -> sequential whole jobs -> exact fresh-process replay.
# Failure or insufficient resources stops; never bypass a gate.
from IPython.display import FileLink,display
output_zip=Path('/kaggle/working/batteryscope_nfr1_remaining19_cpu_v1_results.zip')
try:
    child=subprocess.Popen([sys.executable,'-B','-X','utf8',str(WORKSPACE/'remaining_runtime.py'),
                            'campaign','--bundle',str(bundle)],cwd=WORKSPACE,env=os.environ.copy(),
                            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    try:
        for line in child.stdout: print(line,end='',flush=True)
        returncode=child.wait()
    except BaseException:
        run_dir=WORKSPACE/'output/batteryscope_nfr1_remaining19_cpu_v1'
        if run_dir.exists(): (run_dir/'STOP').touch()
        try:
            child.wait(timeout=30)  # Allow guarded shutdown and failure export first.
        except subprocess.TimeoutExpired:
            pass
        import psutil
        if child.poll() is None:
            owner=psutil.Process(child.pid)
            processes=owner.children(recursive=True)+[owner]
            for p in processes:
                try: p.terminate()
                except psutil.NoSuchProcess: pass
            _,alive=psutil.wait_procs(processes,timeout=5)
            for p in alive:
                try: p.kill()
                except psutil.NoSuchProcess: pass
        raise
    assert returncode==0, 'Run incomplete/failed. Download the ZIP; do not change guards or rerun in this session.'
finally:
    if output_zip.is_file(): display(FileLink(str(output_zip)))
summary_path=WORKSPACE/'output/batteryscope_nfr1_remaining19_cpu_v1/summary.json'
print(summary_path.read_text())
'''


def cell(kind, source):
    result = dict(cell_type=kind, metadata={}, source=source.splitlines(keepends=True))
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def build(refresh=False):
    files, spec = payload()
    raw = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    blob = base64.b64encode(zlib.compress(raw, level=9)).decode()
    code = BOOTSTRAP.replace("__BLOB__", repr(blob)).replace("__SHA__", repr(hashlib.sha256(raw).hexdigest()))
    ast.parse(code); ast.parse(RUN)
    notebook = dict(nbformat=4, nbformat_minor=5,
        metadata=dict(kernelspec=dict(display_name="Python 3", language="python", name="python3"),
                      language_info=dict(name="python"), kaggle=dict(accelerator="none", isInternetEnabled=False)),
        cells=[cell("markdown", "# BatteryScope — 19 jobs còn lại (Kaggle CPU)\n\n"
            "Mount **batteryscope-corrected-inner-v2-kaggle** đã dùng. **Accelerator: None**, Internet off.\n"
            "Notebook chứa code; không cần ZIP code, raw dataset hay kết quả local. Fresh session → Run All.\n\n"
            "13 classical fits + 6 reference jobs; giữ nguyên 28 jobs local. Đây là continuation hậu nghiệm "
            "ở môi trường mới, **không** CPU/Kaggle parity hoặc fairness closure. Mismatch → trả ZIP, không pip install/bypass.\n"),
               cell("code", code), cell("code", RUN),
               cell("markdown", "## Bàn giao\n\nTải `batteryscope_nfr1_remaining19_cpu_v1_results.zip` "
                    "về đúng thư mục notebook này, kể cả run FAILED/partial. Chỉ ghép với 28 jobs local "
                    "sau khi audit hashes, roles, support, environment và replay; không sửa expected hoặc chọn model bằng outer labels.\n")])
    output = OUT / NOTEBOOK
    receipt_path = OUT / "build_receipt.json"
    if output.exists() or receipt_path.exists():
        require(refresh and output.is_file() and receipt_path.is_file(), "REFUSE_OVERWRITE_DELIVERY")
        old = json.loads(output.read_text(encoding="utf-8"))
        previous = json.loads(receipt_path.read_text())
        require(previous["status"] == "CODE_BUILT_LOCAL_QA_PENDING_KAGGLE_NOT_RUN" and
                all(c.get("execution_count") is None and not c.get("outputs") for c in old["cells"]
                    if c["cell_type"] == "code"), "REFUSE_REFRESH_ACCEPTED_OR_EXECUTED_NOTEBOOK")
    OUT.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    receipt = dict(status="CODE_BUILT_LOCAL_QA_PENDING_KAGGLE_NOT_RUN", date="2026-10-10",
        notebook=binding(output), embedded_payload_sha256=hashlib.sha256(raw).hexdigest(),
        builder=binding(Path(__file__)), local_attestation=spec["local_attestation"],
        job_ids=[j["id"] for j in spec["jobs"]], model_fits_executed_by_builder=0,
        data_or_checkpoint_embedded=False, actual_kaggle_execution=False, outer_scoring=False, fairness_closed=False)
    (OUT / "build_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(notebook=str(output), bytes=output.stat().st_size, jobs=19,
                         real_classical_fits=13, references=6, actual_kaggle_execution=False)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-unreleased", action="store_true")
    build(parser.parse_args().refresh_unreleased)
