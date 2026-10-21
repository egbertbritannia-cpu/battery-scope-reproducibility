"""Fresh six-fit GPU timing pilot; no production CLI, outer, retries or resume."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from protocol_and_provenance.neural_classical_fairness_v1.runtime import local_cpu_pilots as local
from protocol_and_provenance.neural_classical_fairness_v1.runtime import f2_trajectory as engine
from protocol_and_provenance.neural_classical_fairness_v1.runtime import kaggle_classical_remaining as inherited

RUN_NAME = "batteryscope_nfr1_f2_gpu_timing_v1"
MANIFEST = "f2_pilot_delivery.json"
DEVICE = "cuda:0"


def jobs():
    return [dict(id=f"pilot.CALB.0.{a}.{c}", source="CALB", inner_fold=0, seed=2026,
                 architecture=a, candidate=c, epochs=3) for a in ("TCN", "GRU") for c in engine.selection.PRIORITY]


def validate_spec(m):
    local.require(m["schema"] == "nfr1-f2-gpu-timing-v1" and m["jobs"] == jobs(), "F2_DELIVERY_QUEUE")
    local.require(m["candidate_recipes"] == engine.selection.candidate_recipes() and m["production_epoch_cap"] == 30 and
                  m["device"] == DEVICE and m["amp"] is False and m["dtype"] == "float32" and
                  m["validation_scope"] == "full_eligible_inner", "F2_RECIPE_DRIFT")
    local.require(m["guards"] == dict(production=False, outer_scoring=False, fairness_closed=False,
        cross_fold_candidate_selection=False, automatic_resume=False, automatic_retry=False), "F2_SCOPE_DRIFT")
    local.require(m["budget"] == dict(wall_seconds=21600, reserve_seconds=1800, job_seconds=3600,
        replay_seconds=300, synthetic_seconds=900, workers=1), "F2_BUDGET_DRIFT")


def manifest():
    m = json.loads((ROOT/MANIFEST).read_text())
    validate_spec(m)
    for name, record in m["files"].items():
        p = (ROOT/name).resolve()
        local.require(p.is_relative_to(ROOT) and p.is_file() and p.stat().st_size == record["bytes"] and
                      local.sha(p) == record["sha256"], "F2_CODE_HASH:"+name)
    return m


def configure(bundle):
    local.BUNDLE = Path(bundle).resolve()
    local.GRU = ROOT / "protocol_and_provenance/data_integrity/e06_gru_model.py"
    local.OUTER = ROOT / "phase2-ouput/batteryscope_stage7_programmatic_v6/stage8_outer_model_executor.py"
    import torch
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False


def environment(m):
    import torch
    from protocol_and_provenance.neural_classical_fairness_v1.runtime.environment_identity import collect
    identity = collect()
    actual = dict(python=platform.python_version(), numpy=identity["packages"].get("numpy"),
        pandas=identity["packages"].get("pandas"), pyarrow=identity["packages"].get("pyarrow"),
        scikit_learn=identity["packages"].get("scikit-learn"), torch=str(torch.__version__),
        cuda_available=torch.cuda.is_available(), torch_cuda_version=torch.version.cuda,
        cudnn_version=torch.backends.cudnn.version(),
        gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    mismatches = {k:dict(expected=v,actual=actual.get(k)) for k,v in m["expected_environment"].items() if actual.get(k) != v}
    count = torch.cuda.device_count()
    if count not in (1,2):
        mismatches["gpu_count"] = dict(expected="1 or 2; cuda:0 only", actual=count)
    return dict(identity=identity, numerical_device_profile=actual, gpu_count=count, device=DEVICE,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_benchmark=torch.backends.cudnn.benchmark, amp=False, tf32=False, threads=2, interop_threads=1), mismatches


def definitions(bundle):
    # The frozen reference imports sibling modules. Fresh QA/replay workers
    # must resolve them from the mounted, hash-verified code directory.
    sys.path.insert(0, str(Path(bundle).resolve()/"code"))
    ref = local.load_module("f2_bound_reference", Path(bundle)/"code/smoke_reference_runtime.py")
    return ref.make_namespace(Path(bundle), json.loads((Path(bundle)/"code/inner_recipe.json").read_text()),
                              json.loads((Path(bundle)/"inner_config.frozen.json").read_text()))


class Guard(local.Guard):
    def check(self):
        super().check()
        import torch
        free, total = torch.cuda.mem_get_info(DEVICE)
        local.require(free >= 2**30, "F2_FREE_VRAM_GUARD")


def worker(mode, bundle, run, job_id):
    m = manifest(); configure(bundle)
    expected = json.loads((run/"environment.lock.json").read_text())
    actual, mismatches = environment(m)
    local.require(not mismatches and actual == expected, "F2_WORKER_ENVIRONMENT_DRIFT")
    session = json.loads((run/"session_started.json").read_text())
    local.require(local.sha(ROOT/MANIFEST) == session["delivery_sha256"] and
        local.sha(ROOT/"protocol_and_provenance/neural_classical_fairness_v1/source_bindings.json") == session["bindings_sha256"], "F2_SESSION_SEAL")
    inherited.input_checks(m,bundle)
    if mode == "qa":
        engine.synthetic_prefix_qa(definitions(bundle), run/"synthetic_qa", DEVICE)
        return
    job = next((j for j in m["jobs"] if j["id"] == job_id), None)
    local.require(job is not None, "F2_JOB_NOT_ASSIGNED")
    directory = run/"jobs"/job_id
    guard = Guard(run, directory, seconds=300 if mode == "replay" else 3600)
    if mode == "replay":
        engine.replay(definitions(bundle),job,directory/"trajectory",guard,DEVICE)
        return
    directory.mkdir(parents=True,exist_ok=False)
    start = time.monotonic()
    local.write(directory/"started.json",dict(job=job,delivery_sha256=local.sha(ROOT/MANIFEST)))
    ns, train, val, cycle, scaler, *_ = local.verified_namespace(job,guard,directory,cycles=True)
    preprocessing_seconds = time.monotonic()-start
    support = dict(all_rows=len(val))
    import numpy as np
    for t in engine.selection.TARGETS:
        ts = ns["TARGET_SPECS"][t]
        support[t] = int((ns["normalize_bool"](val[ts["mask_col"]],ts["mask_col"]) & np.isfinite(val[ts["y_col"]].to_numpy(float))).sum())
        local.require(support[t] > 0 and scaler["targets"][t]["active"], "F2_INACTIVE_ENDPOINT")
    local.write(directory/"expected_support.json",support)
    result = engine.run_trajectory(ns,job,ns["MultitaskCycleWindowIterator"](train,cycle,scaler),
        ns["MultitaskCycleWindowIterator"](val,cycle,scaler),scaler,support,guard,directory/"trajectory",device=DEVICE,epochs=3)
    result.update(preprocessing_seconds=preprocessing_seconds, wall_seconds=time.monotonic()-start,
                  peak_sampled_rss_bytes=guard.peak, execution_host="KAGGLE_GPU_T4_TIMING_PILOT")
    local.write(directory/"result.json",result)


def child(mode,bundle,run,job_id,deadline):
    import psutil
    seconds = 900 if mode == "qa" else 300 if mode == "replay" else 3600
    deadline = min(deadline,time.monotonic()+seconds)
    peak, started = 0, time.monotonic()
    name = job_id or "synthetic_qa"
    with (run/(name+"."+mode+".log")).open("x",encoding="utf-8") as log:
        command = [sys.executable,"-B","-X","utf8",str(Path(__file__)),mode,"--bundle",str(bundle),"--run",str(run)]
        if job_id:
            command += ["--job-id",job_id]
        process = subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
            env=dict(os.environ,OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",
                     CUBLAS_WORKSPACE_CONFIG=":4096:8",PYTHONDONTWRITEBYTECODE="1"),
            creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
        owner = psutil.Process(process.pid); birth = owner.create_time()
        try:
            while process.poll() is None:
                try:
                    owned = [owner]+owner.children(recursive=True)
                except psutil.NoSuchProcess:
                    process.wait(timeout=10); break
                rss = 0
                for p in owned:
                    try: rss += p.memory_info().rss
                    except psutil.NoSuchProcess: pass
                peak = max(peak,rss)
                snapshot = inherited.resources(run)
                local.write(run/"heartbeat.json",dict(state="RUNNING",mode=mode,job_id=job_id,worker_pid=process.pid,
                    worker_create_time=birth,peak_tree_rss_bytes=peak,resource=snapshot,wall_time_unix=time.time()))
                local.require(not (run/"STOP").exists(),"F2_AUTHOR_STOP")
                local.require(time.monotonic() < deadline,"F2_TIME_GUARD")
                local.require(rss <= 4.5*2**30 and snapshot["available_ram_bytes"] >= 1.5*2**30 and
                              snapshot["free_disk_bytes"] >= 8*2**30,"F2_RAM_DISK_GUARD")
                time.sleep(2)
            local.require(process.returncode == 0,"F2_WORKER_FAILED:"+name+":"+mode)
        finally:
            inherited.terminate_owned(process,birth); process.wait(timeout=10)
    local.require(peak*1.3 <= 4.5*2**30,"F2_MEMORY_MARGIN")
    return dict(mode=mode,seconds=time.monotonic()-started,peak_tree_rss_bytes=peak)


def export(run):
    target = ROOT.parent/(RUN_NAME+"_results.zip")
    files = sorted(p for p in run.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    local.write(run/"output_inventory.json",dict(files={p.relative_to(run).as_posix():dict(bytes=p.stat().st_size,
        sha256=local.sha(p)) for p in files},outer_scoring=False,fairness_closed=False))
    with zipfile.ZipFile(target,"w",compression=zipfile.ZIP_DEFLATED) as z:
        for p in files+[run/"output_inventory.json"]:
            z.write(p,RUN_NAME+"/"+p.relative_to(run).as_posix())
    print("DOWNLOAD F2 PILOT RESULTS / FAILURE DIAGNOSTICS:",target,flush=True)


def run(bundle):
    run = ROOT/"output"/RUN_NAME
    local.require(not run.exists(),"F2_FRESH_SESSION_NO_RESUME")
    run.mkdir(parents=True); started = time.monotonic(); m, error, ledger = None, None, []
    try:
        m=manifest(); configure(bundle)
        local.write(run/"delivery_manifest.snapshot.json",m)
        actual,mismatches=environment(m)
        local.write(run/"environment_check.json",dict(actual=actual,mismatches=mismatches))
        local.require(not mismatches,"F2_ENVIRONMENT_MISMATCH:"+json.dumps(mismatches))
        local.write(run/"environment.lock.json",actual)
        mapped=inherited.input_checks(m,bundle); local.write(run/"input_mapping.json",mapped)
        remapped=dict(config=m["source_bindings"]["config"],inputs=[])
        prefix="phase2-ouput/batteryscope_corrected_inner_v2_kaggle/batteryscope_corrected_inner_v2/"
        for r in m["source_bindings"]["inputs"]:
            local.require(r["path"].startswith(prefix),"F2_INPUT_PREFIX")
            name=r["path"][len(prefix):]
            local.require(name in m["input_files"],"F2_INPUT_UNBOUND")
            remapped["inputs"].append(dict(r,path=str((bundle/name).resolve())))
        path=ROOT/"protocol_and_provenance/neural_classical_fairness_v1/source_bindings.json"
        local.write(path,remapped); local.write(run/"input_bindings.remapped.json",remapped)
        snapshot=inherited.resources(run); local.write(run/"preflight.json",snapshot)
        local.require(inherited.can_start(snapshot),"F2_RESOURCE_PREFLIGHT")
        local.write(run/"session_started.json",dict(jobs=m["jobs"],delivery_sha256=local.sha(ROOT/MANIFEST),
            bindings_sha256=local.sha(path),outer_scoring=False,production=False,fairness_closed=False))
        deadline=started+m["budget"]["wall_seconds"]-m["budget"]["reserve_seconds"]
        qa=child("qa",bundle,run,None,deadline); local.write(run/"synthetic_qa_process.json",qa)
        receipt=json.loads((run/"synthetic_qa/summary.json").read_text())
        local.require(receipt["status"] == "ACTUAL_ENGINE_TCN_GRU_PREFIX_RELOAD_SYNTHETIC_PASS","F2_QA_GATE")
        for job in m["jobs"]:
            snapshot=inherited.resources(run)
            local.require(inherited.can_start(snapshot),"F2_START_RESOURCE:"+json.dumps(snapshot))
            local.require(time.monotonic()+1.3*3900 < deadline,"F2_TIME_MARGIN")
            local.write(run/"current_job.json",dict(state="RUNNING",job=job))
            print("[F2 PILOT START]",job["id"],flush=True)
            entry=dict(job=job,state="WORKER_STARTED",processes=[],learned_fits=0,epoch_passes=0)
            ledger.append(entry); local.write(run/"compute_ledger.json",ledger)
            try:
                entry["processes"].append(child("job",bundle,run,job["id"],deadline))
                entry["processes"].append(child("replay",bundle,run,job["id"],deadline))
                entry.update(state="COMPLETE",learned_fits=1,epoch_passes=3)
            except BaseException as exc:
                entry.update(state="FAILED_OR_PARTIAL",error=str(exc))
                raise
            local.write(run/"compute_ledger.json",ledger)
            print("[F2 PILOT COMPLETE]",job["id"],flush=True)
    except BaseException as exc:
        error=str(exc) or type(exc).__name__
        local.write(run/"failure.json",dict(error=error,traceback=traceback.format_exc()))
    finally:
        if m:
            complete={r["job"]["id"] for r in ledger}
            ledger.extend(dict(job=j,state="NOT_ACCEPTED_AFTER_STOP",learned_fits=0) for j in m["jobs"] if j["id"] not in complete)
        local.write(run/"compute_ledger.json",ledger)
        n=sum(r["state"]=="COMPLETE" for r in ledger)
        initializations=len(list((run/"jobs").glob("*/trajectory/trajectory_started.json")))
        saved_epochs=len(list((run/"jobs").glob("*/trajectory/checkpoint_epoch_*.pt")))
        synthetic_initializations=len(list((run/"synthetic_qa").glob("*/trajectory_started.json")))
        summary=dict(status="F2_GPU_TIMING_PILOT_COMPLETE_REVIEW_REQUIRED" if n==6 and error is None else "F2_GPU_TIMING_PILOT_PARTIAL_REVIEW_REQUIRED",
            completed_jobs=n,planned_jobs=6,accepted_pilot_fits=n,accepted_epoch_passes=n*3,error=error,
            actual_battery_fit_initializations=initializations,saved_battery_epoch_checkpoints=saved_epochs,
            synthetic_fit_initializations_separate=synthetic_initializations,
            wall_seconds=time.monotonic()-started,production=False,candidate_selected=False,outer_scoring=False,fairness_closed=False)
        local.write(run/"summary.json",summary)
        local.write(run/"heartbeat.json",dict(state="SUPERVISOR_TERMINAL",**summary))
        local.write(run/"current_job.json",dict(state="SUPERVISOR_TERMINAL",status=summary["status"]))
        export(run); print(json.dumps(summary,indent=2),flush=True)
    return 0 if error is None and n==6 else 1


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("pilot","qa","job","replay"))
    parser.add_argument("--bundle",type=Path,required=True)
    parser.add_argument("--run",type=Path)
    parser.add_argument("--job-id")
    args=parser.parse_args()
    if args.mode == "pilot":
        sys.exit(run(args.bundle.resolve()))
    local.require(args.run is not None and args.run.resolve() == ROOT/"output"/RUN_NAME,"F2_WORKER_OUTPUT_PATH")
    worker(args.mode,args.bundle.resolve(),args.run.resolve(),args.job_id)
