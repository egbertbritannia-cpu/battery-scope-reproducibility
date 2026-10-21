"""Bounded CPU-only continuation; immutable classical helpers, no neural/outer API."""
from __future__ import annotations

import argparse
import hashlib
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

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
MANIFEST = "remaining_delivery.json"
RUN_NAME = "batteryscope_nfr1_remaining19_cpu_v1"


def require(ok, message):
    if not ok:
        raise RuntimeError("NFR1_REMAINING_CPU:" + message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_manifest():
    m = json.loads((ROOT / MANIFEST).read_text(encoding="utf-8"))
    require(m["schema"] == "nfr1-kaggle-remaining19-cpu-v1", "DELIVERY_SCHEMA")
    for name, record in m["files"].items():
        p = (ROOT / name).resolve()
        require(p.is_relative_to(ROOT) and p.is_file() and p.stat().st_size == record["bytes"]
                and sha(p) == record["sha256"], "CODE_HASH:" + name)
    validate_jobs(m)
    return m


def validate_jobs(m):
    from protocol_and_provenance.neural_classical_fairness_v1.runtime import local_cpu_pilots as local
    all_jobs = local.jobs()
    completed = m["local_completed_job_ids"]
    require(len(completed) == len(set(completed)) == 28, "LOCAL_COMPLETION_COUNT")
    require(set(completed) <= {j["id"] for j in all_jobs}, "UNKNOWN_LOCAL_JOB")
    expected = [j for j in all_jobs if j["id"] not in completed]
    require(m["jobs"] == expected and len(expected) == 19, "REMAINING_QUEUE_DRIFT")
    require(sum(j["kind"] == "classical" for j in expected) == 13 and
            sum(j["kind"] == "references" for j in expected) == 6 and
            all(j["kind"] in ("classical", "references") and j["inner_fold"] == 0 and
                j["seed"] == 2026 for j in expected), "REMAINING_SCOPE")
    require(m["classical_grid"] == local.grid() and
            m["prediction_policies"] == local.CLASSICAL_PREDICTION_POLICIES, "RECIPE_DRIFT")
    require(m["outer_scoring"] is False and m["fairness_closed"] is False, "SCORING_DISABLED")


def environment_gate(m):
    from protocol_and_provenance.neural_classical_fairness_v1.runtime.environment_identity import collect
    actual = collect()
    actual["python"] = platform.python_version()
    actual["platform"] = platform.platform()
    actual["cpu_count"] = os.cpu_count()
    import joblib
    import threadpoolctl
    actual["joblib_version"] = joblib.__version__
    actual["threadpoolctl_version"] = threadpoolctl.__version__
    actual["threadpools"] = threadpoolctl.threadpool_info()
    mismatches = {k: dict(expected=v, actual=actual["packages"].get(k))
                  for k, v in m["expected_packages"].items() if actual["packages"].get(k) != v}
    if actual["python"] != m["expected_python"]:
        mismatches["python"] = dict(expected=m["expected_python"], actual=actual["python"])
    # Torch is only an imported dependency of frozen helpers: no tensor fit/device use.
    actual["role"] = "NEW_LINUX_CPU_CLASSICAL_CONTINUATION_NOT_WINDOWS_PARITY"
    actual["torch_used_for_training"] = False
    return actual, mismatches


def input_checks(m, bundle):
    bundle = Path(bundle).resolve()
    mappings = []
    for name, record in m["input_files"].items():
        path = (bundle / name).resolve()
        require(path.is_relative_to(bundle) and path.is_file() and
                path.stat().st_size == record["bytes"] and sha(path) == record["sha256"], "INPUT_HASH:" + name)
        mappings.append(dict(bundle_relative=name, mounted_path=str(path), **record))
    return mappings


def configure(bundle, run, m):
    from protocol_and_provenance.neural_classical_fairness_v1.runtime import local_cpu_pilots as local
    bundle, run = Path(bundle).resolve(), Path(run).resolve()
    require(run == ROOT / "output" / RUN_NAME, "OUTPUT_PATH")
    # Local helpers are unchanged. Only input paths are mapped into the sealed mount.
    local.BUNDLE = bundle
    local.OUTER = ROOT / "phase2-ouput/batteryscope_stage7_programmatic_v6/stage8_outer_model_executor.py"
    return local


def resources(run):
    import psutil
    return dict(available_ram_bytes=psutil.virtual_memory().available,
                free_disk_bytes=shutil.disk_usage(run).free)


def can_start(snapshot):
    return snapshot["available_ram_bytes"] >= 6 * 2**30 and snapshot["free_disk_bytes"] >= 10 * 2**30


def synthetic_gate(local):
    """Two synthetic fits, no battery-label reads; not added to the 13 real fits."""
    import joblib
    import numpy as np
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
    import tempfile
    rng = np.random.default_rng(2026)
    x = rng.normal(size=(96, 8)).astype(np.float32)
    y = rng.random(96)
    checks = []
    for family, model in (("extra_trees", ExtraTreesRegressor(n_estimators=128, n_jobs=2, random_state=2026)),
                          ("hist_gradient_boosting", HistGradientBoostingRegressor(max_iter=3, early_stopping=False, random_state=2026))):
        model.fit(x, y)
        params = model.get_params().copy()
        pred = local.classical_predict(model, x, family)
        require(np.array_equal(pred[:16], local.classical_predict(model, x[:16], family)), "SYNTHETIC_BATCH_DRIFT")
        require(all(np.array_equal(pred, local.classical_predict(model, x, family)) for _ in range(4)), "SYNTHETIC_REPEAT_DRIFT")
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            path = Path(tmp) / "model.joblib"
            joblib.dump(model, path)
            require(np.array_equal(pred, local.classical_predict(joblib.load(path), x, family)), "SYNTHETIC_RELOAD_DRIFT")
        require(params == model.get_params(), "SYNTHETIC_PARAMS_DRIFT")
        checks.append(family)
    return dict(status="SYNTHETIC_PREDICT_RELOAD_PASS", families=checks, synthetic_fits=2,
                battery_fits=0, full_dataset_replay=False)


def worker(mode, bundle, run, job_id):
    m = read_manifest()
    local = configure(bundle, run, m)
    job = next((j for j in m["jobs"] if j["id"] == job_id), None)
    require(job is not None, "JOB_NOT_ASSIGNED")
    session = json.loads((run / "session_started.json").read_text())
    require(sha(ROOT / MANIFEST) == session["delivery_sha256"], "SESSION_DELIVERY_DRIFT")
    expected = json.loads((run / "environment.lock.json").read_text())
    actual, mismatches = environment_gate(m)
    require(not mismatches and actual == expected, "WORKER_ENVIRONMENT_DRIFT")
    require(sha(ROOT / "protocol_and_provenance/neural_classical_fairness_v1/source_bindings.json") ==
            session["remapped_bindings_sha256"], "MAPPING_HASH")
    input_checks(m, bundle)
    directory = run / "jobs" / job_id
    started = time.monotonic()
    try:
        if mode == "replay":
            local.replay(job, directory)
            return
        directory.mkdir(parents=True, exist_ok=False)
        local.write(directory / "started.json", dict(job=job, pid=os.getpid(), delivery_sha256=sha(ROOT / MANIFEST)))
        guard = local.Guard(run, directory, seconds=3600)
        result = local.classical_job(job, run, directory, guard)
        result.update(wall_seconds=time.monotonic()-started, peak_sampled_rss_bytes=guard.peak,
                      execution_host="KAGGLE_LINUX_CPU", production_fit=False, outer_scoring=False, fairness_closed=False)
        local.write(directory / "result.json", result)
    except Exception as exc:
        local.write(directory / ("replay_failure.json" if mode == "replay" else "failure.json"),
                    dict(error=str(exc), traceback=traceback.format_exc(), outer_scoring=False, fairness_closed=False))
        raise


def terminate_owned(child, birth):
    import psutil
    try:
        process = psutil.Process(child.pid)
        require(process.create_time() == birth, "OWNED_PID_REUSED")
        owned = process.children(recursive=True) + [process]
    except psutil.NoSuchProcess:
        return
    for p in reversed(owned):
        try:
            p.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(owned, timeout=5)
    for p in alive:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
    _, survivors = psutil.wait_procs(alive, timeout=5)
    require(not survivors, "OWNED_PROCESS_SURVIVORS")


def child_stage(job, mode, bundle, run, deadline):
    import psutil
    started = time.monotonic()
    deadline = min(deadline, started + (3600 if mode == "job" else 300))
    peak = 0
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
               CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
    with (run / (job["id"] + "." + mode + ".log")).open("x", encoding="utf-8") as log:
        child = subprocess.Popen([sys.executable, "-B", "-X", "utf8", str(ROOT / "remaining_runtime.py"),
                                  mode, "--bundle", str(bundle), "--run", str(run), "--job-id", job["id"]],
                                 cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        process = psutil.Process(child.pid)
        birth = process.create_time()
        try:
            while child.poll() is None:
                try:
                    owned = [process] + process.children(recursive=True)
                except psutil.NoSuchProcess:
                    child.wait(timeout=10)
                    break
                rss = 0
                for p in owned:
                    try:
                        rss += p.memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
                peak = max(peak, rss)
                snapshot = resources(run)
                write(run / "heartbeat.json", dict(state="RUNNING", job_id=job["id"], mode=mode,
                    worker_pid=child.pid, worker_create_time=birth, wall_time_unix=time.time(),
                    elapsed_seconds=time.monotonic()-started, peak_tree_rss_bytes=peak, resource=snapshot))
                require(not (run / "STOP").exists(), "AUTHOR_STOP")
                require(time.monotonic() < deadline, "TIME_GUARD")
                require(rss <= 4.5*2**30 and snapshot["available_ram_bytes"] >= 1.5*2**30, "RAM_GUARD")
                require(snapshot["free_disk_bytes"] >= 8*2**30, "DISK_GUARD")
                time.sleep(2)
            require(child.returncode == 0, "WORKER_FAILED:"+job["id"]+":"+mode)
        finally:
            terminate_owned(child, birth)
            child.wait(timeout=10)
    require(peak * 1.3 <= 4.5*2**30, "MEMORY_MARGIN")
    return dict(mode=mode, seconds=time.monotonic()-started, peak_tree_rss_bytes=peak)


def export(run):
    target = ROOT.parent / (RUN_NAME + "_results.zip")
    # Exclude mounted input, code workspace, caches and original local artifacts.
    files = sorted(p for p in run.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    inventory = {p.relative_to(run).as_posix(): dict(bytes=p.stat().st_size, sha256=sha(p)) for p in files}
    write(run / "output_inventory.json", dict(files=inventory, outer_scoring=False, fairness_closed=False))
    files.append(run / "output_inventory.json")
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, RUN_NAME + "/" + p.relative_to(run).as_posix())
    print("DOWNLOAD RESULTS / FAILURE DIAGNOSTICS:", target, flush=True)
    return target


def run_campaign(bundle):
    run = ROOT / "output" / RUN_NAME
    require(not run.exists(), "USE_FRESH_SESSION_NO_RESUME")
    run.mkdir(parents=True)
    started = time.monotonic()
    ledger, error, m = [], None, None
    try:
        m = read_manifest()
        write(run / "delivery_manifest.snapshot.json", m)
        local = configure(bundle, run, m)
        actual, mismatches = environment_gate(m)
        write(run / "environment_check.json", dict(actual=actual, mismatches=mismatches))
        require(not mismatches, "ENVIRONMENT_MISMATCH:" + json.dumps(mismatches))
        write(run / "environment.lock.json", actual)
        mappings = input_checks(m, bundle)
        write(run / "input_mapping.json", mappings)
        bindings = m["source_bindings"]
        prefix = "phase2-ouput/batteryscope_corrected_inner_v2_kaggle/batteryscope_corrected_inner_v2/"
        remapped = dict(config=bindings["config"], inputs=[])
        for record in bindings["inputs"]:
            require(record["path"].startswith(prefix), "INPUT_MAPPING_PREFIX")
            name = record["path"][len(prefix):]
            require(name in m["input_files"], "INPUT_MAPPING_UNBOUND")
            remapped["inputs"].append(dict(record, path=str((Path(bundle) / name).resolve())))
        mapping_path = ROOT / "protocol_and_provenance/neural_classical_fairness_v1/source_bindings.json"
        write(mapping_path, remapped)
        write(run / "input_bindings.remapped.json", remapped)
        snapshot = resources(run)
        write(run / "preflight.json", snapshot)
        require(can_start(snapshot), "PREFLIGHT_RESOURCE")
        write(run / "synthetic_gate.json", synthetic_gate(local))
        # Metadata-only role seals before real target reads.
        sys.path.insert(0, str(Path(bundle) / "code"))
        ref = local.load_module("remaining_metadata_reference", Path(bundle) / "code/smoke_reference_runtime.py")
        config = json.loads((Path(bundle) / "inner_config.frozen.json").read_text())
        recipe = json.loads((Path(bundle) / "code/inner_recipe.json").read_text())
        ns = ref.make_namespace(Path(bundle), recipe, config, metadata_only=True)
        roles = []
        for source in ("CALB", "CALCE"):
            expected = next(c for c in config["inner_contexts"] if c["source"] == source and c["inner_fold"] == 0)
            payload = ref.validate_inner_roles(ns, source, 0, expected)
            roles.append(dict(source=source, role_hashes=expected, train_cells=len(payload["selection_train"]),
                              validation_cells=len(payload["early_validation"])))
        write(run / "role_seals.json", roles)
        write(run / "session_started.json", dict(jobs=m["jobs"], pid=os.getpid(),
            delivery_sha256=sha(ROOT / MANIFEST), remapped_bindings_sha256=sha(mapping_path),
            local_completed_job_ids=m["local_completed_job_ids"], local_attestation=m["local_attestation"],
            cross_host_parity_claim=False, outer_scoring=False, fairness_closed=False))
        deadline = started + 8*3600 - 1800
        for job in m["jobs"]:
            snapshot = resources(run)
            if not can_start(snapshot):
                ledger.append(dict(**job, state="DEFERRED_RESOURCE", resource=snapshot))
                break
            if time.monotonic()+1.3*3900 >= deadline:
                ledger.append(dict(**job, state="DEFERRED_TIME"))
                break
            write(run / "current_job.json", dict(state="RUNNING", job=job))
            print("[REMAINING CPU START]", job["id"], flush=True)
            t = time.monotonic()
            try:
                executions = [child_stage(job, "job", bundle, run, deadline)]
                result = json.loads((run / "jobs" / job["id"] / "result.json").read_text())
                require(result["status"] in ("PILOT_TRAIN_COMPLETE_RELOAD_PENDING", "REFERENCE_EXECUTIONS_COMPLETE", "NOT_EVALUABLE"), "RESULT_STATUS")
                if result["status"] == "PILOT_TRAIN_COMPLETE_RELOAD_PENDING":
                    executions.append(child_stage(job, "replay", bundle, run, deadline))
                    receipt = json.loads((run / "jobs" / job["id"] / "replay.json").read_text())
                    require(receipt["status"] == "FRESH_PROCESS_FORWARD_FIXTURE_EXACT_PASS", "REPLAY_GATE")
                ledger.append(dict(**job, state="NOT_EVALUABLE" if result["status"] == "NOT_EVALUABLE" else "COMPLETE",
                    wall_seconds=time.monotonic()-t, processes=executions, execution_host="KAGGLE_LINUX_CPU",
                    learned_fits=1 if job["kind"] == "classical" and result["status"] != "NOT_EVALUABLE" else 0,
                    reference_executions=result.get("reference_executions", 0)))
                write(run / "compute_ledger.json", ledger)
                print("[REMAINING CPU COMPLETE]", job["id"], round(time.monotonic()-t, 1), flush=True)
            except Exception as exc:
                ledger.append(dict(**job, state="FAILED", error=str(exc)))
                raise
    except BaseException as exc:
        error = str(exc) or type(exc).__name__
        write(run / "session_failure.json", dict(error=error, traceback=traceback.format_exc()))
    finally:
        if m:
            recorded = {j["id"] for j in ledger}
            ledger.extend(dict(**j, state="NOT_STARTED_AFTER_STOP") for j in m["jobs"] if j["id"] not in recorded)
        write(run / "compute_ledger.json", ledger)
        complete = sum(j["state"] == "COMPLETE" for j in ledger)
        status = "KAGGLE_REMAINING19_COMPLETE_REVIEW_REQUIRED" if error is None and complete == 19 else "KAGGLE_REMAINING19_PARTIAL_REVIEW_REQUIRED"
        summary = dict(status=status, completed_jobs=complete, planned_jobs=19,
            completed_classical_fits=sum(j.get("learned_fits", 0) for j in ledger),
            reference_executions=sum(j.get("reference_executions", 0) for j in ledger),
            wall_seconds=time.monotonic()-started, error=error, execution_host="KAGGLE_LINUX_CPU",
            cross_host_parity_claim=False, automatic_retry=False, outer_scoring=False, fairness_closed=False)
        write(run / "summary.json", summary)
        write(run / "heartbeat.json", dict(state="SUPERVISOR_TERMINAL", **summary, finished_unix=time.time()))
        write(run / "current_job.json", dict(state="SUPERVISOR_TERMINAL", status=status))
        export(run)
        print(json.dumps(summary, indent=2), flush=True)
    return 0 if status == "KAGGLE_REMAINING19_COMPLETE_REVIEW_REQUIRED" else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("campaign", "job", "replay"))
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--run", type=Path)
    p.add_argument("--job-id")
    a = p.parse_args()
    if a.mode == "campaign":
        sys.exit(run_campaign(a.bundle.resolve()))
    require(a.run is not None and a.job_id is not None, "WORKER_ARGUMENTS")
    worker(a.mode, a.bundle, a.run.resolve(), a.job_id)
