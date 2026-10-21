"""Read-only returned-artifact audit. No model deserialization, fits or outer scoring."""
from __future__ import annotations

import ast
import base64
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import sys
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
FOLDER = ROOT / "notebook/neural-classical-fairness-v1/remaining19-cpu-v1"
LOCAL = ROOT / "paper/audits/neural_classical_fairness_2026-10-08_v1/local_cpu_dev_v2"
OUTPUT = ROOT / "paper/audits/neural_classical_fairness_2026-10-08_v1/remaining19_cpu_return_review"
PREFIX = "batteryscope_nfr1_remaining19_cpu_v1/"


def require(ok, reason):
    if not ok:
        raise RuntimeError("REMAINING_RETURN_AUDIT:" + reason)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def binding(path):
    from protocol_and_provenance.neural_classical_fairness_v1.runtime.local_cpu_pilots import sha
    return dict(bytes=path.stat().st_size, sha256=sha(path))


def strict_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "DUPLICATE_JSON_KEY:" + key)
            result[key] = value
        return result
    def reject(value):
        raise RuntimeError("REMAINING_RETURN_AUDIT:NONFINITE_JSON:" + value)
    return json.loads(data, object_pairs_hook=pairs, parse_constant=reject)


def safe_member(name):
    p = PurePosixPath(name)
    require(name.startswith(PREFIX) and not p.is_absolute() and
            all(part not in ("..", ".", "") for part in name.split("/")) and
            "\\" not in name and ":" not in name, "UNSAFE_ZIP_MEMBER:" + name)


def check_metrics(metrics, target, support, *, reference=False):
    require(metrics["status"] == "EVALUABLE" and metrics["anchors"] == support["validation_rows"] and
            metrics["cells"] == support["validation_cells"] and
            metrics["sources"] == len(support["validation_sources"]) and
            set(metrics["by_source"]) == set(support["validation_sources"]), "METRIC_SUPPORT")
    for key, value in metrics["source_balanced"].items():
        require(math.isfinite(value), "NONFINITE_METRIC")
        expected = math.fsum(v[key] for v in metrics["by_source"].values()) / metrics["sources"]
        require(math.isclose(value, expected, rel_tol=1e-14, abs_tol=1e-14), "SOURCE_AGGREGATION:" + key)
        if key in ("mae", "rmse_or_rmsle"):
            require(value >= 0, "NEGATIVE_ERROR_METRIC")
        if key in ("lower_clip", "upper_clip"):
            require(0 <= value <= 1, "INVALID_CLIPPING_FRACTION")
    key = "mae" if target == "soh30" else "rmse_or_rmsle"
    require(metrics["selection_score"] == metrics["source_balanced"][key], "SELECTION_SCORE")
    require(all(math.isfinite(v) for v in metrics["ranges"].values()), "NONFINITE_RANGE")
    require(metrics["ranges"]["bounded_min"] >= 0 and
            metrics["ranges"]["bounded_min"] <= metrics["ranges"]["bounded_max"], "BOUNDED_RANGE")
    if target == "soh30":
        require(metrics["ranges"]["bounded_max"] <= 1.5, "SOH_BOUND")
    if reference:
        require(metrics["raw_clipping_unavailable"] is True and all(
            not {"lower_clip", "upper_clip"} & set(v) for v in
            [metrics["source_balanced"], *metrics["by_source"].values()]), "REFERENCE_CLIPPING")


def notebook_provenance():
    from protocol_and_provenance.neural_classical_fairness_v1 import build_remaining_cpu_notebook as builder
    receipt = strict_json((FOLDER / "build_receipt.json").read_bytes())
    original = FOLDER / builder.NOTEBOOK
    data = original.read_bytes()
    warnings = []
    if binding(original) != receipt["notebook"]:
        # Diagnose, do NOT modify the author's IDE-edited source notebook.
        candidate = data.replace(b'"language": "python",c', b'"language": "python",', 1)
        require(len(candidate) == receipt["notebook"]["bytes"] and
                digest(candidate) == receipt["notebook"]["sha256"], "UNEXPLAINED_SOURCE_NOTEBOOK_DRIFT")
        warnings.append("LOCAL_SOURCE_NOTEBOOK_INVALID_JSON_SINGLE_C_IN_KERNELSPEC_METADATA; "
                        "in-memory removal restores exact frozen SHA; source preserved unchanged")
        data = candidate
    original_nb = strict_json(data)
    returned_path = FOLDER / "batteryscope-nfr1-remaining19-cpu-v1.ipynb"
    returned = strict_json(returned_path.read_bytes())
    sources = lambda nb: ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    require(sources(returned) == sources(original_nb), "RETURNED_NOTEBOOK_CODE_DRIFT")
    require(len(sources(returned)) == 2 and all(c.get("execution_count") is not None
            for c in returned["cells"] if c["cell_type"] == "code"), "UNEXECUTED_NOTEBOOK")
    require(not any(o["output_type"] == "error" for c in returned["cells"]
                    for o in c.get("outputs", [])), "NOTEBOOK_ERROR")
    assignments = {}
    for node in ast.parse(sources(returned)[0]).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ("CODE_BLOB", "EXPECTED_PAYLOAD_SHA256"):
                assignments[node.targets[0].id] = ast.literal_eval(node.value)
    raw = zlib.decompress(base64.b64decode(assignments["CODE_BLOB"], validate=True))
    require(digest(raw) == assignments["EXPECTED_PAYLOAD_SHA256"] == receipt["embedded_payload_sha256"], "PAYLOAD_HASH")
    files = {name: base64.b64decode(value, validate=True) for name, value in strict_json(raw).items()}
    spec = strict_json(files["remaining_delivery.json"])
    require(spec["local_attestation"] == receipt["local_attestation"], "BUILD_ATTESTATION")
    current_files, current_spec = builder.payload()  # hashes only; never loads models
    require(spec == current_spec and files == {n:base64.b64decode(v) for n,v in current_files.items()}, "CURRENT_CODE_INPUT_OR_LOCAL_DRIFT")
    return spec, files, dict(source=binding(original), returned=binding(returned_path),
                            restored_frozen_notebook_sha256=digest(data), code_cells_byte_identical=True,
                            embedded_payload_sha256=digest(raw), warnings=warnings)


def reconstruct_support(spec, output):
    """Independent input re-read and preprocessor reconstruction, inner roles ONLY."""
    import numpy as np
    from types import SimpleNamespace
    from protocol_and_provenance.neural_classical_fairness_v1.runtime import local_cpu_pilots as local
    _, executor = local.classical_adapter()
    supports, contexts, states, fixtures = {}, {}, {}, {}
    for source in ("CALB", "CALCE"):
        directory = output / ("support_" + source)
        directory.mkdir(parents=True, exist_ok=False)
        job = next(j for j in spec["jobs"] if j["source"] == source)
        ns, train, val, _, _, read, train_ids, val_ids = local.verified_namespace(
            job, SimpleNamespace(check=lambda: None), directory, cycles=False)
        contexts[source] = strict_json((directory / "roles.json").read_bytes())
        cols = list(dict.fromkeys(["sample_id", "canonical_cell_id"] + executor.feature_columns))
        ft = read(local.BUNDLE / "phase2/tabular_feature_index.parquet", cols, train_ids)
        fv = read(local.BUNDLE / "phase2/tabular_feature_index.parquet", cols, val_ids)
        for target in local.TARGETS:
            ts = ns["TARGET_SPECS"][target]
            def join(rows, features, ids):
                selected = rows.loc[ns["normalize_bool"](rows[ts["mask_col"]], ts["mask_col"]) &
                                    np.isfinite(rows[ts["y_col"]].to_numpy(float))].copy()
                labels = read(local.BUNDLE / "phase2/sample_target_index.parquet",
                              ["sample_id", "canonical_cell_id", ts["y_col"]], ids)
                selected = selected.drop(columns=ts["y_col"]).merge(labels,
                    on=["sample_id", "canonical_cell_id"], how="left", validate="one_to_one")
                require(np.isfinite(selected[ts["y_col"]].to_numpy(float)).all(), "TARGET_JOIN")
                result = selected.merge(features, on=["sample_id", "canonical_cell_id"], how="left",
                                        validate="one_to_one", indicator=True)
                require(result._merge.eq("both").all(), "FEATURE_JOIN")
                return result.drop(columns="_merge").reset_index(drop=True)
            tr, va = join(train, ft, train_ids), join(val, fv, val_ids)
            key = (source, target)
            supports[key] = dict(training_rows=len(tr), validation_rows=len(va),
                training_cell_hash=local.digest(sorted(set(tr.canonical_cell_id.astype(str)))),
                training_sample_hash=local.digest(sorted(tr.sample_id.astype(str))),
                validation_sample_hash=local.digest(sorted(va.sample_id.astype(str))),
                validation_cells=va.canonical_cell_id.nunique(), validation_sources=sorted(va.source.unique()))
            for family in {spec["classical_grid"][j["candidate"]]["family"] for j in spec["jobs"]
                           if j["source"] == source and j["target"] == target and j["kind"] == "classical"}:
                state = executor._fit_preprocessor(tr, family)
                state["fit_role"] = "inner_train_only_new_runtime_reconstruction"
                states[(source, target, family)] = state
            fixtures[key] = va.iloc[:256][executor.feature_columns]
        print("[AUDIT SUPPORT]", source, flush=True)
    return supports, contexts, states, fixtures


def run(output=OUTPUT):
    from protocol_and_provenance.neural_classical_fairness_v1.runtime import local_cpu_pilots as local
    import numpy as np
    import pandas as pd
    output = Path(output)
    require(not output.exists(), "FRESH_AUDIT_DIRECTORY_REQUIRED")
    spec, embedded, notebook = notebook_provenance()
    archive = FOLDER / "batteryscope_nfr1_remaining19_cpu_v1_results.zip"
    before = binding(archive)
    output.mkdir(parents=True)
    with zipfile.ZipFile(archive) as z:
        members = z.infolist()
        names = [m.filename for m in members]
        require(len(set(names)) == len(names) and len(names) == 212, "ZIP_MEMBER_COVERAGE")
        for m in members:
            safe_member(m.filename)
            require(not m.is_dir() and (m.external_attr >> 16) & 0o170000 != 0o120000, "ZIP_SYMLINK_OR_DIRECTORY")
        get = lambda name: strict_json(z.read(PREFIX + name))
        inventory = get("output_inventory.json")
        require(inventory["outer_scoring"] is False and inventory["fairness_closed"] is False, "INVENTORY_SCOPE")
        require(set(inventory["files"]) == {n[len(PREFIX):] for n in names} - {"output_inventory.json"}, "INVENTORY_COVERAGE")
        # Reading each entire member validates ZIP CRC in addition to sealed SHA/size.
        for name, record in inventory["files"].items():
            data = z.read(PREFIX + name)
            require(dict(bytes=len(data), sha256=digest(data)) == record, "OUTPUT_HASH:" + name)
            if name.endswith(".json"):
                strict_json(data)
        require(not any("failure" in n or n.endswith("/STOP") or n.endswith(".partial") for n in names), "FAILURE_ARTIFACT")
        require(get("delivery_manifest.snapshot.json") == spec, "RETURNED_DELIVERY_DRIFT")
        summary, ledger, session = get("summary.json"), get("compute_ledger.json"), get("session_started.json")
        require(summary["status"] == "KAGGLE_REMAINING19_COMPLETE_REVIEW_REQUIRED" and
                summary["error"] is None and summary["completed_jobs"] == summary["planned_jobs"] == 19 and
                summary["completed_classical_fits"] == 13 and summary["reference_executions"] == 16, "SUMMARY")
        require(summary["execution_host"] == "KAGGLE_LINUX_CPU" and all(summary[k] is False for k in
                ("outer_scoring", "fairness_closed", "automatic_retry", "cross_host_parity_claim")), "SUMMARY_SCOPE")
        require(session["jobs"] == spec["jobs"] and session["local_completed_job_ids"] == spec["local_completed_job_ids"]
                and session["local_attestation"] == spec["local_attestation"] and
                session["delivery_sha256"] == digest(embedded["remaining_delivery.json"]), "SESSION_SEAL")
        require([{k:r[k] for k in j} for r,j in zip(ledger,spec["jobs"])] == spec["jobs"] and
                len(ledger) == 19 and all(r["state"] == "COMPLETE" for r in ledger), "LEDGER_QUEUE")
        require(sum(r["learned_fits"] for r in ledger) == 13 and sum(r["reference_executions"] for r in ledger) == 16, "COMPUTE_COUNTS")
        for record in ledger:
            modes = [p["mode"] for p in record["processes"]]
            require(modes == (["job", "replay"] if record["kind"] == "classical" else ["job"]), "PROCESS_COVERAGE")
            require(all(p["seconds"] > 0 and 0 < p["peak_tree_rss_bytes"] * 1.3 <= 4.5*2**30
                        for p in record["processes"]), "MEMORY_MARGIN")
        env = get("environment_check.json")
        require(env["mismatches"] == {} and env["actual"] == get("environment.lock.json") and
                env["actual"]["python"] == spec["expected_python"] and
                all(env["actual"]["packages"][k] == v for k,v in spec["expected_packages"].items()) and
                env["actual"]["torch_used_for_training"] is False, "ENVIRONMENT")
        for name in ("heartbeat.json", "current_job.json"):
            require(get(name)["state"] == "SUPERVISOR_TERMINAL", "STALE_HEARTBEAT")
        require(get("synthetic_gate.json") == dict(status="SYNTHETIC_PREDICT_RELOAD_PASS",
            families=["extra_trees", "hist_gradient_boosting"], synthetic_fits=2, battery_fits=0,
            full_dataset_replay=False), "SYNTHETIC_GATE")
        mapped = get("input_mapping.json")
        require(len(mapped) == len(spec["input_files"]) and len({r["bundle_relative"] for r in mapped}) == len(mapped), "MAPPING_COVERAGE")
        for record in mapped:
            name = record["bundle_relative"]
            require({k:record[k] for k in ("bytes", "sha256")} == spec["input_files"][name] and
                    binding(local.BUNDLE / name) == spec["input_files"][name] and
                    record["mounted_path"].startswith("/kaggle/input/") and record["mounted_path"].endswith("/" + name), "MAPPING_HASH")
        remapped_data = z.read(PREFIX + "input_bindings.remapped.json")
        require(digest(remapped_data) == session["remapped_bindings_sha256"], "REMAPPED_BINDING_SHA")
        remapped = strict_json(remapped_data)
        require(remapped["config"] == spec["source_bindings"]["config"], "REMAPPED_CONFIG")
        source_inputs = spec["source_bindings"]["inputs"]
        require(len(remapped["inputs"]) == len(source_inputs), "REMAPPED_INPUT_COUNT")
        for record, original in zip(remapped["inputs"], source_inputs):
            require({k:v for k,v in record.items() if k != "path"} == {k:v for k,v in original.items() if k != "path"}
                    and record["path"] in {m["mounted_path"] for m in mapped}, "REMAPPED_INPUT")
        supports, contexts, states, fixtures = reconstruct_support(spec, output)
        jobs_on_disk = {n.split("/")[2] for n in names if n.startswith(PREFIX + "jobs/")}
        require(jobs_on_disk == {j["id"] for j in spec["jobs"]}, "JOB_DIRECTORY_COVERAGE")
        metric_rows = []
        for job in spec["jobs"]:
            p = "jobs/" + job["id"] + "/"
            roles, result, started = get(p + "roles.json"), get(p + "result.json"), get(p + "started.json")
            require(started["job"] == job and started["delivery_sha256"] == session["delivery_sha256"], "JOB_START")
            require(roles == contexts[job["source"]], "INDEPENDENT_ROLE_OR_SCALER_MISMATCH:" + job["id"])
            support = supports[(job["source"], job["target"])]
            hashes = {local.digest(roles["train_cells"]): len(roles["train_cells"]),
                      local.digest(roles["validation_cells"]): len(roles["validation_cells"])}
            access = get(p + "access_log.json")
            require(len(access) == 8 and all(a["table"] in ("sample_target_index.parquet", "window_index.parquet", "tabular_feature_index.parquet")
                    and a["cell_hash"] in hashes and a["cells"] == hashes[a["cell_hash"]] for a in access), "ACCESS_ROLE")
            # Compare complete read sequence with independently reconstructed cell-filtered reads.
            rebuilt = strict_json((output / ("support_" + job["source"]) / "access_log.json").read_bytes())
            base = rebuilt[:4]
            train_feature, val_feature = rebuilt[4:6]
            idx = local.TARGETS.index(job["target"])
            train_label, val_label = rebuilt[6+idx*2:8+idx*2]
            require(access == base + [train_feature, train_label, val_feature, val_label], "ACCESS_TABLE_COLUMNS_OR_ROWS")
            require(all(result[k] is False for k in ("production_fit", "outer_scoring", "fairness_closed")) and
                    result["execution_host"] == "KAGGLE_LINUX_CPU", "RESULT_SCOPE")
            if job["kind"] == "classical":
                seal, replay = get(p + "model.integrity.json"), get(p + "replay.json")
                require(result["status"] == "PILOT_TRAIN_COMPLETE_RELOAD_PENDING" and result["learned_fits"] == 1
                        and result["full_validation"] is True, "FIT_STATUS")
                for key in ("training_rows", "validation_rows", "validation_sample_hash"):
                    require(result[key] == support[key], "ENDPOINT_SUPPORT:" + key)
                for key in ("training_cell_hash", "training_sample_hash"):
                    require(seal[key] == support[key], "TRAIN_HASH:" + key)
                for name, key in (("pipeline.joblib", "sha256"), ("replay_features.parquet", "feature_sha256"), ("replay_expected.npy", "expected_sha256")):
                    require(inventory["files"][p+name]["sha256"] == seal[key], "ARTIFACT_SEAL")
                family = spec["classical_grid"][job["candidate"]]["family"]
                require(seal["feature_state"] == states[(job["source"], job["target"], family)], "PREPROCESSOR_RECONSTRUCTION")
                require(seal["target_state"] == {"transform": "identity" if job["target"] == "soh30" else "log1p"}, "TARGET_TRANSFORM")
                params = spec["classical_grid"][job["candidate"]]["params"]
                require(all(seal["estimator_params"][k] == v for k,v in params.items()), "GRID_PARAMS")
                require(seal["prediction_policy"] == result["prediction_policy"] == spec["prediction_policies"][family] and
                        seal["provenance"] == result["provenance"] == "NEW_RUNTIME_INNER_RECONSTRUCTION_NOT_HISTORICAL_REPLAY", "PROVENANCE_POLICY")
                require(replay == dict(status="FRESH_PROCESS_FORWARD_FIXTURE_EXACT_PASS",
                    scope="first_256_full_validation_rows_only", full_validation_replay=False, labels_read=False), "REPLAY_RECEIPT")
                fixture = pd.read_parquet(io.BytesIO(z.read(PREFIX + p + "replay_features.parquet")))
                pd.testing.assert_frame_equal(fixture, fixtures[(job["source"], job["target"])] .reset_index(drop=True))
                pred = np.load(io.BytesIO(z.read(PREFIX + p + "replay_expected.npy")), allow_pickle=False)
                require(pred.shape == (len(fixture),) and np.isfinite(pred).all() and (pred >= 0).all(), "REPLAY_FIXTURE_VALUES")
                check_metrics(result["metrics"], job["target"], support)
                metric_rows.append(dict(job_id=job["id"], target=job["target"], validation=result["metrics"], host="KAGGLE_LINUX_CPU"))
            else:
                expected_names = {"soh_persistence", "causal_linear_trend"} if job["target"] == "soh30" else {
                    "rul_train_median", "rul_train_log_mean", "causal_linear_trend"}
                require(result["status"] == "REFERENCE_EXECUTIONS_COMPLETE" and result["learned_fits"] == 0 and
                        result["reference_executions"] == len(expected_names) and set(result["metrics"]) == expected_names
                        and result["metrics"] == get(p+"reference_metrics.json"), "REFERENCE_EXECUTIONS")
                for name, metrics in result["metrics"].items():
                    check_metrics(metrics, job["target"], support, reference=True)
                    metric_rows.append(dict(job_id=job["id"], reference=name, target=job["target"], validation=metrics, host="KAGGLE_LINUX_CPU"))
        expected_seals = [dict(source=s, role_hashes=contexts[s]["role_hashes"],
                              train_cells=len(contexts[s]["train_cells"]), validation_cells=len(contexts[s]["validation_cells"]))
                          for s in ("CALB", "CALCE")]
        require(get("role_seals.json") == expected_seals, "METADATA_ROLE_SEALS")
    require(before == binding(archive), "ARCHIVE_CHANGED_DURING_AUDIT")
    # Recheck original 25 protected bindings after all inner support reads.
    launch = strict_json((LOCAL / "launch_manifest.json").read_bytes())
    for record in launch["protected"]:
        require(binding(ROOT / record["path"]) == {k:record[k] for k in ("bytes", "sha256")}, "PROTECTED_CHANGED:"+record["path"])
    review = dict(status="ACCEPTED_BOUNDED_DEVELOPMENT_ARTIFACTS_FAIRNESS_OPEN", date="2026-10-10",
        archive=before, notebook=notebook, zip_members=len(names), inventoried_hashes=len(inventory["files"]),
        completed_kaggle_jobs=19, classical_fits=13, reference_jobs=6, reference_executions=16,
        kaggle_wall_seconds=summary["wall_seconds"], combined_development_jobs=47,
        combined_neural_one_epoch_pilots=8, combined_classical_fits=33, synthetic_fits_separate=2,
        independent_inner_support_and_preprocessor_checks=6, fresh_process_replay_receipts=13,
        reviewer_deserialized_models=False, reviewer_recomputed_full_prediction_metrics=False,
        protected_files_unchanged=len(launch["protected"]), new_model_fits=0, outer_scoring=False,
        cross_host_parity_claim=False, homogeneous_campaign_claim=False, fairness_closed=False,
        remaining_gates=["F2_full_objective_selection_comparison", "F3_full_inner_and_historical_replay",
                         "F4_F7_production_and_fairness"],
        limitations=["Neural results are one-epoch development pilots, not selected/converged systems",
                     "Windows/Linux runtime split is disclosed; no matched-cost superiority conclusion",
                     "Exact replay is first 256 validation rows in Kaggle; local audit verifies receipts/hashes/fixtures, not model execution",
                     "Source-level metric arithmetic checked; per-cell sufficient statistics/full predictions not exported, so full metrics not independently recomputed"])
    local.write(output / "review.json", review)
    local.write(output / "returned_validation_metrics.json", metric_rows)
    print(json.dumps(review, indent=2, ensure_ascii=False), flush=True)
    return review


if __name__ == "__main__":
    run()
