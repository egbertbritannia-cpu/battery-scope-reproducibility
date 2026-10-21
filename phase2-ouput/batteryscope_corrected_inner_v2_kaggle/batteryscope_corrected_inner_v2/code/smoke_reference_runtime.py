"""Corrected delivery v1: bounded inner smoke only; no outer scoring API.

Scientific inner definitions are extracted unchanged from a hash-bound executed
notebook. This adapter supplies fresh namespace, strict paths/read-role guards,
process deadline, access log and diagnostics. Production phases fail closed.
"""
from __future__ import annotations

import ast
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path, PurePosixPath
import queue
import random
import sys
import time
import traceback
from contextlib import nullcontext
from typing import Any, Iterator
import zipfile

from corrected_environment import runtime_environment, verify_environment_profile, write_reproducibility_inventory

MANIFEST = "corrected_delivery_manifest.json"
ZIP_NAME = "batteryscope_corrected_smoke_v3_kaggle.zip"


def require(ok, message):
    if not bool(ok):
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def child(root, relative):
    p = PurePosixPath(relative)
    require(not p.is_absolute() and p.parts and not any(x in ("..", ".") for x in p.parts)
            and "\\" not in relative and ":" not in relative, "UNSAFE_RELATIVE_PATH")
    result = (Path(root) / relative).resolve()
    require(result.is_relative_to(Path(root).resolve()), "PATH_ESCAPES_NAMESPACE")
    return result


def resolve_bundle(input_root="/kaggle/input", working_root="/kaggle/working"):
    input_root = Path(input_root)
    manifests = list(input_root.rglob(MANIFEST))
    require(len(manifests) <= 1, "AMBIGUOUS_CORRECTED_BUNDLES")
    if manifests:
        return manifests[0].parent
    archives = list(input_root.rglob(ZIP_NAME))
    require(len(archives) == 1, "CORRECTED_BUNDLE_NOT_FOUND_OR_AMBIGUOUS")
    working_root = Path(working_root).resolve()
    target = working_root / "corrected_smoke_bundle_extracted_v3"
    require(target.is_relative_to(working_root) and target != working_root, "UNSAFE_EXTRACTION_ROOT")
    require(not target.exists(), "EXTRACTION_EXISTS_USE_FRESH_SESSION")
    with zipfile.ZipFile(archives[0]) as archive:
        infos = archive.infolist()
        require(len(infos) < 2000 and sum(i.file_size for i in infos) < 2_000_000_000, "ZIP_LIMIT_EXCEEDED")
        require(len({i.filename for i in infos}) == len(infos), "DUPLICATE_ZIP_MEMBER")
        for info in infos:
            child(target, info.filename.rstrip("/"))
            require((info.external_attr >> 16) & 0o170000 != 0o120000, "ZIP_SYMLINK_FORBIDDEN")
        archive.extractall(target)
    manifests = list(target.rglob(MANIFEST))
    require(len(manifests) == 1, "EXTRACTED_MANIFEST_MISSING_OR_AMBIGUOUS")
    return manifests[0].parent


def check_environment(expected, actual):
    mismatches = {}
    for name, version in expected["packages"].items():
        if actual["packages"].get(name) != version:
            mismatches[name] = dict(expected=version, actual=actual["packages"].get(name))
    for key in ("torch_cuda_version", "cudnn_version", "gpu_name"):
        if actual.get(key) != expected[key]:
            mismatches[key] = dict(expected=expected[key], actual=actual.get(key))
    if actual.get("cuda_available") is not True:
        mismatches["cuda_available"] = dict(expected=True, actual=actual.get("cuda_available"))
    return mismatches


def environment():
    return runtime_environment()


def validate_delivery_config(config, phase):
    require(config["schema"] == "corrected-smoke-delivery-config-v1", "CONFIG_SCHEMA")
    require(phase == "smoke" and config["enabled_phases"] == ["smoke"], "PRODUCTION_PHASE_DISABLED_PENDING_SMOKE_REVIEW")
    require(config["seed_bases"] == [2026, 2027, 2028], "APPROVED_SEEDS_CHANGED")
    require(config["smoke_pilots"] == [["CALB", 1], ["CALB", 0]], "SMOKE_PILOTS_CHANGED")
    require(config["smoke_max_epochs_per_pilot"] == 1, "SMOKE_EPOCHS_CHANGED")
    require(config["smoke_total_wall_seconds"] == 3600, "SMOKE_BOUND_CHANGED")
    require(config["accepted_production_fit"] is False and config["outer_scoring_enabled"] is False,
            "SMOKE_CANNOT_BECOME_PRODUCTION_OR_SCORE_OUTER")
    require(config["gpu_workers"] == 1, "UNVERIFIED_MULTI_GPU_RECIPE")


def verify_bundle(bundle):
    bundle = Path(bundle).resolve()
    manifest = json.loads((bundle / MANIFEST).read_text(encoding="utf-8"))
    require(manifest["schema"] == "corrected-smoke-delivery-manifest-v1", "DELIVERY_MANIFEST_SCHEMA")
    files = manifest["files"]
    expected = set(files) | {MANIFEST}
    observed = {p.relative_to(bundle).as_posix() for p in bundle.rglob("*") if p.is_file()
                and "__pycache__" not in p.parts}
    require(observed == expected, "DELIVERY_FILE_SET_MISMATCH")
    for relative, record in files.items():
        path = child(bundle, relative)
        require(path.is_file() and path.stat().st_size == record["bytes"], "DELIVERY_SIZE_MISMATCH:"+relative)
        require(sha(path) == record["sha256"], "DELIVERY_HASH_MISMATCH:"+relative)
    config = json.loads((bundle / "smoke_config.frozen.json").read_text(encoding="utf-8"))
    validate_delivery_config(config, "smoke")
    verify_environment_profile(bundle, config)
    recipe = json.loads((bundle / "code/inner_recipe.json").read_text(encoding="utf-8"))
    for record in recipe["definitions"]:
        require(hashlib.sha256(record["source"].encode()).hexdigest() == record["source_sha256"], "RECIPE_DEFINITION_HASH")
        ast.parse(record["source"])
    for record in recipe["source_notebooks"]:
        require(sha(child(bundle, record["bundle_path"])) == record["sha256"], "EXECUTED_NOTEBOOK_HASH")
    current_runtime = Path(__file__).resolve()
    require(sha(current_runtime) == files["code/corrected_kaggle_runtime.py"]["sha256"], "IMPORTED_RUNTIME_HASH_MISMATCH")
    return config, manifest, recipe


def checked_read_cells(path, columns, cells, allowed_cells, log, reader):
    cells = set(cells)
    require(bool(cells) and cells.issubset(allowed_cells), "INNER_READ_OUTSIDE_ALLOWED_TRAIN_VALIDATION")
    frame = reader(path, columns, cells)
    require(set(frame.canonical_cell_id.astype(str)).issubset(cells), "FILTERED_READ_ESCAPED_REQUESTED_CELLS")
    log.append(dict(path=Path(path).name, columns=list(columns), requested_cells=len(cells),
                    requested_cell_hash=json_sha(sorted(cells)), rows=len(frame)))
    return frame


def make_namespace(bundle, recipe, config, *, metadata_only=False):
    # Never execute notebook cells or their top-level historical run/READY code.
    import numpy as np
    import pandas as pd
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq
    if metadata_only:
        torch = nn = F = None
    else:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    from sklearn.metrics import r2_score
    ns = dict(__name__="corrected_extracted_inner_recipe", np=np, pd=pd, ds=ds, pq=pq,
              torch=torch, nn=nn, F=F, gc=gc, hashlib=hashlib, json=json, math=math,
              os=os, random=random, time=time, Path=Path, Any=Any, Iterator=Iterator,
              nullcontext=nullcontext, r2_score=r2_score)
    ns.update(recipe["literal_constants"])
    for record in recipe["definitions"]:
        if metadata_only and record["name"] in ("CausalConv1d", "ResidualTCNBlock", "OrderedQuantileHead", "MultitaskQuantileTCN"):
            continue
        # Future annotations avoid evaluation of optional type globals at import.
        exec(compile("from __future__ import annotations\n" + record["source"],
                     "<bound-inner:"+record["name"]+">", "exec"), ns)
    phase2 = bundle / "phase2"
    feature = json.loads((phase2 / "feature_spec.json").read_text(encoding="utf-8"))
    ns.update(PHASE2_DIR=phase2, MULTISENSOR_STORE_DIR=phase2 / "batteryscope_phase2_multisensor_cycle_store",
              TARGET_INDEX_PATH=phase2 / "sample_target_index.parquet", WINDOW_INDEX_PATH=phase2 / "window_index.parquet",
              VALUE_CHANNELS=feature["value_channels"], VALUE_MASK_COLUMNS=feature["value_mask_columns"],
              STAGE6_ALL_OUTER_SOURCES=config["sources"], SOURCE_BLOCKS={int(k):v for k,v in config["source_blocks"].items()},
              INNER_FOLDS=[0,1,2], SEED=2026, MAX_EPOCHS=1,
              # Fresh smoke-specific config, not a historical authorization rebind.
              PIPELINE_CONFIG_HASH=json_sha(config), INPUT_HASHES=config["input_hashes"])
    require(len(ns["VALUE_CHANNELS"]) == len(ns["VALUE_MASK_COLUMNS"]) == 48, "ACTIVE_LAYOUT_MISMATCH")
    ns["fold_membership"] = pd.read_parquet(phase2 / "battery_phase2_fold_membership.parquet")
    ns["cycle_store_index"] = pd.read_parquet(phase2 / "multisensor_cycle_store_index.parquet").set_index("canonical_cell_id")
    ns["LOCAL_METADATA_ONLY_NO_TRAINING"] = metadata_only
    return ns


def validate_inner_roles(ns, source, fold, expected):
    payload = ns["source_blocked_role_payload"]("LODO", source, fold)
    train, validation = payload["selection_train"], payload["early_validation"]
    # Trace25 uses newline-joined canonical IDs, NOT canonical JSON. Preserve
    # its hash algorithm; checkpoint state separately uses inherited JSON hashes.
    train_hash = hashlib.sha256("\n".join(sorted(train)).encode()).hexdigest()
    validation_hash = hashlib.sha256("\n".join(sorted(validation)).encode()).hexdigest()
    require(train_hash == expected["train_cell_hash"], "TRAIN_MEMBERSHIP_HASH_DRIFT")
    require(validation_hash == expected["validation_cell_hash"], "VALIDATION_MEMBERSHIP_HASH_DRIFT")
    require(not train.intersection(validation), "ROLE_OVERLAP")
    return payload


def run_pilot(bundle, recipe, config, source, fold, run):
    import torch
    ns = make_namespace(bundle, recipe, config)
    expected = next(r for r in config["inner_contexts"] if r["source"] == source and r["inner_fold"] == fold)
    payload = validate_inner_roles(ns, source, fold, expected)
    allowed = payload["selection_train"] | payload["early_validation"]
    access_log = []
    original_read = ns["read_for_cells"]
    ns["read_for_cells"] = lambda path, cols, cells: checked_read_cells(path, cols, cells, allowed, access_log, original_read)
    # Relative release paths only. Never fall back to absolute historical shards.
    index = ns["cycle_store_index"]
    def resolve(cell):
        require(cell in allowed, "PROFILE_READ_OUTSIDE_INNER_ROLES")
        path = child(bundle / "phase2", str(index.loc[cell, "shard_path"]))
        require(path.is_file(), "CORRECTED_SHARD_MISSING")
        return path
    ns["resolve_cycle_shard"] = resolve
    original_load = ns["load_cycle_shard"]
    profile_reads = set()
    def load(cell):
        frame = original_load(cell)
        profile_reads.add(cell)
        return frame
    ns["load_cycle_shard"] = load
    pilot = run / f"pilot_{source}_block{fold}"
    pilot.mkdir()
    started = time.monotonic()
    write_json(pilot / "started.json", dict(source=source, inner_fold=fold, accepted_production_fit=False,
                                           config_sha256=json_sha(config), outer_labels_accessed=False))
    try:
        train_rows = ns["load_multitask_window_targets"](payload["selection_train"])
        validation_rows = ns["load_multitask_window_targets"](payload["early_validation"])
        t0 = time.monotonic()
        target_scaler = ns["fit_target_scaler"](train_rows)
        cycle_scaler = ns["fit_cycle_scaler"](payload["selection_train"])
        preprocessing_seconds = time.monotonic() - t0
        write_json(pilot / "roles_and_scalers.json", dict(audit=payload["audit"], train_cell_hash=expected["train_cell_hash"],
                    validation_cell_hash=expected["validation_cell_hash"], cycle_scaler=cycle_scaler,
                    target_scaler=target_scaler, train_rows=len(train_rows), validation_rows=len(validation_rows)))
        train_iterator = ns["MultitaskCycleWindowIterator"](train_rows, cycle_scaler, target_scaler)
        validation_iterator = ns["MultitaskCycleWindowIterator"](validation_rows, cycle_scaler, target_scaler)
        seed = ns["deterministic_fold_seed"]("LODO", source, fold)
        random.seed(2026); ns["np"].random.seed(2026)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        write_reproducibility_inventory(pilot, config, environment(), bundle, phase="worker_before_training")
        torch.cuda.reset_peak_memory_stats()
        t0 = time.monotonic()
        model, best_epoch, best_score, history = ns["select_tcn_inner_epoch"](
            train_iterator, validation_iterator, pilot / "best_smoke.pt", pilot / "progress_smoke.pt",
            torch.device("cuda"), seed,
            dict(accepted_production_fit=False, phase="smoke", source=source, inner_fold=fold))
        training_validation_seconds = time.monotonic() - t0
        result = dict(status="ONE_EPOCH_INNER_SMOKE_PASS_NOT_PRODUCTION", source=source, inner_fold=fold,
                      best_epoch=best_epoch, validation_rul90_score=best_score, epoch_history=history,
                      fold_seed=seed, config_sha256=json_sha(config), preprocessing_seconds=preprocessing_seconds,
                      training_plus_validation_seconds=training_validation_seconds,
                      total_pilot_seconds=time.monotonic()-started,
                      gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      outer_labels_accessed=False, calibration_labels_accessed=False,
                      accepted_production_fit=False, production_epoch_selection=False)
        write_json(pilot / "pilot_result.json", result)
        del model
        return result
    finally:
        write_json(pilot / "access_log.json", dict(filtered_reads=access_log, profile_cells=sorted(profile_reads),
                    allowed_cell_hash=json_sha(sorted(allowed)), outer_labels_accessed=False,
                    calibration_labels_accessed=False))


def smoke_worker(bundle, run, messages):
    try:
        bundle, run = Path(bundle), Path(run)
        config, _, recipe = verify_bundle(bundle)
        for source, fold in config["smoke_pilots"]:
            result = run_pilot(bundle, recipe, config, source, fold, run)
            print("[SMOKE PILOT PASS]", source, fold, flush=True)
            messages.put(dict(kind="pilot", result=result))
        messages.put(dict(kind="complete"))
    except Exception as exc:
        failure = dict(status="SMOKE_FAILED_NO_OUTER_SCORING", error=f"{type(exc).__name__}: {exc}",
                       traceback=traceback.format_exc(), outer_labels_accessed=False, accepted_production_fit=False)
        write_json(Path(run) / "worker_failure.json", failure)
        messages.put(dict(kind="failure", error=failure["error"]))


def run_smoke(bundle, output_root="/kaggle/working/batteryscope_corrected_smoke_v3"):
    bundle = Path(bundle).resolve()
    run = Path(output_root).resolve()
    require(not run.exists(), "OUTPUT_EXISTS_USE_FRESH_SESSION_NO_SMOKE_RESUME")
    require(not run.is_relative_to(bundle) and not bundle.is_relative_to(run), "OUTPUT_OVERLAPS_READ_ONLY_INPUT")
    run.mkdir(parents=True)
    config, manifest, _ = verify_bundle(bundle)
    actual = environment()
    mismatches = check_environment(config["environment_reference"], actual)
    write_json(run / "environment_check.json", dict(actual=actual, mismatches=mismatches,
                    delivery_manifest_sha256=sha(bundle / MANIFEST), config_sha256=json_sha(config)))
    write_reproducibility_inventory(run, config, actual, bundle, phase="parent_preflight")
    require(not mismatches, "CORRECTED_ENVIRONMENT_MISMATCH:"+json.dumps(mismatches))
    write_json(run / "smoke_started.json", dict(status="BOUNDED_SMOKE_STARTED", max_wall_seconds=3600,
                    pilot_order=config["smoke_pilots"], accepted_production_fit=False,
                    delivery_manifest_sha256=sha(bundle / MANIFEST), config_sha256=json_sha(config),
                    outer_labels_accessed=False, started_at_utc=datetime.now(timezone.utc).isoformat()))
    context = multiprocessing.get_context("spawn")
    messages = context.Queue()
    process = context.Process(target=smoke_worker, args=(str(bundle), str(run), messages))
    process.start()
    started = time.monotonic()
    results = []
    complete = False
    while process.is_alive() and time.monotonic()-started < config["smoke_total_wall_seconds"]:
        try:
            item = messages.get(timeout=1)
            if item["kind"] == "pilot":
                results.append(item["result"])
            elif item["kind"] == "complete":
                complete = True
            elif item["kind"] == "failure":
                break
        except queue.Empty:
            pass
    capped = process.is_alive() and time.monotonic()-started >= config["smoke_total_wall_seconds"]
    if process.is_alive():
        process.terminate()
    process.join(timeout=10)
    if process.is_alive():
        process.kill(); process.join(timeout=10)
    while True:
        try:
            item = messages.get_nowait()
            if item["kind"] == "pilot":
                results.append(item["result"])
            elif item["kind"] == "complete":
                complete = True
        except queue.Empty:
            break
    status = ("SMOKE_CAP_REACHED_NOT_PRODUCTION" if capped else
              "SMOKE_COMPLETE_REVIEW_REQUIRED" if complete and len(results)==2 and process.exitcode==0 else
              "SMOKE_FAILED_REVIEW_REQUIRED")
    write_json(run / "smoke_summary.json", dict(status=status, pilot_results=results,
                worker_exitcode=process.exitcode, worker_wall_seconds=time.monotonic()-started,
                accepted_production_fit=False, outer_scoring=False, recalibration=False,
                production_phases_enabled=False, config_sha256=json_sha(config),
                delivery_manifest_sha256=sha(bundle / MANIFEST)))
    return run


def archive_results(run):
    run = Path(run)
    target = run.parent / (run.name + "_results.zip")
    require(not target.exists(), "REFUSE_OVERWRITE_RESULT_ARCHIVE")
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=4) as archive:
        for path in sorted(run.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=run.name+"/"+path.relative_to(run).as_posix())
    return target
