"""BatteryScope Stage-8 Full-LODO model executor.

This is result-affecting code and must be hash-frozen by Stage 7 before any
outer label is opened.  The module deliberately imports the Kaggle scientific
stack lazily so its access guards and synthetic tests can run without loading
research data.

Scientific contract
-------------------
* one complete dataset/source is held out in each LODO fold;
* development, calibration and test are cell- and duplicate-family-disjoint;
* preprocessing and target scaling are fit on development only;
* calibration labels are used only for grouped conformal calibration;
* outer-test target columns are not read until every candidate prediction and
  interval has been prepared;
* RUL learned targets and conformal residuals use log1p scale;
* SoH remains on its natural fraction scale;
* the locked multitask Quantile TCN is refit for the Stage-6-selected number of
  epochs without outer early stopping;
* Phase-2 Parquet is filtered by cell and cycle shards are streamed one cell at
  a time.  The raw 82 GB corpus is never opened.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence


EXECUTOR_CONTRACT = "BATTERYSCOPE_OUTER_MODEL_EXECUTOR_V1"
IMPLEMENTATION_VERSION = "batteryscope-stage8-outer-model-executor-v1"
SEED = 2026
TARGETS = ("soh30", "rul90", "rul80")
TARGET_SPECS = {
    "soh30": {"y_col": "soh_t_plus_30", "mask_col": "soh30_loss_mask", "transform": "identity"},
    "rul90": {"y_col": "rul90_cycles", "mask_col": "rul90_loss_mask", "transform": "log1p"},
    "rul80": {"y_col": "rul80_cycles", "mask_col": "rul80_loss_mask", "transform": "log1p"},
}
EXPECTED_CANDIDATES = {
    "soh30": (
        "soh_persistence", "causal_linear_trend", "ridge_alpha_1",
        "ridge_alpha_10", "ridge_alpha_100", "hist_gradient_boosting_locked",
        "extra_trees_locked", "quantile_tcn_locked",
    ),
    "rul90": (
        "rul_train_median", "rul_train_log_mean", "causal_linear_trend",
        "ridge_alpha_1", "ridge_alpha_10", "ridge_alpha_100",
        "hist_gradient_boosting_locked", "extra_trees_locked",
        "quantile_tcn_locked",
    ),
    "rul80": (
        "rul_train_median", "rul_train_log_mean", "causal_linear_trend",
        "ridge_alpha_1", "ridge_alpha_10", "ridge_alpha_100",
        "hist_gradient_boosting_locked", "extra_trees_locked",
        "quantile_tcn_locked",
    ),
}
POINT_MODELS = {
    model for models in EXPECTED_CANDIDATES.values() for model in models
    if model != "quantile_tcn_locked"
}
NAIVE_MODELS = {
    "soh_persistence", "rul_train_median", "rul_train_log_mean",
    "causal_linear_trend",
}
SOH_CURRENT_FEATURE = "deg__soh_trailing_med5__last"
SOH_TREND_FEATURE = "deg__soh_trailing_med5__theil_sen_cycle"
METADATA_COLUMNS = [
    "sample_id", "canonical_cell_id", "t_cycle", "duplicate_family_id",
    "source", "protocol_id",
]
TARGET_COLUMNS = [
    "soh_t_plus_30", "soh30_loss_mask", "rul90_cycles", "rul90_loss_mask",
    "rul80_cycles", "rul80_loss_mask",
]
REQUIRED_PHASE2_FILES = {
    "manifest": "manifest.json",
    "gate_results": "gate_results.parquet",
    "feature_spec": "feature_spec.json",
    "fold_membership": "battery_phase2_fold_membership.parquet",
    "target_index": "sample_target_index.parquet",
    "tabular_index": "tabular_feature_index.parquet",
    "window_index": "window_index.parquet",
    "cycle_index": "multisensor_cycle_store_index.parquet",
}
FORBIDDEN_OUTER_PREDICTOR_COLUMNS = set(TARGET_COLUMNS)

RIDGE_CONFIG = {
    "solver": "lsqr", "tol": 1e-6, "max_iter": 10_000,
    "fit_intercept": True,
}
EXTRA_TREES_CONFIG = {
    "n_estimators": 128, "max_depth": 16, "min_samples_leaf": 10,
    "max_features": 0.7, "n_jobs": 2, "bootstrap": False,
}
HIST_GB_CONFIG = {
    "max_leaf_nodes": 31, "max_iter": 200, "learning_rate": 0.05,
    "l2_regularization": 0.01, "early_stopping": False,
}

TCN_BLOCK_CHANNELS = (32, 32, 64, 64)
TCN_DILATIONS = (1, 2, 4, 8)
TCN_KERNEL_SIZE = 3
TCN_DROPOUT = 0.10
TCN_LEARNING_RATE = 1e-3
TCN_WEIGHT_DECAY = 1e-5
TCN_MICRO_BATCH_SIZE = 32
TCN_GRAD_ACCUM_STEPS = 8
TCN_PREDICTION_BATCH_SIZE = 256
TCN_GRADIENT_CLIP_NORM = 1.0
TCN_SCALER_POINTS_PER_CELL = 256
TCN_TARGET_POINTS_PER_CELL = 256
TCN_QUANTILES = (0.10, 0.50, 0.90)
TCN_TASK_WEIGHTS = {"soh30": 1.0, "rul90": 1.0, "rul80": 0.5}

_DEPS: SimpleNamespace | None = None


def require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise RuntimeError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest().upper()


def sha256_file(path: str | Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def atomic_json(path: str | Path, payload: Any) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = file_path.with_suffix(file_path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, file_path)


def _dependencies() -> SimpleNamespace:
    global _DEPS
    if _DEPS is not None:
        return _DEPS
    try:
        import numpy as np
        import pandas as pd
        import pyarrow.dataset as ds
        import pyarrow.parquet as pq
        import sklearn
        from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
        from sklearn.linear_model import Ridge
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover - exercised on Kaggle
        raise RuntimeError(
            "STAGE8_KAGGLE_DEPENDENCIES_UNAVAILABLE: use the frozen Kaggle image; "
            "do not pip-install a different scientific stack"
        ) from exc
    _DEPS = SimpleNamespace(
        np=np, pd=pd, ds=ds, pq=pq, sklearn=sklearn,
        ExtraTreesRegressor=ExtraTreesRegressor,
        HistGradientBoostingRegressor=HistGradientBoostingRegressor,
        Ridge=Ridge, torch=torch, nn=nn, F=F,
    )
    return _DEPS


def normalize_bool_values(values: Sequence[Any], name: str) -> list[bool]:
    mapping = {
        True: True, False: False, 1: True, 0: False,
        "1": True, "0": False, "true": True, "false": False,
        "True": True, "False": False, "yes": True, "no": False,
    }
    output: list[bool] = []
    for value in values:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            output.append(False)
        elif value in mapping:
            output.append(mapping[value])
        else:
            raise RuntimeError(f"INVALID_BOOLEAN_VALUE:{name}:{value!r}")
    return output


def transformed_value(value: float, target: str) -> float:
    value = float(value)
    require(math.isfinite(value), "NONFINITE_TARGET_OR_PREDICTION")
    if target.startswith("rul"):
        require(value >= 0.0, "NEGATIVE_RUL_VALUE")
        return math.log1p(value)
    return value


def inverse_value(value: float, target: str) -> float:
    value = float(value)
    require(math.isfinite(value), "NONFINITE_TRANSFORMED_PREDICTION")
    if target.startswith("rul"):
        require(value <= math.log(float.fromhex("0x1.fffffffffffffp+1023")), "RUL_EXPM1_OVERFLOW")
        return max(0.0, math.expm1(value))
    return min(1.5, max(0.0, value))


def finite_sample_group_quantile(group_scores: Mapping[str, float], alpha: float) -> float:
    require(0.0 < alpha < 1.0, "INVALID_CONFORMAL_ALPHA")
    values = sorted(float(value) for value in group_scores.values())
    require(len(values) >= 4, "GROUPED_CONFORMAL_REQUIRES_AT_LEAST_FOUR_FAMILIES")
    require(all(math.isfinite(value) and value >= 0.0 for value in values), "INVALID_CONFORMAL_SCORE")
    rank = min(len(values), max(1, math.ceil((len(values) + 1) * (1.0 - alpha))))
    return values[rank - 1]


def grouped_point_qhat(rows: Iterable[Mapping[str, Any]], target: str, alpha: float = 0.2) -> float:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        family = str(row["duplicate_family_id"])
        require(family.strip() != "", "MISSING_CALIBRATION_FAMILY")
        truth = transformed_value(float(row["y_true"]), target)
        prediction = transformed_value(float(row["y_pred"]), target)
        grouped.setdefault(family, []).append(abs(truth - prediction))
    return finite_sample_group_quantile(
        {family: max(scores) for family, scores in grouped.items()}, alpha
    )


def grouped_cqr_qhat(rows: Iterable[Mapping[str, Any]], target: str, alpha: float = 0.2) -> float:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        family = str(row["duplicate_family_id"])
        require(family.strip() != "", "MISSING_CALIBRATION_FAMILY")
        truth = transformed_value(float(row["y_true"]), target)
        lower = transformed_value(float(row["q10"]), target)
        upper = transformed_value(float(row["q90"]), target)
        require(lower <= upper, "CROSSED_QUANTILES")
        grouped.setdefault(family, []).append(max(lower - truth, truth - upper, 0.0))
    return finite_sample_group_quantile(
        {family: max(scores) for family, scores in grouped.items()}, alpha
    )


@dataclass(frozen=True)
class ExecutorConfig:
    phase2_dir: Path
    output_dir: Path
    frozen_config_hash: str
    expected_input_hashes: Mapping[str, str]
    conformal_alpha: float = 0.20
    device: str = "cuda"


class BatteryScopeFoldExecutor:
    """Concrete leakage-guarded executor consumed by the frozen runner."""

    def __init__(self, config: ExecutorConfig):
        self.config = config
        self.phase2_dir = Path(config.phase2_dir)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._audit: list[dict[str, Any]] = []
        self._prepared = False
        self._outer_labels_loaded = False
        self._active_source: str | None = None
        self._roles: dict[str, set[str]] | None = None
        self._verified_cycle_shards: dict[Path, str] = {}
        self._verify_input_bundle()
        deps = _dependencies()
        self.feature_spec = json.loads(
            (self.phase2_dir / REQUIRED_PHASE2_FILES["feature_spec"]).read_text(encoding="utf-8")
        )
        self.feature_columns = list(self.feature_spec.get("tabular_feature_columns", []))
        self.feature_mask_columns = list(self.feature_spec.get("tabular_mask_columns", []))
        self.value_channels = list(self.feature_spec.get("value_channels", []))
        self.value_mask_columns = list(self.feature_spec.get("value_mask_columns", []))
        require(len(self.feature_columns) == 112, "TABULAR_FEATURE_COUNT_NOT_112")
        require(len(self.feature_mask_columns) == 112, "TABULAR_MASK_COUNT_NOT_112")
        require(len(self.value_channels) == 48, "TCN_VALUE_CHANNEL_COUNT_NOT_48")
        require(len(self.value_mask_columns) == 48, "TCN_MASK_CHANNEL_COUNT_NOT_48")
        self.fold_membership = deps.pd.read_parquet(
            self.phase2_dir / REQUIRED_PHASE2_FILES["fold_membership"]
        )
        self._validate_fold_membership_schema()
        self._cycle_index = None

    @property
    def input_hash_bundle_sha256(self) -> str:
        return sha256_json({key: str(value).upper() for key, value in sorted(self.config.expected_input_hashes.items())})

    def _verify_input_bundle(self) -> None:
        expected = {str(key): str(value).upper() for key, value in self.config.expected_input_hashes.items()}
        require(set(expected) == set(REQUIRED_PHASE2_FILES), "INPUT_HASH_KEY_SET_MISMATCH")
        for key, name in REQUIRED_PHASE2_FILES.items():
            path = self.phase2_dir / name
            require(path.is_file(), f"PHASE2_INPUT_MISSING:{name}")
            require(sha256_file(path) == expected[key], f"PHASE2_INPUT_SHA256_MISMATCH:{name}")

    def _validate_fold_membership_schema(self) -> None:
        required = {
            "evaluation_mode", "outer_entity", "canonical_cell_id",
            "duplicate_family_id", "split", "inner_group_fold",
        }
        missing = required - set(self.fold_membership.columns)
        require(not missing, f"FOLD_MEMBERSHIP_SCHEMA_MISMATCH:{sorted(missing)}")
        self.fold_membership["canonical_cell_id"] = self.fold_membership["canonical_cell_id"].astype(str)
        self.fold_membership["duplicate_family_id"] = self.fold_membership["duplicate_family_id"].astype(str)
        self.fold_membership["outer_entity"] = self.fold_membership["outer_entity"].astype(str)

    def _roles_for_source(self, outer_source: str) -> dict[str, set[str]]:
        deps = _dependencies()
        fold = self.fold_membership.loc[
            self.fold_membership["evaluation_mode"].astype(str).eq("LODO")
            & self.fold_membership["outer_entity"].astype(str).eq(str(outer_source))
        ].copy()
        require(not fold.empty, f"NO_LODO_MEMBERSHIP:{outer_source}")
        require(fold["canonical_cell_id"].is_unique, f"DUPLICATE_FOLD_CELL:{outer_source}")
        roles = {
            role: set(fold.loc[fold["split"].astype(str).eq(role), "canonical_cell_id"].astype(str))
            for role in ("development", "calibration", "test")
        }
        require(all(roles.values()), f"EMPTY_OUTER_ROLE:{outer_source}")
        cell_family = fold.set_index("canonical_cell_id")["duplicate_family_id"].astype(str).to_dict()
        for left_index, left in enumerate(roles):
            for right in tuple(roles)[left_index + 1:]:
                require(not (roles[left] & roles[right]), f"CELL_ROLE_LEAKAGE:{left}:{right}")
                left_families = {cell_family[cell] for cell in roles[left]}
                right_families = {cell_family[cell] for cell in roles[right]}
                require(not (left_families & right_families), f"FAMILY_ROLE_LEAKAGE:{left}:{right}")
        inner = deps.pd.to_numeric(
            fold.loc[fold["split"].astype(str).eq("development"), "inner_group_fold"],
            errors="coerce",
        )
        require(inner.notna().all() and set(inner.astype(int).unique()) == set(range(5)), "INNER_FOLD_SET_MISMATCH")
        return roles

    def _record_read(self, path: Path, columns: Sequence[str], role: str, cells: set[str]) -> None:
        if role == "outer_test_predictors":
            forbidden = sorted(set(columns) & FORBIDDEN_OUTER_PREDICTOR_COLUMNS)
            require(not forbidden, f"OUTER_TARGET_COLUMN_REQUEST_BEFORE_REVEAL:{forbidden}")
            require(not self._outer_labels_loaded, "OUTER_PREDICTOR_READ_AFTER_LABEL_REVEAL")
        self._audit.append({
            "event": "parquet_read", "file": path.name, "role": role,
            "columns": list(columns), "cell_count": len(cells),
            "outer_labels_loaded": self._outer_labels_loaded,
        })

    def _read_for_cells(self, path: Path, columns: Sequence[str], cells: set[str], role: str):
        deps = _dependencies()
        require(cells, f"EMPTY_CELL_FILTER:{path.name}:{role}")
        schema = set(deps.pq.read_schema(path).names)
        missing = sorted(set(columns) - schema)
        require(not missing, f"PARQUET_COLUMNS_MISSING:{path.name}:{missing}")
        self._record_read(path, columns, role, cells)
        table = deps.ds.dataset(str(path), format="parquet").to_table(
            columns=list(columns),
            filter=deps.ds.field("canonical_cell_id").isin(sorted(cells)),
        )
        frame = table.to_pandas()
        frame["canonical_cell_id"] = frame["canonical_cell_id"].astype(str)
        return frame

    def _tabular_predictors(self, cells: set[str], role: str):
        columns = list(dict.fromkeys(
            METADATA_COLUMNS + self.feature_columns
        ))
        frame = self._read_for_cells(
            self.phase2_dir / REQUIRED_PHASE2_FILES["tabular_index"], columns, cells, role
        )
        require(frame["sample_id"].is_unique, f"DUPLICATE_TABULAR_SAMPLE_ID:{role}")
        if role == "outer_test_predictors":
            require(not (set(frame.columns) & FORBIDDEN_OUTER_PREDICTOR_COLUMNS), "OUTER_TARGET_COLUMN_MATERIALIZED")
        return frame

    def _target_labels(self, cells: set[str], target: str, role: str):
        require(role in {"development_labels", "calibration_labels"}, "EARLY_OUTER_LABEL_ROLE_FORBIDDEN")
        spec = TARGET_SPECS[target]
        columns = list(dict.fromkeys(METADATA_COLUMNS + [spec["y_col"], spec["mask_col"]]))
        frame = self._read_for_cells(
            self.phase2_dir / REQUIRED_PHASE2_FILES["target_index"], columns, cells, role
        )
        masks = normalize_bool_values(frame[spec["mask_col"]].tolist(), spec["mask_col"])
        deps = _dependencies()
        numeric = deps.pd.to_numeric(frame[spec["y_col"]], errors="coerce")
        observed = deps.np.asarray(masks, dtype=bool) & deps.np.isfinite(numeric.to_numpy(dtype=float))
        if target.startswith("rul"):
            require((numeric.to_numpy(dtype=float)[observed] >= 0.0).all(), f"NEGATIVE_{target.upper()}_LABEL")
        frame = frame.loc[observed].copy()
        frame[spec["y_col"]] = numeric.loc[frame.index].astype(float)
        require(frame["sample_id"].is_unique, f"DUPLICATE_TARGET_SAMPLE_ID:{role}:{target}")
        return frame

    def _join_labeled_predictors(self, predictors, labels, target: str):
        spec = TARGET_SPECS[target]
        keep = ["sample_id", "canonical_cell_id", spec["y_col"]]
        merged = predictors.merge(
            labels[keep], on=["sample_id", "canonical_cell_id"], how="inner",
            validate="one_to_one",
        )
        require(not merged.empty, f"NO_OBSERVED_TRAIN_OR_CALIBRATION_ROWS:{target}")
        return merged

    @staticmethod
    def _family(model_id: str) -> str:
        if model_id in NAIVE_MODELS:
            return "naive"
        if model_id.startswith("ridge_alpha_"):
            return "ridge"
        if model_id == "extra_trees_locked":
            return "extra_trees"
        if model_id == "hist_gradient_boosting_locked":
            return "hist_gradient_boosting"
        raise RuntimeError(f"UNKNOWN_POINT_MODEL:{model_id}")

    def _fit_preprocessor(self, train, family: str) -> dict[str, Any]:
        deps = _dependencies()
        values = train[self.feature_columns].to_numpy(dtype=float, copy=False)
        count = len(self.feature_columns)
        median = deps.np.full(count, deps.np.nan)
        scale = deps.np.full(count, deps.np.nan)
        lower = deps.np.full(count, deps.np.nan)
        upper = deps.np.full(count, deps.np.nan)
        active = deps.np.zeros(count, dtype=bool)
        for index in range(count):
            finite = values[:, index][deps.np.isfinite(values[:, index])]
            if len(finite) < 2:
                continue
            median[index] = float(deps.np.median(finite))
            spread = float(deps.np.quantile(finite, 0.75) - deps.np.quantile(finite, 0.25))
            if not math.isfinite(spread) or spread <= 1e-12:
                spread = float(deps.np.std(finite))
            if not math.isfinite(spread) or spread <= 1e-12:
                continue
            scaled = (finite - median[index]) / spread
            active[index] = True
            scale[index] = spread
            lower[index] = float(deps.np.min(scaled))
            upper[index] = float(deps.np.max(scaled))
        require(bool(active.any()), "NO_ACTIVE_TABULAR_FEATURE")
        def finite_or_none(vector):
            return [float(value) if math.isfinite(float(value)) else None for value in vector]
        return {
            "family": family, "feature_columns": self.feature_columns,
            "active": active.tolist(), "median": finite_or_none(median),
            "scale": finite_or_none(scale), "scaled_min": finite_or_none(lower),
            "scaled_max": finite_or_none(upper), "fit_rows": len(train),
            "fit_cells": int(train["canonical_cell_id"].nunique()),
            "fit_role": "outer_development_only",
        }

    def _transform_features(self, frame, state: Mapping[str, Any], family: str):
        deps = _dependencies()
        active = deps.np.asarray(state["active"], dtype=bool)
        values = frame[self.feature_columns].to_numpy(dtype=float, copy=False)[:, active]
        median = deps.np.asarray(state["median"], dtype=float)[active]
        if family in {"ridge", "extra_trees"}:
            values = deps.np.where(deps.np.isfinite(values), values, median)
        if family == "ridge":
            scale = deps.np.asarray(state["scale"], dtype=float)[active]
            lower = deps.np.asarray(state["scaled_min"], dtype=float)[active]
            upper = deps.np.asarray(state["scaled_max"], dtype=float)[active]
            values = deps.np.clip((values - median) / scale, lower, upper)
            require(deps.np.isfinite(values).all(), "RIDGE_NONFINITE_MATRIX")
            return values.astype(float, copy=False)
        return values.astype("float32", copy=False)

    def _naive_predict(self, model_id: str, target: str, train, frame):
        deps = _dependencies()
        spec = TARGET_SPECS[target]
        y_train = train[spec["y_col"]].to_numpy(dtype=float)
        train_median = float(deps.np.median(y_train))
        current = frame[SOH_CURRENT_FEATURE].to_numpy(dtype=float)
        if model_id == "soh_persistence":
            return deps.np.clip(deps.np.where(deps.np.isfinite(current), current, train_median), 0.0, 1.5)
        if model_id == "rul_train_median":
            return deps.np.full(len(frame), train_median)
        if model_id == "rul_train_log_mean":
            return deps.np.full(len(frame), max(0.0, deps.np.expm1(deps.np.log1p(y_train).mean())))
        require(model_id == "causal_linear_trend", f"UNKNOWN_NAIVE_MODEL:{model_id}")
        slope = frame[SOH_TREND_FEATURE].to_numpy(dtype=float)
        if target == "soh30":
            usable = deps.np.isfinite(slope) & (slope < 0.0)
            prediction = deps.np.where(usable, current + 30.0 * slope, current)
            prediction = deps.np.where(deps.np.isfinite(prediction), prediction, train_median)
            return deps.np.clip(prediction, 0.0, 1.5)
        threshold = 0.90 if target == "rul90" else 0.80
        train_max = float(deps.np.max(y_train))
        prediction = deps.np.full(len(frame), train_median)
        reached = deps.np.isfinite(current) & (current <= threshold)
        prediction[reached] = 0.0
        decreasing = deps.np.isfinite(current) & deps.np.isfinite(slope) & (current > threshold) & (slope < 0.0)
        extrapolated = deps.np.full(len(frame), deps.np.nan)
        extrapolated[decreasing] = (threshold - current[decreasing]) / slope[decreasing]
        supported = decreasing & deps.np.isfinite(extrapolated) & (extrapolated >= 0.0) & (extrapolated <= train_max)
        prediction[supported] = extrapolated[supported]
        return deps.np.clip(prediction, 0.0, train_max)

    def _sklearn_predict(self, model_id: str, target: str, train, calibration, test):
        deps = _dependencies()
        family = self._family(model_id)
        state = self._fit_preprocessor(train, family)
        x_train = self._transform_features(train, state, family)
        x_calibration = self._transform_features(calibration, state, family)
        x_test = self._transform_features(test, state, family)
        y_natural = train[TARGET_SPECS[target]["y_col"]].to_numpy(dtype=float)
        y_transformed = deps.np.log1p(y_natural) if target.startswith("rul") else y_natural.copy()
        target_state: dict[str, Any] = {"transform": TARGET_SPECS[target]["transform"]}
        if family == "ridge":
            centre = float(y_transformed.mean())
            scale = float(y_transformed.std())
            if not math.isfinite(scale) or scale <= 1e-12:
                scale = 1.0
            y_fit = (y_transformed - centre) / scale
            target_state.update({
                "centre": centre, "scale": scale,
                "scaled_min": float(y_fit.min()), "scaled_max": float(y_fit.max()),
            })
            alpha = float(model_id.removeprefix("ridge_alpha_"))
            require(alpha in {1.0, 10.0, 100.0}, "UNREGISTERED_RIDGE_ALPHA")
            model = deps.Ridge(alpha=alpha, **RIDGE_CONFIG)
        elif family == "extra_trees":
            y_fit = y_transformed
            model = deps.ExtraTreesRegressor(random_state=SEED, **EXTRA_TREES_CONFIG)
        else:
            y_fit = y_transformed
            model = deps.HistGradientBoostingRegressor(random_state=SEED, **HIST_GB_CONFIG)
        model.fit(x_train, y_fit)

        def restore(raw):
            raw = deps.np.asarray(raw, dtype=float)
            require(deps.np.isfinite(raw).all(), f"NONFINITE_MODEL_OUTPUT:{model_id}:{target}")
            if family == "ridge":
                raw = deps.np.clip(raw, target_state["scaled_min"], target_state["scaled_max"])
                raw = raw * target_state["scale"] + target_state["centre"]
            natural = deps.np.expm1(raw) if target.startswith("rul") else raw
            upper = 1.5 if target == "soh30" else None
            return deps.np.clip(natural, 0.0, upper)

        state["model_id"] = model_id
        state["model_config"] = (
            {"alpha": alpha, **RIDGE_CONFIG} if family == "ridge"
            else EXTRA_TREES_CONFIG if family == "extra_trees" else HIST_GB_CONFIG
        )
        state["target_state"] = target_state
        return restore(model.predict(x_calibration)), restore(model.predict(x_test)), state

    def _checkpoint_frame(self, frame, target: str, model_id: str) -> Path:
        deps = _dependencies()
        path = self.output_dir / "prepared" / target / f"{model_id}.parquet"
        sidecar = path.with_suffix(path.suffix + ".integrity.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".partial")
        temporary.unlink(missing_ok=True)
        frame.to_parquet(temporary, index=False, engine="pyarrow", compression="zstd")
        os.replace(temporary, path)
        atomic_json(sidecar, {
            "path": str(path.relative_to(self.output_dir)), "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path), "rows": len(frame), "target": target,
            "model_id": model_id, "frozen_config_hash": self.config.frozen_config_hash,
            "input_hash_bundle_sha256": self.input_hash_bundle_sha256,
        })
        return path

    def _cached_frame(self, target: str, model_id: str):
        deps = _dependencies()
        path = self.output_dir / "prepared" / target / f"{model_id}.parquet"
        sidecar = path.with_suffix(path.suffix + ".integrity.json")
        if not path.is_file() or not sidecar.is_file():
            return None
        try:
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
            valid = (
                meta.get("frozen_config_hash") == self.config.frozen_config_hash
                and meta.get("input_hash_bundle_sha256") == self.input_hash_bundle_sha256
                and int(meta.get("size_bytes", -1)) == path.stat().st_size
                and str(meta.get("sha256", "")).upper() == sha256_file(path)
            )
            return deps.pd.read_parquet(path) if valid else None
        except Exception:
            return None

    def _point_candidate(self, model_id: str, target: str, train, calibration, test):
        deps = _dependencies()
        cached = self._cached_frame(target, model_id)
        if cached is not None:
            return cached
        if model_id in NAIVE_MODELS:
            calibration_prediction = self._naive_predict(model_id, target, train, calibration)
            test_prediction = self._naive_predict(model_id, target, train, test)
            state = {"model_id": model_id, "family": "naive", "fit_role": "outer_development_only"}
        else:
            calibration_prediction, test_prediction, state = self._sklearn_predict(
                model_id, target, train, calibration, test
            )
        calibration_rows = calibration[
            ["sample_id", "canonical_cell_id", "duplicate_family_id", "source"]
        ].copy()
        calibration_rows["y_true"] = calibration[TARGET_SPECS[target]["y_col"]].to_numpy(dtype=float)
        calibration_rows["y_pred"] = calibration_prediction
        qhat = grouped_point_qhat(
            calibration_rows.to_dict("records"), target, self.config.conformal_alpha
        )
        prediction = deps.np.asarray(test_prediction, dtype=float)
        centres = deps.np.log1p(prediction) if target.startswith("rul") else prediction
        lower_z, upper_z = centres - qhat, centres + qhat
        lower = deps.np.expm1(lower_z) if target.startswith("rul") else lower_z
        upper = deps.np.expm1(upper_z) if target.startswith("rul") else upper_z
        lower = deps.np.maximum(lower, 0.0)
        upper = deps.np.minimum(deps.np.maximum(upper, lower), 1.5) if target == "soh30" else deps.np.maximum(upper, lower)
        require(
            deps.np.isfinite(prediction).all() and deps.np.isfinite(lower).all() and deps.np.isfinite(upper).all(),
            f"NONFINITE_POINT_PREDICTION_OR_INTERVAL:{model_id}:{target}",
        )
        output = test[["sample_id", "canonical_cell_id", "duplicate_family_id", "source"]].copy()
        output["target"] = target
        output["model_id"] = model_id
        output["y_pred"] = prediction
        output["lower"] = lower
        output["upper"] = upper
        output["interval_method"] = "grouped_split_conformal"
        output["calibration_qhat"] = qhat
        output["calibration_family_count"] = int(calibration_rows["duplicate_family_id"].astype(str).nunique())
        state.update({
            "target": target, "conformal_method": "grouped_split_conformal",
            "conformal_scale": "log1p" if target.startswith("rul") else "natural",
            "calibration_qhat": qhat,
        })
        atomic_json(self.output_dir / "states" / target / f"{model_id}.json", state)
        self._checkpoint_frame(output, target, model_id)
        return output

    def _load_cycle_index(self):
        if self._cycle_index is not None:
            return
        deps = _dependencies()
        path = self.phase2_dir / "multisensor_cycle_store_index.parquet"
        require(path.is_file(), "MULTISENSOR_CYCLE_INDEX_MISSING")
        frame = deps.pd.read_parquet(path)
        required = {"canonical_cell_id", "shard_path", "sha256", "rows"}
        require(required.issubset(frame.columns), f"CYCLE_INDEX_SCHEMA_MISMATCH:{sorted(required - set(frame.columns))}")
        frame["canonical_cell_id"] = frame["canonical_cell_id"].astype(str)
        require(frame["canonical_cell_id"].is_unique, "DUPLICATE_CYCLE_INDEX_CELL")
        self._cycle_index = frame.set_index("canonical_cell_id", drop=False)

    def _cycle_shard(self, cell: str):
        deps = _dependencies()
        self._load_cycle_index()
        require(cell in self._cycle_index.index, f"CYCLE_SHARD_INDEX_MISSING:{cell}")
        row = self._cycle_index.loc[cell]
        candidates: list[Path] = []
        if "shard_relpath" in row.index and deps.pd.notna(row["shard_relpath"]):
            relative = Path(str(row["shard_relpath"]))
            require(not relative.is_absolute() and ".." not in relative.parts, f"UNSAFE_CYCLE_SHARD_RELPATH:{cell}")
            candidates.append(self.phase2_dir / relative)
        if "shard_path" in row.index and deps.pd.notna(row["shard_path"]):
            recorded = Path(str(row["shard_path"]))
            candidates.append(self.phase2_dir / "batteryscope_phase2_multisensor_cycle_store" / recorded.name)
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        require(path is not None, f"CYCLE_SHARD_MISSING:{cell}")
        path = path.resolve()
        require(self.phase2_dir.resolve() in path.parents, f"CYCLE_SHARD_OUTSIDE_PHASE2_BUILD:{cell}")
        expected_sha256 = str(row["sha256"]).upper()
        require(len(expected_sha256) == 64, f"CYCLE_SHARD_HASH_MISSING:{cell}")
        if self._verified_cycle_shards.get(path) != expected_sha256:
            require(sha256_file(path) == expected_sha256, f"CYCLE_SHARD_SHA256_MISMATCH:{cell}")
            self._verified_cycle_shards[path] = expected_sha256
        columns = list(dict.fromkeys([
            "canonical_cell_id", "cycle_index", *self.value_channels, *self.value_mask_columns,
        ]))
        frame = deps.pd.read_parquet(path, columns=columns)
        require(frame["canonical_cell_id"].astype(str).eq(cell).all(), f"CYCLE_SHARD_IDENTITY_MISMATCH:{cell}")
        require(len(frame) == int(row["rows"]), f"CYCLE_SHARD_ROW_COUNT_MISMATCH:{cell}")
        return frame.sort_values("cycle_index", kind="mergesort").reset_index(drop=True)

    @staticmethod
    def _evenly_spaced(length: int, maximum: int):
        deps = _dependencies()
        require(length > 0 and maximum > 0, "INVALID_RESERVOIR_SIZE")
        if length <= maximum:
            return deps.np.arange(length, dtype="int64")
        return deps.np.unique(deps.np.rint(deps.np.linspace(0, length - 1, maximum)).astype("int64"))

    def _fit_cycle_scaler(self, cells: set[str]) -> dict[str, Any]:
        deps = _dependencies()
        count = len(self.value_channels)
        sums = deps.np.zeros(count, dtype=float)
        squares = deps.np.zeros(count, dtype=float)
        counts = deps.np.zeros(count, dtype="int64")
        for ordinal, cell in enumerate(sorted(cells), start=1):
            shard = self._cycle_shard(cell)
            selected = self._evenly_spaced(len(shard), TCN_SCALER_POINTS_PER_CELL)
            values = shard.iloc[selected][self.value_channels].to_numpy(dtype=float)
            masks = shard.iloc[selected][self.value_mask_columns].to_numpy(dtype=bool)
            valid = masks & deps.np.isfinite(values)
            safe = deps.np.where(valid, values, 0.0)
            sums += safe.sum(axis=0); squares += (safe * safe).sum(axis=0); counts += valid.sum(axis=0)
            if ordinal % 50 == 0:
                gc.collect()
        safe_counts = deps.np.maximum(counts, 1)
        mean = sums / safe_counts
        variance = deps.np.maximum(squares / safe_counts - mean * mean, 0.0)
        mean[counts == 0] = 0.0
        std = deps.np.sqrt(variance)
        std = deps.np.where(deps.np.isfinite(std) & (std > 1e-8), std, 1.0)
        return {"mean": mean.tolist(), "std": std.tolist(), "fit_role": "outer_development_only"}

    def _all_target_rows(self, cells: set[str], role: str):
        deps = _dependencies()
        require(role in {"development_labels", "calibration_labels"}, "INVALID_TARGET_READ_ROLE")
        frame = self._read_for_cells(
            self.phase2_dir / REQUIRED_PHASE2_FILES["target_index"],
            METADATA_COLUMNS + TARGET_COLUMNS, cells, role,
        )
        any_observed = deps.np.zeros(len(frame), dtype=bool)
        for target, spec in TARGET_SPECS.items():
            values = deps.pd.to_numeric(frame[spec["y_col"]], errors="coerce")
            masks = deps.np.asarray(normalize_bool_values(frame[spec["mask_col"]].tolist(), spec["mask_col"]), dtype=bool)
            observed = masks & deps.np.isfinite(values.to_numpy(dtype=float))
            if target.startswith("rul"):
                require((values.to_numpy(dtype=float)[observed] >= 0.0).all(), f"NEGATIVE_{target.upper()}_LABEL")
            frame[spec["y_col"]] = values.astype("float32")
            frame[spec["mask_col"]] = observed
            any_observed |= observed
        return frame.loc[any_observed].copy()

    def _window_rows(self, cells: set[str], role: str, labels=None):
        deps = _dependencies()
        windows = self._read_for_cells(
            self.phase2_dir / REQUIRED_PHASE2_FILES["window_index"],
            ["sample_id", "canonical_cell_id", "t_cycle"], cells, role,
        )
        metadata = self._read_for_cells(
            self.phase2_dir / REQUIRED_PHASE2_FILES["target_index"],
            METADATA_COLUMNS, cells, role,
        )
        base = windows.merge(metadata, on=["sample_id", "canonical_cell_id", "t_cycle"], how="inner", validate="one_to_one")
        require(len(base) == len(windows), f"WINDOW_METADATA_JOIN_LOSS:{role}")
        if labels is None:
            return base.sort_values(["canonical_cell_id", "t_cycle"], kind="mergesort").reset_index(drop=True)
        columns = ["sample_id", "canonical_cell_id", "t_cycle", *TARGET_COLUMNS]
        merged = base.merge(labels[columns], on=["sample_id", "canonical_cell_id", "t_cycle"], how="inner", validate="one_to_one")
        require(len(merged) == len(labels), f"WINDOW_TARGET_JOIN_LOSS:{role}")
        return merged.sort_values(["canonical_cell_id", "t_cycle"], kind="mergesort").reset_index(drop=True)

    def _fit_target_scaler(self, rows) -> dict[str, Any]:
        deps = _dependencies()
        state: dict[str, Any] = {"fit_role": "outer_development_only", "targets": {}}
        for target, spec in TARGET_SPECS.items():
            sampled = []
            for _, group in rows.groupby("canonical_cell_id", sort=True):
                observed = group[spec["mask_col"]].to_numpy(dtype=bool)
                values = group[spec["y_col"]].to_numpy(dtype=float)[observed]
                if not len(values):
                    continue
                selected = self._evenly_spaced(len(values), TCN_TARGET_POINTS_PER_CELL)
                transformed = deps.np.log1p(values[selected]) if target.startswith("rul") else values[selected]
                sampled.append(transformed)
            require(sampled, f"TCN_NO_DEVELOPMENT_TARGET_SUPPORT:{target}")
            reservoir = deps.np.concatenate(sampled)
            mean = float(reservoir.mean()); std = float(reservoir.std())
            if not math.isfinite(std) or std <= 1e-8:
                std = 1.0
            state["targets"][target] = {"mean": mean, "std": std, "transform": spec["transform"]}
        return state

    def _make_windows(self, cycle_frame, sample_frame, scaler: Mapping[str, Any]):
        deps = _dependencies()
        cycles = cycle_frame["cycle_index"].to_numpy(dtype="int64")
        require(len(cycles) > 0 and (deps.np.diff(cycles) > 0).all(), "CYCLE_INDEX_NOT_STRICT")
        values = cycle_frame[self.value_channels].to_numpy(dtype="float32")
        masks = cycle_frame[self.value_mask_columns].to_numpy(dtype=bool)
        t_cycles = sample_frame["t_cycle"].to_numpy(dtype="int64")
        grid = t_cycles[:, None] - deps.np.arange(29, -1, -1, dtype="int64")[None, :]
        positions = deps.np.searchsorted(cycles, grid)
        safe = deps.np.minimum(positions, len(cycles) - 1)
        observed = (positions < len(cycles)) & (cycles[safe] == grid)
        require(observed[:, -1].all(), "TCN_CAUSAL_ENDPOINT_MISSING")
        raw = values[safe]
        raw_masks = masks[safe] & observed[:, :, None]
        valid = raw_masks & deps.np.isfinite(raw)
        mean = deps.np.asarray(scaler["mean"], dtype="float32")[None, None, :]
        std = deps.np.asarray(scaler["std"], dtype="float32")[None, None, :]
        standardized = deps.np.zeros(raw.shape, dtype="float32")
        z = (raw - mean) / std
        standardized[valid] = z[valid]
        output = deps.np.concatenate([standardized, valid.astype("float32")], axis=2)
        require(output.shape[1:] == (30, 96) and deps.np.isfinite(output).all(), "TCN_INPUT_CONTRACT_FAILURE")
        return output

    def _encode_targets(self, rows, scaler: Mapping[str, Any]):
        deps = _dependencies()
        y = deps.np.zeros((len(rows), 3), dtype="float32")
        masks = deps.np.zeros_like(y, dtype=bool)
        for index, (target, spec) in enumerate(TARGET_SPECS.items()):
            observed = rows[spec["mask_col"]].to_numpy(dtype=bool)
            values = rows[spec["y_col"]].to_numpy(dtype=float)
            transformed = deps.np.log1p(values[observed]) if target.startswith("rul") else values[observed]
            state = scaler["targets"][target]
            y[observed, index] = ((transformed - state["mean"]) / state["std"]).astype("float32")
            masks[observed, index] = True
        require(masks.any(axis=1).all(), "TCN_ROW_WITHOUT_OBSERVED_TARGET")
        return y, masks

    def _make_tcn(self):
        deps = _dependencies(); nn = deps.nn; F = deps.F

        class CausalConv1d(nn.Conv1d):
            def __init__(self, in_channels, out_channels, kernel_size, dilation):
                self.left_padding = (kernel_size - 1) * dilation
                super().__init__(in_channels, out_channels, kernel_size, padding=self.left_padding, dilation=dilation)

            def forward(self, x):
                output = super().forward(x)
                return output[:, :, :-self.left_padding] if self.left_padding else output

        class ResidualBlock(nn.Module):
            def __init__(self, in_channels, out_channels, dilation):
                super().__init__()
                self.conv1 = CausalConv1d(in_channels, out_channels, TCN_KERNEL_SIZE, dilation)
                self.conv2 = CausalConv1d(out_channels, out_channels, TCN_KERNEL_SIZE, dilation)
                self.dropout = nn.Dropout(TCN_DROPOUT)
                self.projection = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

            def forward(self, x):
                hidden = self.dropout(F.relu(self.conv1(x)))
                hidden = self.dropout(F.relu(self.conv2(hidden)))
                return F.relu(hidden + self.projection(x))

        class OrderedHead(nn.Module):
            def __init__(self, width):
                super().__init__(); self.projection = nn.Linear(width, 3)

            def forward(self, hidden):
                raw = self.projection(hidden); q50 = raw[:, 0]
                return deps.torch.stack([q50 - F.softplus(raw[:, 1]), q50, q50 + F.softplus(raw[:, 2])], dim=1)

        class MultitaskTCN(nn.Module):
            def __init__(self):
                super().__init__(); blocks = []; width = 96
                for out_width, dilation in zip(TCN_BLOCK_CHANNELS, TCN_DILATIONS):
                    blocks.append(ResidualBlock(width, out_width, dilation)); width = out_width
                self.trunk = nn.Sequential(*blocks)
                self.heads = nn.ModuleDict({target: OrderedHead(width) for target in TARGETS})

            def forward(self, x):
                require(tuple(x.shape[1:]) == (30, 96), "TCN_TENSOR_SHAPE_MISMATCH")
                hidden = self.trunk(x.transpose(1, 2))[:, :, -1]
                return {target: self.heads[target](hidden) for target in TARGETS}

        return MultitaskTCN()

    def _tcn_loss(self, predictions, targets, masks):
        deps = _dependencies(); torch = deps.torch
        quantiles = torch.tensor(TCN_QUANTILES, device=targets.device).view(1, -1)
        total = None; weight_total = 0.0
        for index, target in enumerate(TARGETS):
            observed = masks[:, index].bool()
            if not bool(observed.any()):
                continue
            error = targets[observed, index].float().view(-1, 1) - predictions[target][observed].float()
            loss = torch.maximum(quantiles * error, (quantiles - 1.0) * error).mean()
            weight = TCN_TASK_WEIGHTS[target]
            total = loss * weight if total is None else total + loss * weight
            weight_total += weight
        require(total is not None and weight_total > 0.0, "TCN_BATCH_HAS_NO_TARGET")
        return total / weight_total

    def _tcn_checkpoint(self) -> Path:
        return self.output_dir / "checkpoints" / "quantile_tcn_locked.progress.pt"

    def _save_tcn_checkpoint(self, payload: Mapping[str, Any]) -> None:
        deps = _dependencies(); path = self._tcn_checkpoint(); path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".partial")
        deps.torch.save(dict(payload), temporary); os.replace(temporary, path)
        atomic_json(path.with_suffix(path.suffix + ".integrity.json"), {
            "size_bytes": path.stat().st_size, "sha256": sha256_file(path),
            "frozen_config_hash": self.config.frozen_config_hash,
            "input_hash_bundle_sha256": self.input_hash_bundle_sha256,
            "outer_source": self._active_source,
        })

    def _load_tcn_checkpoint(self, device):
        deps = _dependencies(); path = self._tcn_checkpoint(); sidecar = path.with_suffix(path.suffix + ".integrity.json")
        if not path.is_file() or not sidecar.is_file():
            return None
        try:
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
            valid = (
                meta.get("frozen_config_hash") == self.config.frozen_config_hash
                and meta.get("input_hash_bundle_sha256") == self.input_hash_bundle_sha256
                and meta.get("outer_source") == self._active_source
                and int(meta.get("size_bytes", -1)) == path.stat().st_size
                and str(meta.get("sha256", "")).upper() == sha256_file(path)
            )
            if not valid:
                return None
            return deps.torch.load(path, map_location=device, weights_only=False)
        except Exception:
            return None

    def _train_tcn(self, rows, cycle_scaler, target_scaler, epochs: int):
        deps = _dependencies(); np = deps.np; torch = deps.torch
        require(1 <= int(epochs) <= 30, "TCN_REFIT_EPOCHS_OUT_OF_RANGE")
        require(self.config.device == "cuda" and torch.cuda.is_available(), "OFFICIAL_TCN_REQUIRES_CUDA")
        torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        seed_key = f"{SEED}|LODO|{self._active_source}|outer_refit|multitask"
        fold_seed = SEED + int(hashlib.sha256(seed_key.encode()).hexdigest()[:8], 16) % 1_000_000
        random.seed(fold_seed); np.random.seed(fold_seed); torch.manual_seed(fold_seed); torch.cuda.manual_seed_all(fold_seed)
        device = torch.device("cuda")
        model = self._make_tcn().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=TCN_LEARNING_RATE, weight_decay=TCN_WEIGHT_DECAY)
        grad_scaler = torch.amp.GradScaler("cuda", enabled=True)
        start_epoch = 1
        checkpoint = self._load_tcn_checkpoint(device)
        if checkpoint is not None:
            require(int(checkpoint.get("target_epochs", -1)) == int(epochs), "TCN_RESUME_EPOCH_TARGET_MISMATCH")
            model.load_state_dict(checkpoint["model_state_dict"]); optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            require("grad_scaler_state_dict" in checkpoint, "TCN_RESUME_GRAD_SCALER_STATE_MISSING")
            grad_scaler.load_state_dict(checkpoint["grad_scaler_state_dict"])
            torch.set_rng_state(checkpoint["torch_rng_state"])
            if checkpoint.get("cuda_rng_state_all") is not None:
                torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
            start_epoch = int(checkpoint["completed_epoch"]) + 1
        groups = {str(cell): group.reset_index(drop=True) for cell, group in rows.groupby("canonical_cell_id", sort=True)}
        require(groups, "TCN_NO_DEVELOPMENT_CELL_GROUPS")
        for epoch in range(start_epoch, int(epochs) + 1):
            model.train(); optimizer.zero_grad(set_to_none=True); microbatches = 0
            rng = np.random.default_rng(fold_seed + 1009 * epoch)
            cells = np.asarray(sorted(groups), dtype=object); rng.shuffle(cells)
            padding = (-len(cells)) % TCN_GRAD_ACCUM_STEPS
            if padding:
                cells = np.concatenate([cells, cells[:padding]])
            for cell_value in cells:
                cell = str(cell_value); group = groups[cell]
                selected = rng.choice(len(group), size=TCN_MICRO_BATCH_SIZE, replace=len(group) < TCN_MICRO_BATCH_SIZE)
                batch = group.iloc[selected].reset_index(drop=True)
                shard = self._cycle_shard(cell)
                x = torch.from_numpy(self._make_windows(shard, batch, cycle_scaler)).to(device)
                y_np, mask_np = self._encode_targets(batch, target_scaler)
                y = torch.from_numpy(y_np).to(device); masks = torch.from_numpy(mask_np).to(device)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                    predictions = model(x)
                    loss = self._tcn_loss(predictions, y, masks)
                grad_scaler.scale(loss / TCN_GRAD_ACCUM_STEPS).backward(); microbatches += 1
                if microbatches % TCN_GRAD_ACCUM_STEPS == 0:
                    grad_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), TCN_GRADIENT_CLIP_NORM)
                    grad_scaler.step(optimizer); grad_scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                del shard, x, y, masks, predictions, loss
            require(microbatches > 0 and microbatches % TCN_GRAD_ACCUM_STEPS == 0, "TCN_INCOMPLETE_ACCUMULATION")
            self._save_tcn_checkpoint({
                "model_state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                "optimizer_state_dict": optimizer.state_dict(), "completed_epoch": epoch,
                "grad_scaler_state_dict": grad_scaler.state_dict(),
                "target_epochs": int(epochs), "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
                "frozen_config_hash": self.config.frozen_config_hash,
            })
        return model, device

    def _predict_tcn(self, model, device, rows, cycle_scaler, target_scaler):
        deps = _dependencies(); np = deps.np; torch = deps.torch
        parts = {target: [] for target in TARGETS}; model.eval()
        with torch.no_grad():
            for ordinal, (cell, group) in enumerate(rows.groupby("canonical_cell_id", sort=True), start=1):
                shard = self._cycle_shard(str(cell)); group = group.reset_index(drop=True)
                for start in range(0, len(group), TCN_PREDICTION_BATCH_SIZE):
                    batch = group.iloc[start:start + TCN_PREDICTION_BATCH_SIZE].reset_index(drop=True)
                    x = torch.from_numpy(self._make_windows(shard, batch, cycle_scaler)).to(device)
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                        outputs = model(x)
                    for target in TARGETS:
                        values = outputs[target].detach().float().cpu().numpy()
                        state = target_scaler["targets"][target]
                        restored = values * state["std"] + state["mean"]
                        restored = np.expm1(restored) if target.startswith("rul") else restored
                        restored = np.clip(restored, 0.0, 1.5 if target == "soh30" else None)
                        require(np.isfinite(restored).all(), f"TCN_NONFINITE_PREDICTION:{target}")
                        require((restored[:, 0] <= restored[:, 1]).all() and (restored[:, 1] <= restored[:, 2]).all(), "TCN_CROSSED_QUANTILES")
                        part = batch[["sample_id", "canonical_cell_id", "duplicate_family_id", "source"]].copy()
                        part["q10"] = restored[:, 0]; part["q50"] = restored[:, 1]; part["q90"] = restored[:, 2]
                        parts[target].append(part)
                    del x, outputs
                del shard
                if ordinal % 50 == 0:
                    gc.collect()
        return {target: deps.pd.concat(values, ignore_index=True) for target, values in parts.items()}

    def _prepare_tcn(self, plan, development_labels, calibration_labels, test_rows):
        deps = _dependencies(); np = deps.np
        cached = {target: self._cached_frame(target, "quantile_tcn_locked") for target in TARGETS}
        if all(frame is not None for frame in cached.values()):
            return cached
        development_rows = self._window_rows(self._roles["development"], "development_labels", development_labels)
        calibration_rows = self._window_rows(self._roles["calibration"], "calibration_labels", calibration_labels)
        cycle_scaler = self._fit_cycle_scaler(self._roles["development"])
        target_scaler = self._fit_target_scaler(development_rows)
        model, device = self._train_tcn(development_rows, cycle_scaler, target_scaler, int(plan.tcn_refit_epochs))
        calibration_predictions = self._predict_tcn(model, device, calibration_rows, cycle_scaler, target_scaler)
        test_predictions = self._predict_tcn(model, device, test_rows, cycle_scaler, target_scaler)
        outputs = {}
        for target in TARGETS:
            spec = TARGET_SPECS[target]
            calibration = calibration_predictions[target].merge(
                calibration_rows[["sample_id", "canonical_cell_id", spec["y_col"], spec["mask_col"]]],
                on=["sample_id", "canonical_cell_id"], how="inner", validate="one_to_one",
            )
            calibration = calibration.loc[calibration[spec["mask_col"]].astype(bool)].copy()
            calibration["y_true"] = calibration[spec["y_col"]].astype(float)
            qhat = grouped_cqr_qhat(calibration.to_dict("records"), target, self.config.conformal_alpha)
            output = test_predictions[target].copy()
            lower_z = np.log1p(output["q10"].to_numpy(dtype=float)) if target.startswith("rul") else output["q10"].to_numpy(dtype=float)
            upper_z = np.log1p(output["q90"].to_numpy(dtype=float)) if target.startswith("rul") else output["q90"].to_numpy(dtype=float)
            lower = np.expm1(lower_z - qhat) if target.startswith("rul") else lower_z - qhat
            upper = np.expm1(upper_z + qhat) if target.startswith("rul") else upper_z + qhat
            output["target"] = target; output["model_id"] = "quantile_tcn_locked"
            output["y_pred"] = output["q50"].astype(float)
            output["lower"] = np.maximum(lower, 0.0)
            output["upper"] = np.minimum(np.maximum(upper, output["lower"]), 1.5) if target == "soh30" else np.maximum(upper, output["lower"])
            output["interval_method"] = "grouped_cqr"
            output["calibration_qhat"] = qhat
            output["calibration_family_count"] = int(calibration["duplicate_family_id"].astype(str).nunique())
            self._checkpoint_frame(output, target, "quantile_tcn_locked")
            outputs[target] = output
        atomic_json(self.output_dir / "states" / "quantile_tcn_locked.json", {
            "model_id": "quantile_tcn_locked", "fit_role": "outer_development_only",
            "refit_epochs": int(plan.tcn_refit_epochs), "cycle_scaler": cycle_scaler,
            "target_scaler": target_scaler, "conformal_method": "grouped_cqr",
            "rul_conformal_scale": "log1p",
        })
        del model; gc.collect(); deps.torch.cuda.empty_cache()
        return outputs

    def prepare_without_outer_labels(self, plan) -> dict[str, Any]:
        """Fit/calibrate every candidate and prepare outer predictors without labels."""
        require(not self._prepared and not self._outer_labels_loaded, "EXECUTOR_PREPARE_CALLED_TWICE")
        source = str(plan.outer_source)
        self._active_source = source
        self._roles = self._roles_for_source(source)
        deps = _dependencies()
        development = self._tabular_predictors(self._roles["development"], "development_predictors")
        calibration_predictors = self._tabular_predictors(self._roles["calibration"], "calibration_predictors")
        test_predictors = self._tabular_predictors(self._roles["test"], "outer_test_predictors")
        require(set(test_predictors["source"].astype(str)) == {source}, "OUTER_TEST_PREDICTOR_SOURCE_MISMATCH")
        require(source not in set(development["source"].astype(str)), "HELD_OUT_SOURCE_IN_DEVELOPMENT")
        development_labels_by_target = {
            target: self._target_labels(self._roles["development"], target, "development_labels")
            for target in TARGETS
        }
        calibration_labels_by_target = {
            target: self._target_labels(self._roles["calibration"], target, "calibration_labels")
            for target in TARGETS
        }
        predictions: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        for target in TARGETS:
            train = self._join_labeled_predictors(development, development_labels_by_target[target], target)
            calibration = self._join_labeled_predictors(calibration_predictors, calibration_labels_by_target[target], target)
            for model_id in EXPECTED_CANDIDATES[target]:
                if model_id == "quantile_tcn_locked":
                    continue
                last_exception: Exception | None = None
                completed = False
                for retry_count in (0, 1):
                    try:
                        frame = self._point_candidate(model_id, target, train, calibration, test_predictors)
                        predictions.extend(frame.to_dict("records"))
                        completed = True
                        break
                    except Exception as exc:
                        last_exception = exc
                        if retry_count == 0:
                            gc.collect()
                if not completed:
                    require(last_exception is not None, "MODEL_FAILURE_WITHOUT_EXCEPTION")
                    failures.append({
                        "outer_source": source, "target": target, "model_id": model_id,
                        "stage": "prepare_without_outer_labels", "exception_class": type(last_exception).__name__,
                        "reason": str(last_exception), "retry_count": 1,
                        "occurred_before_outer_label_access": True,
                    })
            del train, calibration
        del (
            development, calibration_predictors, test_predictors,
            development_labels_by_target, calibration_labels_by_target,
        )
        gc.collect()
        development_all = self._all_target_rows(self._roles["development"], "development_labels")
        calibration_all = self._all_target_rows(self._roles["calibration"], "calibration_labels")
        test_windows = self._window_rows(self._roles["test"], "outer_test_predictors")
        tcn_exception: Exception | None = None
        tcn_complete = False
        for retry_count in (0, 1):
            try:
                tcn = self._prepare_tcn(plan, development_all, calibration_all, test_windows)
                for target in TARGETS:
                    predictions.extend(tcn[target].to_dict("records"))
                tcn_complete = True
                break
            except Exception as exc:
                tcn_exception = exc
                if retry_count == 0:
                    gc.collect()
        if not tcn_complete:
            require(tcn_exception is not None, "TCN_FAILURE_WITHOUT_EXCEPTION")
            for target in TARGETS:
                failures.append({
                    "outer_source": source, "target": target, "model_id": "quantile_tcn_locked",
                    "stage": "prepare_without_outer_labels", "exception_class": type(tcn_exception).__name__,
                    "reason": str(tcn_exception), "retry_count": 1,
                    "occurred_before_outer_label_access": True,
                })
        self._prepared = True
        audit_path = self.output_dir / "outer_access_audit.pre_reveal.json"
        atomic_json(audit_path, {
            "schema_version": "batteryscope-outer-access-audit-v1",
            "outer_source": source, "prepared_complete": True,
            "outer_labels_loaded": False, "events": self._audit,
            "prediction_rows": len(predictions), "failures": failures,
        })
        return {"predictions": predictions, "failures": failures}

    def load_outer_labels_once(self, outer_source: str) -> list[dict[str, Any]]:
        """The only method permitted to request held-out target columns."""
        require(self._prepared, "OUTER_LABEL_REQUEST_BEFORE_PREPARATION")
        require(not self._outer_labels_loaded, "OUTER_LABEL_REQUEST_MORE_THAN_ONCE")
        require(str(outer_source) == self._active_source, "OUTER_LABEL_SOURCE_MISMATCH")
        require(self._roles is not None, "OUTER_ROLE_STATE_MISSING")
        deps = _dependencies(); path = self.phase2_dir / REQUIRED_PHASE2_FILES["target_index"]
        columns = list(dict.fromkeys(METADATA_COLUMNS + TARGET_COLUMNS))
        self._audit.append({
            "event": "outer_label_reveal", "file": path.name, "role": "outer_test_labels",
            "columns": columns, "cell_count": len(self._roles["test"]),
            "outer_labels_loaded": False,
        })
        self._outer_labels_loaded = True
        table = deps.ds.dataset(str(path), format="parquet").to_table(
            columns=columns,
            filter=deps.ds.field("canonical_cell_id").isin(sorted(self._roles["test"])),
        )
        frame = table.to_pandas(); rows: list[dict[str, Any]] = []
        require(set(frame["source"].astype(str)) == {str(outer_source)}, "OUTER_LABEL_HELD_SOURCE_MISMATCH")
        for target, spec in TARGET_SPECS.items():
            values = deps.pd.to_numeric(frame[spec["y_col"]], errors="coerce")
            masks = deps.np.asarray(normalize_bool_values(frame[spec["mask_col"]].tolist(), spec["mask_col"]), dtype=bool)
            observed = masks & deps.np.isfinite(values.to_numpy(dtype=float))
            if target.startswith("rul"):
                require((values.to_numpy(dtype=float)[observed] >= 0.0).all(), "NEGATIVE_OUTER_RUL_LABEL")
            selected = frame.loc[observed, ["sample_id", "source"]].copy()
            selected["target"] = target; selected["y_true"] = values.loc[selected.index].astype(float)
            rows.extend(selected.to_dict("records"))
        atomic_json(self.output_dir / "outer_access_audit.post_reveal.json", {
            "schema_version": "batteryscope-outer-access-audit-v1",
            "outer_source": outer_source, "outer_labels_loaded": True,
            "events": self._audit, "observed_evaluable_label_rows": len(rows),
        })
        require(rows, "NO_EVALUABLE_OUTER_LABELS_FOR_SOURCE")
        return rows


__all__ = [
    "EXECUTOR_CONTRACT", "IMPLEMENTATION_VERSION", "ExecutorConfig",
    "BatteryScopeFoldExecutor", "grouped_point_qhat", "grouped_cqr_qhat",
    "finite_sample_group_quantile", "normalize_bool_values", "sha256_file",
]
