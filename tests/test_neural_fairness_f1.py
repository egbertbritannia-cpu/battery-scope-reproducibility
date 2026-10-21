"""NFR1 code/guard/synthetic QA, not actual Kaggle or historical replay."""
import ast
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "protocol_and_provenance/neural_classical_fairness_v1"
VERSION = "v4"
OUT = ROOT / f"notebook/neural-classical-fairness-v1/diagnostic-{VERSION}"
PAYLOAD = OUT / f"batteryscope_nfr1_diagnostic_{VERSION}"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def synthetic_namespace():
    import gc, math, random, time
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    recipe = json.loads((ROOT / "phase2-ouput/batteryscope_corrected_inner_v2_kaggle/batteryscope_corrected_inner_v2/code/inner_recipe.json").read_text())
    ns = dict(np=np, pd=pd, torch=torch, nn=nn, F=F, gc=gc, math=math,
              random=random, time=time, hashlib=hashlib, json=json, Path=Path)
    ns.update(recipe["literal_constants"])
    feature = json.loads((ROOT / "phase2-ouput/batteryscope_corrected_inner_v2_kaggle/batteryscope_corrected_inner_v2/phase2/feature_spec.json").read_text())
    ns.update(VALUE_CHANNELS=feature["value_channels"], VALUE_MASK_COLUMNS=feature["value_mask_columns"], SEED=2026)
    for record in recipe["definitions"]:
        if hashlib.sha256(record["source"].encode()).hexdigest() != record["source_sha256"]:
            raise RuntimeError("BOUND_DEFINITION_DRIFT")
        exec(compile("from __future__ import annotations\n" + record["source"], "<synthetic-bound-recipe>", "exec"), ns)
    return ns


class FairnessF1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)
        sys.path.insert(0, str(PAYLOAD / "scripts"))
        cls.builder = load("nfr1_builder", HERE / "build_diagnostic_delivery.py")
        cls.checks = load("nfr1_checks", HERE / "runtime/forensic_checks.py")
        cls.runtime = load("nfr1_gpu_runtime", PAYLOAD / "scripts/neural_kaggle_runtime.py")
        cls.reviewer = load("nfr1_reviewer", HERE / "review_diagnostic_results.py")
        cls.manifest = json.loads((PAYLOAD / "neural_diagnostic_delivery_manifest.json").read_text())
        cls.cfg = json.loads((PAYLOAD / f"protocol_and_provenance/data_integrity/nfr1_diagnostic_{VERSION}_A.json").read_text())

    @contextmanager
    def synthetic_role_context(self):
        with patch.object(self.reviewer, "bound_context", return_value=dict(
            train_cell_hash=hashlib.sha256(b"synthetic.train").hexdigest(),
            validation_cell_hash=hashlib.sha256(b"synthetic.val").hexdigest())), \
             patch.object(self.reviewer, "replay_target_support", return_value=dict(
                 train_cells=["synthetic.train"], represented_cells=["synthetic.train"], train_rows=1,
                 target_scaler=dict(fit_cell_hash=self.reviewer.digest(["synthetic.train"])),
                 excluded_from_target_scaler=[], train_only_reads=[])):
            yield

    def test_adapter_reproduces_and_preserves_original(self):
        original = (PAYLOAD / "scripts/gpu_v1_reference.frozen.py").read_text()
        current = (PAYLOAD / "scripts/neural_kaggle_runtime.py").read_text()
        self.assertEqual(self.builder.adapt(original, release=int(VERSION[1:])), current)
        base = ROOT / "notebook/neural-inner-diagnostic-kaggle-v1/batteryscope_neural_kaggle_diagnostic_v1/scripts/neural_kaggle_runtime.py"
        self.assertEqual(base.read_bytes(), (PAYLOAD / "scripts/gpu_v1_reference.frozen.py").read_bytes())

    def test_exact_six_whole_jobs_no_duplicates(self):
        jobs = [j for values in self.manifest["shard_jobs"].values() for j in values]
        self.assertEqual(sorted((j["inner_fold"], j["seed_base"]) for j in jobs),
                         [(f, s) for f in range(3) for s in (2026, 2027)])
        self.assertEqual([len(self.manifest["shard_jobs"][s]) for s in "ABCD"], [2, 2, 1, 1])
        for s in "ABCD":
            cfg = json.loads((PAYLOAD / f"protocol_and_provenance/data_integrity/nfr1_diagnostic_{VERSION}_{s}.json").read_text())
            self.assertEqual(cfg["jobs"], self.manifest["shard_jobs"][s])
            self.assertEqual(cfg["campaign"]["max_epochs_per_job"], 20)
            self.assertFalse(any(cfg["guards"].values()))
            self.assertFalse(cfg["amp"])
            self.assertFalse(cfg["campaign"]["automatic_resume"])

    def test_code_zip_bound_and_contains_no_dataset(self):
        with zipfile.ZipFile(OUT / f"batteryscope_nfr1_diagnostic_{VERSION}_code.zip") as z:
            self.assertIsNone(z.testzip())
            self.assertEqual(len(z.namelist()), len(self.manifest["files"]) + 1)
            for name, record in self.manifest["files"].items():
                data = z.read(PAYLOAD.name + "/" + name)
                self.assertEqual(hashlib.sha256(data).hexdigest(), record["sha256"])
                self.assertEqual(len(data), record["bytes"])
            self.assertFalse(any(n.endswith((".parquet", ".pt", ".pkl", ".joblib")) for n in z.namelist()))

    def test_notebooks_empty_and_shard_specific_and_output_only(self):
        workspaces = set()
        for s in "ABCD":
            nb = json.loads((OUT / f"BatteryScope_NFR1_Diagnostic_{s}_{VERSION}.ipynb").read_text())
            code = []
            for cell in nb["cells"]:
                if cell["cell_type"] == "code":
                    self.assertEqual(cell["outputs"], []); self.assertIsNone(cell["execution_count"])
                    text = "".join(cell["source"]); compile(text, "notebook-cell", "exec"); code.append(text)
            self.assertIn(f"nfr1_diagnostic_{VERSION}_{s}.json", code[0])
            self.assertIn(f"batteryscope_nfr1_diagnostic_{VERSION}_{s}')", code[0])
            self.assertIn("receipt['files']", code[1]); self.assertNotIn("shutil.make_archive", code[1])
            self.assertIn("finally:", code[1])
            self.assertLess(code[1].index("run_stage('pilot')"), code[1].index("run_stage('campaign')"))
            workspaces.add(next(line for line in code[0].splitlines() if line.startswith("WORKSPACE =")))
        self.assertEqual(len(workspaces), 4)

    def test_config_direct_launch_and_tampering_fail_closed(self):
        # Real local payload verification doesn't import Torch or open input data.
        with patch.object(self.runtime, "ROOT", PAYLOAD), patch.object(self.runtime, "PROTOCOL", PAYLOAD / f"protocol_and_provenance/data_integrity/nfr1_diagnostic_{VERSION}_A.json"):
            self.assertEqual(self.runtime.config()["jobs"], self.cfg["jobs"])
        modified = json.loads(json.dumps(self.cfg)); modified["jobs"] = [dict(inner_fold=1, seed_base=2026)]
        with patch.object(self.runtime, "ROOT", PAYLOAD), patch.object(self.runtime, "PROTOCOL", PAYLOAD / f"protocol_and_provenance/data_integrity/nfr1_diagnostic_{VERSION}_A.json"), patch.object(Path, "read_text", return_value=json.dumps(modified)):
            with self.assertRaisesRegex(RuntimeError, "SHARD_JOB_DRIFT"): self.runtime.config()

    def test_all_bound_synthetic_functional_checks(self):
        result = self.checks.run_checks(synthetic_namespace())
        self.assertEqual(result["status"], "F1_SYNTHETIC_BOUND_FUNCTION_CHECKS_PASS")
        self.assertGreaterEqual(len(result["checks"]), 16)
        self.assertFalse(result["outer_labels_accessed"])

    def test_synthetic_checks_do_not_consume_rng(self):
        import torch
        torch.manual_seed(783)
        state = torch.get_rng_state().clone()
        self.checks.run_checks(synthetic_namespace())
        self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_raw_inverse_bounds_and_order(self):
        import numpy as np
        scaler = dict(targets={"soh30": dict(mean=0., std=1., transform="identity"),
                              "rul90": dict(mean=0., std=1., transform="log1p")})
        transformed, raw, bounded = self.checks.raw_inverse(np.array([[-1., .2, 2.]]), "soh30", scaler)
        self.assertEqual(raw.tolist(), [[-1., .2, 2.]])
        self.assertEqual(bounded.tolist(), [[0., .2, 1.5]])
        _, raw, bounded = self.checks.raw_inverse(np.array([[-2., 0., 2.]]), "rul90", scaler)
        self.assertLess(raw[0, 0], 0.); self.assertEqual(bounded[0, 0], 0.)

    def test_seed_before_constructor_replay_sampler_not_changed(self):
        ns = synthetic_namespace()
        with patch.object(self.runtime, "DEVICE", "cpu"):
            a, _, seed = self.runtime.initialize(ns, 2027, 0)
            b, _, other = self.runtime.initialize(ns, 2027, 0)
        self.assertEqual(seed, other)
        self.assertEqual(self.checks.parameter_digest(a), self.checks.parameter_digest(b))
        self.assertEqual(ns["SEED"], 2026)

    def test_training_accumulation_and_instrumentation(self):
        import numpy as np
        import torch
        ns = synthetic_namespace()
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.trunk = torch.nn.Linear(2, 3)
                self.heads = torch.nn.ModuleDict({t: torch.nn.Linear(3, 3) for t in self_targets})
            def forward(self, x):
                return {t: head(self.trunk(x)) for t, head in self.heads.items()}
        self_targets = self.runtime.TARGETS
        model = Model(); optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-5)
        class Iterator:
            def iter_cell_balanced_microbatches(self, epoch):
                for _ in range(8):
                    yield np.ones((2, 2), dtype=np.float32), np.ones((2, 3), dtype=np.float32), np.ones((2, 3), dtype=bool), None
        class Guard:
            def check(self): pass
        before = self.checks.parameter_digest(model)
        with patch.object(self.runtime, "DEVICE", "cpu"):
            result = self.runtime.train_epoch(ns, model, optimizer, Iterator(), 1, Guard())
        self.assertEqual(result["optimizer_updates"], 1); self.assertEqual(result["microbatches"], 8)
        self.assertEqual(result["observed_target_exposures"], {t: 16 for t in self_targets})
        self.assertTrue(result["parameters_and_optimizer_finite"])
        self.assertEqual(result["first_development_batch_gradient_probe"]["optimizer_updates"], 0)
        self.assertNotEqual(before, self.checks.parameter_digest(model))
        total_norm = sum(v*v for v in result["gradient_norms_after_clip_mean"].values()) ** .5
        self.assertLessEqual(total_norm, 1.00001)

    def test_output_seal_excludes_extracted_data_and_cache(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "_review_tmp") as tmp:
            d = Path(tmp)
            for n in ("input_extracted/raw.pkl", "_review_tmp/cache/x", "scripts/x.py", "protocol.json", "paper/audits/job/result.json"):
                p = d / n; p.parent.mkdir(parents=True, exist_ok=True); p.write_text("synthetic")
            result = self.checks.seal_output(d, dict(shard_id="A", jobs=[]), d / "scripts/x.py", d / "protocol.json")
            self.assertNotIn("input_extracted/raw.pkl", result["files"])
            self.assertNotIn("_review_tmp/cache/x", result["files"])
            self.assertIn("paper/audits/job/result.json", result["files"])
            with self.assertRaisesRegex(RuntimeError, "refuse_overwrite_output_seal"):
                self.checks.seal_output(d, dict(shard_id="A", jobs=[]), d / "scripts/x.py", d / "protocol.json")

    def test_merger_rejects_missing_shards_before_reads(self):
        with self.assertRaisesRegex(RuntimeError, "FOUR_SHARDS_REQUIRED"):
            self.reviewer.merge([])

    def synthetic_return(self, directory, shard, mutate=None):
        """A fake return used only to test the reviewer, never research evidence."""
        version = PAYLOAD.name.rsplit("_", 1)[1]
        cfg_name = f"protocol_and_provenance/data_integrity/nfr1_diagnostic_{version}_{shard}.json"
        cfg = json.loads((PAYLOAD / cfg_name).read_text())
        ph = self.manifest["files"][cfg_name]["sha256"]
        rh = self.manifest["files"]["scripts/neural_kaggle_runtime.py"]["sha256"]
        files = {n: (PAYLOAD / n).read_bytes() for n in self.manifest["files"]}
        files["neural_diagnostic_delivery_manifest.json"] = (PAYLOAD / "neural_diagnostic_delivery_manifest.json").read_bytes()
        out = cfg["output_root"] + "/"
        def add(name, value):
            files[out + name] = json.dumps(value, allow_nan=False).encode()
        env = dict(cfg["kaggle_environment"], gpu_count=1,
                   packages={"numpy":"2.1.3", "pandas":"2.3.3", "pyarrow":"23.0.1", "scikit-learn":"1.6.1"})
        if version in ("v3", "v4"):
            env["packages"].update(torch=env["torch"], psutil="synthetic")
            env.update(identity_source="imported_module_version_and_file",
                imported_modules={k: dict(version=v, file="/synthetic/" + k + "/__init__.py")
                                  for k, v in env["packages"].items()},
                distribution_inventory={"numpy": [dict(version="2.1.3", location="/synthetic/local"),
                                                   dict(version="1.26.4", location="/synthetic/system")]},
                duplicate_distributions={"numpy": [dict(version="2.1.3", location="/synthetic/local"),
                                                    dict(version="1.26.4", location="/synthetic/system")]})
        add("environment.lock.json", env); add("protocol.lock.json", cfg)
        for mode, status in (("pilot", "KAGGLE_DIAGNOSTIC_PILOT_PASS"), ("campaign", "KAGGLE_DIAGNOSTIC_COMPLETE")):
            add(mode + "/result.json", dict(status=status, outer_scoring=False, completed_jobs=cfg["jobs"]))
            add(mode + "/started.json", dict(protocol_sha256=ph, runner_sha256=rh))
            add(mode + "/supervisor_complete.json", dict(worker_exit_code=0))
        pd = "pilot/inner.CALB.0.seed2026/"
        add(pd + "tiny_fit_result.json", dict(relative_loss_reduction=.5, maximum_parameter_delta=.1,
            transform_roundtrip="PASS", last_gradient_norms={t:.1 for t in ("trunk", *self.runtime.TARGETS)}))
        add(pd + "tiny_fit_history.json", [dict(step=s) for s in range(65)])
        layout = json.loads((ROOT / "phase2-ouput/batteryscope_corrected_inner_v2_kaggle/batteryscope_corrected_inner_v2/phase2/feature_spec.json").read_text())
        functional = dict(status="F1_SYNTHETIC_BOUND_FUNCTION_CHECKS_PASS",
            feature_layout=dict(value_channels=layout["value_channels"], value_mask_columns=layout["value_mask_columns"],
                                coordinate_column="cycle_index", perturbed_value_channels=[v for v in layout["value_channels"] if v != "cycle_index"]),
            checks=["synthetic"] * 18 + ["future_perturbation_preserves_cycle_coordinates",
                "masked_perturbation_preserves_cycle_coordinates", "masked_coordinate_feature_zero_lookup_coordinate_retained",
                "empty_cycle_index_rejected", "descending_cycle_index_rejected", "duplicate_cycle_index_rejected"])
        add(pd + "functional_checks.json", functional)
        add(pd + "pilot_epoch.json", dict(training=dict(optimizer_updates=1, microbatches=8)))
        for job in cfg["jobs"]:
            d = f"campaign/inner.CALB.{job['inner_fold']}.seed{job['seed_base']}/"
            train = ["synthetic.train"]
            scaler = dict(fit_cell_hash=self.reviewer.digest(train))
            add(d + "roles_and_scalers.json", dict(train_cells=train, validation_cells=["synthetic.val"],
                train_rows=1,
                pilot_subset=False, outer_labels_accessed=False, calibration_labels_accessed=False,
                target_scaler=scaler, cycle_scaler=scaler))
            add(d + "access_log.json", [dict(cell_ids=train), dict(cell_ids=["synthetic.val"])])
            add(d + "cycle_shard_bindings.json", {"synthetic.train": {}, "synthetic.val": {}})
            add(d + "complete.json", dict(status="DIAGNOSTIC_JOB_COMPLETE", official_fit=False,
                outer_labels_accessed=False, calibration_labels_accessed=False))
            add(d + "functional_checks.json", functional)
            add(d + "initialization_check.json", dict(replay="PASS"))
            add(d + "parameter_update_check.json", dict(changed=True))
            training = dict(microbatches=8, optimizer_updates=1, parameters_and_optimizer_finite=True,
                observed_target_exposures={t:16 for t in self.runtime.TARGETS},
                gradient_norms_after_clip_mean={t:.1 for t in ("trunk", *self.runtime.TARGETS)},
                gradient_norms_mean_per_update={t:.1 for t in ("trunk", *self.runtime.TARGETS)})
            validation = {t:dict(status="EVALUABLE", ranges={}) for t in self.runtime.TARGETS}
            add(d + "epoch_history.json", [dict(epoch=e, training=training, validation=validation) for e in range(1,21)])
            checkpoint = b"synthetic-not-a-real-checkpoint"
            files[out + d + "last.pt"] = checkpoint
            add(d + "last.integrity.json", dict(epoch=20, protocol_sha256=ph,
                bytes=len(checkpoint), sha256=hashlib.sha256(checkpoint).hexdigest()))
        if mutate:
            mutate(files, cfg)
        records = {n:dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest()) for n, data in files.items()}
        seal = dict(schema="nfr1-diagnostic-output-v2", shard_id=shard, jobs=cfg["jobs"],
                    protocol_sha256=ph, runtime_sha256=rh, files=records, outer_scoring=False, diagnostic_only=True)
        files["output_manifest.json"] = json.dumps(seal).encode()
        p = directory / f"{shard}.zip"
        with zipfile.ZipFile(p, "w") as z:
            for n, data in files.items():
                z.writestr(f"{PAYLOAD.name}_{shard}/" + n, data)
        return p

    def test_four_shard_synthetic_merger_and_duplicate_rejection(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "_review_tmp") as tmp, self.synthetic_role_context():
            directory = Path(tmp)
            paths = [self.synthetic_return(directory, s) for s in "ABCD"]
            result = self.reviewer.merge(paths)
            self.assertEqual(result["completed_diagnostic_fits"], 6)
            self.assertEqual(result["completed_epochs"], 120)
            self.assertFalse(result["fairness_closed"])
            with self.assertRaisesRegex(RuntimeError, "DUPLICATE_OR_MISSING_SHARD"):
                self.reviewer.merge([paths[0], paths[0], paths[2], paths[3]])

    def test_return_rejects_missing_epoch_even_with_valid_file_hashes(self):
        def mutation(files, cfg):
            key = next(n for n in files if n.endswith("epoch_history.json"))
            history = json.loads(files[key]); history.pop(); files[key] = json.dumps(history).encode()
        with tempfile.TemporaryDirectory(dir=ROOT / "_review_tmp") as tmp, self.synthetic_role_context():
            p = self.synthetic_return(Path(tmp), "A", mutation)
            with self.assertRaisesRegex(RuntimeError, "NOT_20_WHOLE_EPOCHS"):
                self.reviewer.review_one(p)

    def test_return_rejects_forbidden_role_even_with_valid_file_hashes(self):
        def mutation(files, cfg):
            key = next(n for n in files if n.endswith("access_log.json"))
            files[key] = json.dumps([dict(cell_ids=["synthetic.OUTER_FORBIDDEN"])]).encode()
        with tempfile.TemporaryDirectory(dir=ROOT / "_review_tmp") as tmp, self.synthetic_role_context():
            p = self.synthetic_return(Path(tmp), "A", mutation)
            with self.assertRaisesRegex(RuntimeError, "FORBIDDEN_LABEL_ROLE"):
                self.reviewer.review_one(p)

    def test_return_rejects_runtime_drift_even_with_new_output_hashes(self):
        def mutation(files, cfg):
            files["scripts/neural_kaggle_runtime.py"] += b"\n# synthetic tamper\n"
        with tempfile.TemporaryDirectory(dir=ROOT / "_review_tmp") as tmp:
            p = self.synthetic_return(Path(tmp), "A", mutation)
            with self.assertRaisesRegex(RuntimeError, "CODE_OR_PROTOCOL_DRIFT"):
                self.reviewer.review_one(p)

    def test_return_rejects_wrong_role_contract(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "_review_tmp") as tmp:
            p = self.synthetic_return(Path(tmp), "A")
            # Do not patch: the real frozen CALB roles must reject synthetic IDs.
            with self.assertRaisesRegex(RuntimeError, "BOUND_ROLE_MEMBERSHIP_DRIFT"):
                self.reviewer.review_one(p)

    def test_zip_safety_and_nonfinite_curves(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "_review_tmp") as tmp:
            for i, name in enumerate(("../escape", "/absolute", "C:/escape")):
                p = Path(tmp) / f"bad{i}.zip"
                with zipfile.ZipFile(p, "w") as z: z.writestr(name, "synthetic")
                with zipfile.ZipFile(p) as z:
                    with self.assertRaisesRegex(RuntimeError, "ZIP_UNSAFE_PATH"): self.reviewer.safe_zip(z)
        self.assertFalse(self.reviewer.finite({"curves": [float("nan")]}))

    def test_inventory_does_not_unpickle_or_train_or_import_selector(self):
        source = (HERE / "inventory.py").read_text()
        tree = ast.parse(source)
        called = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        self.assertFalse(called & {"fit", "backward", "step"})
        self.assertNotIn("torch.load", source); self.assertNotIn("pickle.load", source)


if __name__ == "__main__": unittest.main()
