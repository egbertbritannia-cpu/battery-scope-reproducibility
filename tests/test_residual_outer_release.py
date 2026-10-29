"""Checks for complete seed/source support and equal-source aggregation."""
import importlib.util
from pathlib import Path
import statistics
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('released_residual',ROOT/'recompute_residual_outer.py')
residual=importlib.util.module_from_spec(spec)
spec.loader.exec_module(residual)


class ResidualReleaseTests(unittest.TestCase):
    def records(self):
        records=[]
        for i in range(12):
            for j,seed in enumerate(residual.SEEDS):
                mae=.01+i*.001+j*.0001
                records.append(dict(source=f'source{i:02}',seed=seed,system=residual.SYSTEM,
                    endpoint='soh30',mae=mae,rmse=mae+.003,bias=mae/2,cells=i+1,windows=(i+1)*100))
        return records

    def test_equal_source_weight_and_sample_sd(self):
        metrics,sources=residual.aggregate(self.records())
        expected=[.0155+j*.0001 for j in range(5)]
        self.assertAlmostEqual(metrics['mae']['mean'],statistics.mean(expected))
        self.assertAlmostEqual(metrics['mae']['sample_seed_sd'],statistics.stdev(expected))
        self.assertEqual(len(sources),12)

    def test_missing_or_duplicate_source_seed_is_rejected(self):
        original=self.records()
        for rows in (original[:-1],original[:-1]+[original[0]]):
            with self.assertRaises(ValueError):residual.aggregate(rows)

    def test_inconsistent_support_and_error_metrics_are_rejected(self):
        rows=self.records();rows[0]['windows']=1
        with self.assertRaises(ValueError):residual.aggregate(rows)
        rows=self.records();rows[0]['bias']=1
        with self.assertRaises(ValueError):residual.aggregate(rows)

    def test_released_aggregates_and_unassessed_rul(self):
        result=residual.reconstruct()
        self.assertEqual(len(result['sources']),12)
        self.assertAlmostEqual(result['metrics']['mae']['mean'],.022371865766392718)
        self.assertAlmostEqual(result['table3_row']['soh30_skill_percent'],-6.930466668680912)
        for field in ('rul90','rul90_seed_sd','rul90_skill_percent','rul80','rul80_seed_sd'):
            self.assertEqual(result['table3_row'][field],'')


if __name__=='__main__':unittest.main()
