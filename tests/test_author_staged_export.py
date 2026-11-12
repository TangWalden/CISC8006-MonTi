"""Synthetic saved-count/export guards only; no real records or ML imports."""
import copy
import pathlib
import sys
import unittest

ROOT=pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from experiments.author_staged import export_saved as module


def mock_rows(differences):
    rows=[]
    for seed,delta in zip(module.SEEDS,differences):
        for group in range(909):
            n=12 if group<516 else 11
            for arm in module.ARMS:
                change=delta if arm=='quota_learned' else 0
                row=dict(seed=seed,group=group,arm=arm,targets=n)
                for detector in ('gaga','gcn'):
                    value=change if detector=='gaga' else 0
                    row[detector]=dict(occurrences=n,clean_wrong=1,attacked_wrong=5+value,new_wrong=4+value,recoveries=0)
                rows.append(row)
    return rows


class SyntheticAuthorExportChecks(unittest.TestCase):
    def test_complete_matrix_and_positive_fixed_rule(self):
        metrics=module.case_metrics(mock_rows((1,1,1)))
        self.assertEqual(len(metrics),9)
        self.assertTrue(all(r['occurrences']==10515 and r['groups']==909 for r in metrics))
        contrasts,qualified=module.intervals(metrics)
        self.assertTrue(qualified)
        self.assertEqual(set(contrasts),{'native_monti','quota_random'})
        self.assertEqual(contrasts['native_monti']['nominal_paired_t95_pp'][0],contrasts['native_monti']['mean_pp'])
    def test_zero_and_negative_outcomes_stay_unqualified(self):
        metrics=module.case_metrics(mock_rows((1,0,-1)))
        contrasts,qualified=module.intervals(metrics)
        self.assertFalse(qualified)
        self.assertGreater(contrasts['native_monti']['seed_differences_pp'][0],0)
        self.assertEqual(contrasts['native_monti']['seed_differences_pp'][1],0)
        self.assertLess(contrasts['native_monti']['seed_differences_pp'][2],0)
        zero,qualified=module.intervals(module.case_metrics(mock_rows((0,0,0))))
        self.assertEqual(zero['native_monti']['nominal_paired_t95_pp'],[0.,0.]);self.assertFalse(qualified)
    def test_missing_duplicate_and_broken_transition_fail(self):
        rows=mock_rows((0,0,0))
        with self.assertRaises(RuntimeError):module.case_metrics(rows[:-1])
        with self.assertRaises(RuntimeError):module.case_metrics(rows[:-1]+[rows[0]])
        bad=copy.deepcopy(rows);bad[0]['gaga']['attacked_wrong']=0
        with self.assertRaises(RuntimeError):module.case_metrics(bad)
    def test_opaque_key_matches_auditor_namespace(self):
        self.assertEqual(module.opaque(0,123),module.opaque(0,123))
        self.assertEqual(len(module.opaque(0,123)),20)
        self.assertNotEqual(module.opaque(0,123),module.opaque(1,123))
    def test_public_audit_summary_keeps_raw_identity_and_filters_bulk(self):
        provenance=dict(source_commit='1'*40,runtime_commit='2'*40,protocol_sha256='3'*64)
        source_sha='4'*64;raw_sha='5'*64
        protocol=dict(source_manifest=dict(source_commit='1'*40,files={'experiments/author_staged/audit_saved.py':source_sha}))
        audited=dict(provenance=provenance,saved_rows=9090,clean_groups=909,attacked_cases=8181,
            occurrence_denominator_per_arm_seed=10515,unique_test_targets=1221,serialized_cache_files_hashed=5538,
            serialized_plan_files_hashed=2727,terminal_checkpoint_files_hashed=3,training_rows_recounted=11244,
            producer_logical_counters=dict(training_GCN_forwards=11244))
        raw=dict(status='passed',auditor_source_sha256=source_sha,recount=audited,limitations=['Synthetic scope'],
            model_queries=0,tensor_deserializations=0,checkpoint_deserializations=0,optimizer_updates=0,API_calls=0,GPU_hours=0,
            input_file_hashes={'private/example/path':'6'*64})
        value=module.public_audit_summary(raw,'saved_logit_recount',raw_sha,provenance,protocol)
        self.assertTrue(value['public_summary']);self.assertEqual(value['full_raw_audit_sha256'],raw_sha)
        self.assertEqual(value['core_counts']['attacked_cases'],8181);self.assertEqual(value['counters']['model_queries'],0)
        self.assertNotIn('input_file_hashes',value);self.assertNotIn('producer_logical_counters',value['counters'])


if __name__=='__main__':unittest.main()
