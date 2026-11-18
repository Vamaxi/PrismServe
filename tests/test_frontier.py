"""Frontier outcome/selection protocols must retain negative outcomes."""
import json
from pathlib import Path
import unittest

from prismserve.frontier import outcome,select_audits,rescue_cells,validate_config


def run(policy,quota=600,completed=4,rejected=0,seed=7,quantized=0,prompt=384):
    return {'summary':{'policy':policy,'device_budget_mib':quota,'prompt_tokens':prompt,'seed':seed,
                       'requests':4,'completed':completed,'rejected':rejected},
            'exposure':{'quantized_iterations':quantized}}


class FrontierTests(unittest.TestCase):
    def test_rejected_partial_complete_and_failed_stay_distinct(self):
        self.assertEqual(outcome(run('full')['summary']),'complete')
        self.assertEqual(outcome(run('full',completed=0,rejected=4)['summary']),'rejected')
        self.assertEqual(outcome(run('full',completed=2,rejected=2)['summary']),'partial')
        self.assertEqual(outcome(dict(run('full')['summary'],worker_failed=True)),'failed')
        with self.assertRaises(ValueError):
            outcome(run('full',completed=2,rejected=1)['summary'])

    def test_audit_selection_requires_all_repeats_complete_and_real_exposure(self):
        rows=[run('morph_only',580,quantized=4),run('morph_only',580,seed=19,completed=0,rejected=4),
              run('morph_only',600,quantized=1),run('morph_only',600,seed=19,quantized=1),
              run('joint',600),run('joint',620,quantized=1),run('full',580,quantized=1)]
        selected=select_audits(rows)
        self.assertEqual({(r['policy'],r['device_budget_mib']) for r in selected},{('morph_only',600),('joint',620)})

    def test_rescue_requires_same_seed_context_quota_and_full_rejection(self):
        rows=[run('full',completed=0,rejected=4),run('joint',quantized=4),
              run('morph_only',quota=620,quantized=4),run('offload_only',seed=19)]
        result=rescue_cells(rows)
        self.assertEqual(len(result),1)
        self.assertEqual(result[0]['policy'],'joint')

    def test_context_and_duplicate_grid_validation(self):
        c=json.loads(Path('configs/capacity_frontier.json').read_text())
        validate_config(c)
        with self.assertRaises(ValueError):
            validate_config(dict(c,prompt_lengths=[1024]))
        with self.assertRaises(ValueError):
            validate_config(dict(c,device_quotas_mib=[600,600]))
        with self.assertRaises(ValueError):
            validate_config(dict(c,policies=['full','joint']))


if __name__=='__main__':
    unittest.main()
