"""Protocol tests: fair orders, exact targets, exposure and telemetry parsing."""
from collections import Counter
from pathlib import Path
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import torch

from prismserve.batching import BatchOutput,BatchResult
from prismserve.study_monitor import parse_gpu_csv,summarize_gpu
from prismserve.validation_study import balanced_order,disjoint_trace,fixed_quality,paired_interval,require_power


class FakeTokenizer:
    def encode(self,*args,**kwargs):
        return list(range(64))*80


class FakeExecutor:
    tokenizer=FakeTokenizer()
    context_limit=1024
    def set_quantized_layers(self,q):
        self.q=q


def fake_batch(executor,inputs):
    outputs=[]
    for item in inputs:
        length=len(item.token_ids)+(0 if item.cache is None else item.cache[0][0].shape[2])
        cache=[(torch.zeros(1,1,length,1),torch.zeros(1,1,length,1))]
        logits=torch.arange(64,dtype=torch.float32).unsqueeze(0)*.01
        outputs.append(BatchOutput(item.request_id,63,logits,cache))
    return BatchResult(outputs,sum(len(i.token_ids) for i in inputs),48)


class StudyTests(unittest.TestCase):
    def test_ac_condition_rejects_a_midrun_battery_sample(self):
        monitor=SimpleNamespace(before={'power_plugged':True},after={'power_plugged':True},
                                host=[{'power_plugged':True},{'power_plugged':False}])
        with self.assertRaises(RuntimeError):
            require_power({'required_ac_power':True},monitor)
        monitor.host=[{'power_plugged':True}]
        require_power({'required_ac_power':True},monitor)
        monitor.after['power_plugged']=False
        require_power({},monitor)

    def test_balanced_order_covers_each_position_once(self):
        policies=['full_v03','joint_v03','full','joint']
        for scenario in range(3):
            orders=[balanced_order(policies,i,scenario) for i in range(4)]
            for position in range(4):
                self.assertEqual(Counter(o[position] for o in orders),Counter(policies))

    def test_quality_windows_are_disjoint_and_never_wrap(self):
        config={'requests':16,'prompt_tokens':128,'output_tokens':128,'duration_s':1}
        trace=disjoint_trace(FakeExecutor(),config)
        positions=[set(range(r['dataset_token_start'],r['dataset_token_start']+256)) for r in trace]
        self.assertEqual(sum(len(p) for p in positions),len(set.union(*positions)))
        self.assertEqual(sum(len(r['reference_ids']) for r in trace),2048)
        with self.assertRaises(ValueError):
            disjoint_trace(FakeExecutor(),dict(config,requests=50))

    def test_fixed_schedules_score_the_same_targets_and_track_hidden_prefill_exposure(self):
        trace=disjoint_trace(FakeExecutor(),{'requests':2,'prompt_tokens':128,'output_tokens':4,'duration_s':1})
        results={}
        with patch('prismserve.batching.execute_batch',side_effect=fake_batch):
            for schedule in ('fp32','static_int4','early_prefill_int4_restore','decode_int4','cpu_attention'):
                results[schedule]=fixed_quality(FakeExecutor(),trace,schedule,(0,),{})
        for result in results.values():
            self.assertEqual(result['scored_targets'],8)
            self.assertEqual(result['trace'],trace)
            self.assertAlmostEqual(result['nll'],results['fp32']['nll'])
        self.assertEqual(results['static_int4']['quantized_scored_targets'],8)
        self.assertEqual(results['decode_int4']['quantized_scored_targets'],6)
        self.assertEqual(results['early_prefill_int4_restore']['quantized_scored_targets'],0)
        self.assertEqual(results['early_prefill_int4_restore']['quantized_prefill_iterations'],1)
        self.assertEqual(results['cpu_attention']['cpu_attention_scored_targets'],6)

    def test_gpu_null_fields_flags_and_time_window(self):
        line='2026/11/18 18:00:00.000, 0, GPU-test, 54, 2100, 7001, [N/A], 50, 700, P3, Active, Not Active, [N/A]'
        rows=parse_gpu_csv(line+'\nunsupported output')
        self.assertEqual(len(rows),1)
        self.assertIsNone(rows[0]['power.draw'])
        self.assertTrue(rows[0]['clocks_event_reasons.sw_power_cap'])
        self.assertFalse(rows[0]['clocks_event_reasons.hw_thermal_slowdown'])
        summary=summarize_gpu(rows,rows[0]['unix_s']-.1,rows[0]['unix_s']+.1)
        self.assertEqual(summary['temperature.gpu']['mean'],54)
        self.assertIsNone(summary['power.draw'])
        self.assertEqual(summarize_gpu(rows,0,1)['samples'],0)

    def test_paired_interval_is_seeded_and_retains_negative_outcomes(self):
        first=paired_interval([-5,-2,-1,0,2],samples=1000)
        self.assertEqual(first,paired_interval([-5,-2,-1,0,2],samples=1000))
        self.assertLess(first['mean'],0)
        self.assertLessEqual(first['lower_95'],first['upper_95'])
        with self.assertRaises(ValueError):
            paired_interval([])


if __name__=='__main__':
    unittest.main()
