"""Verify M10 protocols and evidence without executing model inference."""
from collections import defaultdict
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from prismserve.validation_study import STAGES


def load(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_replay(run,requests,tokens,quality=False):
    s=run['summary']
    assert not run['metrics']['fatal_error']
    assert s['requests']==s['completed']+s['rejected']==requests
    assert s['completed']==requests and s['rejected']==0
    assert s['mode']==('teacher_forced_quality_audit' if quality else 'greedy_generation')
    for r in run['requests']:
        assert r['completion_tokens']==tokens
        assert len(r['output_token_times_s'])==tokens
        assert len(r['decode_token_ms'])==tokens-1
        assert len(r['target_nll'])==(tokens if quality else 0)
    for event in run['iterations']:
        assert event['projection_calls']==48
        assert event['quality_proxy']<=run['config']['quality_budget']+1e-12
        if s.get('device')=='cpu':
            assert event['device_reserved_bytes']+event['host_reserved_bytes']<=run['config']['host_memory_budget_bytes']
            assert not event['cpu_request_ids']
        elif run['config']['device_memory_budget_bytes']:
            assert event['device_reserved_bytes']<=run['config']['device_memory_budget_bytes']
        if event['plan_reused']:
            assert event['resource_checks']>=1
    if s['peak_cuda_allocated_bytes'] is not None and run['config']['device_memory_budget_bytes']:
        assert s['peak_cuda_allocated_bytes']<=run['config']['device_memory_budget_bytes']


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'results/validation_study_final')
    parser.add_argument('--check-current',action='store_true')
    args=parser.parse_args()
    out=args.output
    c=load(out/'config.json')
    m=load(out/'manifest.json')
    assert m['status']=='complete' and set(m['completed_stages'])==set(STAGES)
    assert not list((out/'failures').rglob('*.json')), 'failed attempts must be reported and cannot be silently accepted'
    with zipfile.ZipFile(out/'source_snapshot.zip') as archive:
        for name,value in m['source_sha256'].items():
            assert hashlib.sha256(archive.read('prismserve/'+name)).hexdigest()==value
            if args.check_current:
                assert digest(ROOT/'prismserve'/name)==value
    assert digest(out/'source_snapshot.zip')==m['source_snapshot_sha256']
    for name,value in m['calibration_sha256'].items():
        assert digest(out/name)==value
    for split,value in m['data_sha256'].items():
        assert digest(ROOT/f'data/wikitext2/wiki.{split}.prefix.txt')==value
    cal_cpu=load(out/'calibration_cpu.json')
    cal_gpu=load(out/'calibration_cuda.json')
    assert cal_cpu['hardware']['device_kind']=='cpu'
    assert cal_gpu['hardware']['device_kind']=='cuda'
    assert cal_cpu['hardware']['link_bandwidth_gbps']==0
    assert all(s['cpu_requests']==0 for s in cal_cpu['timing']['samples'])
    runs=[load(p) for p in sorted((out/'gpu_runs').glob('*.json'))]
    assert len(runs)==len(c['policies'])*len(c['seeds'])*len(c['scenarios'])
    traces={}
    sampled=0
    for r in runs:
        check_replay(r,c['workload']['requests'],c['workload']['output_tokens'])
        s=r['summary']; key=(s['scenario'],s['seed'])
        assert traces.setdefault(key,r['trace'])==r['trace']
        assert s['policy_order'][s['order_position']]==s['policy']
        sampled+=r['instrumentation']['gpu']['samples']>0
    if c.get('required_ac_power'):
        for path in (out/'telemetry').rglob('*.json'):
            record=load(path)
            assert record['before']['power_plugged'] is True, f'AC precondition failed: {path}'
            assert record['after']['power_plugged'] is True, f'AC condition changed: {path}'
            assert all(s['power_plugged'] is True for s in record['host_samples']), f'AC sample failed: {path}'
    # At least one driver sample is required per main timed GPU replay.
    # A tiny smoke replay can finish before driver startup, and is labeled so.
    if c['workload']['requests']>=12:
        assert sampled==len(runs),'missing GPU instrumentation in a main replay'
    port=[load(p) for p in (out/'portability_runs').glob('*.json')]
    assert len(port)==2*len(c['portability']['policies'])*len(c['portability']['seeds'])
    port_traces={}
    for r in port:
        check_replay(r,c['portability']['requests'],c['portability']['output_tokens'])
        seed=r['summary']['seed']
        assert port_traces.setdefault(seed,r['trace'])==r['trace']
    probe=load(out/'prediction_probes.json')
    observations=defaultdict(list)
    for row in probe['observations']:
        assert math.isfinite(row['actual_execution_ms']) and row['actual_execution_ms']>0
        assert math.isfinite(row['predicted_execution_ms']) and row['predicted_execution_ms']>0
        observations[(row['shape'],row['repeat'])].append(row)
    assert len(observations)==probe['shapes']*probe['repeats']
    for rows in observations.values():
        assert {r['predictor'] for r in rows}=={'old_static','fresh_static','fresh_legacy_ewma','fresh_phase_ewma'}
        assert len({r['actual_execution_ms'] for r in rows})==1
        assert all(r['features']['batch_size']==rows[0]['features']['batch_size'] for r in rows)
    trace=load(out/'quality_trace.json')
    q=c['quality']; width=q['prompt_tokens']+q['output_tokens']; positions=set()
    targets=q['requests']*q['output_tokens']
    for r in trace:
        span=set(range(r['dataset_token_start'],r['dataset_token_start']+width))
        assert not positions & span
        positions|=span
    dynamic=[load(out/'quality_dynamic'/f'{p}.json') for p in q['policies']]
    for r in dynamic:
        check_replay(r,q['requests'],q['output_tokens'],quality=True)
        assert r['trace']==trace
        assert r['summary']['teacher_forced_targets']==targets
    fixed={p:load(out/'quality_fixed'/f'{p}.json') for p in q['fixed_schedules']}
    for r in fixed.values():
        assert r['trace']==trace and r['scored_targets']==targets
        assert sum(len(x['target_nll']) for x in r['requests'])==targets
        assert sum(e['scored_targets'] for e in r['events'])==targets
    def nll_map(requests):
        return {r['request_id']:r['target_nll'] for r in requests}
    ref=nll_map(fixed['fp32']['requests'])
    errors={}
    for label,requests in [('dynamic_full',next(r for r in dynamic if r['summary']['policy']=='full')['requests']),
                           ('cpu_attention',fixed['cpu_attention']['requests'])]:
        values=nll_map(requests)
        delta=max(abs(a-b) for key in ref for a,b in zip(ref[key],values[key]))
        assert delta<.002, (label,delta)
        errors[label]=delta
    early=fixed['early_prefill_int4_restore']
    assert early['quantized_prefill_iterations']>0 and early['quantized_scored_targets']==0
    assert fixed['static_int4']['quantized_scored_targets']==targets
    tests=load(ROOT/'results/validation.json')
    assert tests['successful']
    http_path=out/'http_smoke.json'
    if http_path.exists():
        assert load(http_path)['passed']
    result={'passed':True,'version':m['version'],'gpu_runs':len(runs),
            'gpu_requests':sum(r['summary']['completed'] for r in runs),
            'gpu_runs_with_clock_temperature_samples':sampled,'portability_runs':len(port),
            'portability_requests':sum(r['summary']['completed'] for r in port),
            'prediction_shapes':probe['shapes'],'matched_prediction_executions':len(observations),
            'quality_targets_per_arm':targets,'quality_arms':len(dynamic)+len(fixed),
            'quality_target_evaluations':targets*(len(dynamic)+len(fixed)),
            'fp32_protocol_and_cpu_attention_max_abs_nll_error':errors,
            'tests_passed':tests['tests_run'],'archived_sources_verified':True,
            'paired_cpu_cuda_traces_verified':True,'rejected_requests':0,
            'required_ac_power_verified':bool(c.get('required_ac_power')),
            'recorded_failed_attempts_in_this_series':0,
            'concurrent_http_passed':http_path.exists(),
            'notes':['CPU/CUDA portability is on one physical laptop.',
                     'GPU clocks are observed, not locked; AC is checked at every recorded host sample when required.',
                     'Target evaluations across arms reuse the same held-out targets.']}
    (out/'validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
