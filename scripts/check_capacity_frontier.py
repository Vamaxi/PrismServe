"""Validate M11 paired inputs, rejections, quality identities and source snapshots."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from prismserve.frontier import outcome,select_audits,rescue_cells


def load(p):
    return json.loads(p.read_text())


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def check_run(run,tokens):
    s=run['summary'];outcome(s)
    assert not run['metrics']['fatal_error']
    assert len(run['requests'])==s['completed']
    assert len(run['rejections'])==s['rejected']
    completed={r['request_id'] for r in run['requests']}
    rejected={r['request_id'] for r in run['rejections']}
    assert not completed&rejected
    assert completed|rejected=={r['request_id'] for r in run['trace']}
    for r in run['requests']:
        assert r['completion_tokens']==tokens
        assert len(r['decode_token_ms'])==tokens-1
        assert len(r['precision_history'])==tokens
        assert len(r['target_nll'])==(tokens if s['mode']=='teacher_forced_quality_audit' else 0)
    failed=s['rejected']+sum(r['slo_violated'] for r in run['requests'])
    assert abs(s['slo_failure_rate']-failed/s['requests'])<1e-10
    if s['completed']==0:
        assert s['ttft_p50_ms'] is None and s['ttft_p95_ms'] is None
        assert s['output_tokens_per_s']==0
        assert s['slo_failure_rate']==1
    for e in run['iterations']:
        assert e['device_reserved_bytes']<=run['config']['device_memory_budget_bytes']
        assert e['host_reserved_bytes']<=run['config']['host_memory_budget_bytes']
        assert e['quality_proxy']<=run['config']['quality_budget']+1e-12
        assert e['projection_calls']==48
        if e['plan_reused']:
            assert e['resource_checks']>=1


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=ROOT/'results/capacity_frontier')
    p.add_argument('--check-current',action='store_true')
    args=p.parse_args();out=args.output
    m=load(out/'manifest.json');c=load(out/'config.json')
    assert m['status']=='complete' and set(m['completed_stages'])=={'serving','quality'}
    assert not list((out/'failures').rglob('*.json'))
    assert not list((out/'condition_rejected').rglob('*.json'))
    assert digest(out/'source_snapshot.zip')==m['source_snapshot_sha256']
    assert digest(out/'calibration_cuda.json')==m['calibration_sha256']
    with zipfile.ZipFile(out/'source_snapshot.zip') as z:
        for name,value in m['source_sha256'].items():
            assert hashlib.sha256(z.read('prismserve/'+name)).hexdigest()==value
            if args.check_current:
                assert digest(ROOT/'prismserve'/name)==value
    for split,value in m['data_sha256'].items():
        assert digest(ROOT/f'data/wikitext2/wiki.{split}.prefix.txt')==value
    runs=[load(p) for p in sorted((out/'runs').glob('*.json'))]
    assert len(runs)==len(c['policies'])*len(c['seeds'])*len(c['device_quotas_mib'])*len(c['prompt_lengths'])
    traces={};peaks_above=0
    for r in runs:
        check_run(r,c['workload']['output_tokens'])
        s=r['summary'];key=(s['prompt_tokens'],s['seed'])
        assert traces.setdefault(key,r['trace'])==r['trace']
        assert s['policy_order'][s['order_position']]==s['policy']
        assert r['summary']['mode']=='greedy_generation'
        peaks_above+=s['peak_cuda_allocated_bytes']>s['device_budget_mib']*2**20
    if c.get('required_ac_power'):
        for p in (out/'telemetry').rglob('*.json'):
            record=load(p)
            assert record['before']['power_plugged'] is True
            assert record['after']['power_plugged'] is True
            assert all(s['power_plugged'] is True for s in record['host_samples'])
    cases=load(out/'quality_selection.json')['cases']
    assert cases==select_audits(runs)
    quality_targets=0;quantized_audits=0
    for case in cases:
        prompt=case['prompt_tokens'];quota=case['device_budget_mib'];policy=case['policy']
        r=load(out/'quality_runs'/f'p{prompt}__m{quota}__{policy}.json')
        ref=load(out/'quality_reference'/f'p{prompt}.json')
        trace=load(out/'quality_traces'/f'p{prompt}.json')
        assert r['trace']==ref['trace']==trace
        assert all(len(t['prompt_ids'])==prompt for t in trace)
        assert all(t['output_tokens']==c['workload']['output_tokens'] for t in trace)
        for run in (r,ref):
            check_run(run,c['workload']['output_tokens'])
            assert outcome(run['summary'])=='complete'
            assert run['summary']['mode']=='teacher_forced_quality_audit'
            assert all(math.isfinite(v) for row in run['requests'] for v in row['target_nll'])
        spans=set()
        for t in trace:
            window=set(range(t['dataset_token_start'],t['dataset_token_start']+prompt+t['output_tokens']))
            assert not spans&window
            spans|=window
        expected=len(trace)*c['workload']['output_tokens']
        assert r['summary']['teacher_forced_targets']==ref['summary']['teacher_forced_targets']==expected
        quality_targets+=expected
        quantized_audits+=r['exposure']['quantized_iterations']>0
    references=[load(p) for p in (out/'quality_reference').glob('*.json')]
    quality_targets+=sum(r['summary']['teacher_forced_targets'] for r in references)
    tests=load(ROOT/'results/validation.json');assert tests['successful']
    result={'passed':True,'version':m['version'],'serving_runs':len(runs),
        'requested_requests':sum(r['summary']['requests'] for r in runs),
        'completed_requests':sum(r['summary']['completed'] for r in runs),
        'rejected_requests':sum(r['summary']['rejected'] for r in runs),
        'complete_runs':sum(outcome(r['summary'])=='complete' for r in runs),
        'all_rejected_runs':sum(outcome(r['summary'])=='rejected' for r in runs),
        'partial_runs':sum(outcome(r['summary'])=='partial' for r in runs),
        'rescue_policy_seed_cells':len(rescue_cells(runs)),
        'selected_adaptive_quality_audits':len(cases),'audits_with_actual_quantization':quantized_audits,
        'fp32_quality_references':len(references),'quality_target_evaluations':quality_targets,
        'actual_cuda_peak_over_quota_runs':peaks_above,'tests_passed':tests['tests_run'],
        'paired_traces_verified':True,'reserved_quota_checks_passed':True,'rejections_count_as_slo_failures':True,
        'source_snapshot_verified':True,'ac_samples_verified':bool(c.get('required_ac_power')),
        'notes':['No rejection is a zero-latency observation. Quotas apply to timed warm-start serving.',
                 'Sampled capacity belongs to this conservative transition/admission contract, not an optimal physical frontier.']}
    (out/'validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
