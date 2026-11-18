"""One command: inspect validated evidence, replay a live capacity case, test HTTP."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import gc
import json
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--evidence-only',action='store_true',help='Show saved validated results without model execution')
    p.add_argument('--skip-http',action='store_true')
    p.add_argument('--json',action='store_true',help='Print full saved evidence as JSON')
    args=p.parse_args()
    source=ROOT/'results/capacity_frontier'
    validation=json.loads((source/'validation.json').read_text());assert validation['passed']
    analysis=json.loads((source/'analysis.json').read_text())
    evidence={'mode':'saved_validated_evidence','serving_runs':validation['serving_runs'],
              'completed_requests':validation['completed_requests'],'rejected_requests':validation['rejected_requests'],
              'sampled_minimum_quotas':analysis['sampled_minimum_quotas'],'quality':analysis['quality']}
    if args.json:
        print(json.dumps(evidence,indent=2),flush=True)
    else:
        print('PrismServe 1.0: saved validated evidence (actual GPT-2)',flush=True)
        print(f"{validation['serving_runs']} replays | {validation['completed_requests']} completed requests | {validation['rejected_requests']} explicit rejections",flush=True)
        print('Lowest tested complete quota (MiB): prompt / FP32 / quantization / offload / joint',flush=True)
        for prompt in sorted({r['prompt_tokens'] for r in analysis['sampled_minimum_quotas']}):
            values=[next(r['lowest_tested_complete_quota_mib'] for r in analysis['sampled_minimum_quotas'] if r['prompt_tokens']==prompt and r['policy']==policy) for policy in ('full','morph_only','offload_only','joint')]
            print(f"  {prompt}: "+' / '.join(str(v) for v in values),flush=True)
        print('Saved quality audits: prompt / policy / targets / PPL change / INT4 outputs',flush=True)
        for r in analysis['quality']:
            print(f"  {r['prompt_tokens']} / {r['policy']} / {r['targets']} / {r['relative_ppl_pct']:+.2f}% / {r['int4_output_pct']:.1f}%",flush=True)
    if args.evidence_only:
        return
    import torch
    from prismserve.live_engine import ServingConfig
    from prismserve.live_experiments import replay
    from prismserve.pretrained import GPT2Executor
    config=json.loads((source/'config.json').read_text())
    candidates=analysis['rescue_policy_seed_cells']
    preferred=[r for r in candidates if r['policy']=='joint'] or candidates
    if not preferred:
        raise RuntimeError('No saved rescue case exists; inspect the negative frontier result with --evidence-only')
    case=preferred[0]
    name=f"p{case['prompt_tokens']}__m{case['device_budget_mib']}__{case['policy']}__{case['seed']}.json"
    saved=json.loads((source/'runs'/name).read_text())
    trace=saved['trace']
    calibration=json.loads((source/'calibration_cuda.json').read_text())
    settings=ServingConfig(**config['serving'],device_memory_budget_bytes=case['device_budget_mib']*2**20,
                           host_memory_budget_bytes=config['host_quota_mib']*2**20)
    torch.set_num_threads(min(4,torch.get_num_threads()))
    executor=GPT2Executor.from_local(ROOT/config['model_path'],'cuda')
    destination=ROOT/'results/final_demo'
    release_path=ROOT/'deliverables/release.json'
    if release_path.exists():
        from prismserve import __version__
        if json.loads(release_path.read_text())['version']==__version__:
            destination=ROOT/'results/demo_sessions'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    destination.mkdir(parents=True,exist_ok=True)
    print(f'Live demonstration output: {destination}',flush=True)
    summary=[]
    try:
        for policy in ('full',case['policy']):
            result=replay(executor,calibration,trace,settings,policy,
                          failure_output=destination/f'{policy}_failed.json')
            (destination/f'{policy}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
            row={'mode':'live_capacity_demonstration_not_a_controlled_benchmark','policy':policy,
                 'prompt_tokens':case['prompt_tokens'],'quota_mib':case['device_budget_mib'],
                 'completed':result['summary']['completed'],'rejected':result['summary']['rejected'],
                 'max_batch':result['summary']['max_execution_batch_size'],
                 'quantized_iterations':sum(bool(e['quantized_layers']) for e in result['iterations']),
                 'cpu_attention_iterations':sum(bool(e['cpu_request_ids']) for e in result['iterations'])}
            if result['requests']:
                row['sample_base_model_completion']=result['requests'][0]['text'][:140]
            if policy==case['policy']:
                assert row['completed']==len(trace) and row['quantized_iterations']>0
            else:
                assert row['rejected']==len(trace)
            summary.append(row);print(json.dumps(row,indent=2),flush=True)
    finally:
        del executor;gc.collect();torch.cuda.empty_cache()
    (destination/'capacity_summary.json').write_text(json.dumps({'passed':True,'runs':summary},indent=2),encoding='utf-8')
    if not args.skip_http:
        # Release the first model before the temporary server loads another.
        subprocess.run([sys.executable,str(ROOT/'scripts/smoke_batch_api.py'),'--output',str(destination),
                        '--calibration',str(source/'calibration_cuda.json')],cwd=ROOT,check=True)
    result={'passed':True,'live_capacity_verified':True,'concurrent_http_verified':not args.skip_http,
            'note':'Demonstration timings are not substituted for controlled experiment measurements.'}
    (destination/'demo_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
