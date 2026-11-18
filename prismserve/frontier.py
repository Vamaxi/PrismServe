"""M11: measured warm-start serving capacity, including explicit rejections.

The frontier concerns the existing conservative admission/transition contract,
not physical allocator OOM, cold checkpoint loading or an optimal capacity bound.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import random
import zipfile

from . import __version__
from .experiments import write_json
from .live_experiments import _requests
from .pretrained_assets import sha256_file
from .study_monitor import machine_conditions
from .validation_study import (_calibration, _replay, balanced_order, disjoint_trace,
                               require_power, source_hashes)


def validate_config(config):
    if config.get('schema_version')!=1:
        raise ValueError('frontier schema_version must be 1')
    if set(config['policies'])!={'full','morph_only','offload_only','joint'} or len(config['policies'])!=4:
        raise ValueError('frontier requires exactly four mechanism controls')
    for key in ('device_quotas_mib','prompt_lengths','seeds'):
        values=config[key]
        if not values or len(set(values))!=len(values):
            raise ValueError(f'{key} must be nonempty and unique')
    if any(v<=0 for v in config['device_quotas_mib']):
        raise ValueError('memory quotas must be positive')
    w=config['workload']
    if w['requests']<1 or w['output_tokens']<1 or w['duration_s']<0:
        raise ValueError('invalid workload limits')
    if any(p<1 or p+w['output_tokens']>1024 for p in config['prompt_lengths']):
        raise ValueError('prompt and output must fit GPT-2 context')
    if config['warmup_passes']<1 or config['calibration_repeats']<1:
        raise ValueError('positive warmup/calibration required')


def outcome(summary):
    """Rejected work never becomes a zero-latency successful observation."""
    if summary.get('worker_failed'):
        return 'failed'
    completed,rejected,total=summary['completed'],summary['rejected'],summary['requests']
    if completed+rejected!=total:
        raise ValueError('incomplete request accounting')
    if completed==total:
        return 'complete'
    return 'rejected' if completed==0 else 'partial'


def select_audits(runs):
    """A deterministic observed-trajectory selection, declared before replay."""
    groups={}
    for run in runs:
        row=run['summary']
        key=(row['prompt_tokens'],row['policy'],row['device_budget_mib'])
        groups.setdefault(key,[]).append(run)
    chosen={}
    for (prompt,policy,quota),rows in sorted(groups.items()):
        if policy not in ('morph_only','joint'):
            continue
        if not all(outcome(r['summary'])=='complete' for r in rows):
            continue
        if not any(r['exposure']['quantized_iterations'] for r in rows):
            continue
        key=(prompt,policy)
        if key not in chosen or quota<chosen[key]['device_budget_mib']:
            chosen[key]={'prompt_tokens':prompt,'policy':policy,'device_budget_mib':quota}
    return [chosen[key] for key in sorted(chosen)]


def rescue_cells(runs):
    by_key={}
    for run in runs:
        s=run['summary']
        by_key[(s['prompt_tokens'],s['device_budget_mib'],s['seed'],s['policy'])]=s
    result=[]
    for key,s in sorted(by_key.items()):
        prompt,quota,seed,policy=key
        ref=by_key.get((prompt,quota,seed,'full'))
        if policy!='full' and ref and outcome(ref)=='rejected' and outcome(s)=='complete':
            result.append({'prompt_tokens':prompt,'device_budget_mib':quota,'seed':seed,'policy':policy})
    return result


def initialize(config_path,output):
    config=json.loads(config_path.read_text())
    validate_config(config)
    output.mkdir(parents=True,exist_ok=True)
    sources=source_hashes()
    path=output/'manifest.json'
    if path.exists():
        manifest=json.loads(path.read_text())
        if sources!=manifest['source_sha256'] or sha256_file(config_path)!=manifest['config_sha256']:
            raise ValueError('frontier output belongs to another source/config; select a new output directory')
    else:
        with zipfile.ZipFile(output/'source_snapshot.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
            for p in sorted(Path(__file__).parent.glob('*.py')):
                archive.write(p,'prismserve/'+p.name)
        manifest={'version':__version__,'milestone':'M11','status':'in_progress','completed_stages':[],
            'created_utc':datetime.now(timezone.utc).isoformat(),'config_sha256':sha256_file(config_path),
            'source_sha256':sources,'source_snapshot_sha256':sha256_file(output/'source_snapshot.zip'),
            'machine':machine_conditions(),
            'model_source':json.loads(Path(config['model_path'],'prismserve_source.json').read_text()),
            'data_sha256':{s:sha256_file(Path(f'data/wikitext2/wiki.{s}.prefix.txt')) for s in ('validation','test')},
            'notes':['Actual warm-start serving under enforced reservation quotas; no physical OOM stress.',
                     'FP32 checkpoint load, calibration and warmup precede timed quota-constrained execution.',
                     'No initial quantized checkpoint, prefill CPU offload or cache eviction is introduced.',
                     'Precision transitions must fit from the actual FP32 starting state.',
                     'A rejected request counts as an SLO failure; absent latency values are not zero-latency wins.',
                     'One physical laptop with synchronous reference kernels; no universal-capacity or vendor-portability claim.']}
        write_json(path,manifest);write_json(output/'config.json',config)
    return config,manifest


def run_frontier(config_path:Path,output:Path,stage='all'):
    import torch
    from .live_engine import ServingConfig
    from .pretrained import GPT2Executor
    config,manifest=initialize(config_path,output)
    if stage not in ('all','serving','quality'):
        raise ValueError('unknown frontier stage')
    torch.set_num_threads(min(4,torch.get_num_threads()));torch.manual_seed(7)
    require_power(config)
    executor=GPT2Executor.from_local(Path(config['model_path']),'cuda')
    calibration=_calibration(executor,config,output)
    serving=replace(ServingConfig(**config['serving']),host_memory_budget_bytes=config['host_quota_mib']*2**20)
    selected=('serving','quality') if stage=='all' else (stage,)
    try:
        for phase in selected:
            if phase in manifest['completed_stages']:
                continue
            if source_hashes()!=manifest['source_sha256']:
                raise RuntimeError('runtime changed during frontier study')
            if phase=='serving':
                cells=[(p,q) for p in config['prompt_lengths'] for q in config['device_quotas_mib']]
                for repeat,seed in enumerate(config['seeds']):
                    ordered=list(enumerate(cells));random.Random(810+repeat).shuffle(ordered)
                    for cell_index,(prompt,quota) in ordered:
                        w=config['workload']
                        trace=_requests(executor,w['requests'],prompt,w['output_tokens'],w['duration_s'],w['traffic'],seed)
                        settings=replace(serving,device_memory_budget_bytes=int(quota*2**20))
                        order=balanced_order(config['policies'],repeat,cell_index)
                        for position,policy in enumerate(order):
                            result=_replay(executor,calibration,trace,settings,policy,config,output,
                                output/'runs'/f'p{prompt}__m{quota}__{policy}__{seed}.json',
                                metadata={'prompt_tokens':prompt,'device_budget_mib':quota,'seed':seed,
                                          'repeat':repeat,'order_position':position,'policy_order':order})
                            if outcome(result['summary'])=='failed':
                                raise RuntimeError('worker failure cannot be an accepted frontier cell')
                runs=[json.loads(p.read_text()) for p in sorted((output/'runs').glob('*.json'))]
                write_json(output/'quality_selection.json',{'cases':select_audits(runs),
                    'selection':config['quality']['selection'],
                    'note':'Selection uses completion and precision exposure, never test NLL. Quality is a separate trajectory.'})
            else:
                selection=output/'quality_selection.json'
                if not selection.exists():
                    raise RuntimeError('run the serving grid before selecting quality cases')
                ids=executor.tokenizer.encode(Path('data/wikitext2/wiki.test.prefix.txt').read_text(encoding='utf-8'),
                                              add_special_tokens=False,verbose=False)
                cases=json.loads(selection.read_text())['cases']
                for prompt in sorted({c['prompt_tokens'] for c in cases}):
                    count=min(config['quality']['max_requests'],len(ids)//(prompt+config['workload']['output_tokens']))
                    qconfig={'requests':count,'prompt_tokens':prompt,'output_tokens':config['workload']['output_tokens'],
                             'duration_s':config['quality']['duration_s']}
                    trace=disjoint_trace(executor,qconfig)
                    write_json(output/'quality_traces'/f'p{prompt}.json',trace)
                    reference=replace(serving,device_memory_budget_bytes=config['quality']['reference_quota_mib']*2**20)
                    _replay(executor,calibration,trace,reference,'full',config,output,
                        output/'quality_reference'/f'p{prompt}.json',quality=True,
                        metadata={'prompt_tokens':prompt,'device_budget_mib':config['quality']['reference_quota_mib']})
                    for case in (c for c in cases if c['prompt_tokens']==prompt):
                        settings=replace(serving,device_memory_budget_bytes=int(case['device_budget_mib']*2**20))
                        _replay(executor,calibration,trace,settings,case['policy'],config,output,
                            output/'quality_runs'/f"p{prompt}__m{case['device_budget_mib']}__{case['policy']}.json",
                            quality=True,metadata=case)
            if source_hashes()!=manifest['source_sha256']:
                raise RuntimeError('runtime changed while measuring frontier')
            manifest['completed_stages'].append(phase)
            manifest['status']='complete' if set(manifest['completed_stages'])=={'serving','quality'} else 'in_progress'
            manifest['updated_utc']=datetime.now(timezone.utc).isoformat()
            manifest['calibration_sha256']=sha256_file(output/'calibration_cuda.json')
            write_json(output/'manifest.json',manifest)
            print(f'Completed M11 stage: {phase}',flush=True)
    finally:
        del executor;gc.collect();torch.cuda.empty_cache()
    return {'output':str(output.resolve()),'status':manifest['status'],'completed_stages':manifest['completed_stages']}
