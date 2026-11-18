"""M10: instrumented repeats, matched prediction probes and CPU/CUDA validation.

This harness changes no serving decisions. It calibrates each execution device,
retains complete raw observations and separates quality from greedy serving.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import random
from statistics import mean
import time
import zipfile

from . import __version__
from .experiments import write_json
from .live_experiments import _calibrate, _requests, _sync, _warm_batch_shapes, replay
from .study_monitor import RunMonitor, machine_conditions
from .pretrained_assets import sha256_file

STAGES = ('gpu', 'prediction', 'cuda_portability', 'quality', 'cpu_portability')


def require_power(config, monitor=None):
    if not config.get('required_ac_power'):
        return
    states=([machine_conditions()['power_plugged']] if monitor is None else
            [monitor.before['power_plugged'],monitor.after['power_plugged']] +
            [sample.get('power_plugged') for sample in monitor.host])
    if not states or any(state is not True for state in states):
        raise RuntimeError('AC-conditioned study requires plugged-in power at every recorded sample; raw diagnostics are retained. Plug in and resume.')


def source_hashes():
    return {p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def balanced_order(policies, round_index, scenario_index=0):
    """Four cyclic positions balanced over each block of four repeats."""
    order = list(policies)
    random.Random(101 + round_index // len(order)).shuffle(order)
    offset = (round_index + scenario_index) % len(order)
    return order[offset:] + order[:offset]


def disjoint_trace(executor, config):
    ids = executor.tokenizer.encode(Path('data/wikitext2/wiki.test.prefix.txt').read_text(encoding='utf-8'),
                                    add_special_tokens=False, verbose=False)
    width = config['prompt_tokens'] + config['output_tokens']
    if width > executor.context_limit or config['requests'] * width > len(ids):
        raise ValueError('disjoint audit windows exceed available held-out text or model context')
    result = []
    for i in range(config['requests']):
        begin = i * width
        split = begin + config['prompt_tokens']
        result.append({'request_id': f'a{i:04d}', 'arrival_s': i * config['duration_s']/max(1,config['requests']-1),
                       'prompt_ids': ids[begin:split], 'reference_ids': ids[split:begin+width],
                       'output_tokens': config['output_tokens'], 'dataset_token_start': begin})
    return result


def exposure(run):
    events = run['iterations']
    return {'iterations': len(events),
            'quantized_iterations': sum(bool(e['quantized_layers']) for e in events),
            'quantized_prefill_iterations': sum(bool(e['quantized_layers']) and bool(e['prefill_tokens']) for e in events),
            'quantized_decode_only_iterations': sum(bool(e['quantized_layers']) and not e['prefill_tokens'] for e in events),
            'cpu_attention_iterations': sum(bool(e['cpu_request_ids']) for e in events)}


def paired_interval(values, seed=7, samples=4000):
    """Exploratory percentile bootstrap of paired repeat differences."""
    if not values:
        raise ValueError('paired sample is empty')
    rng = random.Random(seed)
    means = sorted(mean(rng.choices(values, k=len(values))) for _ in range(samples))
    return {'mean': mean(values), 'lower_95': means[int(.025*samples)],
            'upper_95': means[min(samples-1,int(.975*samples))], 'paired_repeats': len(values),
            'note': 'exploratory paired bootstrap; repeats share one laptop and do not establish population confidence'}


def _initialize(config_path, output):
    config = json.loads(config_path.read_text())
    if config.get('schema_version') != 1:
        raise ValueError('study config schema_version must be 1')
    if len(config['policies']) < 2 or len(set(config['policies'])) != len(config['policies']):
        raise ValueError('comparison policies must be unique')
    if config['warmup_passes'] < 1 or not config['seeds']:
        raise ValueError('positive warmup and nonempty repeats required')
    output.mkdir(parents=True, exist_ok=True)
    sources = source_hashes()
    manifest_path = output / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['source_sha256'] != sources or manifest['config_sha256'] != sha256_file(config_path):
            raise ValueError('output belongs to a different source/config; use a new output folder')
    else:
        with zipfile.ZipFile(output/'source_snapshot.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(Path(__file__).parent.glob('*.py')):
                archive.write(path,'prismserve/'+path.name)
        manifest = {'version': __version__, 'milestone': 'M10', 'created_utc': datetime.now(timezone.utc).isoformat(),
                    'config_sha256': sha256_file(config_path), 'source_sha256': sources,
                    'source_snapshot_sha256': sha256_file(output/'source_snapshot.zip'),
                    'machine': machine_conditions(), 'status': 'in_progress', 'completed_stages': [],
                    'model_source': json.loads(Path(config['model_path'],'prismserve_source.json').read_text()),
                    'data_sha256': {split: sha256_file(Path(f'data/wikitext2/wiki.{split}.prefix.txt')) for split in ('validation','test')},
                    'notes': ['One physical laptop; CPU and CUDA are separate execution backends, not separate machines.',
                              'Read-only instrumentation; GPU clocks, temperature and AC power are observed, not fixed.',
                              'No GPU tests or other project model workloads run concurrently with this harness.',
                              'Quality audits use disjoint token windows but remain one small WikiText-2 test prefix.']}
        write_json(manifest_path, manifest)
        write_json(output/'config.json',config)
    return config, manifest


def _calibration(executor, config, output):
    target = output / f'calibration_{executor.device.type}.json'
    if target.exists():
        payload = json.loads(target.read_text())
        if payload['device'].split(':')[0] != executor.device.type:
            raise ValueError('calibration/executor mismatch')
        return payload
    print(f'FRESH calibration: {executor.device}',flush=True)
    require_power(config)
    with RunMonitor(output/'telemetry'/f'calibration_{executor.device.type}',executor.device.type=='cuda') as monitor:
        payload = _calibrate(executor, config['calibration_repeats'])
    require_power(config,monitor)
    if executor.device.type == 'cpu':
        # The legacy helper measures host rates correctly but inherits the GPU
        # inventory name when CUDA is installed. Label the actual CPU executor.
        h = payload['hardware']
        h.update(name=platform.processor() or platform.machine(), device_memory_bytes=h['host_memory_bytes'],
                 device_kind='cpu', source='measured_host_microbenchmarks_cpu_executor')
    payload['model_path'] = config['model_path']
    write_json(target,payload)
    return payload


def _serving(config, **overrides):
    from .live_engine import ServingConfig
    return replace(ServingConfig(**config['serving']), **overrides)


def _replay(executor, calibration, trace, serving, policy, config, output, target, quality=False, metadata=None):
    if target.exists():
        result = json.loads(target.read_text())
        if result['trace'] != trace or result['summary']['policy'] != policy:
            raise ValueError('cached run inputs do not match requested replay')
        return result
    executor.set_quantized_layers(())
    require_power(config)
    # replay itself supplies one additional identical warmup pass.
    for _ in range(config['warmup_passes']-1):
        _warm_batch_shapes(executor, serving)
    relative = target.relative_to(output).with_suffix('')
    print(f'REAL {executor.device.type}: {relative}',flush=True)
    with RunMonitor(output/'telemetry'/relative,executor.device.type=='cuda') as monitor:
        result = replay(executor,calibration,trace,serving,policy,quality=quality,
                        failure_output=output/'failures'/target.relative_to(output))
    row = result['summary']
    row.update(metadata or {})
    result['instrumentation'] = monitor.summary(row['measurement_start_unix_s'],row['measurement_end_unix_s'])
    result['exposure'] = exposure(result)
    try:
        require_power(config,monitor)
    except RuntimeError:
        write_json(output/'condition_rejected'/target.relative_to(output),result)
        raise
    write_json(target,result)
    return result


def gpu_repeats(executor, calibration, config, output):
    for repeat, seed in enumerate(config['seeds']):
        scenarios = list(enumerate(config['scenarios']))
        random.Random(711+repeat).shuffle(scenarios)
        for index, scenario in scenarios:
            w = config['workload']
            trace = _requests(executor,w['requests'],w['prompt_tokens'],w['output_tokens'],w['duration_s'],scenario['traffic'],seed)
            serving = _serving(config,device_memory_budget_bytes=scenario['device_budget_mib']*2**20)
            order = balanced_order(config['policies'],repeat,index)
            for position, policy in enumerate(order):
                _replay(executor,calibration,trace,serving,policy,config,output,
                        output/'gpu_runs'/f"{scenario['name']}__{policy}__{seed}.json",
                        metadata={'scenario':scenario['name'],'seed':seed,'repeat':repeat,
                                  'order_position':position,'policy_order':order,'device_budget_mib':scenario['device_budget_mib']})


def portability(executor, calibration, config, output):
    p = config['portability']
    serving = _serving(config,max_batch_size=p['max_batch_size'],device_memory_budget_bytes=p['device_budget_mib']*2**20,
                       host_memory_budget_bytes=p['host_budget_mib']*2**20)
    for repeat, seed in enumerate(p['seeds']):
        trace = _requests(executor,p['requests'],p['prompt_tokens'],p['output_tokens'],p['duration_s'],'bursty',seed)
        for policy in balanced_order(p['policies'],repeat):
            _replay(executor,calibration,trace,serving,policy,config,output,
                    output/'portability_runs'/f'{executor.device.type}__{policy}__{seed}.json',
                    metadata={'device':executor.device.type,'seed':seed,'repeat':repeat})


def matched_predictions(executor, calibration, config, output):
    target = output/'prediction_probes.json'
    if target.exists():
        return
    import torch
    from .batching import BatchInput, execute_batch
    from .efficient_policy import EfficientController
    from .measured_policy import MeasuredController, TimingCalibration
    from .types import ControlState, Decision, HardwareProfile, PlannerConfig, Plan, WorkItem
    old = json.loads(Path('results/live/calibration.json').read_text())
    hardware = HardwareProfile(**calibration['hardware'])
    model = executor.model_spec(tuple(calibration['sensitivity']))
    qsets = tuple(dict.fromkeys(s['quantized_layers'] and tuple(s['quantized_layers']) or () for s in calibration['timing']['samples']))
    kwargs = dict(profile=hardware,model=model,config=PlannerConfig(quality_budget=1,memory_fraction=1),
                  quantization_candidates=qsets)
    predictors = {
        'old_static': MeasuredController(calibration=TimingCalibration.from_dict(old['timing']),**kwargs),
        'fresh_static': MeasuredController(calibration=TimingCalibration.from_dict(calibration['timing']),**kwargs),
        'fresh_legacy_ewma': MeasuredController(calibration=TimingCalibration.from_dict(calibration['timing']),**kwargs),
        'fresh_phase_ewma': EfficientController(calibration=TimingCalibration.from_dict(calibration['timing']),**kwargs),
    }
    ids = executor.tokenizer.encode(Path('data/wikitext2/wiki.validation.prefix.txt').read_text(encoding='utf-8'),
                                    add_special_tokens=False,verbose=False)
    specs = []
    for q in qsets:
        for count in config['prediction']['batch_sizes']:
            for context in config['prediction']['contexts']:
                for phase in ('prefill','mixed','decode'):
                    eligible = 0 if phase=='prefill' else count if phase=='decode' else count//2
                    for cpu in sorted({0,eligible//2}):
                        specs.append((q,count,context,phase,cpu))
    random.Random(401).shuffle(specs)
    observations = []
    print(f'Matched prediction probes: {len(specs)} shapes',flush=True)
    require_power(config)
    with RunMonitor(output/'telemetry/prediction',True) as monitor:
        with torch.inference_mode():
            for shape_index,(q,count,context,phase,cpu_count) in enumerate(specs):
                executor.set_quantized_layers(q)
                old_count = 0 if phase=='prefill' else count if phase=='decode' else count//2
                prepared = execute_batch(executor,[BatchInput(f'p{i}',ids[:context],prefill=True) for i in range(old_count)]) if old_count else None
                chunk = 32 if context==config['prediction']['contexts'][0] else 96
                inputs, items = [], []
                for i in range(count):
                    existing = context if i<old_count else 0
                    tokens = [ids[context]] if existing else ids[:chunk]
                    cache = prepared.outputs[i].cache if existing else None
                    if i<cpu_count:
                        cache = [(k.cpu(),v.cpu()) for k,v in cache]
                    inputs.append(BatchInput(f'p{i}',tokens,cache,i<cpu_count,not existing))
                    items.append(WorkItem(f'p{i}',existing+len(tokens)+8,0 if existing else len(tokens),existing))
                plan = Plan(q,tuple(f'p{i}' for i in range(cpu_count)))
                state = ControlState(q,plan.cpu_request_ids)
                warm = execute_batch(executor,inputs)
                del warm
                _sync(executor)
                for repeat in range(config['prediction']['repeats']):
                    decisions = {name: Decision(plan,p._estimate(items,plan,state),'joint',0,'matched probe') for name,p in predictors.items()}
                    _sync(executor)
                    begin = time.perf_counter()
                    result = execute_batch(executor,inputs)
                    _sync(executor)
                    elapsed = (time.perf_counter()-begin)*1000
                    del result
                    for name,decision in decisions.items():
                        if not decision.estimate.feasible:
                            raise RuntimeError(f'probe prediction missing/unsafe: {name}, shape {shape_index}')
                        observations.append({'shape':shape_index,'repeat':repeat,'phase':phase,'batch_size':count,
                            'context_tokens':context,'quantized_layers':list(q),'cpu_requests':cpu_count,'predictor':name,
                            'actual_execution_ms':elapsed,'predicted_execution_ms':decision.estimate.iteration_ms,
                            'features':decision.estimate.components['timing_features'],
                            'nearest_sample':decision.estimate.components.get('nearest_sample')})
                        if name.endswith('ewma'):
                            predictors[name].observe(decision,elapsed)
                del inputs,items,prepared,cache
                if shape_index%12==0:
                    print(f'Prediction shape {shape_index+1}/{len(specs)}',flush=True)
    require_power(config,monitor)
    executor.set_quantized_layers(())
    write_json(target,{'measurement_kind':'matched_actual_batches_prequential_prediction',
        'shapes':len(specs),'repeats':config['prediction']['repeats'],'observations':observations,
        'notes':['All four predictors forecast the same execution before observing its duration.',
                 'Static predictors do not update; feedback predictors update only after that execution.',
                 'Shape contexts 96/320 and prefill chunks 32/96 differ from calibration shapes.',
                 'All shape preparation uses validation text; no quality-test targets enter timing calibration.',
                 'This separates predictor error from the different actions/trajectories chosen during serving.']})


def _score(logits, targets):
    import torch
    from torch.nn import functional as F
    values = torch.cat(logits).float()
    target = torch.tensor(targets,device=values.device)
    return F.cross_entropy(values,target,reduction='none').cpu().tolist()


def fixed_quality(executor, trace, schedule, qset, config):
    """Score the same continuation under controlled precision/KV histories."""
    import torch
    from .batching import BatchInput,execute_batch
    if schedule not in ('fp32','static_int4','early_prefill_int4_restore','decode_int4','cpu_attention'):
        raise ValueError('unknown fixed quality schedule')
    current_q = None
    requests,events = [],[]
    started = time.time()
    with torch.inference_mode():
        for begin in range(0,len(trace),4):
            batch = trace[begin:begin+4]
            caches = [None]*len(batch)
            nll = [[] for _ in batch]
            for cursor in range(0,len(batch[0]['prompt_ids']),64):
                last = cursor+64 >= len(batch[0]['prompt_ids'])
                q = qset if schedule=='static_int4' or (schedule=='early_prefill_int4_restore' and not last) else ()
                if q!=current_q:
                    executor.set_quantized_layers(q)
                    current_q=q
                inputs=[BatchInput(r['request_id'],r['prompt_ids'][cursor:cursor+64],caches[i],prefill=True) for i,r in enumerate(batch)]
                result=execute_batch(executor,inputs)
                caches=[r.cache for r in result.outputs]
                if last:
                    values=_score([r.logits for r in result.outputs],[r['reference_ids'][0] for r in batch])
                    for i,v in enumerate(values):
                        nll[i].append(v)
                events.append({'phase':'prefill','quantized_layers':list(q),'scored_targets':len(batch) if last else 0,
                               'cpu_attention':False,'request_ids':[r['request_id'] for r in batch]})
                del result,inputs
            for step in range(1,batch[0]['output_tokens']):
                q=qset if schedule in ('static_int4','decode_int4') else ()
                if q!=current_q:
                    executor.set_quantized_layers(q)
                    current_q=q
                offload=schedule=='cpu_attention'
                if offload and step==1:
                    caches=[[(k.cpu(),v.cpu()) for k,v in cache] for cache in caches]
                inputs=[BatchInput(r['request_id'],[r['reference_ids'][step-1]],caches[i],cpu_attention=offload) for i,r in enumerate(batch)]
                result=execute_batch(executor,inputs)
                caches=[r.cache for r in result.outputs]
                values=_score([r.logits for r in result.outputs],[r['reference_ids'][step] for r in batch])
                for i,v in enumerate(values):
                    nll[i].append(v)
                events.append({'phase':'decode','quantized_layers':list(q),'scored_targets':len(batch),
                               'cpu_attention':offload,'request_ids':[r['request_id'] for r in batch]})
                del result,inputs
            requests.extend({'request_id':r['request_id'],'target_nll':nll[i]} for i,r in enumerate(batch))
            del caches
    executor.set_quantized_layers(())
    flat=[v for r in requests for v in r['target_nll']]
    return {'measurement_kind':'controlled_teacher_forced_precision_history_audit','schedule':schedule,
            'nll':mean(flat),'perplexity':math.exp(mean(flat)),'scored_targets':len(flat),
            'measurement_start_unix_s':started,'measurement_end_unix_s':time.time(),
            'quantized_scored_targets':sum(e['scored_targets'] for e in events if e['quantized_layers']),
            'quantized_prefill_iterations':sum(e['phase']=='prefill' and bool(e['quantized_layers']) for e in events),
            'cpu_attention_scored_targets':sum(e['scored_targets'] for e in events if e['cpu_attention']),
            'requests':requests,'events':events,'trace':trace,
            'notes':['Controlled schedules diagnose exposure; they are not adaptive-serving throughput measurements.',
                     'Early-prefill schedule restores FP32 before first output; retained KV still contains a quantized history.']}


def quality_audits(executor,calibration,config,output):
    q=config['quality']
    trace=disjoint_trace(executor,q)
    write_json(output/'quality_trace.json',trace)
    serving=_serving(config,device_memory_budget_bytes=q['device_budget_mib']*2**20)
    for policy in q['policies']:
        _replay(executor,calibration,trace,serving,policy,config,output,output/'quality_dynamic'/f'{policy}.json',quality=True)
    qset=max((tuple(s['quantized_layers']) for s in calibration['timing']['samples']),key=len)
    for schedule in q['fixed_schedules']:
        target=output/'quality_fixed'/f'{schedule}.json'
        if target.exists():
            continue
        print(f'CONTROLLED quality: {schedule}',flush=True)
        require_power(config)
        with RunMonitor(output/'telemetry/quality_fixed'/schedule,True) as monitor:
            result=fixed_quality(executor,trace,schedule,qset,config)
        require_power(config,monitor)
        result['instrumentation']=monitor.summary(result['measurement_start_unix_s'],result['measurement_end_unix_s'])
        write_json(target,result)


def run_study(config_path:Path,output:Path,stage='all'):
    import torch
    from .pretrained import GPT2Executor
    config,manifest=_initialize(config_path,output)
    if stage!='all' and stage not in STAGES:
        raise ValueError('unknown study stage')
    torch.set_num_threads(min(4,torch.get_num_threads()))
    torch.manual_seed(7)
    selected=STAGES if stage=='all' else (stage,)
    for name in selected:
        if name in manifest['completed_stages']:
            print(f'Already complete: {name}',flush=True)
            continue
        if source_hashes()!=manifest['source_sha256']:
            raise RuntimeError('runtime source changed during study; snapshot no longer describes execution')
        device='cpu' if name=='cpu_portability' else 'cuda'
        require_power(config)
        if device=='cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA stage requested but no CUDA device; use cpu_portability stage on a CPU machine')
        executor=GPT2Executor.from_local(Path(config['model_path']),device)
        calibration=_calibration(executor,config,output)
        try:
            if name=='gpu':
                gpu_repeats(executor,calibration,config,output)
            elif name=='prediction':
                matched_predictions(executor,calibration,config,output)
            elif name.endswith('portability'):
                portability(executor,calibration,config,output)
            else:
                quality_audits(executor,calibration,config,output)
        finally:
            del executor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if source_hashes()!=manifest['source_sha256']:
            raise RuntimeError('runtime source changed while measuring; stage is not accepted')
        manifest['completed_stages'].append(name)
        manifest['status']='complete' if set(manifest['completed_stages'])==set(STAGES) else 'in_progress'
        manifest['updated_utc']=datetime.now(timezone.utc).isoformat()
        manifest['calibration_sha256']={p.name:sha256_file(p) for p in output.glob('calibration_*.json')}
        write_json(output/'manifest.json',manifest)
        print(f'Completed M10 stage: {name}',flush=True)
    return {'output':str(output.resolve()),'completed_stages':manifest['completed_stages'],'status':manifest['status']}
