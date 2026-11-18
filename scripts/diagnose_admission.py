"""Instrument one reproduction without changing the archived serving modules."""
from dataclasses import asdict,replace
from pathlib import Path
import argparse
import json
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    import torch
    from prismserve.live_engine import ContinuousEngine,ServingConfig
    from prismserve.live_experiments import _requests,replay
    from prismserve.pretrained import GPT2Executor
    from prismserve.types import Plan
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,default=31)
    parser.add_argument('--output',type=Path,default=Path('results/admission_diagnostic'))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    payload=json.loads(Path('results/validation_study_ac/calibration_cuda.json').read_text())
    executor=GPT2Executor.from_local(Path('models/gpt2'),'cuda')
    trace=_requests(executor,12,384,24,2,'bursty',args.seed)
    observations=[]
    original=ContinuousEngine._tick
    def traced_tick(engine):
        try:
            original(engine)
            if engine._events:
                e=engine._events[-1]
                observations.append({'iteration':e['iteration'],'device_limit':engine.controller.profile.device_memory_bytes,
                                     'host_limit':engine.controller.profile.host_memory_bytes,'device_reserved':e['device_reserved_bytes'],
                                     'host_reserved':e['host_reserved_bytes'],'host_available':e['telemetry']['host_available_bytes'],
                                     'quantized_layers':e['quantized_layers'],'cpu_ids':e['cpu_request_ids']})
        except BaseException as error:
            active=list(engine._active);chunks=engine._chunks(active);work=engine._work(active,chunks)
            plans=[]
            eligible=[r.request_id for r in work if not r.prefill_tokens]
            for q in engine.controller.quantization_candidates:
                for count in range(len(eligible)+1):
                    plan=Plan(q,tuple(eligible[:count]))
                    est=engine.controller._estimate(work,plan,engine._state)
                    plans.append({'plan':asdict(plan),'estimate':asdict(est)})
            detail={'error':str(error),'state':asdict(engine._state),'work':[asdict(w) for w in work],
                    'profile_at_failure':asdict(engine.controller.profile),'plans':plans,
                    'memory':engine.executor.memory_report(),'recent_iterations':observations[-12:]}
            (args.output/'failure.json').write_text(json.dumps(detail,indent=2),encoding='utf-8')
            raise
    ContinuousEngine._tick=traced_tick
    try:
        result=replay(executor,payload,trace,ServingConfig(device_memory_budget_bytes=700*2**20),'joint')
        (args.output/'successful_reproduction.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(result['summary'],indent=2))
    finally:
        ContinuousEngine._tick=original
        (args.output/'per_iteration_limits.json').write_text(json.dumps(observations,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
