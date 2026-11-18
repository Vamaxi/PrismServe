"""Aggregate M10 evidence, draw scientific figures and update submission sources."""
from collections import defaultdict
from pathlib import Path
from statistics import mean,stdev,median
import json
import math
import sys
import argparse

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from prismserve.validation_study import paired_interval
from scripts.update_controller_report import insert_block

OUT=ROOT/'results/validation_study_final'


def load(path):
    return json.loads(path.read_text())


def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--analysis-only',action='store_true')
    args=parser.parse_args()
    OUT=args.output
    result_folder=OUT.name
    validation=load(OUT/'validation.json')
    assert validation['passed']
    config=load(OUT/'config.json')
    groups=defaultdict(list)
    for path in sorted((OUT/'gpu_runs').glob('*.json')):
        run=load(path)
        groups[(run['summary']['scenario'],run['summary']['policy'])].append(run)
    scenarios=[s['name'] for s in config['scenarios']]
    policies=config['policies']
    def average(s,p,k):
        return mean(r['summary'][k] for r in groups[(s,p)])
    comparisons=[]
    for scenario in scenarios:
        for alternative,reference in (('joint','joint_v03'),('full','full_v03'),('joint','full')):
            refs={r['summary']['seed']:r['summary'] for r in groups[(scenario,reference)]}
            changes=[100*(r['summary']['output_tokens_per_s']/refs[r['summary']['seed']]['output_tokens_per_s']-1)
                     for r in groups[(scenario,alternative)]]
            comparisons.append({'scenario':scenario,'alternative':alternative,'reference':reference,
                                'paired_throughput_change_pct':paired_interval(changes),
                                'individual_paired_changes_pct':changes})
    gpu_table=['| Scenario | Policy | Mean token/s | Median token/s | SD | CV % | Control % | P95 TTFT ms | P95 TBT ms |',
               '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in scenarios:
        for p in policies:
            values=[r['summary']['output_tokens_per_s'] for r in groups[(s,p)]]
            sd=stdev(values) if len(values)>1 else 0
            gpu_table.append(f"| {s} | {p} | {mean(values):.2f} | {median(values):.2f} | {sd:.2f} | {100*sd/mean(values):.1f} | {100*average(s,p,'search_admission_wall_fraction'):.2f} | {average(s,p,'ttft_p95_ms'):.1f} | {average(s,p,'tbt_p95_ms'):.1f} |")
    concise=['| Scenario | Old joint token/s | New joint token/s | Paired change % | Exploratory 95% interval | Old/new control % |',
             '|---|---:|---:|---:|---|---|']
    for s in scenarios:
        interval=next(r['paired_throughput_change_pct'] for r in comparisons if r['scenario']==s and r['reference']=='joint_v03')
        concise.append(f"| {s} | {average(s,'joint_v03','output_tokens_per_s'):.2f} | {average(s,'joint','output_tokens_per_s'):.2f} | {interval['mean']:+.1f} | [{interval['lower_95']:+.1f}, {interval['upper_95']:+.1f}] | {100*average(s,'joint_v03','search_admission_wall_fraction'):.2f} / {100*average(s,'joint','search_admission_wall_fraction'):.2f} |")
    joint_runs=[r for (s,p),rows in groups.items() if p=='joint' for r in rows]
    below=sum(r['summary']['search_admission_wall_fraction']<.05 for r in joint_runs)
    overhead_note=(f"Optimized joint control stays below 5% of workload wall time in {below}/{len(joint_runs)} runs; "
                   f"its worst observed fraction is {100*max(r['summary']['search_admission_wall_fraction'] for r in joint_runs):.2f}%. "
                   "This is a new session, not a re-labeling of M9's six-run overhead result. The fraction depends on "
                   "execution speed as well as absolute controller cost. All request latency includes telemetry, planning, transitions and drain.")
    best_full=sum(average(s,'full','output_tokens_per_s')>average(s,'joint','output_tokens_per_s') for s in scenarios)
    serving_interpretation=(f"Optimized full precision has higher mean throughput than optimized joint control in {best_full}/{len(scenarios)} scenarios. "
        "Improvements over the older joint controller do not establish an advantage over optimized full precision. "
        "Decision reuse also trades admission responsiveness for lower control cost: compare TTFT as well as throughput. "
        "Raw action histories distinguish control-overhead savings from gains due to actual quantization/offload.")
    if all(r['summary']['quantized_output_fraction']==0 and r['summary']['cpu_attention_output_fraction']==0 for r in joint_runs):
        serving_interpretation += ' Optimized joint control selects FP32 and device attention for every emitted token in this final session; its measured gains therefore concern control cost/admission, not active quantization or CPU offload.'
    action_table=['| Scenario | Policy | INT4 outputs % | CPU-attention outputs % | Max active | SLO failure % |',
                  '|---|---|---:|---:|---:|---:|']
    for s in scenarios:
        for p in policies:
            action_table.append(f"| {s} | {p} | {100*average(s,p,'quantized_output_fraction'):.2f} | {100*average(s,p,'cpu_attention_output_fraction'):.2f} | {max(r['summary']['max_active'] for r in groups[(s,p)])} | {100*average(s,p,'slo_failure_rate'):.2f} |")
    telemetry=[]
    all_samples=[]
    plugged=[]
    for rows in groups.values():
        for r in rows:
            summary=r['summary']; instrument=r['instrumentation']; g=instrument['gpu']
            telemetry.append({'scenario':summary['scenario'],'policy':summary['policy'],'seed':summary['seed'],**g,
                              'power_plugged_before':instrument['power_plugged_before'],
                              'power_plugged_after':instrument['power_plugged_after']})
            plugged.extend([instrument['power_plugged_before'],instrument['power_plugged_after']])
            all_samples.append(g)
    def span(field):
        values=[x[field] for x in all_samples if x.get(field)]
        return (min(v['min'] for v in values),max(v['max'] for v in values)) if values else (None,None)
    clock_range=span('clocks.current.sm'); temperature_range=span('temperature.gpu')
    range_text=lambda values,unit: 'unavailable' if values[0] is None else f'{values[0]:.0f}–{values[1]:.0f} {unit}'
    conditions=(f"Observed SM clocks span {range_text(clock_range,'MHz')}; GPU temperatures span "
                f"{range_text(temperature_range,'°C')}. AC-state observations are {sorted(set(str(p) for p in plugged))}. "
                "Driver power/thermal reason flags are retained per sample. These observations do not prove that a particular "
                "performance change was caused by throttling; clock settings and external OS activity were not controlled.")
    power_diagnostic=[]
    diagnostic_path=ROOT/'results/validation_study'
    if config.get('required_ac_power') and (diagnostic_path/'validation.json').exists():
        strata=defaultdict(list)
        for path in sorted((diagnostic_path/'gpu_runs').glob('steady_roomy__joint__*.json')):
            r=load(path);m=r['instrumentation']
            state=('AC' if m['power_plugged_before'] is True and m['power_plugged_after'] is True else
                   'battery' if m['power_plugged_before'] is False and m['power_plugged_after'] is False else 'transition')
            strata[state].append(r)
        power_diagnostic=['The first complete study changed power source during measurement and is retained separately in `results/validation_study/`. It is diagnostic, not the primary controlled comparison. The new study requires AC power before execution and at every recorded 500 ms host sample, including calibration; a failed condition refuses the affected measurement. Observed clock settings still vary.','',
            '| Original steady joint subgroup | Runs | Mean token/s | Mean observed SM MHz |','|---|---:|---:|---:|']
        for state,rows in sorted(strata.items()):
            clocks=[r['instrumentation']['gpu']['clocks.current.sm']['mean'] for r in rows if r['instrumentation']['gpu'].get('clocks.current.sm')]
            power_diagnostic.append(f"| {state} | {len(rows)} | {mean(r['summary']['output_tokens_per_s'] for r in rows):.2f} | {mean(clocks):.0f} |" if clocks else f'| {state} | {len(rows)} | N/A | N/A |')
        power_diagnostic+=['','This subgroup split is observational and also differs in round/order. It exposes a power/clock confound; it does not isolate a causal effect. Primary throughput comparisons below use the separate AC-conditioned session.','']
        if (ROOT/'results/validation_study_ac/STATUS.md').exists():
            power_diagnostic += ['An intermediate AC attempt stopped when a constrained joint worker reported no feasible plan for admitted work. It remains incomplete in `results/validation_study_ac/` and is excluded from successful-run aggregates. Available host RAM was about 0.8–1.0 GiB during that attempt; the same trace later completed with more than 2 GiB available. Its exact original cause remains unproven. The final source adds failure-only state/reservation snapshots and saves failed replay records; it does not bypass the safety check. Completion and zero-rejection statements below refer to the final accepted series, not every diagnostic attempt.','']
    probes=load(OUT/'prediction_probes.json')
    prediction_groups=defaultdict(list)
    for r in probes['observations']:
        prediction_groups[(r['predictor'],'all')].append(r)
        prediction_groups[(r['predictor'],r['phase'])].append(r)
    prediction_summary=[]
    prediction_table=['| Predictor | Phase | Executions | MAE ms | Bias ms | MAPE % |','|---|---|---:|---:|---:|---:|']
    for (predictor,phase),rows in sorted(prediction_groups.items()):
        errors=[r['predicted_execution_ms']-r['actual_execution_ms'] for r in rows]
        stats={'predictor':predictor,'phase':phase,'executions':len(rows),
               'mae_ms':mean(abs(e) for e in errors),'bias_ms':mean(errors),
               'mape_pct':100*mean(abs(e)/r['actual_execution_ms'] for e,r in zip(errors,rows))}
        prediction_summary.append(stats)
        prediction_table.append(f"| {predictor} | {phase} | {len(rows)} | {stats['mae_ms']:.2f} | {stats['bias_ms']:+.2f} | {stats['mape_pct']:.1f} |")
    def pred_mae(name):
        return next(r['mae_ms'] for r in prediction_summary if r['predictor']==name and r['phase']=='all')
    diagnosis=(f"On identical executed batches, overall MAE is {pred_mae('old_static'):.2f} ms with the old static table, "
               f"{pred_mae('fresh_static'):.2f} ms with the fresh static table, {pred_mae('fresh_legacy_ewma'):.2f} ms "
               f"with legacy feedback and {pred_mae('fresh_phase_ewma'):.2f} ms with phase feedback. "
               "These forecasts share actual execution durations and are made before observing each result. "
               "They therefore diagnose calibration/feedback without conflating different serving trajectories. "
               "Feedback remains dependent on probe order, action-family granularity and unobserved shapes; "
               "the best predictor here need not produce the best admission policy.")
    port_runs=[load(p) for p in sorted((OUT/'portability_runs').glob('*.json'))]
    ports=defaultdict(list)
    for r in port_runs:
        ports[(r['summary']['device'],r['summary']['policy'])].append(r)
    port_table=['| Backend | Policy | Mean token/s | P95 TTFT ms | P95 TBT ms | Control % | Max batch | INT4 output % |',
                '|---|---|---:|---:|---:|---:|---:|---:|']
    for (device,policy),rows in sorted(ports.items()):
        get=lambda key:mean(r['summary'][key] for r in rows)
        port_table.append(f"| {device} | {policy} | {get('output_tokens_per_s'):.2f} | {get('ttft_p95_ms'):.1f} | {get('tbt_p95_ms'):.1f} | {100*get('search_admission_wall_fraction'):.2f} | {max(r['summary']['max_execution_batch_size'] for r in rows)} | {100*get('quantized_output_fraction'):.2f} |")
    full_cpu={r['summary']['seed']:r for r in ports[('cpu','full')]}
    matches=total=0
    for run in ports[('cuda','full')]:
        reference={r['request_id']:r['token_ids'] for r in full_cpu[run['summary']['seed']]['requests']}
        for r in run['requests']:
            matches+=sum(a==b for a,b in zip(r['token_ids'],reference[r['request_id']]))
            total+=len(r['token_ids'])
    portability_note=(f"All {len(port_runs)} matched backend runs complete. CPU/CUDA full-precision greedy outputs match in "
                      f"{matches}/{total} token positions. CPU-primary execution selects no CPU-offload requests, as required "
                      "by its capability contract. Both use the same controller/batching source and separate calibration. "
                      "This establishes execution portability across two backends on one laptop, not physical multi-machine "
                      "or GPU-vendor portability. CPU's combined host quota and CUDA's separate memory pools are not equivalent resources.")
    fixed={p:load(OUT/'quality_fixed'/f'{p}.json') for p in config['quality']['fixed_schedules']}
    base=fixed['fp32']['nll']
    quality_table=['| Audit arm | Targets | Perplexity | Relative PPL % | INT4 output % | Quantized prefill iterations |',
                   '|---|---:|---:|---:|---:|---:|']
    quality_summary=[]
    for policy in config['quality']['policies']:
        r=load(OUT/'quality_dynamic'/f'{policy}.json');s=r['summary']
        row={'arm':'dynamic_'+policy,'targets':s['teacher_forced_targets'],'nll':s['teacher_forced_nll'],
             'perplexity':s['teacher_forced_perplexity'],'int4_output_pct':100*s['quantized_output_fraction'],
             'quantized_prefill_iterations':r['exposure']['quantized_prefill_iterations']}
        quality_summary.append(row)
    for name,r in fixed.items():
        quality_summary.append({'arm':'fixed_'+name,'targets':r['scored_targets'],'nll':r['nll'],'perplexity':r['perplexity'],
            'int4_output_pct':100*r['quantized_scored_targets']/r['scored_targets'],
            'quantized_prefill_iterations':r['quantized_prefill_iterations']})
    for r in quality_summary:
        r['relative_ppl_pct']=100*math.expm1(r['nll']-base)
        display_change=0 if abs(r['relative_ppl_pct'])<.0005 else r['relative_ppl_pct']
        quality_table.append(f"| {r['arm']} | {r['targets']} | {r['perplexity']:.3f} | {display_change:+.3f} | {r['int4_output_pct']:.2f} | {r['quantized_prefill_iterations']} |")
    quality_note=(f"Each arm scores the same {validation['quality_targets_per_arm']} distinct held-out positions in 16 disjoint "
                  "prompt/continuation windows; eight arms repeat those targets. Early-prefill quantization restores FP32 "
                  "before output emission but retains affected KV. Controlled schedules isolate this history effect and "
                  "bypass admission; their times are not serving benchmarks. Dynamic teacher forcing has its own policy "
                  "trajectory. These subset perplexities are not downstream-task accuracy or proof of a quality bound. "
                  f"FP32 dynamic/fixed token NLL and CPU-attention comparisons pass the 0.002 tolerance; the maximum "
                  f"observed token-NLL discrepancy across both checks is {max(validation['fp32_protocol_and_cpu_attention_max_abs_nll_error'].values()):.2e}.")
    if all(r['int4_output_pct']==0 and r['quantized_prefill_iterations']==0 for r in quality_summary if r['arm'].startswith('dynamic_')):
        quality_note += ' All three dynamic audits remain unquantized in this session. Their equal perplexities do not certify adaptive quantized outputs; the controlled schedules supply the separate precision-history evidence.'
    summary={'paired_gpu_comparisons':comparisons,'prediction':prediction_summary,'quality':quality_summary,
             'portability_full_fp32_matching_tokens':matches,'portability_full_fp32_total_tokens':total,
             'gpu_run_telemetry':telemetry,'gpu_sm_clock_range_mhz':clock_range,'gpu_temperature_range_c':temperature_range}
    (OUT/'analysis.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures=OUT/'figures';figures.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,len(scenarios),figsize=(13.5,4),squeeze=False)
    for ax,s in zip(axes[0],scenarios):
        for i,p in enumerate(policies):
            values=[r['summary']['output_tokens_per_s'] for r in groups[(s,p)]]
            ax.bar(i,mean(values),color='#176b87' if p=='joint' else '#91b9c7')
            ax.scatter([i+(j-2)*.045 for j in range(len(values))],values,color='#203c4d',s=15)
        ax.set_xticks(range(len(policies)),policies,rotation=40)
        ax.set_title(s);ax.set_ylabel('Actual output tokens/s');ax.grid(axis='y',alpha=.2)
    fig.suptitle('Fresh calibration, five rounds, balanced policy positions; individual runs shown')
    fig.tight_layout();fig.savefig(figures/'repeated_throughput.png',dpi=160);plt.close(fig)
    predictors=['old_static','fresh_static','fresh_legacy_ewma','fresh_phase_ewma']
    fig,ax=plt.subplots(figsize=(9,4))
    for i,name in enumerate(predictors):
        vals=[next(r['mae_ms'] for r in prediction_summary if r['predictor']==name and r['phase']==p) for p in ('prefill','mixed','decode')]
        ax.bar([j+i*.2 for j in range(3)],vals,width=.19,label=name)
    ax.set_xticks([j+.3 for j in range(3)],['prefill','mixed','decode']);ax.set_ylabel('Matched execution MAE (ms)')
    ax.set_title('Each predictor forecasts the same tensor executions');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(figures/'matched_prediction_mae.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,s in zip(axes,[scenarios[0],scenarios[-1]]):
        for p in policies:
            rows=groups[(s,p)]
            points=[(r['instrumentation']['gpu'].get('clocks.current.sm'),r['summary']['output_tokens_per_s']) for r in rows]
            ax.scatter([x['mean'] for x,y in points if x],[y for x,y in points if x],label=p,s=24)
        ax.set_title(s);ax.set_xlabel('Mean observed SM clock (MHz)');ax.set_ylabel('Actual token/s');ax.grid(alpha=.2)
    axes[1].legend(fontsize=8);fig.suptitle('Observed clocks versus throughput; association is not causation')
    fig.tight_layout();fig.savefig(figures/'clock_throughput.png',dpi=160);plt.close(fig)
    fixed_rows=[r for r in quality_summary if r['arm'].startswith('fixed_')]
    fig,ax=plt.subplots(figsize=(10,4))
    ax.bar([r['arm'].removeprefix('fixed_') for r in fixed_rows],[r['relative_ppl_pct'] for r in fixed_rows],
           color=['#cf864b' if 'early_prefill' in r['arm'] else '#176b87' for r in fixed_rows])
    ax.set_ylabel('Perplexity change versus FP32 (%)');ax.tick_params(axis='x',rotation=18)
    ax.set_title('2048 held-out targets: early precision changes can persist through KV')
    ax.axhline(0,color='#203c4d',linewidth=.8);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(figures/'quality_history.png',dpi=160);plt.close(fig)
    evidence=(f"{validation['gpu_runs']} instrumented GPU runs complete {validation['gpu_requests']} requests, "
              f"plus {validation['portability_runs']} matched CPU/CUDA runs ({validation['portability_requests']} requests). "
              f"There are zero rejections. {validation['prediction_shapes']} prediction shapes supply "
              f"{validation['matched_prediction_executions']} matched executions. Eight quality arms score "
              f"{validation['quality_targets_per_arm']} held-out targets each. All {validation['tests_passed']} tests pass.")
    limitations=("Five rounds and a single laptop remain a limited sample. Paired-change intervals bootstrap paired percent "
                 "changes, not independent machines; they are exploratory, not population confidence guarantees. "
                 "Steady traces repeat unchanged; burst arrivals vary by seed. Fresh calibration and standardized warmup "
                 "change conditions relative to M9, so cross-session absolute rates are not controlled speedup evidence. "
                 "TTFT measures engine token readiness in a non-streaming service. Reference INT4 still unpacks to floating "
                 "point, and CPU attention remains synchronous. Broader hardware/model/task validation remains open.")
    report=['# M10: instrumented reliability and backend portability study','',evidence,'',*power_diagnostic,
            '## Repeated GPU comparison','',*concise,'',overhead_note,'',serving_interpretation,'',*gpu_table,'',*action_table,'',conditions,'',
            '![Repeated real throughput](figures/repeated_throughput.png)','',
            '![Observed GPU clocks](figures/clock_throughput.png)','',
            '## Prediction on identical actual executions','',diagnosis,'',*prediction_table,'',
            '![Matched prediction error](figures/matched_prediction_mae.png)','',
            '## CPU/CUDA execution portability','',*port_table,'',portability_note,'',
            '## Expanded history-aware quality audit','',*quality_table,'',quality_note,'',
            '![Controlled precision-history audit](figures/quality_history.png)','',
            '## Evidence limits','',limitations,'',
            'Protocol: [validation_study.md](../../docs/validation_study.md). Raw timing, telemetry, sources and target histories are retained alongside this report.','']
    if not config.get('required_ac_power'):
        report[1:1]=['','**Diagnostic mixed-power session. Primary accepted results: `results/validation_study_final/`. See STATUS.md.**','']
    (OUT/'report.md').write_text('\n'.join(report),encoding='utf-8')
    content='\n'.join(['### Reliability and backend portability follow-up (v0.5)','',evidence,'',
        'M10 retains the serving policy while adding fresh calibration, three warmup passes, five GPU rounds, balanced policy positions, shuffled scenario order and read-only hardware instrumentation. It separately tests predictor accuracy on identical executed batches, uses device-specific calibration for CPU/CUDA serving, and expands independent quality audits.','',
        *power_diagnostic,
        *concise,'',overhead_note,'',serving_interpretation,'',conditions,'',
        '![Repeated GPU throughput with individual rounds](../results/validation_study/figures/repeated_throughput.png)','',
        diagnosis,'',
        '![Prediction errors on identical batches](../results/validation_study/figures/matched_prediction_mae.png)','',
        *port_table,'',portability_note,'',*quality_table,'',quality_note,'',
        '![Precision history affects later FP32 likelihoods](../results/validation_study/figures/quality_history.png)','',limitations,'',
        'Complete policy tails, paired intervals, observed clock plots and error breakdowns are in `results/validation_study/report.md`; `docs/validation_study.md` specifies the protocol.',''])
    content=content.replace('../results/validation_study/',f'../results/{result_folder}/').replace('`results/validation_study/report.md`',f'`results/{result_folder}/report.md`')
    if not args.analysis_only:
        insert_block(ROOT/'docs/final_report.md','VALIDATION_STUDY_RESULTS',content,'## 7. Analytical simulation results')
    slide_table=['| Scenario | Old joint token/s | New joint token/s | New control % |','|---|---:|---:|---:|']
    for s in scenarios:
        slide_table.append(f"| {s} | {average(s,'joint_v03','output_tokens_per_s'):.1f} | {average(s,'joint','output_tokens_per_s'):.1f} | {100*average(s,'joint','search_admission_wall_fraction'):.1f} |")
    slides='\n'.join(['# 16. Reliability: five instrumented rounds','',
        '60 real GPU runs; fresh calibration, three warmups, balanced order.','',*slide_table,'',
        'Clock/temperature/power logs retained; clock settings are observed, not fixed.','',
        '<!-- Speaker notes: The full report includes paired exploratory intervals and every repeat. Five repeats on a laptop do not establish universal wins. Clocks and driver flags do not prove a thermal explanation. M9 and M10 are separate sessions with different calibration and warmup. -->','',
        '---','','# 17. Portability, prediction and quality evidence','',
        f"- Same serving controller on CPU/CUDA: {validation['portability_runs']} paired-trace runs complete.",
        f'- FP32 greedy output agreement: {matches}/{total} token positions.',
        f"- {validation['prediction_shapes']} matched shapes compare four predictors before observation.",
        f"- Eight quality arms, {validation['quality_targets_per_arm']} held-out targets each.",
        '- Early quantized prefill can affect later FP32 outputs through KV history.',
        f"- Controlled early-prefill audit: {next(r['relative_ppl_pct'] for r in quality_summary if r['arm']=='fixed_early_prefill_int4_restore'):+.2f}% perplexity.",
        f"- {validation['tests_passed']} tests pass; all source/data snapshots verified.",'',
        'One laptop with two execution backends; physical multi-machine validation remains.','',
        '<!-- Speaker notes: Controlled quality schedules are diagnostic, bypass admission, and are not serving-speed measurements. Targets repeat across arms. Matched probes remove the trajectory confound from comparing predictor errors. Consult the raw phase errors before attributing effects to phase feedback. -->','','---',''])
    if not args.analysis_only:
        deck=ROOT/'docs/presentation.md'
        text=deck.read_text(encoding='utf-8').replace('# 16. What remains uncertain','# 18. What remains uncertain').replace('# 17. Answer and next step','# 19. Answer and next step')
        deck.write_text(text,encoding='utf-8')
        insert_block(deck,'VALIDATION_STUDY_SLIDES',slides,'# 18. What remains uncertain')
    print(evidence)
    print(diagnosis)
    print('Updated M10 analysis, four figures, report section and two slides.')


if __name__=='__main__':
    main()
