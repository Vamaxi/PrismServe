"""Create the capacity map and integrate the last experiment into submission."""
from collections import defaultdict
from pathlib import Path
from statistics import mean
import argparse
import csv
import json
import math
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from prismserve.frontier import outcome,rescue_cells
from scripts.update_controller_report import insert_block


def load(p):
    return json.loads(p.read_text())


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=ROOT/'results/capacity_frontier')
    args=p.parse_args();out=args.output
    v=load(out/'validation.json');assert v['passed']
    c=load(out/'config.json')
    runs=[load(p) for p in sorted((out/'runs').glob('*.json'))]
    groups=defaultdict(list)
    summaries=[]
    for r in runs:
        s=r['summary'];groups[(s['prompt_tokens'],s['device_budget_mib'],s['policy'])].append(r)
        summaries.append(dict(s,outcome=outcome(s),**r['exposure']))
    with (out/'summary.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    cells=[]
    for (prompt,quota,policy),rows in sorted(groups.items()):
        complete=all(outcome(r['summary'])=='complete' for r in rows)
        avg=lambda name:mean(r['summary'][name] for r in rows)
        cells.append({'prompt_tokens':prompt,'quota_mib':quota,'policy':policy,'repeats':len(rows),
            'all_repeats_complete':complete,'mean_completed_requests':avg('completed'),
            'completion_fraction':avg('completed')/c['workload']['requests'],
            'mean_output_tokens_per_s_if_complete':avg('output_tokens_per_s') if complete else None,
            'mean_ttft_p95_ms_if_complete':avg('ttft_p95_ms') if complete else None,
            'mean_tbt_p95_ms_if_complete':avg('tbt_p95_ms') if complete else None,
            'slo_failure_rate':avg('slo_failure_rate'),'int4_output_fraction':avg('quantized_output_fraction'),
            'cpu_attention_output_fraction':avg('cpu_attention_output_fraction'),
            'max_active':max(r['summary']['max_active'] for r in rows),
            'max_cuda_peak_mib':max(r['summary']['peak_cuda_allocated_bytes'] for r in rows)/2**20})
    minima=[]
    for prompt in c['prompt_lengths']:
        for policy in c['policies']:
            feasible=[r['quota_mib'] for r in cells if r['prompt_tokens']==prompt and r['policy']==policy and r['all_repeats_complete']]
            minima.append({'prompt_tokens':prompt,'policy':policy,'lowest_tested_complete_quota_mib':min(feasible) if feasible else None})
    minimum_table=['| Prompt tokens | FP32 | Quantization only | Offload only | Joint |','|---:|---:|---:|---:|---:|']
    for prompt in c['prompt_lengths']:
        values=[next(r['lowest_tested_complete_quota_mib'] for r in minima if r['prompt_tokens']==prompt and r['policy']==p) for p in c['policies']]
        minimum_table.append(f"| {prompt} | "+' | '.join('N/A' if q is None else str(q) for q in values)+' |')
    rescues=rescue_cells(runs)
    rescue_groups=defaultdict(list)
    for r in rescues:
        rescue_groups[(r['prompt_tokens'],r['device_budget_mib'],r['policy'])].append(r['seed'])
    rescue_table=['| Prompt | Quota MiB | Policy completing all requests | Paired repeats with FP32 rejecting all |',
                  '|---:|---:|---|---:|']
    for (prompt,quota,policy),seeds in sorted(rescue_groups.items()):
        rescue_table.append(f'| {prompt} | {quota} | {policy} | {len(seeds)}/{len(c["seeds"])} |')
    if not rescues:
        rescue_table.append('| N/A | N/A | No rescue observed | 0 |')
    quality=[]
    for case in load(out/'quality_selection.json')['cases']:
        prompt=case['prompt_tokens'];quota=case['device_budget_mib'];policy=case['policy']
        r=load(out/'quality_runs'/f'p{prompt}__m{quota}__{policy}.json')
        ref=load(out/'quality_reference'/f'p{prompt}.json')
        s=r['summary'];base=ref['summary']
        quality.append(dict(case,targets=s['teacher_forced_targets'],perplexity=s['teacher_forced_perplexity'],
            reference_perplexity=base['teacher_forced_perplexity'],
            relative_ppl_pct=100*math.expm1(s['teacher_forced_nll']-base['teacher_forced_nll']),
            int4_output_pct=100*s['quantized_output_fraction'],
            quantized_prefill_iterations=r['exposure']['quantized_prefill_iterations'],
            cpu_output_pct=100*s['cpu_attention_output_fraction']))
    quality_table=['| Prompt | Quota MiB | Policy | Audit targets | FP32 PPL | Actual PPL | PPL change % | INT4 output % |',
                   '|---:|---:|---|---:|---:|---:|---:|---:|']
    for r in quality:
        quality_table.append(f"| {r['prompt_tokens']} | {r['device_budget_mib']} | {r['policy']} | {r['targets']} | {r['reference_perplexity']:.3f} | {r['perplexity']:.3f} | {r['relative_ppl_pct']:+.3f} | {r['int4_output_pct']:.2f} |")
    if not quality:
        quality_table.append('| N/A | N/A | No quantized completed cell | 0 | N/A | N/A | N/A | N/A |')
    complete_table=['| Prompt | Quota | Policy | Completed/requested per repeat | Token/s if all complete | P95 TTFT ms | P95 TBT ms | SLO failure % | INT4 % | CPU attention % | Max active |',
                    '|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|']
    fmt=lambda x:'N/A' if x is None else f'{x:.2f}'
    for r in cells:
        complete_table.append(f"| {r['prompt_tokens']} | {r['quota_mib']} | {r['policy']} | {r['mean_completed_requests']:.2f}/{c['workload']['requests']} | {fmt(r['mean_output_tokens_per_s_if_complete'])} | {fmt(r['mean_ttft_p95_ms_if_complete'])} | {fmt(r['mean_tbt_p95_ms_if_complete'])} | {100*r['slo_failure_rate']:.2f} | {100*r['int4_output_fraction']:.2f} | {100*r['cpu_attention_output_fraction']:.2f} | {r['max_active']} |")
    analysis={'cells':cells,'sampled_minimum_quotas':minima,'rescue_policy_seed_cells':rescues,'quality':quality,
              'notes':['Lowest tested quota is a sampled frontier, not an exact continuous minimum.',
                       'All-complete throughput cells only; rejection data remain in raw CSV/JSON.',
                       'Quality selection is trajectory-aware, representative, and does not certify all greedy outputs.']}
    (out/'analysis.json').write_text(json.dumps(analysis,indent=2),encoding='utf-8')
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures=out/'figures';figures.mkdir(exist_ok=True)
    for key,label,name in [('completion_fraction','Mean fraction of requests completed','capacity_frontier'),
                           ('max_active','Maximum observed active requests','concurrency'),
                           ('mean_output_tokens_per_s_if_complete','Output token/s: only all-complete cells','throughput')]:
        fig,axes=plt.subplots(1,len(c['policies']),figsize=(15,4),squeeze=False,layout='constrained')
        finite=[r[key] for r in cells if r[key] is not None]
        upper=1 if key=='completion_fraction' else max(finite,default=1)
        for ax,policy in zip(axes[0],c['policies']):
            matrix=np.full((len(c['prompt_lengths']),len(c['device_quotas_mib'])),np.nan)
            for i,prompt in enumerate(c['prompt_lengths']):
                for j,quota in enumerate(c['device_quotas_mib']):
                    value=next(r[key] for r in cells if r['prompt_tokens']==prompt and r['quota_mib']==quota and r['policy']==policy)
                    matrix[i,j]=np.nan if value is None else value
                    text='N/A' if value is None else f'{100*value:.0f}%' if key=='completion_fraction' else f'{value:.0f}'
                    ax.text(j,i,text,ha='center',va='center',fontsize=8,color='white' if value is not None and value>=.65*upper else '#172f41')
            im=ax.imshow(matrix,vmin=0,vmax=upper,cmap='YlGnBu',aspect='auto',alpha=.65)
            ax.set_xticks(range(len(c['device_quotas_mib'])),c['device_quotas_mib'],rotation=45)
            ax.set_yticks(range(len(c['prompt_lengths'])),c['prompt_lengths'])
            ax.set_xlabel('Device quota (MiB)');ax.set_ylabel('Prompt tokens');ax.set_title(policy)
        fig.colorbar(im,ax=list(axes[0]),shrink=.75,label=label)
        fig.suptitle(f'Real GPT-2: {label}; {len(c["seeds"])} repeats per cell')
        fig.savefig(figures/f'{name}.png',dpi=165);plt.close(fig)
    if quality:
        fig,ax=plt.subplots(figsize=(9,3.5),layout='constrained')
        labels=[f"{r['policy']}\np{r['prompt_tokens']} / {r['device_budget_mib']} MiB" for r in quality]
        ax.bar(labels,[r['relative_ppl_pct'] for r in quality],color='#176b87')
        ax.set_ylabel('PPL change versus paired FP32 (%)');ax.axhline(0,color='#203c4d',linewidth=.7)
        ax.set_title('Actual frontier-policy quality; target counts reported separately');ax.grid(axis='y',alpha=.2)
        fig.savefig(figures/'quality.png',dpi=165);plt.close(fig)
    evidence=(f"{v['serving_runs']} actual serving runs request {v['requested_requests']} completions: "
              f"{v['completed_requests']} complete and {v['rejected_requests']} are explicitly rejected. "
              f"There are {v['complete_runs']} all-complete runs, {v['all_rejected_runs']} all-rejected runs and "
              f"{v['partial_runs']} partial runs. No worker failure is accepted. All {v['tests_passed']} tests pass.")
    interpretation=(f"There are {len(rescues)} paired policy/seed observations where an adaptive policy completes every request "
        "while FP32 rejects every request under the same quota and inputs. Quantization and offload are evaluated independently "
        "as well as jointly. Joint control need not beat the single mechanisms. The measured frontier includes full-context KV "
        "reservation, calibration-derived workspace, headroom and FP32-to-INT4 transition staging. It is conditional on this "
        "warm-start implementation, not a physical-memory optimum or a cold-load feasibility result.")
    if all(next(r['lowest_tested_complete_quota_mib'] for r in minima if r['prompt_tokens']==p and r['policy']=='joint') ==
           next(r['lowest_tested_complete_quota_mib'] for r in minima if r['prompt_tokens']==p and r['policy']=='morph_only') for p in c['prompt_lengths']):
        interpretation += ' Joint and quantization-only control share the same sampled minimum quotas here; a capacity advantage of joint control over quantization alone is not demonstrated.'
    quality_note=("Quality audits keep each selected cell's exact prompt/output lengths and quota. For each prompt and precision-capable "
        "policy, the lowest all-repeat-complete quota with observed quantization is selected without consulting test NLL. "
        "Disjoint held-out windows are compared with the same targets under FP32 at a roomy quota. Target counts are limited "
        "by the stored prefix and long prompts. Audit trajectories remain separate from greedy trajectories; this supplements "
        "M10's larger controlled audits and does not prove a task-accuracy bound. "
        f"Actual quantization occurs in {v['audits_with_actual_quantization']}/{v['selected_adaptive_quality_audits']} selected adaptive audits; "
        "an audit that remains FP32 cannot certify the quantized trajectory observed during its separate greedy replay.")
    limitations=("The first pilot missed a narrow transition-feasible region. Before the full sweep, validation-calibration reservation "
        "bounds motivated the final grid (580/600/630/675/700/1400 MiB); both pilots and the diagnostic preflight are retained. "
        "Cold model loading/calibration/warmup precede timed enforcement. Reference INT4 has unpacking costs, CPU attention is "
        "synchronous, and prefill CPU offload is unsupported. Rejections count as SLO failures, and missing latency is N/A. "
        "Finite grid spacing and three seeds on one GPU limit generalization. Resource limits are sampled and cannot isolate "
        "the worker from competing processes. Actual CUDA peaks above quota are separately reported: "
        f"{v['actual_cuda_peak_over_quota_runs']} runs.")
    report=['# M11: measured capacity frontier','',evidence,'',
        '## Lowest tested quota completing every request in every repeat','',*minimum_table,'',interpretation,'',
        '![Actual completion frontier](figures/capacity_frontier.png)','',*rescue_table,'',
        '![Observed concurrency](figures/concurrency.png)','',
        '## Quality where the policy actually quantizes','',*quality_table,'',quality_note,'']
    if quality: report+=['![Frontier quality tradeoff](figures/quality.png)','']
    report+=['## Complete cell results','',*complete_table,'',
             '![Throughput among fully completed cells](figures/throughput.png)','',limitations,'',
             'See [protocol and reproduction](../../docs/capacity_frontier.md) and [demo/viva guide](../../docs/demo_and_viva.md).','']
    (out/'report.md').write_text('\n'.join(report),encoding='utf-8')
    content='\n'.join(['### Capacity frontier and submission release (M11, v1.0)','',evidence,'',
        *minimum_table,'',interpretation,'',*rescue_table,'',
        '![Measured warm-start completion frontier](../results/capacity_frontier/figures/capacity_frontier.png)','',
        *quality_table,'',quality_note,'',limitations,'',
        'Full cell-level completion, throughput, tail latency, SLO failures, actual memory peaks and precision/offload exposure are in `results/capacity_frontier/report.md`. The final demonstration and viva guide are in `docs/demo_and_viva.md`.',''])
    insert_block(ROOT/'docs/final_report.md','FRONTIER_RESULTS',content,'## 7. Analytical simulation results')
    slides='\n'.join(['# 18. Where adaptation becomes necessary','',
        f"{v['serving_runs']} real quota/context replays; FP32, quantization only, offload only and joint.",'',*minimum_table,'',
        'Lowest tested complete quotas (MiB), not exact physical limits.','',
        '<!-- Speaker notes: Include transition staging and future KV in admission. Cold loading and warmup precede the timed quota. The complete grid and every rejection are retained. Joint control is not assumed to beat either mechanism alone. -->','','---','',
        '# 19. Capacity gain has a quality cost','',*rescue_table,'',
        f"- {v['selected_adaptive_quality_audits']} trajectory-aware quality audits, plus paired FP32 references.",
        '- Audit prompt/output lengths match the selected frontier cell.',
        '- Report quantization exposure and target counts with perplexity.',
        '- Rejected requests are SLO failures; their latency is N/A.','',
        '<!-- Speaker notes: Read the actual audit PPL values from the report. The long-context audit is small and supplements M10; it does not prove task accuracy. Reference quantization can expand capacity while slowing execution. Explain what the common hardware contract does and what remains kernel/device-specific. -->','','---',''])
    deck=ROOT/'docs/presentation.md'
    text=deck.read_text(encoding='utf-8').replace('# 18. What remains uncertain','# 20. What remains uncertain').replace('# 19. Answer and next step','# 21. Answer and next step')
    deck.write_text(text,encoding='utf-8')
    insert_block(deck,'FRONTIER_SLIDES',slides,'# 20. What remains uncertain')
    print(evidence)
    print('Updated frontier tables, figures, report section and two slides.')


if __name__=='__main__':
    main()
