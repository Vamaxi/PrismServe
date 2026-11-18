"""Build a comprehensive companion without mutating the frozen v1.0 release.

Run from the project root: python reports/detailed/build_report.py
All numerical tables and added plots derive from existing archived measurements.
"""
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
from statistics import mean
import csv
import hashlib
import html
import json
import math
import re
import shutil
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from diagrams import make_diagrams
from renderers import prepare_html,export_pdf,export_html,export_atlas

ASSETS=HERE/'assets';TABLES=HERE/'tables';OUTPUT=ROOT/'deliverables'
INPUTS=set()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text(path):
    path=ROOT/path if not Path(path).is_absolute() else Path(path)
    INPUTS.add(path)
    return path.read_text(encoding='utf-8')


def data(path):return json.loads(text(path))


def csv_rows(path):
    from io import StringIO
    return list(csv.DictReader(StringIO(text(path))))


def number(value,places=2):
    if value is None or value=='':return 'N/A'
    try:return f'{float(value):,.{places}f}'
    except (ValueError,TypeError):return str(value)


class Book:
    def __init__(self):
        self.parts=[];self.figures=[];self.tables=[];self.used_plots=set();self.equations=[]
    def add(self,content):self.parts.append(content.strip()+'\n')
    def chapter(self,title):self.add('## '+title)
    def sub(self,title):self.add('### '+title)
    def table(self,caption,headers,rows,source='',split=True):
        source=str(source)
        if len(headers)>8 and split:
            for start in range(3,len(headers),5):
                indices=list(range(3))+list(range(start,min(start+5,len(headers))))
                self.table(caption+f' (field panel {1+(start-3)//5})',[headers[i] for i in indices],
                           [[row[i] for i in indices] for row in rows],source,False)
            return
        if len(rows)>18:
            for start in range(0,len(rows),18):
                self.table(caption+f' (rows {start+1}-{min(start+18,len(rows))})',headers,rows[start:start+18],source,False)
            return
        n=len(self.tables)+1;self.tables.append({'number':n,'caption':caption,'rows':len(rows),'source':source})
        self.add(f'<p class="caption" id="table-{n}"><strong>Table {n}.</strong> {html.escape(caption)}</p>')
        clean=lambda v:str(v).replace('|','/').replace('\n',' ')
        self.add('| '+' | '.join(clean(x) for x in headers)+' |\n| '+' | '.join('---' for _ in headers)+' |\n'+
                 '\n'.join('| '+' | '.join(clean(v) for v in row)+' |' for row in rows))
        if source:self.add('<p class="small">Source: '+html.escape(str(source))+'</p>')
    def figure(self,path,caption,kind='Measured experiment',source=None):
        path=Path(path)
        if not path.is_absolute():path=ROOT/path
        if not path.exists():raise FileNotFoundError(path)
        n=len(self.figures)+1
        if path.is_relative_to(ROOT/'results'):
            INPUTS.add(path);self.used_plots.add(path)
        if path.parent!=ASSETS:
            target=ASSETS/f'figure_{n:02d}_{path.stem}.png';shutil.copyfile(path,target)
        else:target=path
        rel=target.relative_to(HERE).as_posix()
        src=source or path.relative_to(ROOT).as_posix()
        self.figures.append({'number':n,'caption':caption,'kind':kind,'asset':rel,'source':src})
        self.add(f'<div class="figure" id="figure-{n}"><img src="{rel}" alt="Figure {n}: {html.escape(caption,quote=True)}" />'
                 f'<p class="caption"><strong>Figure {n}.</strong> {html.escape(caption)} <em>[{html.escape(kind)}]</em><br />'
                 f'Source: {html.escape(src)}</p></div>')
    def equation(self,expression,explanation):
        import matplotlib.pyplot as plt
        n=len(self.equations)+1;path=ASSETS/f'equation_{n:02d}.png'
        fig,ax=plt.subplots(figsize=(11,1.0));ax.axis('off')
        ax.text(.5,.5,'$'+expression+'$',ha='center',va='center',fontsize=17,color='#193b50')
        fig.savefig(path,dpi=210,bbox_inches='tight',facecolor='white');plt.close(fig)
        self.equations.append({'number':n,'expression':expression,'asset':path.relative_to(HERE).as_posix()})
        self.add(f'<div class="figure"><img src="assets/{path.name}" alt="Equation {n}" />'
                 f'<p class="caption"><strong>Equation {n}.</strong> {html.escape(explanation)}</p></div>')
    def import_md(self,path,drop_title=True,heading_offset=1,asset_base=None):
        source=ROOT/path;raw=text(path)
        source_ref='docs/final_report.md' if source.parent==HERE and source.name.startswith('_') else str(path)
        initial={'_tiny_source':'Tiny-model quality and storage','_pretrained_source':'Pretrained GPT-2 measured variants',
                 '_simulation_source':'Analytical synthetic-profile comparison'}
        lines=raw.splitlines();i=0;context=initial.get(source.stem,source.stem);first=True
        while i<len(lines):
            line=lines[i]
            if first and drop_title and line.startswith('# '):first=False;i+=1;continue
            if line.startswith('#'):
                level=len(line)-len(line.lstrip('#'));context=line.lstrip('# ').strip()
                self.add('#'*min(4,level+heading_offset)+' '+context);i+=1;continue
            match=re.search(r'!\[([^]]*)\]\(([^)]+)\)',line)
            if match:
                target=((asset_base or source.parent)/match.group(2)).resolve()
                mode='Analytical simulation' if 'simulation' in str(source) or 'load_sweep' in str(source) else 'Measured experiment'
                self.figure(target,match.group(1),mode);i+=1;continue
            if line.strip().startswith('|') and i+1<len(lines) and re.match(r'^\s*\|?\s*:?-{2,}',lines[i+1]):
                headers=[x.strip() for x in line.strip().strip('|').split('|')];rows=[];i+=2
                while i<len(lines) and lines[i].strip().startswith('|'):
                    rows.append([x.strip() for x in lines[i].strip().strip('|').split('|')]);i+=1
                self.table(context+' — published results',headers,rows,source_ref)
                continue
            line=re.sub(r'\[([^]]+)\]\((?!https?://)([^)]+)\)',lambda m:m.group(1)+' (`'+m.group(2)+'` in the source artifact)',line)
            self.parts.append(line+'\n');i+=1
    def export_csv(self,name,rows):
        target=TABLES/name
        if not rows:return
        fields=list(dict.fromkeys(k for row in rows for k in row))
        with target.open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            for row in rows:writer.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()})


def new_plots():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    outputs={}
    def save(fig,name):
        fig.tight_layout()
        for ext in ('png','svg'):fig.savefig(ASSETS/f'{name}.{ext}',dpi=190,bbox_inches='tight')
        plt.close(fig);outputs[name]=ASSETS/f'{name}.png'
    tiny=data('results/demo/real_backend.json')['training']
    fig,ax=plt.subplots(figsize=(8,3.4));ax.plot([r['step'] for r in tiny['history']],[r['train_nll'] for r in tiny['history']],marker='o',color='#176b87')
    ax.set(xlabel='Training step',ylabel='Sampled training NLL',title='Actual tiny-decoder training: 149,248 parameters, 80 steps');ax.grid(alpha=.2);save(fig,'tiny_training')
    cal=data('results/pretrained/calibration.json')
    fig,ax=plt.subplots(figsize=(9,3.7));chosen=set(cal['layer_order'][:3])
    ax.bar([r['layer'] for r in cal['layers']],[r['mean_logit_kl'] for r in cal['layers']],color=['#cf864b' if r['layer'] in chosen else '#176b87' for r in cal['layers']])
    ax.set(xticks=list(range(12)),xlabel='GPT-2 transformer block (zero-based)',ylabel='Mean logit KL',title='Validation-only sensitivity; orange marks selected blocks 1, 7 and 9');ax.grid(axis='y',alpha=.2);save(fig,'layer_sensitivity')
    adapt=data('results/pretrained/adaptive.json');rows=adapt['decisions']
    fig,axes=plt.subplots(2,1,figsize=(9,5),sharex=True)
    x=[r['token_index'] for r in rows]
    axes[0].step(x,[int(bool(r['plan']['cpu_request_ids'])) for r in rows],where='post',color='#176b87');axes[0].set(yticks=[0,1],yticklabels=['device KV','CPU KV'],title='Actual adaptive GPT-2 placement under changing declared quota')
    axes[1].bar(x,[r['application_and_migration_ms_measured'] for r in rows],color='#cf864b');axes[1].set(xlabel='Recorded decode/control token index',ylabel='Application / migration ms')
    for ax in axes:ax.grid(alpha=.2)
    save(fig,'adaptive_placement')
    q=data('results/live/quality.json')['policy_summaries']
    fig,ax=plt.subplots(figsize=(9,3.5));ax.bar([r['policy'] for r in q],[r['relative_ppl_change_pct'] for r in q],color='#176b87');ax.tick_params(axis='x',rotation=22)
    ax.set(ylabel='Relative audit PPL change (%)',title='M8 separate teacher-forced quality: 64 targets per policy');ax.grid(axis='y',alpha=.2);save(fig,'live_quality')
    ports=defaultdict(list)
    for p in sorted((ROOT/'results/validation_study_final/portability_runs').glob('*.json')):
        r=data(p)['summary'];ports[(r['device'],r['policy'])].append(r)
    fig,axes=plt.subplots(1,2,figsize=(10,3.7))
    labels=[f'{d}\n{p}' for d,p in sorted(ports)]
    for ax,key,title in zip(axes,['output_tokens_per_s','ttft_p95_ms'],['Output token/s','P95 engine TTFT (ms)']):
        ax.bar(labels,[mean(r[key] for r in ports[k]) for k in sorted(ports)],color=['#176b87','#5b9aaf']*2)
        ax.set_title(title);ax.grid(axis='y',alpha=.2)
    fig.suptitle('M10 actual CPU/CUDA serving; same small paired traces, three seeds');save(fig,'cpu_cuda')
    rows=csv_rows('results/capacity_frontier/summary.csv');counts=defaultdict(lambda:[0,0])
    for r in rows:
        counts[r['policy']][0]+=int(r['completed']);counts[r['policy']][1]+=int(r['rejected'])
    labels=['full','morph_only','offload_only','joint'];fig,ax=plt.subplots(figsize=(8.5,3.7))
    ax.bar(labels,[counts[k][0] for k in labels],label='Completed',color='#176b87')
    ax.bar(labels,[counts[k][1] for k in labels],bottom=[counts[k][0] for k in labels],label='Explicitly rejected',color='#d79578')
    ax.set(ylabel='Requested completions across grid',title='M11 outcome accounting: all 864 requests remain in the denominator');ax.legend();ax.grid(axis='y',alpha=.2);save(fig,'frontier_outcomes')
    a=data('results/capacity_frontier/analysis.json');fig,ax=plt.subplots(figsize=(8.5,3.8))
    for policy,marker in zip(labels,['o','s','^','D']):
        selected=sorted([r for r in a['sampled_minimum_quotas'] if r['policy']==policy],key=lambda r:r['prompt_tokens'])
        ax.plot([r['prompt_tokens'] for r in selected],[r['lowest_tested_complete_quota_mib'] for r in selected],marker=marker,label=policy,alpha=.8)
    ax.set(xlabel='Prompt tokens (16 output tokens reserved)',ylabel='Lowest tested complete quota (MiB)',title='Sampled minima; connecting lines guide the eye only');ax.legend();ax.grid(alpha=.2);save(fig,'sampled_minima')
    return outputs


def extract(raw,start,end):
    return raw.split(start,1)[1].split(end,1)[0].strip()


def build():
    ASSETS.mkdir(parents=True,exist_ok=True);TABLES.mkdir(exist_ok=True)
    release=data('deliverables/release.json')
    original_paths={ROOT/p for p in [*release['source_sha256'],*release['evidence_sha256']]}
    original_paths|={ROOT/'deliverables'/p for p in ['PrismServe_Project.zip','PrismServe_Final_Report.pdf','PrismServe_Presentation.pptx','PrismServe_Presentation.pdf','release.json']}
    original_hashes={p:digest(p) for p in original_paths}
    for p in ['Hardware_Project_Proposal (1).pdf','mid_project_ppt.pdf','MORPHSERVE_paper_reference.pdf','neo_saving_gpu_research_paper_reference.pdf']:INPUTS.add(ROOT/p)
    final=text('docs/final_report.md');b=Book();diagrams=make_diagrams(ASSETS);plots=new_plots()
    frontier=data('results/capacity_frontier/analysis.json');m10=data('results/validation_study_final/analysis.json')
    vf=data('results/capacity_frontier/validation.json');tests=data('results/validation.json')
    b.chapter('1. Executive summary and principal findings')
    b.add('''PrismServe is an independently implemented, hardware-profile-driven research prototype for coordinating reversible layer quantization and CPU attention placement during language-model serving. It takes its two mechanisms from the research directions represented by MorphServe and NEO [1,2], but uses its own model adapter, request scheduler, cost model and experiment harness. The main actual model is pretrained GPT-2 with 124,439,808 parameters. A small trained decoder supports fast correctness checks; separate analytical experiments explore an 8B-shaped model.

The central question is whether a common resource contract can make useful precision/placement decisions without hard-coded product-name rules. The implementation demonstrates this engineering interface on CPU and CUDA on one laptop. The results support conditional capacity and quality tradeoffs, rather than universal speedups, guaranteed task accuracy, optimized low-bit arithmetic, or physical multi-vendor portability.

The strongest capacity observation occurs in the final warm-start grid. With four requests, 768 prompt tokens, 16 output tokens and a 675 MiB serving quota, FP32 and offload-only reject all requests. Quantization-only and joint control complete all requests across three paired repeats. Their lowest tested complete quota is 675 MiB, against 700 MiB for the other two policies. Joint control does not improve that sampled frontier over quantization alone. The corresponding joint audit increases perplexity by 2.752% over the paired FP32 reference on 96 held-out targets; this small audit is a measured cost, not a quality guarantee.

In the final reliability study, optimized full precision has higher mean throughput than optimized joint control in all three scenarios. Nevertheless, efficient joint scheduling improves over the earlier joint controller by about 2.1%, 3.2% and 21.9% in paired mean percent changes. All 15 optimized-joint repetitions remain below 5% control overhead (worst 4.97%). On identical prediction probes, fresh calibration reduces static MAE from 71.88 to 10.31 ms. Legacy feedback reaches 4.55 ms, outperforming phase feedback at 5.54 ms. The additional phase structure is therefore not presented as an accuracy improvement.

This companion preserves the frozen submission v1.0.0. It adds original architecture/flow diagrams, every archived result chart, complete run-level tables and editable document output. No new model training, serving benchmark or quality measurement is performed while producing it. Tables and added plots are regenerated from archived records; schematics are labeled explicitly.''')
    b.table('Evidence map and headline counts',['Evidence stream','Model / execution','Size','Interpretation'],[
        ['Correctness and tiny training','149,248-parameter decoder','80 training steps; 480 held-out targets','Functional mechanisms, not general language competence'],
        ['Pretrained model evaluation','Real GPT-2, CPU/CUDA','4095 subset-quality targets per variant','Actual model outputs, storage and timing'],
        ['Analytical studies','Synthetic 8B-shaped model','63 main + 25 load runs','Modeled sensitivity, not physical 8B execution'],
        ['M8 measured continuous serving','Real GPT-2','42 serving runs; 7 quality audits','Shared dense batches and queue-aware wall time'],
        ['M9 controller efficiency','Real GPT-2','36 serving runs; 6 quality audits','Old/new comparison and two ablations'],
        ['M10 final AC study','Real GPT-2','60 GPU + 12 CPU/CUDA runs; 54 matched shapes','Reliability, feedback diagnosis and backend portability'],
        ['M11 capacity frontier','Real GPT-2','216 runs: 552 completed / 312 rejected','Sampled quota/context capacity and quality cost'],
        ['Frozen submission validation','Version 1.0.0',f"{tests['tests_run']} tests pass",'Source snapshot, raw evidence and verified live demo']],
        'Archived manifests and validation files; stage counts are not independent-machine replications.')
    b.figure(plots['sampled_minima'],'Lowest tested complete serving quotas across prompt lengths. Lines connect sampled points only; they do not identify an exact continuous physical frontier.',source='Derived from results/capacity_frontier/analysis.json')

    b.chapter('2. Research motivation, paper inspiration and scope')
    b.add('''LLM serving competes for device capacity between model weights and growing attention history. Larger batches can amortize dense computation, but a device can run out of useful capacity before exhausting its arithmetic throughput. Quantization can reclaim weight memory; CPU attention can reclaim device KV memory. Both introduce coupled costs: host recovery storage, CPU bandwidth, device-link traffic, temporary copies and switching latency. A controller must evaluate those costs together.

MorphServe uses workload-aware runtime quantized layer swapping and KV-capacity resizing. The resizing repurposes freed device memory; it should not be described as quantizing or deleting the live KV history. NEO offloads selected requests' attention computation and KV state to the CPU, using load-aware scheduling and asymmetric CPU/GPU pipelining. Its CPU placement is more than parking KV bytes and repeatedly fetching the entire history back to the device. PrismServe adopts these mechanism ideas while implementing synchronous reference execution [1,2].''')
    b.table('Research mechanisms and this implementation',['Aspect','MorphServe inspiration','NEO inspiration','PrismServe evidence'],[
        ['Main control','Runtime layer precision','Request attention / KV placement','Joint plan Q (layers), C (request IDs)'],
        ['Capacity benefit','Freed device weight storage','Freed device KV capacity','Measured admitted workload region'],
        ['Host resource','Recovery / packed variants','Persistent host KV and attention','Host backups, copies and KV reserved jointly'],
        ['Execution engineering','Asynchronous state-preserving swaps','Asymmetric pipeline and load-aware scheduling','Synchronous reference transitions and attention'],
        ['Portability scope','Paper-specific evaluated engines','Paper-specific evaluated engines','Common profile contract; actual CPU/CUDA laptop'],
        ['What is not reproduced','Original optimized full system','Original optimized full system','No paper-speedup, vLLM-equivalence or vendor-portability claim']],
        'User-supplied MorphServe and NEO papers; docs/research_review.md')
    b.add('''The supplied proposal and midterm slides are planning context. The delivered scope was revised to fit the available hardware and independently available code: real GPT-2 serving plus an explicitly separate simulator. The proposal's larger-model, downstream-accuracy and universal-latency targets are not converted into measured accomplishments. A working GPT-2 implementation is not a physical Llama-8B deployment, and a resource-profile interface is not a universal optimized kernel.

The hypotheses are deliberately falsifiable. A common controller should respond to capacity/rate changes through its inputs; exact cached and restored paths should preserve the intended numerical invariants; adaptation should help in at least some constrained regimes; and adding complexity should justify its overhead. The studies retain counterexamples: FP32 often wins, reference quantization is slower, phase feedback does not beat simpler feedback in matched probes, and joint control does not expand the final frontier beyond quantization alone.''')
    b.figure(diagrams['architecture'],'End-to-end architecture: concurrent request handling, common control interface, one worker, owned request state and measured feedback.','Original architecture schematic','prismserve/{types,efficient_policy,live_engine,batching,batch_service}.py')

    b.chapter('3. System architecture, request lifecycle and true batching')
    b.add('''HTTP handlers submit requests and asynchronously await concurrent Futures. One background worker owns mutable global precision and per-request KV placement. This ownership prevents two clients from independently changing the same layer weights. Request IDs, prompt tokens, current prompt cursor, generated outputs, cache, timing history and completion Future remain request-specific.

At an iteration boundary, the engine observes live capacities, snapshots active/pending work, allocates a bounded prefill token budget, and evaluates admission prefixes and allowed execution plans. Existing active work advances rather than being silently evicted. New requests enter in FIFO prefix order. The worker applies any precision or placement changes, executes one shared batch, records output times and updates learned timing corrections. Finished requests release their owned KV.

The batch primitive flattens all current token chunks into one activation matrix. Each of GPT-2's 12 blocks performs its four dense projections once, giving 48 projection calls per iteration. Attention then operates over each request's own causal history and placement. The vocabulary head uses only the last current token of each request. This establishes shared dense execution, while retaining a Python-level ragged attention loop. It is not a fused paged-attention kernel.''')
    b.figure(diagrams['lifecycle'],'Request states and explicit unsuccessful outcomes. A rejection is different from worker failure, and neither is silently treated as a completed request.','Original flowchart','prismserve/live_engine.py')
    b.figure(diagrams['batching'],'Selective batching: dense work is shared while causal positions, histories and output ownership remain separate. Illustrative token counts are not measured data.','Original architecture schematic','prismserve/batching.py')
    b.table('Implementation map',['Component','Principal module','Responsibility'],[
        ['Capability and plan contracts','types.py','HardwareProfile, ModelSpec, WorkItem, Plan, Estimate, Decision'],
        ['Reference analytical path','estimator.py / policy.py / simulator.py','8B-shaped modeled times and restricted plan search'],
        ['Measured control','measured_policy.py / efficient_policy.py','Timing families, resource bounds, feedback and conservative gate'],
        ['Pretrained adapter','pretrained.py / quantization.py','Real GPT-2, packed INT4, exact masters, cached forward'],
        ['Actual shared batches','batching.py','Flattened dense tokens, ragged independent attention, last-token head'],
        ['Continuous worker','live_engine.py','Admission, chunking, transitions, request timestamps and failure diagnosis'],
        ['Concurrent API','batch_service.py','Async handlers, bounded queues, health and metrics'],
        ['Experiment and provenance','live_experiments.py / validation_study.py / frontier.py','Actual replay, audit targets, archived source and all outcomes']])
    b.add('''The API is non-streaming: clients receive a full response after generation completes. Engine TTFT measures internal first-token readiness from scheduled arrival, not the time at which the client receives its first streamed byte. Concurrent HTTP success and true shared batching are validated separately from a large active-request count.''')

    b.chapter('4. Precision, placement and memory accounting')
    b.sub('4.1 Grouped INT4 storage and exact weight restoration')
    b.add('''The portable quantizer uses symmetric per-group values in [-7, 7], with a group size of 64 and FP32 scales. For an all-zero group the scale is replaced by one to avoid division by zero. Two encoded values are stored in each byte; padding handles incomplete groups and odd packed lengths. Embeddings, normalization parameters, biases and tied vocabulary weights remain floating point. GPT-2's four projection matrices in a selected block are morphable.

A CPU FP32 master supports exact restoration of the original stored matrix. During INT4 execution, the reference implementation temporarily unpacks one matrix and performs floating-point matmul. Packed storage is genuine; accelerated native INT4 arithmetic is not claimed. Quantization can reduce resident model storage while increasing total host-plus-device accounting and runtime.''')
    b.equation(r's_g=\max_{i\in g}|w_i|/7,\qquad q_i=\mathrm{clip}(\mathrm{round}(w_i/s_g),-7,7)',
               'Per-group reference quantization; reconstruction uses the stored scale and integer value.')
    b.figure(diagrams['precision'],'Precision transition and restoration lifecycle, including host masters, staging and temporary unpack scratch.','Original mechanism diagram','prismserve/quantization.py and prismserve/pretrained.py')
    b.sub('4.2 Persistent CPU attention and cache movement')
    b.add('''Offloaded decode requests keep their accumulated KV on CPU and execute attention there. Current query and newly projected K/V travel from the primary device, and attention output returns for subsequent dense operations. The full historical KV does not make a round trip every steady-state token. An actual placement change copies the existing history and must reserve overlapping source/destination lifetimes. Prefill stays on the primary device. CPU-primary execution disallows the nonsensical action of offloading attention to itself.''')
    b.figure(diagrams['offload'],'Placement and link traffic in the reference CPU-attention path. Migration differs from steady-state offloaded decode.','Original data-flow diagram','prismserve/batching.py and prismserve/pretrained.py')
    b.sub('4.3 Future KV and transition-safe admission')
    b.equation(r'm_{KV}=2L H_{KV}d\,(b_{KV}/8)', 'KV bytes per cached token; the factor two counts keys and values, and KV heads support grouped-query model geometry.')
    b.table('Derived GPT-2 reservation geometry',['Quantity','Value','Interpretation'],[
        ['Layers / KV heads / head dimension','12 / 12 / 64','FP32 KV, 4 bytes per element'],
        ['KV bytes per token','73,728 bytes = 72 KiB','Derived geometry, not an allocator-peak observation'],
        ['128 prompt + 16 reserved outputs','10.125 MiB KV per request','Maximum declared context 144'],
        ['384 prompt + 16 reserved outputs','28.125 MiB KV per request','Maximum declared context 400'],
        ['768 prompt + 16 reserved outputs','55.125 MiB KV per request','Maximum declared context 784'],
        ['Four requests at maximum context 784','220.5 MiB future KV','Before weights, temporary copies, workspace and margins'],
        ['One morphable FP32 block matrix set','27 MiB','7,077,888 projection weights'],
        ['Its packed INT4 plus scales','3.796875 MiB','64-weight groups and FP32 scales'],
        ['Three-block ideal projection-weight saving','69.609375 MiB','Consistent with measured 486.70 to 417.09 MiB resident storage']])
    b.add('''Admission reserves prompt length plus configured maximum output length, rather than assuming oracle knowledge of early EOS. It checks both host and device pools and the maximum of steady and transition states. The measured workspace bound is monotone in admitted request count and reserved context, with fixed/per-request scratch fitted from calibration, reserved KV copies and safety headroom. Static buffers and a 64 MiB backend margin further reduce usable primary-device capacity. CPU-primary mode checks the combined host footprint.

Reservations intentionally differ from reported physical allocation. CUDA allocated bytes, allocator-reserved bytes, owned KV, retained masters and sampled free memory are different measurements. A logical quota is not a physical VRAM exhaustion event. Sampling does not provide isolation from unrelated processes: if capacity becomes unsafe, the engine stops and preserves diagnosis rather than forcing work to finish.''')
    b.figure(diagrams['memory'],'Separate memory pools and temporary lifetimes. The diagram is schematic and not to scale.','Original memory-accounting diagram','prismserve/estimator.py, measured_policy.py and efficient_policy.py')

    b.chapter('5. Planning algorithm, calibrated predictions and safety')
    b.add('''A plan specifies a global quantized-layer set Q and per-request CPU-attention set C. Candidate precision sets follow a validation-derived sensitivity order; CPU assignments use a bounded set of eligible decode prefixes, preferring shorter contexts. The planner evaluates memory, additive quality risk and a compatible measured timing family before comparing cost. Unknown timing families are infeasible in the measured path. This is a bounded heuristic search, not a global solution over every layer/request subset.

The engine and plan selector have distinct objectives. Within a candidate admitted batch, the measured selector combines predicted iteration time, transition cost amortized over a horizon of 16, and an additive quality penalty. Across admission prefixes, the worker ranks predicted time per serviced request with a TBT penalty. Meeting an iteration-time proxy does not guarantee a request's queue-inclusive TTFT or every inter-token gap.''')
    b.equation(r'J(Q,C)=\frac{\widehat T_{iter}+\lambda_t\widehat T_{change}/H}{T_{SLO}}+\lambda_q\frac{\widehat R(Q)}{R_{budget}}',
               'Conceptual execution-plan objective; the engine separately compares admission sizes and charges actual changes in full.')
    b.equation(r'\widehat R(Q)=\sum_{\ell\in Q}s_\ell,\qquad \widehat R(Q)\leq R_{budget}',
               'Additive validation-logit KL surrogate. This is not a downstream-task or perplexity guarantee.')
    b.figure(plots['layer_sensitivity'],'Validation-only one-block logit-KL sensitivity. Blocks 1, 7 and 9 form the selected three-block set.','Derived from measured validation data','results/pretrained/calibration.json; 128 validation input tokens')
    b.sub('5.1 Measured lookup and feedback')
    b.add('''Calibration covers actual batch sizes, current contexts, prompt/decode mixtures, precision and CPU-attention placement. The controller finds a compatible nearest shape in log feature space and conservatively scales upward for greater work. Features include batch count, total prefill tokens, maximum current context and CPU request count. Current lengths drive timing; future reservations drive memory. An EWMA ratio updates only the executed family after its duration is observed. New phase-specific keys separate prefill, mixed and decode; observations do not reveal the true counterfactual runtime of unchosen plans.

The efficient path compiles resource terms and timing families, bounds lookup/reservation caches, avoids repeated model-tensor scans, and skips no-op placement/precision applications. It can reuse a stable plan during three intervening iterations of a four-iteration refresh interval. New request IDs, a changed active set/phase or a failed resource check trigger search. Every reused plan still receives current capacity checks. This reduces search cost but can postpone reconsidering an already-seen pending request for up to three ticks.''')
    b.figure(diagrams['controller_flow'],'A control tick with plan reuse, fresh feasibility, search, execution and explicit failure paths.','Original algorithm flowchart','prismserve/live_engine.py and efficient_policy.py')
    b.sub('5.2 Conservative adaptation and observability')
    b.add('''The conservative gate compares uncertainty-adjusted predicted time per serviced request against a feasible FP32 reference, which may have a smaller batch. It requires a 10% minimum benefit after margins; transition amortization is included. If FP32 has no feasible reference, adaptation remains allowed. Uncertainty margins are engineering heuristics, not statistical confidence bounds. Cooldown can retain a current feasible action, while safety constraints take priority.

Failure snapshots preserve the last sampled profile, control state, active work and current-plan estimate before request state is cleared. Replay saves failed partial observations and propagates the error. Such an attempt is not accepted as a successful benchmark. These checks improve diagnosis; they do not make a shared laptop immune to external memory pressure.''')
    b.add('''```text
for each worker tick:
    sample capacities; build active work and FIFO candidate additions
    if stable reuse is eligible:
        re-evaluate the current plan with fresh resource limits
    otherwise, or if that plan no longer fits:
        check FP32 prefixes; search permitted precision/placement candidates
        reject plans exceeding host/device/transition or quality budgets
        compare admitted-batch cost and apply the conservative FP32 gate
    if no plan fits:
        reject an unadmitted request, or stop unsafe active work with diagnosis
    else:
        apply precision/cache movement; execute the actual shared batch
        record token readiness, update feedback, complete/release requests
```''')

    b.chapter('6. Experimental design, metrics and reproducibility')
    b.add('''All stages identify their evidence type. Actual timing uses tensor execution and synchronized CUDA or host wall-clock observations; the simulator advances virtual time from an analytical estimator. These quantities are not combined into a single speedup. The real pretrained studies use local pinned safetensors and the original GPT-2 BPE tokenizer. Model assets are not downloaded implicitly by generation paths.

The final GPU studies pair prompt IDs, arrivals and output limits across policies. Kernel warmup, device synchronization, explicit EOS disabling, per-request state isolation and complete queue drain avoid easy work-count confounds. Calibration precedes replay. GPU tests and other project model jobs do not run concurrently with timed studies. Policy order rotates and cell/scenario order is shuffled where specified. Read-only telemetry records AC state, clocks, thermal/power-limit flags, utilization, available RAM and process RSS.

Quality selection uses the validation split, while audits use a separate stored test prefix. Teacher-forced quality is a separate trajectory from greedy serving: input continuations and timing decisions can change both precision choices and cached history. Reference comparisons therefore publish target identity/count and precision exposure. Repeated evaluations across arms do not multiply the number of independent target positions.''')
    b.figure(diagrams['validation'],'Separation of calibration, greedy serving, quality auditing and preserved experiment provenance.','Original validation flowchart','prismserve/live_experiments.py, validation_study.py and frontier.py')
    b.equation(r'TTFT=t_1-a,\qquad TBT_i=t_i-t_{i-1},\qquad TPOT=(t_n-t_1)/(n-1)',
               'Token-readiness timing definitions; gaps and TPOT are undefined for a single-token completion.')
    b.equation(r'NLL=-\frac{1}{N}\sum_i\log p(y_i\mid x,y_{<i}),\qquad PPL=\exp(NLL)',
               'Held-out teacher-forced likelihood and perplexity, evaluated on identical target positions within each comparison.')
    b.table('Metric definitions and common interpretation errors',['Metric','Implemented definition','Interpretation'],[
        ['Output token/s','Completed output tokens / full trace-and-drain wall time','Includes admission, transition and scheduling costs in actual replay'],
        ['Request SLO failure','Rejections plus completed requests violating TTFT or any available TBT limit','Rejecting work never improves the apparent violation fraction'],
        ['Control overhead','Telemetry + planning/admission time / workload wall time','A fraction depends on execution speed as well as absolute search cost'],
        ['TTFT P95','Per-run percentile of completed request readiness times','Means of run P95 values are not pooled-request P95'],
        ['TBT P95','Per-run percentile of observed successive output gaps','Not end-to-end latency divided by output count'],
        ['Quality relative PPL','100 x expm1(NLL_variant - NLL_reference)','Corpus/protocol-specific; not task-accuracy percentage loss'],
        ['Capacity frontier','Lowest sampled quota completing all configured requests in every repeat','Neither an interpolated exact minimum nor a physical OOM limit']])
    env=data('results/environment.json')
    b.table('Recorded software and model environment',['Item','Recorded value'],[
        ['CPU/GPU environment',env['platform']],['Python',env['python'].split(' ')[0]],
        *[[name,version] for name,version in env['packages'].items()],
        ['Actual GPU','NVIDIA GeForce RTX 4050 Laptop GPU, about 6 GB VRAM'],
        ['Host memory','About 16 GB installed; free capacity varies substantially'],
        ['GPT-2 shape','12 layers, width 768, 12 attention/KV heads, head dimension 64, 1024 positions'],
        ['GPT-2 source revision','607a30d783dfa663caf39e06633721c8d4cfcd7e']], 'results/environment.json and pinned model manifest')
    b.add('''Memory units in this document are MiB = 2^20 bytes unless explicitly described otherwise. A driver-reported power/thermal reason flag is an observation, not proof that it caused a particular latency change. Bootstrap intervals over a few same-laptop repeats are exploratory and conditional; they do not establish independent-machine population confidence.''')

    b.chapter('7. Tiny-decoder training, correctness and storage checks')
    tiny=extract(final,'## 4. Measured tiny-model results','## 5. Real pretrained GPT-2 results')
    # Temporary source text is local to this report, not a mutation of the release.
    temp=HERE/'_tiny_source.md';temp.write_text(tiny.split('![Measured resident storage')[0],encoding='utf-8')
    b.import_md(temp,drop_title=False,asset_base=ROOT/'docs')
    b.figure(plots['tiny_training'],'Sampled training NLL for the actual 80-step tiny-decoder run. This demonstrates executable training, not general language capability.','Derived measured training curve','results/demo/real_backend.json')
    b.figure('results/demo/figures/measured_storage_latency.png','Actual tiny-decoder resident storage and mean reference latency. Packed storage does not imply accelerated arithmetic.')
    tinyq=data('results/demo/quality.json')
    b.add(f"The tiny model scores {tinyq['evaluation_tokens']} held-out byte targets. Fresh restored weights produce maximum logit error {tinyq['restore_max_abs_logit_error']:.1f}. A tiny negative PPL perturbation on authored text is not evidence of improved general model quality.")

    b.chapter('8. Actual pretrained GPT-2: parity, quality and execution')
    body=extract(final,'## 5. Real pretrained GPT-2 results','<!-- LIVE_RESULTS_BEGIN -->')
    # End at the next chapter if a marker layout changes.
    body=body.split('## 6. Real continuous-batching traffic experiments')[0]
    temp=HERE/'_pretrained_source.md';temp.write_text(body,encoding='utf-8')
    b.import_md(temp,drop_title=False,heading_offset=1,asset_base=ROOT/'docs')
    b.figure(plots['adaptive_placement'],'Real GPT-2 placement transitions under declared capacity changes. Measured application/migration cost is displayed separately from execution.','Derived measured execution trace','results/pretrained/adaptive.json')
    parity=data('results/pretrained/parity.json');bp=data('results/live/real_batch_parity.json')
    b.table('Independent numerical and batching checks',['Check','Maximum absolute error / observation','Declared tolerance'],[
        ['Original GPT-2 forward',f"{parity['hf_forward_max_abs_error']:.8f}",'0.002'],
        ['Cached decode vs full prefix',f"{parity['cached_decode_max_abs_error']:.8f}",'0.002'],
        ['CPU attention placement',f"{parity['cpu_attention_max_abs_error']:.8f}",'0.002'],
        ['Actual unequal-context batch',number(max(bp['prefill_max_abs_logit_errors']),8),'0.002'],
        ['Mixed CPU/device batch decode',number(max(bp['mixed_placement_decode_max_abs_logit_errors']),8),'0.002'],
        ['Dense projection calls per iteration',str(bp['projection_calls']),'48 expected across 12 blocks']])

    b.chapter('9. Analytical 8B-shaped study and offered-load sensitivity')
    b.add('**Evidence label: analytical simulation.** No physical 8B inference or original-paper engine reproduction is implied by this section.')
    sim=extract(final,'## 7. Analytical simulation results','## 8. Interpretation and threats to validity')
    temp=HERE/'_simulation_source.md';temp.write_text(sim,encoding='utf-8');b.import_md(temp,drop_title=False,asset_base=ROOT/'docs')
    b.figure('results/simulation/figures/slo_violation_rate.png','Analytical request-SLO violation rates across synthetic profiles and policies. These are virtual-service outcomes.','Analytical simulation')
    b.figure('results/load_sweep/failures_vs_load.png','Analytical offered-load sensitivity of request-SLO failure. One load-sweep seed limits uncertainty claims.','Analytical simulation')

    b.chapter('10. M8: real continuous batching and concurrent serving')
    b.add('''This stage changes the evidence from serialized local generation to actual scheduled-arrival continuous serving. Twelve requests use 384-token prompts, 24 output tokens, a two-second window and maximum batch eight. Shared dense computation, independent KV and chunked prefill execute in real wall time. Seventy-five calibrated shapes cover batch sizes 1/4/8, current contexts, mixtures, precision sets and CPU-attention families. The three scenarios use steady/bursty arrivals and 1400/700 MiB enforced quotas. Two seeds/orderings yield 42 policy runs; a separate 64-target audit runs for each policy.

The serialized control uses the same optimized batch primitive with maximum batch one. It is therefore a useful concurrency control, not a comparison with a different vocabulary-head implementation. Tables below report means over two runs; the individual repeats are retained in the appendices and source CSV. Large timing variation makes a mean an insufficient universal speedup claim.''')
    b.import_md('results/live/report.md')
    b.figure(plots['live_quality'],'Separate M8 teacher-forced policy-quality changes. Exposure and 64-target counts must accompany the small PPL differences.','Derived measured audit data','results/live/quality.json')

    b.chapter('11. M9: efficient control and gate/reuse ablations')
    b.add('''This stage compares fresh legacy full/joint controllers, efficient full/joint paths, joint without the conservative gate and joint without decision reuse. Workload geometry, quotas and saved calibration match M8. The comparison addresses whether controller work can be reduced; it does not assume that greater active concurrency or additional feedback improves throughput. Original calibration/resource formulas are retained in the compiled implementation and regression-tested against randomized plans.''')
    b.import_md('results/controller/comparison.md')
    b.figure('results/controller/figures/output_tokens_per_s.png','M9 automatic throughput view, retained alongside the focused comparison figure for complete chart coverage. Individual repeats are visible.')
    b.figure('results/controller/figures/ttft_p95_ms.png','M9 automatic P95 TTFT view. Lower planning cost does not by itself imply improved tail latency.')

    b.chapter('12. M10: controlled repeats, matched prediction and portability')
    b.add('''The primary M10 session uses fresh calibration, three warmup passes, five rounds, rotating policy positions and shuffled scenario order. The main comparison preserves the v0.4 serving policy while improving measurement controls. AC is checked before execution and at every recorded 500 ms host sample; clocks and ambient conditions remain observed rather than fixed. The four predictors in the matched test forecast identical tensor executions before observing duration, removing the different-policy-trajectory confound from MAE comparisons.

CPU/CUDA portability uses separate 33/75-shape timing calibration and the same serving/controller source. Its smaller workload is four 128-token prompts with eight outputs, three seeds and two policies. The CPU checks combined host allocation; CUDA checks separate memory pools. Equal quota values do not make these backends physically equivalent.''')
    b.import_md('results/validation_study_final/report.md')
    b.figure(plots['cpu_cuda'],'Measured backend throughput and P95 engine TTFT for the matched smaller CPU/CUDA workload. These are means of three repeats.','Derived measured portability results','results/validation_study_final/portability_runs/*.json')
    b.figure(diagrams['history'],'Controlled precision history explains why FP32 restoration can leave later quality effects through retained KV. The 0.881% observation belongs to the M10 controlled audit.','Original causal mechanism diagram','results/validation_study_final/quality_fixed/early_prefill_int4_restore.json')

    b.chapter('13. M11: capacity frontier and quality during actual adaptation')
    b.add('''The final study freezes a 6-quota x 3-prompt-length x 4-policy x 3-repeat grid before the full sweep. Inputs are paired across quotas and policies; maximum batch size is four and every request reserves 16 outputs. Unlike M10, tight quotas actively exercise quantization and CPU placement. The default FP32 starting state and transition workspace must fit: the study does not start from a specially prepacked low-memory checkpoint.

The first pilot missed a narrow transition-feasible region. A calibration-based, validation-side resource preflight motivated adding a 675 MiB quota before the main experiment. Both pilots and the estimate are retained. This protocol refinement is disclosed rather than described as an untouched initial grid. Quality cases are selected using completion and observed quantization, never test NLL, and keep the same output reservation so the capacity boundary is not shifted by the audit.''')
    b.import_md('results/capacity_frontier/report.md')
    b.figure(plots['frontier_outcomes'],'Complete requested-work accounting across all frontier cells. Rejections remain in the denominator and count as SLO failures.','Derived measured outcome counts','results/capacity_frontier/summary.csv')

    b.chapter('14. Synthesis: supported claims, negative results and limitations')
    b.table('Research conclusions and their evidence bounds',['Question','Observed finding','Supported conclusion / limit'],[
        ['Can one policy interface drive CPU and CUDA?','Same controller, separate calibration, 12 portability runs; 96/96 FP32 outputs match','Backend execution portability on one laptop; not physical multi-vendor validation'],
        ['Can precision enlarge feasible workloads?','768-token prompts complete at 675 MiB with quantization, FP32 rejects','A sampled warm-start capacity benefit under this reservation contract'],
        ['Does joint control beat quantization alone in capacity?','Both have the same sampled minima in M11','A joint capacity advantage over quantization alone is not established'],
        ['Does adaptation always improve speed?','Optimized FP32 has the highest M10 mean throughput in all scenarios','No universal throughput or SLO win'],
        ['Does controller optimization reduce overhead?','M10 optimized joint below 5% in 15/15 runs; worst 4.97%','Observed scheduling-cost reduction in the tested sessions'],
        ['Is phase feedback the most accurate predictor?','Matched MAE: legacy 4.55 ms, phase 5.54 ms','Fresh calibration matters; phase isolation did not establish better MAE'],
        ['Does restoring FP32 remove all error?','Early-prefill audit: +0.881% PPL after restoration','Stored weights can restore exactly while historical KV remains affected'],
        ['Is the additive risk budget a task-quality guarantee?','M11 joint frontier audit +2.752% PPL on 96 targets','No downstream-accuracy guarantee follows from validation KL'],
        ['Are simulated large-model gains physical results?','88 analytical main/load records on synthetic profiles','Sensitivity evidence only; no 8B hardware deployment claim']])
    b.add('''The study's internal controls are useful but finite. Synchronous Python attention, dequantize-and-matmul INT4, conservative full-context admission and a small pretrained model differ from optimized production serving. A fixed sensitivity order ignores layer interactions and prompt dependence. Actual quality audits have their own state trajectories and several protocols/target counts; their PPL values must not be ranked across incompatible contexts as a single leaderboard. One audit selected from a quantized greedy cell remains FP32, explicitly showing that greedy exposure does not certify audit exposure.

Clock and available-memory variation remain real confounders on a shared laptop. The mixed-power diagnostic is retained separately, and the incomplete AC attempt's exact original cause is unproven. A later same-trace reproduction completed with more free RAM; this association does not establish causality. New failure-only snapshots improve future diagnosis but do not retroactively supply the lost original plan state.

The next research extensions are optimized low-bit kernels, asynchronous placement/stream ownership, paged KV, larger model adapters, downstream task tests and independently calibrated physical machines. These are research directions, not unfinished requirements for the delivered course prototype. The version 1.0.0 result is a reproducible implementation and a measured capacity/quality tradeoff with explicit negative regimes.''')

    b.chapter('15. Live demonstration, reproduction and submission artifacts')
    b.add('''The one-command demonstration first labels and prints saved validated evidence. It then performs an actual matched capacity replay under the selected tight quota, followed by a temporary concurrent HTTP server check. The server is stopped automatically. Live demonstration timings are not substituted for the controlled experiment. After release freeze, new demo outputs are placed in timestamped session folders, preserving the released reference evidence.''')
    b.add('''```powershell
# Run from the project folder with the prepared local CUDA model
python scripts/demo_project.py

# Read saved validated evidence without model execution
python scripts/demo_project.py --evidence-only

# Verify the frozen accepted experiment and packaged submission
python scripts/check_capacity_frontier.py --check-current
python scripts/check_submission.py --package
```''')
    demo=data('results/final_demo/capacity_summary.json')
    b.table('Recorded live capacity demonstration',['Policy','Prompt / quota','Completed','Rejected','Max batch','Quantized iterations','CPU-attention iterations'],[
        [r['policy'],f"{r['prompt_tokens']} / {r['quota_mib']} MiB",r['completed'],r['rejected'],r['max_batch'],r['quantized_iterations'],r['cpu_attention_iterations']] for r in demo['runs']],
        'results/final_demo/capacity_summary.json; demonstration, not a controlled benchmark')
    b.add('''The frozen project ZIP contains code, configurations, tests, the original report/slides, raw result traces, source snapshots, the tiny checkpoint and model/data provenance. The 124M-parameter pretrained weights are deliberately prepared separately from their pinned public source. This detailed companion adds a PDF, self-contained HTML, figure atlas and reproducible report sources without altering that submission freeze.

For exact software setup and API calls, use the repository README. For the presentation and viva, `docs/demo_and_viva.md` includes an architecture walkthrough and answers on hardware agnosticism, transition peaks, CPU compute, quality scope and unsuccessful outcomes. Reproducing historical runs with newer code requires a new output folder; archived wheels/snapshots retain the exact implementations that generated earlier physical measurements.''')

    b.chapter('16. References and source attribution')
    b.add('''[1] Zhaoyuan Su, Zeyu Zhang, Tingfeng Lan, Zirui Wang, Haiying Shen, Juncheng Yang and Yue Cheng. *MorphServe: Efficient and Workload-Aware LLM Serving via Runtime Quantized Layer Swapping and KV Cache Resizing.* MLSys 2026. User-supplied 20-page paper. Mechanism discussion is based on its runtime layer swapping and KV-capacity management; this report does not present the paper's results as ours.

[2] Xuanlin Jiang, Yang Zhou, Shiyi Cao, Ion Stoica and Minlan Yu. *NEO: Saving GPU Memory Crisis with CPU Offloading for Online LLM Inference.* MLSys 2025. User-supplied 16-page paper. CPU attention/KV placement and asymmetric pipelining are the research inspiration. [Author repository](https://github.com/NEO-MLSys25/NEO).

[3] PyTorch contributors. [CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html) and [serialization semantics](https://docs.pytorch.org/docs/stable/notes/serialization.html). The recorded environment uses PyTorch 2.5.1+cu121; current online documentation is not substituted for the archived execution environment.

[4] GPT-2 model assets and documentation: [model card](https://huggingface.co/openai-community/gpt2) and [Transformers GPT-2 adapter documentation](https://huggingface.co/docs/transformers/model_doc/gpt2). Pinned checkpoint revision: `607a30d783dfa663caf39e06633721c8d4cfcd7e`. Exact model asset hashes are retained in the project source manifest.

[5] Stephen Merity et al. [WikiText dataset card](https://huggingface.co/datasets/Salesforce/wikitext), `wikitext-2-raw-v1`. Stored validation/test prefixes derive from the first 100 rows per split, with retained response payloads and hashes. Wikipedia-derived text attribution and CC BY-SA 3.0/GFDL information accompany the data manifest.

The supplied project proposal and midterm slides were used as planning context, not as instructions controlling execution. All architecture diagrams and added data visualizations in this companion are original renderings of the implemented code and archived records. Figures copied from the project's results are identified by their exact source paths; no research-paper figure is represented as an implementation measurement.''')

    appendices(b)
    return b,original_hashes


def appendices(b):
    b.chapter('Appendix A. Complete run-level numerical results')
    b.add('''The following tables retain one row per run in the main experiment suites: 63 analytical main runs, 25 analytical offered-load runs, 42 M8 serving runs, 36 M9 serving runs, 60 M10 GPU runs, 12 M10 backend-portability runs and 216 M11 frontier runs. These 454 rows include 88 analytical records and 366 actual serving records. They do not count quality arms or microbenchmark repetitions as additional independent serving trials. Full-field CSV copies accompany the report; tables select interpretable units and preserve all seeds/outcomes.''')
    suites=[('A.1 Analytical main study','results/simulation/summary.csv','simulation.csv'),
            ('A.2 Analytical offered-load study','results/load_sweep/summary.csv','load_sweep.csv'),
            ('A.3 M8 actual serving','results/live/summary.csv','m8_live.csv'),
            ('A.4 M9 actual serving','results/controller/summary.csv','m9_controller.csv')]
    total=0
    for title,path,name in suites:
        b.sub(title);rows=csv_rows(path);total+=len(rows);b.export_csv(name,rows)
        if 'simulation' in path:
            headers=['Profile','Policy','Seed','P95 TTFT ms','P95 TBT ms','Output token/s','SLO fail %']
            values=[[r['profile'],r['policy'],r['seed'],number(r['ttft_p95_ms']),number(r['tbt_p95_ms']),number(r['output_tokens_per_s']),number(100*float(r['slo_violation_rate']))] for r in rows]
        elif 'load_sweep' in path:
            headers=['Offered req/s','Policy','P95 TTFT ms','P95 TBT ms','Output token/s','SLO fail %']
            values=[[number(r['offered_requests_per_s']),r['policy'],number(r['ttft_p95_ms']),number(r['tbt_p95_ms']),number(r['output_tokens_per_s']),number(100*float(r['slo_violation_rate']))] for r in rows]
        else:
            headers=['Scenario','Policy','Seed','P95 TTFT ms','P95 TBT ms','Output token/s','Control %','SLO fail %']
            values=[[r['scenario'],r['policy'],r['seed'],number(r['ttft_p95_ms']),number(r['tbt_p95_ms']),number(r['output_tokens_per_s']),number(100*float(r['search_admission_wall_fraction'])),number(100*float(r['slo_failure_rate']))] for r in rows]
        b.table(title+' — every recorded run',headers,values,path)
    for title,folder,name in [('A.5 M10 final GPU repeats','gpu_runs','m10_gpu.csv'),('A.6 M10 backend portability','portability_runs','m10_portability.csv')]:
        b.sub(title)
        paths=sorted((ROOT/'results/validation_study_final'/folder).glob('*.json'))
        rows=[data(p)['summary'] for p in paths];total+=len(rows);b.export_csv(name,rows)
        headers=['Scenario / backend','Policy','Seed','P95 TTFT ms','P95 TBT ms','Output token/s','Control %','SLO fail %']
        values=[[r.get('scenario',r.get('device')),r['policy'],r['seed'],number(r['ttft_p95_ms']),number(r['tbt_p95_ms']),number(r['output_tokens_per_s']),number(100*r['search_admission_wall_fraction']),number(100*r['slo_failure_rate'])] for r in rows]
        b.table(title+' — every recorded run',headers,values,'results/validation_study_final/'+folder+'/*.json')
    b.sub('A.7 M11 quota/context grid — every repeat')
    rows=csv_rows('results/capacity_frontier/summary.csv');total+=len(rows);b.export_csv('m11_frontier.csv',rows)
    values=[]
    for r in rows:
        complete=int(r['completed'])==int(r['requests'])
        values.append([r['prompt_tokens'],r['device_budget_mib'],r['policy'],r['seed'],f"{r['completed']}/{r['rejected']}",number(r['output_tokens_per_s']) if complete else 'N/A',number(r['ttft_p95_ms']) if complete else 'N/A',number(100*float(r['slo_failure_rate']))])
    b.table('M11 all 216 repeats; C/R means completed/rejected',['Prompt','MiB quota','Policy','Seed','C/R','Token/s if complete','P95 TTFT ms','SLO fail %'],values,'results/capacity_frontier/summary.csv')
    b.add(f'Run-index consistency check: **{total} rows**. Report generation asserts that this equals 454. Each complete CSV retains fields omitted from the compact printed tables, including memory, exposure and timing metadata.')
    assert total==454
    b.run_rows=total

    b.chapter('Appendix B. Diagnostic sessions and functional-test charts')
    b.add('''These charts are included to make the figure collection complete, but they are not additional accepted benchmark evidence. The initial M10 session switched from battery to AC; its observed subgroup clock/throughput difference is a confound, not a controlled causal estimate. The incomplete AC attempt stopped when active work became infeasible. Its precise original cause remains unproven. Original source snapshots and STATUS files retain these distinctions.

Smoke/regression charts below validate that experiment pipelines, numerical paths and resource checks execute. They use smaller or diagnostic protocols and must not be pooled with the main benchmark tables. Preview images of report pages/slides are presentation checks and are not scientific figures.''')
    for name,caption in [
        ('results/validation_study/figures/repeated_throughput.png','Mixed-power diagnostic session: repeated throughput; excluded from the primary AC comparison.'),
        ('results/validation_study/figures/clock_throughput.png','Mixed-power diagnostic clock/throughput association; correlation does not establish causality.'),
        ('results/validation_study/figures/matched_prediction_mae.png','Matched prediction errors in the earlier mixed-power diagnostic session; use the final AC probe table for primary conclusions.'),
        ('results/live_smoke/figures/output_tokens_per_s.png','Functional actual-serving smoke chart; small-protocol diagnostic only.'),
        ('results/live_smoke/figures/ttft_p95_ms.png','Functional actual-serving smoke TTFT chart; not a production performance result.'),
        ('results/live_regression/figures/output_tokens_per_s.png','Serving regression chart retained for pipeline verification.'),
        ('results/live_regression/figures/ttft_p95_ms.png','Serving regression TTFT chart retained for pipeline verification.'),
        ('results/smoke/figures/ttft_p95_ms.png','Analytical smoke-test TTFT chart; neither the main simulation nor physical serving.'),
        ('results/smoke/figures/slo_violation_rate.png','Analytical smoke-test SLO chart; diagnostic only.'),
        ('results/smoke/figures/controller_timeline.png','Analytical smoke-test control timeline; diagnostic only.')]:
        b.figure(name,caption,'Diagnostic / functional test')
    b.chapter('Appendix C. Provenance, validation and reader checklist')
    for title,path in [('Final unit suite','results/validation.json'),('M10 final validation','results/validation_study_final/validation.json'),('M11 final validation','results/capacity_frontier/validation.json')]:
        record=data(path)
        b.table(title,['Field','Recorded value'],[[k,json.dumps(v) if isinstance(v,(dict,list)) else str(v)] for k,v in record.items() if k!='notes'],path)
    b.add('''The report bundle includes a figure registry, complete CSV copies, input hashes and report-generation sources. A separate high-resolution graph/diagram atlas puts each registered figure on its own portrait or landscape page for enlarged inspection. The Markdown report remains available as an editable source. The standalone HTML embeds all images and supports internal navigation offline.

Before presenting a number, identify: (1) actual or analytical execution, (2) model/checkpoint, (3) trace and quota, (4) completed and rejected work, (5) whether setup and drain are charged, (6) quality target/exposure protocol, and (7) physical session and uncertainty scope. The preserved frozen submission and its release checks remain the authority for the original implementation; this companion is a richer presentation of those archived observations.''')


def main():
    book,original_hashes=build()
    existing={p for p in (ROOT/'results').rglob('*.png') if 'preview' not in p.name and 'page_' not in p.name}
    missing=existing-book.used_plots
    if missing:raise AssertionError('Existing chart omitted: '+','.join(str(p) for p in missing))
    # A generated front matter index uses the registry after all chapters exist.
    indices=['## Figure index','']
    indices += [f"- [Figure {f['number']}. {f['caption']}](#figure-{f['number']})" for f in book.figures]
    indices += ['','## Table index','']
    indices += [f"- [Table {t['number']}. {t['caption']}](#table-{t['number']})" for t in book.tables]
    md='\n'.join(indices+['']+book.parts)
    source=HERE/'PrismServe_Detailed_Report.md';source.write_text(md,encoding='utf-8')
    cover='''<div class="cover"><p class="eyebrow">IIT GUWAHATI · DA4022 · HARDWARE AWARE DEEP LEARNING</p>
<h1>PRISMSERVE</h1><p class="subtitle">Hardware-Agnostic Control of LLM Serving<br />Comprehensive Technical Report</p>
<p class="cover-note">Architecture · Algorithms · Flowcharts<br />Measured Results · All Graphs · Complete Run Tables</p>
<p class="cover-note">Siddharth Khatod · Akash Iyer · Vamaxi Maheshwari<br />Kanika Bansal · Yashvardhan Kant</p>
<p>Companion to frozen submission <strong>v1.0.0</strong><br />18 November 2026</p>
<p class="small">Actual pretrained GPT-2 execution, CPU/CUDA validation and a separately labeled analytical 8B-shaped study. All diagrams are original implementation schematics. This companion regenerates presentation artifacts from archived evidence and preserves the original source/release package.</p></div>'''
    body,headings=prepare_html(md,cover)
    pdf=OUTPUT/'PrismServe_Detailed_Report.pdf';web=OUTPUT/'PrismServe_Detailed_Report.html'
    pdf_info=export_pdf(body,pdf,HERE);print('PDF',json.dumps(pdf_info),flush=True)
    export_html(body,web,HERE)
    atlas=OUTPUT/'PrismServe_Graph_and_Diagram_Atlas.pdf';atlas_pages=export_atlas(book.figures,atlas,HERE)
    for p,value in original_hashes.items():
        assert digest(p)==value,'Frozen artifact changed: '+str(p)
    with pymupdf_open(pdf) as d:
        assert len(d)>25 and all(p.get_text().strip() for p in d)
        combined=''.join(p.get_text() for p in d)
        for required in ('675','312','116','768','71.88','2.752'):
            assert required in combined,required
    manifest={'title':'PrismServe Comprehensive Technical Report','companion_to_version':'1.0.0',
        'created_utc':datetime.now(timezone.utc).isoformat(),'pdf':pdf_info,
        'figures':book.figures,'tables':book.tables,'equations':book.equations,
        'existing_result_charts_covered':len(existing),'run_level_table_rows':book.run_rows,'atlas_pages':atlas_pages,
        'input_sha256':{p.relative_to(ROOT).as_posix():digest(p) for p in sorted(INPUTS) if not p.is_relative_to(HERE)},
        'original_release_and_submission_unchanged':True,
        'protected_original_sha256':{p.relative_to(ROOT).as_posix():value for p,value in original_hashes.items()},
        'artifact_sha256':{p.name:digest(p) for p in (pdf,web,atlas)},
        'notes':['All numbers derive from archived measurements or explicitly derived model geometry.',
                 'Architecture/flowcharts are original schematics; diagnostic and analytical charts are labeled.',
                 'PDF layout, figure coverage, HTML links and frozen submission hashes were checked.']}
    (HERE/'report_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    from validate_report import main as validate_companion
    validate_companion()
    bundle=OUTPUT/'PrismServe_Detailed_Report_Package.zip'
    with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in (pdf,web,atlas):z.write(p,p.name)
        for p in sorted(HERE.rglob('*')):
            if (p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc' and 'previews' not in p.parts
                    and not p.name.startswith(('debug','_'))):
                z.write(p,'source/'+p.relative_to(HERE).as_posix())
    print(json.dumps({'pdf_pages':pdf_info['pages'],'figures':len(book.figures),'tables':len(book.tables),
        'equations':len(book.equations),'all_existing_graphs_included':len(existing),'run_rows':book.run_rows,
        'atlas_pages':atlas_pages,'bundle_mib':round(bundle.stat().st_size/2**20,2),'original_release_unchanged':True},indent=2),flush=True)


def pymupdf_open(path):
    import pymupdf
    return pymupdf.open(path)


if __name__=='__main__':
    main()
