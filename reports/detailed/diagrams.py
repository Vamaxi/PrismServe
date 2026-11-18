"""Original vector schematics for the detailed report; no runtime measurements."""
from pathlib import Path
import textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Polygon

COLORS={'blue':'#e3f0f6','green':'#e5f1e9','orange':'#fff0dc','red':'#f8e4e1','gray':'#eef0f3'}


def draw(destination,name,title,nodes,edges,note='',size=(13,7)):
    fig,ax=plt.subplots(figsize=size)
    ax.set_xlim(0,16);ax.set_ylim(0,10);ax.axis('off')
    lookup={}
    for item in nodes:
        key,x,y,w,h,label,*other=item
        color=COLORS.get(other[0] if other else 'blue','#e3f0f6')
        diamond=len(other)>1 and other[1]=='decision'
        patch=(Polygon([(x,y+h/2),(x+w/2,y+h),(x+w,y+h/2),(x+w/2,y)],closed=True,
                       facecolor=color,edgecolor='#355267',lw=1.1)
               if diamond else FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.045,rounding_size=.12',
                                              facecolor=color,edgecolor='#355267',lw=1.1))
        ax.add_patch(patch)
        ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=10.2,color='#183245',linespacing=1.3)
        lookup[key]=(x,y,w,h)
    for item in edges:
        start,end,*options=item
        label=options[0] if options else ''
        sx,sy,sw,sh=lookup[start];tx,ty,tw,th=lookup[end]
        a=(sx+sw/2,sy+sh/2);b=(tx+tw/2,ty+th/2)
        dx,dy=b[0]-a[0],b[1]-a[1]
        if abs(dx)/max(sw,tw)>abs(dy)/max(sh,th):
            a=(sx+sw if dx>0 else sx,sy+sh/2);b=(tx if dx>0 else tx+tw,ty+th/2)
        else:
            a=(sx+sw/2,sy+sh if dy>0 else sy);b=(tx+tw/2,ty if dy>0 else ty+th)
        curve=options[1] if len(options)>1 else 0
        arrow=FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,lw=1.1,color='#416075',
                              connectionstyle=f'arc3,rad={curve}',shrinkA=4,shrinkB=4,zorder=1)
        ax.add_patch(arrow)
        if label:
            ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.12,label,ha='center',va='center',fontsize=7.6,
                    bbox={'facecolor':'white','edgecolor':'none','pad':1.6},color='#304e62')
    fig.suptitle(title,fontsize=14,fontweight='bold',color='#123b50',y=.98)
    if note:fig.text(.5,.025,note,ha='center',va='bottom',fontsize=8,color='#466174')
    fig.tight_layout(rect=(0,.055,1,.95))
    for suffix in ('png','svg'):fig.savefig(Path(destination)/f'{name}.{suffix}',dpi=210,bbox_inches='tight')
    plt.close(fig)
    return Path(destination)/f'{name}.png'


def make_diagrams(destination):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    output={}
    def add(name,title,nodes,edges,note='',size=(13,7)):
        output[name]=draw(destination,name,title,nodes,edges,note,size)
    add('architecture','PrismServe: execution and control architecture',[
        ('clients',.3,7.7,3,1.25,'Concurrent HTTP clients\n/v1/completions','gray'),
        ('api',4.2,7.7,3,1.25,'Async API handlers\nsubmit and await Futures'),
        ('queue',8.2,7.7,3,1.25,'FIFO pending queue\nowned request state'),
        ('results',12.3,7.7,3.2,1.25,'Completion response\nusage and engine timing','green'),
        ('profile',.3,4.5,3.1,1.6,'Capability contract\ncapacity / rates / device\nmeasured calibration'),
        ('planner',4.3,4.5,6.5,1.6,'Joint planner and admission\nQ: quantized layer set | C: CPU request IDs\nresource checks + quality proxy + fallback','green'),
        ('worker',12.2,4.5,3.2,1.6,'One model worker\napply plan at tick boundary'),
        ('telemetry',.3,1.2,3.1,1.6,'Live capacity samples\nand actual timing feedback','orange'),
        ('batch',4.4,1.2,6.3,1.6,'Shared transformer projections\nragged attention / per-request KV\nFP32 or packed-INT4 reference path'),
        ('hardware',12.2,1.2,3.2,1.6,'Primary device + host\nactual tensors / transfers','gray')],
        [('clients','api'),('api','queue'),('queue','planner'),('profile','planner'),('planner','worker'),
         ('worker','batch'),('batch','hardware'),('batch','telemetry'),('telemetry','planner'),('worker','results')],
        'Original schematic. The same policy consumes CPU/CUDA profiles; numerical kernels and performance remain device-specific.')
    add('controller_flow','A serving tick: feasibility before execution',[
        ('start',5.8,8.6,4.4,.9,'Sample current device and host limits'),
        ('reuse',5.7,6.9,4.6,1.1,'Stable structure and reuse interval?','green','decision'),
        ('cached',.4,5,4.5,1.15,'Evaluate existing plan\nfresh reservation / quality check'),
        ('search',10.8,5,4.7,1.15,'Search FIFO admission prefixes\nand allowed precision / CPU families'),
        ('fit',5.7,3.2,4.6,1.25,'Any feasible execution plan?','green','decision'),
        ('execute',5.4,1.25,5.3,1.15,'Apply changes; execute shared batch\nobserve duration; emit/update requests','green'),
        ('empty',11.8,1.2,3.8,1.3,'No active work: reject\nActive work: stop safely\nand preserve failure state','red')],
        [('start','reuse'),('reuse','cached','yes'),('reuse','search','no'),('cached','fit','if feasible'),
         ('cached','search','if infeasible'),('search','fit'),
         ('fit','execute','yes'),('fit','empty','no')],
        'A cached plan is never cached feasibility. Precision changes are gated against feasible FP32; all transitions are charged.')
    add('batching','Selective continuous batching with independent KV ownership',[
        ('r1',.3,7.3,3.7,1.2,'Request A: prefill chunk\nillustrative 4 current tokens'),
        ('r2',6.1,7.3,3.7,1.2,'Request B: decode\nillustrative 1 current token'),
        ('r3',11.8,7.3,3.7,1.2,'Request C: prefill chunk\nillustrative 2 current tokens'),
        ('flat',3.2,5,9.6,1.25,'Flatten current tokens: 4 + 1 + 2 = 7\nEach block performs its four dense projections once','green'),
        ('a',.3,2.7,4.3,1.2,'Attention A\nA query + A history only'),
        ('b',5.8,2.7,4.3,1.2,'Attention B\nB query + B history only'),
        ('c',11.2,2.7,4.3,1.2,'Attention C\nC query + C history only'),
        ('head',3.2,.45,9.6,1.2,'Last-token vocabulary head per request\nIndependent cache / causal position / output timestamp','green')],
        [('r1','flat'),('r2','flat'),('r3','flat'),('flat','a'),('flat','b'),('flat','c'),('a','head'),('b','head'),('c','head')],
        'Illustrative token counts, not a measured batch. GPT-2: 12 blocks x 4 projections = 48 projection calls per iteration.')
    add('offload','CPU attention placement: what crosses the device link',[
        ('input',.5,7.6,5.5,1.25,'Primary device: hidden states\nand dense Q/K/V projections'),
        ('local',.5,4.8,5.5,1.25,'Device-attention requests\nKV remains on primary device'),
        ('host',9.7,4.8,5.6,1.55,'CPU-attention decode requests\npersistent host KV; attention on CPU','orange'),
        ('merge',.5,1.5,5.5,1.55,'Primary device: attention output\nprojection / MLP / next-token head'),
        ('migration',9.7,1.5,5.6,1.55,'Placement change: migrate existing KV\nreserve source + destination lifetime','red')],
        [('input','local'),('input','host','current Q + new K/V'),('host','merge','attention output'),
         ('local','merge'),('migration','host','on transition')],
        'History stays on the host during steady offloaded decode. This implementation synchronizes; it does not implement NEO overlap.')
    add('precision','Reversible precision and transition accounting',[
        ('fp',.3,6.7,3,1.7,'Resident FP32\nprojection matrices'),
        ('reserve',4.3,6.7,3.2,1.7,'Reserve transition peak\ncreate CPU master\npack groups + scales','orange'),
        ('q',8.5,6.7,3.1,1.7,'Packed INT4\ntwo nibbles per byte\nFP32 group scales','green'),
        ('restore',12.5,6.7,3.1,1.7,'Restore FP32\nfrom exact master\ncheck staging again'),
        ('compute',7.8,3.5,4.4,1.6,'Reference execution\nunpack to floating point\nthen dense matmul','orange'),
        ('kv',2.1,.9,11.7,1.5,'Existing request KV is retained across precision changes\nExact weight restoration does not erase approximation already written into KV','red')],
        [('fp','reserve'),('reserve','q'),('q','restore'),('q','compute'),('compute','kv'),('restore','kv')],
        'Embeddings, normalization, biases and tied vocabulary weights remain floating point. Packed storage is not a fused INT4 arithmetic claim.')
    add('memory','Memory-reservation model: distinct pools and temporary lifetimes',[
        ('gpu',.5,7.8,6.6,1,'Primary-device budget','blue'),
        ('cpu',8.8,7.8,6.6,1,'Host-memory budget','orange'),
        ('gw',.5,5.8,3.0,1.2,'Resident weights\nFP32 / selected INT4'),
        ('gkv',4.1,5.8,3.0,1.2,'Future device KV\nprompt + max outputs'),
        ('cw',8.8,5.8,3.0,1.2,'Recovery masters\nand model variants','orange'),
        ('ckv',12.4,5.8,3.0,1.2,'Persistent host KV\n+ concatenation copies','orange'),
        ('gs',.5,3.5,6.6,1.35,'Workspace + static buffers + backend margin\n+ precision transition / cache copy peak'),
        ('cs',8.8,3.5,6.6,1.35,'Existing host KV + destination during migration\n+ explicit temporary copy reservations','orange'),
        ('check',2,1,12,1.25,'Admit only when both pools fit the sampled limits\nCPU-primary mode checks the combined host footprint','green')],
        [('gpu','gw'),('gpu','gkv'),('cpu','cw'),('cpu','ckv'),('gw','gs'),('gkv','gs'),('cw','cs'),('ckv','cs'),('gs','check'),('cs','check')],
        'Schematic, not to scale. Logical quotas and measured allocator peaks are reported separately; other processes can change available capacity.')
    add('validation','Data separation and experiment provenance',[
        ('validation',.4,7.4,4.3,1.45,'WikiText-2 validation prefix\nlayer-sensitivity calibration\nno test-NLL selection'),
        ('profile',5.8,7.4,4.3,1.45,'Hardware / kernel probes\nactual batch timings\nseparate CPU/CUDA calibration'),
        ('test',11.2,7.4,4.3,1.45,'Held-out test prefix\nfixed request inputs\ndisjoint audit target windows','green'),
        ('freeze',3.5,4.6,9,1.5,'Freeze source + configuration + trace protocol\nrecord model/data hashes; instrument power, clocks and RAM'),
        ('greedy',.5,1.5,4.3,1.7,'Greedy serving replay\nwall-clock latency / throughput\ncompletion and rejections'),
        ('audit',5.8,1.5,4.3,1.7,'Separate teacher forcing\nactual precision/KV history\nNLL / PPL / exposure','green'),
        ('report',11.1,1.5,4.3,1.7,'Validation + report\nall baselines / failures\nsource snapshot and raw records')],
        [('validation','freeze'),('profile','freeze'),('test','freeze'),('freeze','greedy'),('freeze','audit'),('freeze','report'),('audit','report')],
        'Analytical 8B-shaped simulations are a separate evidence stream. They are never labeled as physical GPT-2 or 8B measurements.')
    add('history','Why restoring FP32 can leave a quantization effect',[
        ('fa',.4,7,4,1.35,'Reference: FP32\nfirst 64 prompt tokens'),
        ('fb',6,7,4,1.35,'FP32\nremaining prompt'),
        ('fc',11.5,7,4,1.35,'FP32 output likelihood\nreference KV history','green'),
        ('qa',.4,3.6,4,1.35,'Early quantized prefill\nfirst 64 prompt tokens','orange'),
        ('qb',6,3.6,4,1.35,'Restore FP32 weights\nretain existing KV'),
        ('qc',11.5,3.6,4,1.35,'FP32 output likelihood\nperturbed historical KV','red'),
        ('result',3,.7,10,1.25,'M10 controlled audit: +0.881% PPL on the same 2048 targets\nwith zero INT4 output exposure','green')],
        [('fa','fb'),('fb','fc'),('qa','qb'),('qb','qc'),('fc','result'),('qc','result')],
        'Only the early precision history differs in this controlled schedule. This finding is subset-specific and not a task-accuracy guarantee.')
    add('lifecycle','Request lifecycle and explicit unsuccessful outcomes',[
        ('request',.5,7.3,3.2,1.2,'Validate and enqueue'),('admit',5.3,7.3,4.7,1.2,'Memory / quality admission','green'),
        ('reject',12,7.3,3.4,1.2,'Reject explicitly\nno TTFT observation','red'),
        ('prefill',5.3,4.7,4.7,1.3,'Chunked prompt processing\nfirst output at final chunk'),
        ('decode',5.3,2.1,4.7,1.3,'Cached decode steps\nrecord each token time'),
        ('done',.5,2.1,3.2,1.3,'Completion\nrelease owned KV','green'),
        ('failed',12,2.1,3.4,1.3,'Capacity becomes unsafe\nstop + preserve diagnosis','red')],
        [('request','admit'),('admit','reject','no plan'),('admit','prefill','fits'),('prefill','decode'),('decode','done','EOS / limit'),('decode','failed','guard failure')],
        'Rejections are counted as SLO failures. The non-streaming API returns the full completion; engine TTFT is internal token readiness.')
    return output
