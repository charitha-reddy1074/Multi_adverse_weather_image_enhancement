"""Read-only project evidence extraction. Writes only beneath PPT_EVIDENCE."""
from pathlib import Path
import sys, os, json, csv, re, hashlib, shutil, textwrap, zipfile, ast
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image, ImageDraw, ImageFont
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'PPT_EVIDENCE'
sys.path.insert(0,str(ROOT))
os.environ['PYTHONDONTWRITEBYTECODE']='1'
NAVY='#16324F'; TEAL='#168B91'; GOLD='#DB9639'; GRAY='#5A6876'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'axes.spines.top':False,'axes.spines.right':False,'text.color':NAVY,'axes.labelcolor':NAVY,'savefig.dpi':160})
for d in ['diagrams','code_screenshots','datasets','training','results','restoration_samples','comparison','loss','sources']:(OUT/d).mkdir(parents=True,exist_ok=True)
ASSETS=[]
def read(p):return (ROOT/p).read_text(encoding='utf-8-sig',errors='replace')
def js(p):return json.loads(read(p))
def write(p,s): (OUT/p).write_text(s,encoding='utf8')
def csvwrite(p,rows,fields=None):
    rows=list(rows)
    with (OUT/p).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)
def hashfile(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def asset(p,slide,source,limit='',kind=None):
    ASSETS[:]=[r for r in ASSETS if r['asset_filename']!=p]
    ASSETS.append(dict(asset_filename=p,slide_number=str(slide),asset_type=kind or Path(p).suffix[1:],original_source_path=source,verified='yes',known_limitations=limit))
def save(fig,p,slide,source,limit='',svg=False):
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    for txt in fig.texts:
        bounds=txt.get_window_extent(renderer)
        max_width=fig.bbox.width*.91
        if bounds.width>max_width:txt.set_fontsize(txt.get_fontsize()*max_width/bounds.width)
    fig.savefig(OUT/p,facecolor='white');plt.close(fig);asset(p,slide,source,limit)
    if svg:
        # Re-save via the still valid figure object.
        sp=str(Path(p).with_suffix('.svg'));fig.savefig(OUT/sp,facecolor='white');asset(sp,slide,source,limit)
def canvas(title,subtitle='',size=(16,9)):
    fig=plt.figure(figsize=size);fig.text(.045,.94,title,fontsize=24,weight='bold',color=NAVY)
    if subtitle:fig.text(.045,.895,subtitle,fontsize=11,color=GRAY)
    return fig
def panel(title,lines,p,slide,source,limit=''):
    fig=canvas(title);y=.83
    for line in lines:
        for t in textwrap.wrap(str(line),105):fig.text(.055,y,t,fontsize=16);y-=.047
        y-=.018
    fig.text(.045,.035,'Evidence: '+source,fontsize=9,color=GRAY,wrap=True)
    save(fig,p,slide,source,limit)
def tablepic(title,headers,rows,p,slide,source,subtitle='',widths=None):
    fig=canvas(title,subtitle);ax=fig.add_axes([.045,.14,.91,.69]);ax.axis('off')
    wrapped=[[textwrap.fill(str(v),32) for v in row] for row in rows]
    t=ax.table(cellText=wrapped,colLabels=headers,loc='center',cellLoc='left',colWidths=widths)
    t.auto_set_font_size(False);t.set_fontsize(12);t.scale(1,3.0 if len(rows)<9 else 2.35)
    for (r,c),cell in t.get_celld().items():
        cell.set_edgecolor('#D9E3EA')
        if r==0:cell.set_facecolor(NAVY);cell.set_text_props(color='white',weight='bold')
        else:cell.set_facecolor('#EFF7F7' if r%2 else 'white')
    fig.text(.045,.035,'Source: '+source,fontsize=9,color=GRAY)
    save(fig,p,slide,source,subtitle)
def diagram(title,subtitle=''):
    f=canvas(title,subtitle);a=f.add_axes([.035,.11,.93,.74]);a.set(xlim=(0,16),ylim=(0,8.4));a.axis('off');return f,a
def box(a,x,y,w,h,t,color='#E8F3F5',fs=12):
    a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.06,rounding_size=0.12',facecolor=color,edgecolor=TEAL,lw=1.2));a.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=fs,color=NAVY)
def arrow(a,p,q,color=NAVY,style='-',curve=0):
    a.add_patch(FancyArrowPatch(p,q,arrowstyle='-|>',mutation_scale=13,lw=1.8,color=color,linestyle=style,connectionstyle=f'arc3,rad={curve}'))
def flow(title,steps,p,slide,source,note=''):
    f,a=diagram(title,note);n=len(steps)
    for i,s in enumerate(steps):
        row=i//4;col=i%4;col=col if row%2==0 else 3-col;x=.2+col*4;y=5.3-row*3
        box(a,x,y,3.3,1.5,str(i+1)+'. '+s,fs=13)
        if i%4:
            arrow(a,(x-.65,y+.75),(x-.08,y+.75)) if row%2==0 else arrow(a,(x+3.95,y+.75),(x+3.38,y+.75))
        if i==4:arrow(a,(13.85,5.22),(13.85,3.88))
    save(f,p,slide,source,note)
BASE='outputs/four_weather_test_epoch20_20261109'
summary=js(BASE+'/summary.json');groups=summary['groups'];prov=summary['provenance'];overall=groups['overall']
duration=js(BASE+'/duration_summary.json')
checkpoint=ROOT/prov['checkpoint']
import torch
torch.set_num_threads(4)
state=torch.load(checkpoint,map_location='cpu',weights_only=True)
from models.airformer import AIRFormer
model=AIRFormer(**state['model_config']);model.load_state_dict(state['model']);model.eval()
params=sum(p.numel() for p in model.parameters() if p.requires_grad)
assert params==612435 and state['epoch']==20
cfg=state['config'];model_cfg=state['model_config']
hash_checks={}
sig_paths={'checkpoint':checkpoint,'test_manifest':ROOT/cfg['test_manifest'],'train_manifest':ROOT/cfg['train_manifest'],'evaluator':ROOT/'scripts/test_restoration.py','attention':ROOT/'models/attention.py','inference':ROOT/'utils/inference.py'}
for key,path in sig_paths.items():hash_checks[key]=dict(path=str(path),expected=prov['signature'][key],actual=hashfile(path));hash_checks[key]['matches']=hash_checks[key]['expected']==hash_checks[key]['actual']
assert all(v['matches'] for k,v in hash_checks.items() if k!='evaluator'),hash_checks
hash_checks['evaluator']['investigation']='Current evaluator differs from recorded evaluation hash. docs/LOG.md 2026-11-09 records later report routing to docs/reports. Original evaluator bytes not available as a verified matching snapshot. Numeric CSV reconciliation and twelve predictions provide separate checks; do not claim all source hashes match.'
with (ROOT/BASE/'per_image_metrics.csv').open(encoding='utf8',newline='') as f:rows=list(csv.DictReader(f))
assert len(rows)==14695
WEATHER=['rain','raindrop','snow','haze']
manifests={}
for split in ['train','test']:
    with (ROOT/cfg[split+'_manifest']).open(encoding='utf8',newline='') as f:manifests[split]=list(csv.DictReader(f))
counts={s:Counter(r['weather_type'] for r in rr) for s,rr in manifests.items()}
assert len(manifests['train'])==58766 and len(manifests['test'])==14695
metric_checks=[]
for w in ['overall']+WEATHER:
    rr=rows if w=='overall' else [r for r in rows if r['weather']==w]
    assert len(rr)==groups[w]['images']
    for k in ['input_psnr','restored_psnr','input_ssim','restored_ssim','input_mae','restored_mae','input_mse','restored_mse','input_rmse','restored_rmse']:
        val=float(np.mean([float(r[k]) for r in rr]));err=abs(val-groups[w][k]);assert err<1e-8,(w,k,err)
        metric_checks.append(dict(weather=w,metric=k,csv_mean=val,summary_value=groups[w][k],absolute_difference=err))
csvwrite('sources/metric_reconciliation.csv',metric_checks)
write('sources/checkpoint_verification.json',json.dumps(dict(epoch=state['epoch'],parameters=params,model_config=model_cfg,training_config=cfg,hash_checks=hash_checks),indent=2))
for name in ['summary.json','provenance.json','status.json','duration_summary.json','preflight.json','final_test.json']:shutil.copy2(ROOT/BASE/name,OUT/'sources'/name)
shutil.copy2(ROOT/'docs/architecture/trained_model_trace.json',OUT/'sources/trained_model_trace.json')
shutil.copy2(ROOT/'docs/architecture/figures/provenance.json',OUT/'sources/paper_figure_provenance.json')
for name in ['figure4_airformer.png','figure5_queried_attention.png','equations11_13.png','table13_efficiency.png']:
    shutil.copy2(ROOT/'docs/architecture/figures'/name,OUT/'comparison'/name)
    asset('comparison/'+name,4 if name.startswith('figure') else 11 if name.startswith('equations') else 14,'docs/architecture/figures/'+name,'Unmodified supplied-paper excerpt; Gao et al., IEEE TCSVT 2024, DOI 10.1109/TCSVT.2023.3299324. Licensed source; public redistribution permission not established.')
print('Checkpoint and five provenance hashes match; evaluator hash drift documented. All summary metrics verified.',flush=True)

# Four-stage implementation diagram. Shapes are per image; B omitted.
f,a=diagram('Actually trained AIRFormer-inspired model','128 x 128 example; C x H x W. Navy: spatial features. Teal dashed: prior guidance. Gold: encoder skips.')
box(a,.1,6.2,2,1.1,'RGB input\n3 x 128 x 128')
box(a,2.7,6.2,2.5,1.1,'Patch embed\n16 x 128 x 128')
arrow(a,(2.15,6.75),(2.65,6.75))
xs=[.2,4.2,8.2,12.2];sh=['16 x 128 x 128','32 x 64 x 64','64 x 32 x 32','128 x 16 x 16']
arrow(a,(3.95,6.15),(1.7,5.4))
for i,x in enumerate(xs):
    box(a,x,4.15,3.1,1.2,f'Encoder {i}: 1 block\n{sh[i]}')
    if i<3:
        arrow(a,(x+3.17,4.75),(x+3.92,4.75));a.text(x+3.53,5.05,'/2',ha='center',fontsize=11)
        box(a,x,2.15,3.1,1.15,f'SP{i}: Haar + fusion\n{sh[i+1]}',color='#D6F0EB')
        arrow(a,(x+1.55,3.35),(x+1.55,4.1),TEAL,'--') if False else None
        arrow(a,(x+1.55,3.35),(x+4.05,4.38),TEAL,'--')
        if i>0:arrow(a,(x+1.55,4.1),(x+1.55,3.37),TEAL,'--')
        if i<2:arrow(a,(x+3.18,2.72),(x+3.92,2.72),TEAL,'--')
a.text(.25,1.6,'SP0 receives zero feature + zero prior; SP1/SP2 receive preceding enhanced feature + preceding prior.',fontsize=11,color=TEAL)
for i,x in enumerate(xs[:3]):
    box(a,x,-.1,3.1,1.2,f'Decoder {2-i}: up + skip\nC -> S -> W; {sh[i]}',color='#FFF0D9',fs=11)
    a.plot([x+2.8,x+2.8,x+3.52,x+3.52,x+2.8],[4.1,3.72,3.72,1.45,1.45],color=GOLD,lw=1.8)
    arrow(a,(x+2.8,1.45),(x+2.8,1.18),GOLD)
    if i<2:arrow(a,(x+4.12,.5),(x+3.18,.5))
arrow(a,(13.75,4.1),(10.1,1.18),NAVY,curve=-.2)
box(a,12.2,-.1,3.2,1.2,'RGB head + residual\nCrop + clamp -> restored RGB\n3 x 128 x 128',fs=11)
a.plot([1.1,1.1,15.8,15.8],[7.38,7.9,7.9,1.55],color=GRAY,lw=1.2,linestyle=':')
arrow(a,(15.8,1.55),(15.4,.65),GRAY,':')
a.text(8,7.55,'Input image residual',fontsize=10,color=GRAY,ha='center')
arrow(a,(.25,-.15),(13.6,-.15),NAVY,curve=.2) if False else None
# Output flow is routed below the three decoder stages, preserving decoder direction.
a.set_ylim(-1.9,8.4)
arrow(a,(1.75,-.15),(13.5,-.15),NAVY,curve=.14)
f.text(.045,.04,'Source: checkpoint model_config, models/*.py, and docs/architecture/trained_model_trace.json. Chosen reconstruction, query and prior internals differ from paper.',fontsize=9,color=GRAY)
save(f,'diagrams/actual_airformer_architecture.png',5,'models/*.py; final checkpoint; docs/architecture/trained_model_trace.json','Shapes for 128 x 128 input; batch omitted. Architecture interpretation, not author equivalence.',svg=True)
f,a=diagram('Supplementary Prior: actual internal processing','C x H x W input. First module receives zeros. No IDWT is used inside this prior.')
items=[(.2,5.6,3,1.3,'Feature F\nC x H x W'),(4.1,5.6,3,1.3,'Haar DWT\n4C x H/2 x W/2'),(8.1,5.6,3.3,1.3,'LL unchanged here\nLH/HL/HH: DW 3x3'),(12.1,5.6,3.3,1.3,'Pack + bilinear resize\n4C x H x W'),(.2,2.3,3.2,1.3,'Previous prior\n1x1 project -> C'),(4.3,2.3,3.2,1.3,'Concat F / bands / prev\n6C x H x W'),(8.5,2.3,3,1.3,'Fusion + residual\nrefinement -> C'),(12.3,2.3,3.1,1.3,'Stride-2 projection\n2C x H/2 x W/2')]
for x,y,w,h,t in items:box(a,x,y,w,h,t,fs=12)
for p,q in [((3.25,6.25),(4,6.25)),((7.15,6.25),(8,6.25)),((11.45,6.25),(12,6.25)),((13.75,5.55),(6.1,3.65)),((3.45,2.95),(4.2,2.95)),((7.55,2.95),(8.4,2.95)),((11.55,2.95),(12.2,2.95)),((1.7,5.5),(5,3.65))]:arrow(a,p,q,TEAL)
save(f,'diagrams/supplementary_prior.png',5,'models/supplementary_prior.py; models/encoder.py','LL unchanged only during detail filtering; fusion may modify low-frequency information.')
flow('Frequency-refined decoder: repeated at three skip levels',['Upsample to encoder skip\nConcatenate + 1x1 project','3x3 convolution + GELU\nC x H x W','Channel attention\nMix channels per head','Space attention\nFull queries, pooled K/V','Wavelet attention\nDWT K/V + IDWT branch','Project + residual + MLP\nC x H x W'],'diagrams/frequency_refined_decoder.png',5,'models/decoder.py; models/attention.py','Sequence is Channel -> Space -> Wavelet at widths 64, 32, 16; 2 heads; K/V cap 256.')
flow('Model lifecycle: completed experiment',['Local paired sources\nRain / drop / snow / haze','Verify identities + group\nCustom 80/20 split','Preprocess + augment\n128 x 128 crops','Train fixed 20 epochs\nAdamW + CUDA AMP','Save resumable checkpoint\nEpoch 20, 612,435 params','Held-out evaluation\n14,695 full-size pairs','Inference on RGB image\nPad -> restore -> crop','Document results + limits\nNo matched paper benchmark'],'diagrams/model_lifecycle.png',7,'datasets/splitting.py; airformer/data.py; scripts/train.py; scripts/test_restoration.py; utils/inference.py')

# Architecture-to-code excerpts from AST, preserving source text exactly.
def segment(path,name):
    src=read(path);tree=ast.parse(src)
    for n in ast.walk(tree):
        if isinstance(n,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name:
            return n.lineno,n.end_lineno,src.splitlines()[n.lineno-1:n.end_lineno]
    raise ValueError((path,name))
def code_image(path,start,end,dest,slide):
    from pygments import lex
    from pygments.lexers import PythonLexer
    from pygments.styles import get_style_by_name
    fontpath='C:/Windows/Fonts/consola.ttf';font=ImageFont.truetype(fontpath,20);head=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',21)
    lines=read(path).splitlines()[start-1:end];line_h=29;cw=font.getlength('M');width=max(1400,int(max(map(len,lines))*cw+140))
    im=Image.new('RGB',(width,100+line_h*len(lines)), '#F6F8FA');d=ImageDraw.Draw(im);d.text((25,20),f'Actual repository source: {path}  [lines {start}-{end}]',font=head,fill=NAVY)
    style=get_style_by_name('friendly')
    for i,line in enumerate(lines):
        y=72+i*line_h;d.text((16,y),str(start+i),font=font,fill='#738090');x=90
        for token,txt in lex(line,PythonLexer()):
            txt=txt.rstrip('\n');color=style.style_for_token(token).get('color') or '172B3D';d.text((x,y),txt,font=font,fill='#'+color);x+=font.getlength(txt)
    im.save(OUT/dest);asset(dest,slide,f'{path}:{start}-{end}','Rendered genuine source excerpt, not a terminal/editor screenshot.')
components=[
('Model initialization','models/airformer.py','AIRFormer','3 x H x W','3 x H x W','Builds four widths, three priors, skip decoder and RGB head. Widths/depths are experiment choices.'),
('Patch embedding','models/patch_embed.py','PatchEmbed','3 x H x W','16 x H x W','Dense 3x3 stride-1 projection with GroupNorm, not disjoint image patches.'),
('Encoder forward pass','models/encoder.py','HierarchicalEncoder','16 x H x W','four feature maps','Stride-2 convolution and prior addition before each next stage. First SP gets zeros.'),
('Supplementary Prior','models/supplementary_prior.py','SupplementaryPrior','C x H x W plus prior','2C x H/2 x W/2','Haar bands, detail-only filtering, 6C fusion, refinement and stride-2 alignment are assumed internals.'),
('Wavelet decomposition','models/supplementary_prior.py','dwt2','C x H x W','C x 4 x ceil(H/2) x ceil(W/2)','Fixed orthonormal Haar; odd boundaries replicated. Family not specified by paper.'),
('Channel attention','models/attention.py','QueriedAttention','C x H x W','C x H x W','Channel mode transposes to channel affinities; per-head matrix (C/2) x (C/2). Additive learned query is assumed.'),
('Spatial attention','models/attention.py','QueriedAttention','C x H x W','C x H x W','Spatial queries retain HW positions; K/V adaptively pooled to at most 16x16 in trained configuration.'),
('Wavelet attention','models/attention.py','QueriedAttention','C x H x W','C x H x W','Normalize before DWT. Project packed 4C to C at H/2,W/2; K/V from this. Separate IDWT produces C/4 spatial channels.'),
('Decoder forward pass','models/decoder.py','HierarchicalDecoder','128 x 16 x 16 + skips','16 x 128 x 128','Three bilinear upsample/concat/project stages; each applies Channel, Space, Wavelet.'),
('Reconstruction','models/airformer.py','forward','16 x H x W','3 x H x W','RGB 3x3 convolution, input residual, original crop and clamp. This tail is an adaptation.'),
('Multi-domain loss','losses/multidomain.py','MultiDomainConsistencyLoss','prediction + target: B x 3 x H x W','scalar + detached parts','Elementwise Smooth L1 mean plus .04 smoothed per-image DCT norm; differs from literal Eq.12.'),
('Training step','scripts/train.py','main','B=8, 3 x 128 x 128','loss, gradients and checkpoint','Autocast, GradScaler, gradient clipping, AdamW, cosine scheduler and state recovery.'),
('Evaluation/inference','scripts/test_restoration.py','predict','H x W x 3 RGB float','same-size RGB float','Uses _restore_tensor symmetric multiple-of-128 padding; FP32 full resolution, no tiling.')]
mapping='# Architecture to code\n\nShapes are NCHW with batch omitted unless stated. Dimensions below use the actual 128 x 128 training crop. Line ranges refer to the source hashes recorded in sources/source_inventory.csv.\n\n'
for title,path,name,inp,out,explain in components:
    st,en,lines=segment(path,name)
    # Use a concise genuine window per long symbol, retaining full symbol ranges separately.
    if title=='Model initialization':st=next(i for i,s in enumerate(read(path).splitlines(),1) if 'channels=tuple' in s);en=st+15
    elif title=='Training step':st=next(i for i,s in enumerate(read(path).splitlines(),1) if 'for step,(x, target' in s);en=st+25
    elif title=='Supplementary Prior':st=next(i for i,s in enumerate(read(path).splitlines(),1) if 'def forward(self, encoder_feature' in s);en=segment(path,'SupplementaryPrior')[1]
    elif title=='Reconstruction':st=next(i for i,s in enumerate(read(path).splitlines(),1) if 'restored=self.reconstruct' in s);en=st+6
    elif title in ['Channel attention','Spatial attention','Wavelet attention']:
        ls=read(path).splitlines();needle={'Channel attention':'if self.mode == "channel":','Spatial attention':'if self.mode != "channel" and self.spatial_kv_pooling','Wavelet attention':'if self.mode == "wavelet":'}[title]
        if title=='Wavelet attention':st=next(i for i,s in enumerate(ls,1) if 'coeffs, size = dwt2' in s)-1;en=st+5
        else:st=next(i for i,s in enumerate(ls,1) if needle in s);en=st+(10 if title=='Channel attention' else 6)
    else:en=min(en,st+31)
    snippet='\n'.join(f'{i}: {s}' for i,s in enumerate(read(path).splitlines()[st-1:en],st))
    mapping+=f'## {title}\n\n- Paper concept: {title}.\n- Implemented symbol: `{path}::{name}` (AST symbol lines {segment(path,name)[0]}-{segment(path,name)[1]}).\n- Input: {inp}. Output: {out}.\n- Behavior and assumptions: {explain}\n\n```python\n{snippet}\n```\n\n'
    if title in ['Supplementary Prior','Wavelet decomposition','Channel attention','Spatial attention','Wavelet attention','Multi-domain loss','Training step','Reconstruction']:
        code_image(path,st,en,'code_screenshots/'+title.lower().replace(' ','_').replace('/','_')+'.png',11 if title=='Multi-domain loss' else 9)
write('03_ARCHITECTURE_TO_CODE.md',mapping)

# Dataset plots and exact counts from actual manifests.
sources={'rain':('Rain14000','datasets/sources.csv; scripts/prepare_available_data.py'),'raindrop':('Raindrop','datasets/sources.csv; scripts/prepare_local_archives.py'),'snow':('Snow100K local training archive','datasets/sources.csv; scripts/prepare_local_archives.py'),'haze':('Unofficial RESIDE-ITS mirror val','docs/RESIDE_MIRROR_VERIFICATION.json')}
details=[dict(weather=w,dataset=sources[w][0],train_pairs=counts['train'][w],test_pairs=counts['test'][w],total_pairs=counts['train'][w]+counts['test'][w],source_evidence=sources[w][1],split='Custom grouped 80/20, no validation',limitation='Not official AIR40K membership; near duplicates not ruled out'+('; upstream original haze release unverified' if w=='haze' else '')) for w in WEATHER]
csvwrite('datasets/dataset_details.csv',details)
f=canvas('Dataset composition','Pair counts from the exact trained manifests; total 73,461 pairs');a=f.add_axes([.1,.17,.82,.65]);x=np.arange(4)
a.bar(x,[counts['train'][w] for w in WEATHER],color=NAVY,label='Train');a.bar(x,[counts['test'][w] for w in WEATHER],bottom=[counts['train'][w] for w in WEATHER],color=TEAL,label='Test');a.set_xticks(x,WEATHER);a.set_ylabel('Paired images');a.legend()
for i,w in enumerate(WEATHER):a.text(i,counts['train'][w]+counts['test'][w]+650,f"{counts['train'][w]+counts['test'][w]:,}",ha='center')
save(f,'datasets/dataset_distribution.png',8,'data/manifests/four_weather/train.csv; data/manifests/four_weather/test.csv')
f=canvas('Custom training and testing distribution','Scene-identity groups, identical clean pixels and identical input bytes stay together');a=f.add_axes([.15,.17,.7,.65]);bs=a.bar(['Training','Held-out testing'],[58766,14695],color=[NAVY,TEAL]);a.bar_label(bs,fmt='{:,.0f}');a.set_ylabel('Paired images');save(f,'datasets/train_test_split.png',8,'data/manifests/four_weather/*.csv; datasets/splitting.py')
flow('Dataset preparation and integrity checks',['Read local source archives\nDataset-specific name mapping','Explicit input-target pairs\nNever pair by sorted order','Group scene / clean pixels\nInput-byte identities','Exclude known official tests\nCustom 80/20 groups','Check files + image shapes\nDecode/hash audit records','Paired crops + flips + turns\nMixUp probability 0.5','Weather-balanced sampler\nReplacement during training','Full-resolution held-out test\nNo training augmentation'],'datasets/dataset_pipeline.png',8,'scripts/prepare_local_archives.py; datasets/pairing.py; datasets/splitting.py; airformer/data.py','Near-duplicate scenes and unknown upstream haze membership remain limitations.')

# Logged history, combining only the four retained resume sessions.
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
runs=['four_weather_20261104_fast','four_weather_20261105_resume','four_weather_resume_epoch11','four_weather_resume_epoch17']
history={}
for run in runs:
    ac=EventAccumulator(str(ROOT/'checkpoints'/run/'runs'));ac.Reload()
    losses={v.step:v for v in ac.Scalars('loss/train')};lrs={v.step:v for v in ac.Scalars('learning_rate')}
    for ep,v in losses.items():history[ep]=dict(epoch=ep,training_loss=v.value,logged_lr_after_epoch=lrs[ep].value,event_wall_time=v.wall_time,source='checkpoints/'+run+'/runs',run=run)
assert sorted(history)==list(range(1,21))
csvwrite('training/epoch_history.csv',[history[i] for i in sorted(history)])
epochs=sorted(history);losses=[history[i]['training_loss'] for i in epochs]
for key,title,ylabel,dest in [('training_loss','Recorded training loss','Mean training loss','training_loss_curve.png'),('logged_lr_after_epoch','Logged cosine learning rate','Learning rate after scheduler.step()','learning_rate_curve.png')]:
    f=canvas(title,'TensorBoard scalar events from the four retained sessions; no reconstructed values');a=f.add_axes([.09,.17,.85,.65]);vals=[history[i][key] for i in epochs];a.plot(epochs,vals,'o-',color=TEAL,lw=2);a.set(xlabel='Completed epoch',ylabel=ylabel,xticks=range(1,21));a.grid(alpha=.2)
    if key=='training_loss':a.annotate(f'First: {vals[0]:.6f}',(1,vals[0]),xytext=(2,vals[0]),fontsize=11);a.annotate(f'Last: {vals[-1]:.6f}',(20,vals[-1]),xytext=(13,vals[-1]+.07),fontsize=11)
    else:a.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    save(f,'training/'+dest,10,'checkpoints/{'+','.join(runs)+'}/runs; scripts/train.py')
panel('Final checkpoint training configuration',[f'{params:,} trainable parameters; widths 16 / 32 / 64 / 128; depths [1,1,1,1]','2 attention heads; spatial/wavelet K/V cap 256 tokens','128 x 128 paired crops; microbatch 8; accumulation 1; effective batch 8','AdamW: LR 2e-4, minimum 1e-6, betas (0.9, 0.999), weight decay 1e-4','Cosine schedule over 20 epochs; Smooth L1 + 0.04 DCT norm','Flips, 90-degree rotations, MixUp p=0.5; balanced replacement sampler','CUDA AMP; NVIDIA RTX 2000 Ada Generation; recorded 16.00 GiB VRAM','No validation split or validation-based checkpoint selection'],'training/training_configuration.png',10,'final checkpoint config; configs/four_weather_fast.yaml; training logs')
logpath='checkpoints/four_weather_resume_epoch17/training.log'
final_lines=[s for s in read(logpath).splitlines() if re.match(r'epoch=\d+ ',s)]
panel('Training completion: rendered actual log excerpt',['Rendered excerpt, not a terminal screenshot',*final_lines,'Epoch-20 weights were saved before the subsequent CUDA test failure.','Separate full evaluation completed 9 November 2026 on all 14,695 test pairs.'],'training/training_completion_evidence.png',10,logpath+'; '+BASE+'/status.json')
panel('Final checkpoint verification',[str(checkpoint.relative_to(ROOT)),f"Saved epoch: {state['epoch']}; trainable parameters re-counted: {params:,}",'SHA-256: '+hash_checks['checkpoint']['actual'],'Checkpoint weights loaded strictly into AIRFormer(model_config).','Checkpoint, train/test manifests, attention and inference hashes match.','Evaluator hash differs; later report routing change is documented in LOG.md.','Training state includes optimizer, scheduler, AMP scaler and random states.'],'training/checkpoint_summary.png',10,'saved checkpoint; '+BASE+'/provenance.json; docs/LOG.md')

# Results: recomputed means were checked against summary above.
metrics=[]
for w in ['overall']+WEATHER:
    g=groups[w];metrics.append([w.title(),f"{g['images']:,}",f"{g['input_psnr']:.3f}",f"{g['restored_psnr']:.3f}",f"{g['mean_psnr_gain_db']:+.3f}",f"{g['restored_ssim']:.4f}"])
tablepic('Completed held-out evaluation',['Group','Pairs','Input dB','Output dB','Gain dB','Output SSIM'],metrics,'results/final_metrics_table.png',12,BASE+'/summary.json','Equal weight per image; custom split; RGB float data range 1',widths=[.16,.13,.17,.18,.17,.19])
def bars(metric,title,dest):
    f=canvas(title,'Same held-out images before and after restoration');a=f.add_axes([.09,.17,.85,.65]);xx=np.arange(5);ww=['overall']+WEATHER
    for off,prefix,color,label in [(-.2,'input_',NAVY,'Degraded input'),(.2,'restored_',TEAL,'Restored')]:
        b=a.bar(xx+off,[groups[w][prefix+metric] for w in ww],.38,color=color,label=label);a.bar_label(b,fmt='%.3f',fontsize=10,padding=3)
    a.set_xticks(xx,[w.title() for w in ww]);a.set_ylabel('PSNR (dB)' if metric=='psnr' else 'SSIM');a.legend();a.set_ylim(0,38 if metric=='psnr' else 1);save(f,'results/'+dest,12,BASE+'/summary.json')
bars('psnr','PSNR before and after restoration','psnr_before_after.png');bars('ssim','SSIM before and after restoration','ssim_before_after.png')
f=canvas('PSNR gain by weather','Mean of per-image gains on each custom held-out group');a=f.add_axes([.1,.17,.8,.65]);b=a.bar(WEATHER,[groups[w]['mean_psnr_gain_db'] for w in WEATHER],color=TEAL);a.bar_label(b,fmt='%+.3f dB');a.set(ylabel='Restored PSNR - input PSNR (dB)',ylim=(0,11));save(f,'results/psnr_improvement_by_weather.png',12,BASE+'/summary.json')
f=canvas('Mean absolute and root mean square errors','Per-image errors averaged across images. Lower is better. Raindrop MAE increases.');axs=f.subplots(1,2);f.subplots_adjust(top=.81,bottom=.17,wspace=.25)
for a,metric in zip(axs,['mae','rmse']):
    xx=np.arange(5)
    for off,pref,color,label in [(-.2,'input_',NAVY,'Input'),(.2,'restored_',TEAL,'Restored')]:a.bar(xx+off,[groups[w][pref+metric] for w in ['overall']+WEATHER],.38,color=color,label=label)
    a.set_xticks(xx,['All']+WEATHER,rotation=15);a.set_title(metric.upper());a.set_ylabel('RGB error, range 0..1');a.legend()
save(f,'results/mae_rmse_comparison.png',12,BASE+'/summary.json','Mean per-image RMSE is not sqrt(mean MSE).')
gain=np.array([float(r['psnr_gain_db']) for r in rows]);improved=float((gain>0).mean());assert abs(improved-overall['psnr_improved_fraction'])<1e-12
f=canvas('Distribution of per-image PSNR gains',f'{improved:.2%} of 14,695 images improved; {int((gain<=0).sum()):,} had nonpositive gain');a=f.add_axes([.09,.17,.85,.65]);a.hist(gain,bins=75,color=TEAL);a.axvline(0,color=GOLD,lw=2);a.set(xlabel='PSNR gain (dB)',ylabel='Image pairs');save(f,'results/improvement_distribution.png',12,BASE+'/per_image_metrics.csv')
mae_reduction=(1-overall['restored_mae']/overall['input_mae'])*100
panel('Final evaluation dashboard',[f"14,695 held-out pairs; completed checkpoint epoch {state['epoch']}",f"PSNR: {overall['input_psnr']:.4f} -> {overall['restored_psnr']:.4f} dB (+{overall['mean_psnr_gain_db']:.4f})",f"SSIM: {overall['input_ssim']:.6f} -> {overall['restored_ssim']:.6f}",f"Mean MAE: {overall['input_mae']:.6f} -> {overall['restored_mae']:.6f} ({mae_reduction:.2f}% reduction)",f"Positive PSNR gains: {improved:.2%}; mean inference {overall['mean_inference_seconds']:.6f} s/image",'Full test job: 7,865 s; test loop: 7,807.230 s; inference intervals: 5,596.097 s','Full-resolution FP32, batch 1, multiple-of-128 padding, no tiling.','Limit: snow accounts for 68.06% of test pairs; raindrop is weakest.'],'results/performance_dashboard.png',12,BASE+'/summary.json; duration_summary.json')
csvwrite('results/verified_metrics.csv',[dict(weather=w,**{k:v for k,v in groups[w].items() if not isinstance(v,dict)}) for w in ['overall']+WEATHER])
print('Diagrams, code excerpts, dataset and training/results graphics generated.',flush=True)

# Genuine qualitative examples: existing galleries do not preserve separate predictions.
# Preserve/reuse the original galleries, then infer only the twelve selected pairs.
from scripts.test_restoration import predict,pair_arrays
from utils.metrics import image_metrics
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model=model.to(device)
selected=[];sample_images={};sample_manifest=[]
for w in WEATHER:
    old=ROOT/BASE/'visualizations'/('examples_'+w+'.png')
    # Existing composite galleries remain untouched at their original paths.
    rr=sorted([r for r in rows if r['weather']==w],key=lambda r:float(r['psnr_gain_db']))
    for label,r in [('best',rr[-1]),('median',rr[len(rr)//2]),('worst',rr[0])]:
        idx=int(r['index']);m=manifests['test'][idx]
        assert m['input_path']==r['input_path'] and m['target_path']==r['target_path']
        degraded,target=pair_arrays((Path(r['input_path']),Path(r['target_path'])))
        _,restored,seconds=predict(model,degraded,device,128)
        p,s=image_metrics(restored,target);dp=abs(p-float(r['restored_psnr']));ds=abs(s-float(r['restored_ssim']))
        assert dp<.005 and ds<.0001,(w,label,dp,ds)
        f=canvas(f'{w.title()}: {label} PSNR-gain case',f"Test index {idx}, scene {r['scene_id']}, {r['width']} x {r['height']}; selected by PSNR gain, not visual preference")
        aa=f.subplots(1,3);f.subplots_adjust(top=.80,bottom=.18,left=.025,right=.975,wspace=.025)
        for a,arr,t in zip(aa,[degraded,restored,target],['Degraded input','Model restoration','Clean ground truth']):a.imshow(arr);a.set_title(t,fontsize=14);a.axis('off');a.set_aspect('equal')
        f.text(.045,.09,f"Recorded FP32 PSNR: {float(r['input_psnr']):.3f} -> {float(r['restored_psnr']):.3f} dB   Gain: {float(r['psnr_gain_db']):+.3f} dB",fontsize=14)
        f.text(.045,.05,f"Recorded SSIM: {float(r['input_ssim']):.4f} -> {float(r['restored_ssim']):.4f}   |   Newly inferred with verified final weights; metrics agree within tolerance",fontsize=11,color=GRAY)
        dest=f'restoration_samples/{w}_{label}.png';save(f,dest,13,r['input_path']+'; '+r['target_path']+'; '+str(checkpoint),'Small-sample FP32 inference only. Display keeps native aspect ratio. Title metrics are original float evaluation, before PNG quantization.')
        sample_images[(w,label)]=(degraded,restored,target,r)
        Image.fromarray((np.clip(restored,0,1)*255).round().astype('uint8')).save(OUT/'restoration_samples'/f'{w}_{label}_prediction.png')
        asset(f'restoration_samples/{w}_{label}_prediction.png',13,r['input_path']+'; '+str(checkpoint),'8-bit display copy of newly inferred prediction; metrics calculated on float output.')
        sample_manifest.append(dict(weather=w,case=label,test_index=idx,scene_id=r['scene_id'],input_path=r['input_path'],target_path=r['target_path'],checkpoint=str(checkpoint),checkpoint_sha256=hash_checks['checkpoint']['actual'],comparison_path=str(OUT/dest),prediction_path=str(OUT/'restoration_samples'/f'{w}_{label}_prediction.png'),input_psnr=r['input_psnr'],recorded_output_psnr=r['restored_psnr'],recomputed_output_psnr=p,input_ssim=r['input_ssim'],recorded_output_ssim=r['restored_ssim'],recomputed_output_ssim=s,psnr_delta_abs=dp,ssim_delta_abs=ds,inference_seconds=seconds,selection='minimum/upper-median/maximum PSNR gain over all weather-specific test pairs',rendering='aspect preserved; original float metrics'))
        print(f'Qualitative {w} {label}: PSNR {p:.4f}; CSV difference {dp:.7f}',flush=True)
csvwrite('restoration_samples/sample_manifest.csv',sample_manifest)
f=canvas('Four-weather restoration overview','Median PSNR-gain sample from each weather; full best/median/worst evidence is supplied separately',size=(16,12));aa=f.subplots(4,3);f.subplots_adjust(top=.86,bottom=.04,left=.07,right=.98,wspace=.03,hspace=.16)
for i,w in enumerate(WEATHER):
    degraded,restored,target,r=sample_images[(w,'median')]
    for j,arr in enumerate([degraded,restored,target]):
        aa[i,j].imshow(arr);aa[i,j].axis('off')
        if i==0:aa[i,j].set_title(['Degraded input','Model restoration','Clean target'][j])
    aa[i,0].text(-.04,.5,w.title(),transform=aa[i,0].transAxes,rotation=90,va='center',ha='right',weight='bold')
    aa[i,1].text(.5,-.055,f"PSNR {float(r['input_psnr']):.2f} -> {float(r['restored_psnr']):.2f}; SSIM {float(r['input_ssim']):.3f} -> {float(r['restored_ssim']):.3f}",transform=aa[i,1].transAxes,ha='center',va='top',fontsize=10)
save(f,'restoration_samples/four_weather_overview.png',13,'restoration_samples/sample_manifest.csv','Median by PSNR gain; no cropping or aspect stretching. Tall overview for four rows.')
f=canvas('Genuine local dataset input / clean pairs','Representative median-gain held-out scenes; original paths in restoration_samples/sample_manifest.csv');aa=f.subplots(2,4);f.subplots_adjust(top=.81,bottom=.08,left=.03,right=.97,wspace=.08,hspace=.20)
for j,w in enumerate(WEATHER):
    degraded,_,target,r=sample_images[(w,'median')]
    for i,arr in enumerate([degraded,target]):aa[i,j].imshow(arr);aa[i,j].axis('off');aa[i,j].set_title(w.title()+(' / degraded' if i==0 else ' / clean'),fontsize=12)
save(f,'datasets/sample_image_pairs.png',8,'restoration_samples/sample_manifest.csv','Native aspect ratios preserved; all images are actual local pairs.')

# Contextual paper comparison; values checked visually in supplied PDF Table XI/XIII.
context='Contextual comparison only — different datasets and evaluation protocols.'
paper_scores={'rain':(30.25,.949),'raindrop':(28.26,.911),'snow':(31.75,.954),'haze':(32.33,.989)}
comparison_rows=[['Parameters','58.34 million','0.612435 million'],['Training data','AIR40K, reported 40,561','Custom local 58,766 pairs'],['Training crop','256 x 256','128 x 128'],['Test data','Official multi-benchmark suite','14,695 custom held-out pairs'],['FLOPs at 256 x 256','5.746 billion','Not measured'],['Inference conditions','0.086 s, RTX 3090, 256x256','0.380816 s, RTX 2000 Ada, variable sizes'],['Learning-rate minimum','1e-7','1e-6']]
tablepic('Paper and implemented experiment',['Setting','Published AIRFormer','Completed local model'],comparison_rows,'comparison/paper_vs_implementation_table.png',14,'Supplied PDF, Sec. IV-B, Table XIII; final checkpoint; summary.json',context,widths=[.22,.36,.42])
f=canvas('Parameter counts: different model implementations',context);a=f.add_axes([.15,.17,.7,.65]);b=a.bar(['Paper AIRFormer','Trained local model'],[58.34,.612435],color=[NAVY,TEAL]);a.bar_label(b,fmt='%.6g M');a.set(ylabel='Trainable parameters (millions)',ylim=(0,66));save(f,'comparison/parameter_comparison.png',14,'Supplied PDF Table XIII; checkpoint parameter re-count',context+' Fewer parameters do not prove faster inference or equivalent quality.')
tablepic('Published weather summaries and local results',['Weather','Paper PSNR / SSIM','Local PSNR / SSIM','Local pairs'],[[w.title(),f'{paper_scores[w][0]:.2f} / {paper_scores[w][1]:.3f}',f"{groups[w]['restored_psnr']:.3f} / {groups[w]['restored_ssim']:.4f}",f"{groups[w]['images']:,}"] for w in WEATHER],'comparison/paper_vs_our_results.png',14,'Supplied PDF Table XI final row; final summary.json',context,widths=[.15,.29,.34,.22])
panel('Architecture differences to state explicitly',['Independent reconstruction; author code/complete supplementary settings unavailable.','Four encoder positions match the broad graph; widths and depths are chosen.','Supplementary Prior uses detail filters + resized coefficients + 6C fusion.','Learned query vectors are added to projected features; composition is assumed.','Space/wavelet K/V pooling changes the attention graph; token cap is 256.','Bilinear skip fusion, RGB residual head and clamp are implementation choices.','Elementwise Smooth L1 differs from the literal printed Equation 12.','No local FLOPs, matched latency comparison or component ablation claims.'],'comparison/architecture_difference_summary.png',6,'models/*.py; losses/multidomain.py; docs/architecture_assumptions.md; supplied PDF')
f,a=diagram('Actual multi-domain consistency loss','Two parallel loss branches. Unweighted orthonormal DCT preserves the total squared-error energy.')
for x,y,w,h,t in [(0.2,3.2,3,1.4,'Restored + clean RGB\nB x 3 x H x W'),(4.4,5.4,4.1,1.4,'Spatial branch\nElementwise Smooth L1\nMean across B/C/H/W'),(4.4,1.2,4.1,1.4,'Frequency branch\nDCT-II of each RGB image\nDifference of coefficients'),(9.6,1.2,4.7,1.4,'Per-image smoothed L2 norm\nMean across batch\nMultiply by lambda = 0.04'),(10,5.4,4.3,1.4,'Sum both branch losses\nL = spatial + .04 frequency')]:box(a,x,y,w,h,t,fs=12)
for p,q in [((3.25,4.15),(4.3,6.1)),((3.25,3.5),(4.3,1.9)),((8.6,6.1),(9.9,6.1)),((8.6,1.9),(9.5,1.9)),((12,2.7),(12,5.3))]:arrow(a,p,q,TEAL)
save(f,'loss/multidomain_loss_diagram.png',11,'losses/multidomain.py; supplied PDF Eq.11-13','Unweighted orthonormal transform norm, not selective high-frequency weighting.')
f=canvas('Implemented loss and evaluation metrics','Actual reduction differs from the literal printed Equation 12')
texts=[r'$e=\hat{y}-y,\quad \rho(e)=\begin{cases}0.5e^2,&|e|<1\\ |e|-0.5,&|e|\geq1\end{cases}$',r'$L_{spatial}=\mathrm{mean}_{B,C,H,W}\,\rho(e)$',r'$L_{freq}=\mathrm{mean}_{b}\sqrt{\sum_{c,u,v}[DCT(\hat{y}_b)-DCT(y_b)]^2+10^{-6}}$',r'$L=L_{spatial}+0.04L_{freq}$',r'$PSNR=10\log_{10}(1/MSE),\quad MAE=\mathrm{mean}|e|,\quad RMSE=\sqrt{MSE}$']
# Matplotlib mathtext does not support cases, so render its piecewise definition as text.
texts[0]='Smooth L1: rho(e) = 0.5 e^2 for |e| < 1, otherwise |e| - 0.5'
for i,t in enumerate(texts):f.text(.055,.80-i*.115,t,fontsize=19)
f.text(.055,.15,'SSIM compares local luminance, contrast and structure (RGB skimage, range 1, window up to 7).',fontsize=13)
f.text(.055,.09,'Paper Eq.12: rho(mean spatial residual). Code: mean(rho(elementwise residual)). Cancellation makes these different.',fontsize=12)
f.text(.055,.045,'No frequency weighting is applied: an orthonormal DCT L2 norm equals the spatial L2 norm up to the same epsilon.',fontsize=12,color=TEAL)
save(f,'loss/loss_equations.png',11,'losses/multidomain.py; utils/metrics.py; supplied PDF Eq.11-13')
st,en,_=segment('losses/multidomain.py','forward');code_image('losses/multidomain.py',st,en,'loss/loss_code_evidence.png',11)

# Source inventory: relevant files, timestamps, hashes and architecture symbol coverage.
inventory=[]
scanroots=['models','airformer','losses','configs','datasets','scripts','checkpoints','outputs','docs','data/manifests','tests','utils']
for dirname in scanroots:
    for p in (ROOT/dirname).rglob('*'):
        if p.is_file() and (p.suffix.lower() in ['.md','.json','.yaml','.csv','.py','.log','.xml','.txt'] or p.name.startswith('events.out.tfevents')):
            inventory.append(dict(path=str(p.relative_to(ROOT)).replace('\\','/'),bytes=p.stat().st_size,modified_ns=p.stat().st_mtime_ns,sha256=hashfile(p)))
csvwrite('sources/source_inventory.csv',inventory)
write('sources/build_facts.json',json.dumps(dict(parameters=params,checkpoint_epoch=state['epoch'],first_loss=losses[0],last_loss=losses[-1],mae_reduction_percent=mae_reduction,positive_psnr_fraction=improved,example_count=len(sample_manifest),source_files=len(inventory)),indent=2))
print('All numeric extraction and qualitative inference complete. Documentation follows.',flush=True)

# Documentation is authored separately to keep extraction repeatable.
exec(compile((OUT/'write_documentation.py').read_text(encoding='utf8'),str(OUT/'write_documentation.py'),'exec'))
