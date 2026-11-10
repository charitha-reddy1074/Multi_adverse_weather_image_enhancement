"""Re-render reviewed assets and verify/package existing extracted evidence; no inference."""
from pathlib import Path
import sys,os,json,csv,re,hashlib,shutil,textwrap,zipfile,ast
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'PPT_EVIDENCE'
NAVY='#16324F';TEAL='#168B91';GOLD='#DB9639';GRAY='#5A6876'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'axes.spines.top':False,'axes.spines.right':False,'text.color':NAVY,'axes.labelcolor':NAVY,'savefig.dpi':160})
src=(OUT/'build_evidence.py').read_text(encoding='utf8');tree=ast.parse(src)
for node in tree.body:
    if isinstance(node,ast.FunctionDef):exec(compile(ast.Module(body=[node],type_ignores=[]),str(OUT/'build_evidence.py'),'exec'))
with (OUT/'PPT_ASSET_MANIFEST.csv').open(encoding='utf-8-sig',newline='') as f:ASSETS=list(csv.DictReader(f))
for w in ['rain','raindrop','snow','haze']:
    p=OUT/'restoration_samples'/('original_gallery_'+w+'.png')
    assert p.resolve().is_relative_to(OUT.resolve())
    p.unlink(missing_ok=True)
ASSETS[:]=[r for r in ASSETS if 'original_gallery_' not in r['asset_filename']]
BASE='outputs/four_weather_test_epoch20_20261109';WEATHER=['rain','raindrop','snow','haze']
summary=js(BASE+'/summary.json');groups=summary['groups'];overall=groups['overall'];prov=summary['provenance'];duration=js(BASE+'/duration_summary.json')
verified=json.loads((OUT/'sources/checkpoint_verification.json').read_text(encoding='utf8'));hash_checks=verified['hash_checks'];params=verified['parameters'];cfg=verified['training_config'];model_cfg=verified['model_config'];state={'epoch':verified['epoch']};checkpoint=Path(prov['checkpoint'])
with (ROOT/BASE/'per_image_metrics.csv').open(encoding='utf8',newline='') as f:rows=list(csv.DictReader(f))
gain=np.array([float(r['psnr_gain_db']) for r in rows]);improved=float((gain>0).mean());mae_reduction=(1-overall['restored_mae']/overall['input_mae'])*100
with (OUT/'training/epoch_history.csv').open(encoding='utf-8-sig',newline='') as f:history={int(r['epoch']):{**r,'training_loss':float(r['training_loss']),'logged_lr_after_epoch':float(r['logged_lr_after_epoch'])} for r in csv.DictReader(f)}
losses=[history[i]['training_loss'] for i in sorted(history)]
with (OUT/'datasets/dataset_details.csv').open(encoding='utf-8-sig',newline='') as f:details=[{**r,**{k:int(r[k]) for k in ['train_pairs','test_pairs','total_pairs']}} for r in csv.DictReader(f)]
with (OUT/'restoration_samples/sample_manifest.csv').open(encoding='utf-8-sig',newline='') as f:sample_manifest=list(csv.DictReader(f))
exec(compile(src[src.index('# Four-stage implementation diagram.'):src.index('# Architecture-to-code excerpts')],'<diagram-render>','exec'))
for node in tree.body:
    if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='flow':
        if node.value.args[0].value=='Dataset preparation and integrity checks':exec(compile(ast.Module(body=[node],type_ignores=[]),'<dataset-render>','exec'))
exec(compile(src[src.index("f,a=diagram('Actual multi-domain"):src.index('# Source inventory:')],'<loss-render>','exec'))
path='models/supplementary_prior.py';st=next(i for i,s in enumerate(read(path).splitlines(),1) if 'def forward(self, encoder_feature' in s);en=segment(path,'SupplementaryPrior')[1]
code_image(path,st,en,'code_screenshots/supplementary_prior.png',9)
for name,st,en in [('channel_attention',77,83),('spatial_attention',71,76),('wavelet_attention',66,70)]:code_image('models/attention.py',st,en,'code_screenshots/'+name+'.png',9)
# Improve overview caption clearance using the twelve already saved predictions.
f=canvas('Four-weather restoration overview','Median PSNR-gain sample from each weather; full best/median/worst evidence is supplied separately',size=(16,12));aa=f.subplots(4,3);f.subplots_adjust(top=.86,bottom=.04,left=.07,right=.98,wspace=.03,hspace=.21)
for i,w in enumerate(WEATHER):
    r=next(r for r in sample_manifest if r['weather']==w and r['case']=='median')
    with Image.open(r['input_path']) as im:degraded=np.asarray(im.convert('RGB'))
    with Image.open(r['target_path']) as im:target=np.asarray(im.convert('RGB'))
    with Image.open(r['prediction_path']) as im:restored=np.asarray(im.convert('RGB'))
    for j,arr in enumerate([degraded,restored,target]):
        aa[i,j].imshow(arr);aa[i,j].axis('off')
        if i==0:aa[i,j].set_title(['Degraded input','Model restoration','Clean target'][j])
    aa[i,0].text(-.04,.5,w.title(),transform=aa[i,0].transAxes,rotation=90,va='center',ha='right',weight='bold')
    aa[i,1].text(.5,-.055,f"PSNR {float(r['input_psnr']):.2f} -> {float(r['recorded_output_psnr']):.2f}; SSIM {float(r['input_ssim']):.3f} -> {float(r['recorded_output_ssim']):.3f}",transform=aa[i,1].transAxes,ha='center',va='top',fontsize=10)
save(f,'restoration_samples/four_weather_overview.png',13,'restoration_samples/sample_manifest.csv','Median-gain cases; native aspect ratios preserved. Tall overview; float metrics before display quantization.')
doc=(OUT/'03_ARCHITECTURE_TO_CODE.md').read_text(encoding='utf8')
new='\n'.join(f'{i}: {s}' for i,s in enumerate(read(path).splitlines()[st-1:en],st))
doc=re.sub(r'(## Supplementary Prior\n.*?```python\n).*?(\n```)',lambda m:m[1]+new+m[2],doc,flags=re.S)
write('03_ARCHITECTURE_TO_CODE.md',doc)
context='Contextual comparison only — different datasets and evaluation protocols.'
# Include small, genuine source snapshots and exact log lines for offline traceability.
snapshots=['models/airformer.py','models/encoder.py','models/patch_embed.py','models/supplementary_prior.py','models/attention.py','models/decoder.py','losses/multidomain.py','airformer/data.py','datasets/splitting.py','utils/metrics.py','utils/inference.py','scripts/train.py','scripts/test_restoration.py','configs/four_weather_fast.yaml','datasets/sources.csv']
for rel in snapshots:
    dest=OUT/'sources/code'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dest)
    asset(str(dest.relative_to(OUT)).replace('\\','/'),'support',rel,'Exact current source snapshot; evaluator differs from original evaluation hash as documented.')
for rel in ['docs/RESIDE_MIRROR_VERIFICATION.json','data/manifests/four_weather/preflight.json','docs/architecture/system_validation.json']:
    dest='sources/'+Path(rel).stem+'_source.json';shutil.copy2(ROOT/rel,OUT/dest);asset(dest,'support',rel,'Historical verified record, not a fresh full-data scan.')
final_log='checkpoints/four_weather_resume_epoch17/training.log'
lines=[(i,s) for i,s in enumerate(read(final_log).splitlines(),1) if re.match(r'epoch=\d+ ',s)]
write('sources/final_epoch_log_excerpt.txt','Rendered source excerpt from '+final_log+'\n\n'+'\n'.join(str(i)+': '+s for i,s in lines)+'\n')
asset('sources/final_epoch_log_excerpt.txt',10,final_log,'Exact selected completion lines with original line numbers; not a terminal capture.')
write('sources/evaluator_hash_discrepancy.md',f'''# Evaluator hash discrepancy

Recorded final-evaluation evaluator SHA-256: `{hash_checks['evaluator']['expected']}`.

Current `scripts/test_restoration.py` SHA-256: `{hash_checks['evaluator']['actual']}`.

`docs/LOG.md`, 9 November, records moving Markdown reports to `docs/reports/` after completion. Current evaluator includes `markdown_report_path()` and report-link routing consistent with that record. This explains a plausible cause but does not prove byte-for-byte scope of all edits because the exact old evaluator snapshot was not recovered.

Checkpoint, train manifest, test manifest, attention and inference SHA-256 all match. Fifty summary means reconcile with the stored CSV; selected outputs reproduce all twelve recorded PSNR/SSIM values. No numeric artifact was replaced, and no full evaluation was rerun.
''')
asset('sources/evaluator_hash_discrepancy.md','support','final provenance; scripts/test_restoration.py; docs/LOG.md','Original evaluator bytes unavailable; report-routing explanation is documented, not an exact byte diff.')
exec(compile((OUT/'write_documentation.py').read_text(encoding='utf8'),str(OUT/'write_documentation.py'),'exec'))
