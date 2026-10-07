"""Text-conditioned CLIP crops: final.pdf Eqs. (11), (16), (17)."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

SCALES = [(s,s) for s in (1.,.95,.9,.85,.8,.75,.7,.65,.6,.55,.5,.45,.4,.35)] + [
    (.8,.65),(.65,.8),(.9,.7),(.7,.9),(.55,.8),(.8,.55),(.75,.5),(.5,.75),
    (.65,.5),(.5,.65),(.45,.7),(.7,.45),(.35,.55),(.55,.35)]
ANCHORS = (0.,.12,.25,.38,.5,.62,.75,.88,1.)

def detect_content_box(image, white_threshold=245, trim_margin=8):
    ys,xs = np.where(np.any(np.asarray(image.convert('RGB')) < white_threshold, axis=2))
    if not len(xs):
        return (0,0,image.width,image.height)
    return (max(0,int(xs.min())-trim_margin), max(0,int(ys.min())-trim_margin),
            min(image.width,int(xs.max())+1+trim_margin), min(image.height,int(ys.max())+1+trim_margin))

def generate_candidate_windows(image, content_box, limit=360):
    """Evenly subsample template/grid enumeration; retain both fallback boxes."""
    if not 2 <= limit <= 360:
        raise ValueError('window limit must be between 2 and 360')
    x1,y1,x2,y2 = content_box
    cw,ch = x2-x1,y2-y1
    fixed = list(dict.fromkeys([(0,0,image.width,image.height), tuple(content_box)]))
    boxes = {}
    for sw,sh in SCALES:
        w,h = min(cw,max(32,round(cw*sw))),min(ch,max(32,round(ch*sh)))
        for ax in ANCHORS:
            for ay in ANCHORS:
                x,y = x1+round((cw-w)*ax),y1+round((ch-h)*ay)
                boxes[(x,y,x+w,y+h)] = None
    remaining = [box for box in boxes if box not in fixed]
    count = min(len(remaining),limit-len(fixed))
    positions = np.linspace(0,len(remaining)-1,count,dtype=int) if count else []
    return fixed+[remaining[i] for i in positions]

def intersection_over_union(left,right):
    iw = max(0,min(left[2],right[2])-max(left[0],right[0]))
    ih = max(0,min(left[3],right[3])-max(left[1],right[1]))
    intersection = iw*ih
    union = ((left[2]-left[0])*(left[3]-left[1])+
             (right[2]-right[0])*(right[3]-right[1])-intersection)
    return intersection/union if union else 0.

def select_windows_budgeted(scored_windows,keep_top_k=3,iou_threshold=.5,max_total_area_ratio=.55):
    """Greedy feasible set in decreasing S_w order; fallback is applied by caller."""
    if keep_top_k < 1 or not 0 < max_total_area_ratio <= 1:
        raise ValueError('keep_top_k must be positive and area budget must be in (0,1]')
    selected,area = [],0.
    for row in sorted(scored_windows,key=lambda r:-r['retain_score']):
        if area+row['area_ratio'] > max_total_area_ratio+1e-9:
            continue
        if any(intersection_over_union(row['box'],p['box']) >= iou_threshold for p in selected):
            continue
        selected.append(dict(row,rank=len(selected)+1))
        area += row['area_ratio']
        if len(selected) >= keep_top_k:
            break
    return selected

def select_with_paper_fallback(scored):
    selected = select_windows_budgeted(scored)
    # Appendix B.3 explicitly retains one top candidate if none is feasible.
    return (selected,False) if selected else ([dict(max(scored,key=lambda r:r['retain_score']),rank=1)],True)

def load_clip(model_path):
    import torch
    from transformers import CLIPModel,CLIPTokenizer,CLIPImageProcessor
    torch.set_num_threads(4)
    return (CLIPModel.from_pretrained(str(model_path),local_files_only=True).eval(),
            CLIPTokenizer.from_pretrained(str(model_path),local_files_only=True),
            CLIPImageProcessor.from_pretrained(str(model_path),local_files_only=True))

def score_windows(image,boxes,query,snippets,runtime):
    import torch
    from torch.nn.functional import normalize
    model,tokenizer,processor = runtime
    with torch.inference_mode():
        tokens = tokenizer([query]+snippets,padding=True,truncation=True,max_length=77,return_tensors='pt')
        text = normalize(model.get_text_features(**tokens).float(),dim=-1)
        rows = []
        for start in range(0,len(boxes),12):
            batch = boxes[start:start+12]
            pixels = processor(images=[image.crop(box) for box in batch],return_tensors='pt')
            visual = normalize(model.get_image_features(**pixels).float(),dim=-1)
            similarities = (visual@text.T).cpu().numpy()
            for box,cosines in zip(batch,similarities):
                area = (box[2]-box[0])*(box[3]-box[1])/(image.width*image.height)
                redundancy = float(cosines[1:].max()) if snippets else 0.
                relevance = float(cosines[0])
                rows.append({'box':list(box),'area_ratio':area,'query_score':relevance,
                             'text_redundancy':redundancy,'retain_score':relevance-.75*redundancy-.08*area})
    return sorted(rows,key=lambda r:-r['retain_score'])

def retained_figure_context(bundle,graph_dir,figure_node_id=None):
    """Select from final retained figures; T_f(q) is the retained section text."""
    figures = {f['node_id']:f for f in map(json.loads,(Path(graph_dir)/'figures.jsonl').read_text(encoding='utf-8').splitlines())}
    for paper in bundle['papers']:
        for section in paper['sections']:
            for selected in section['figures']:
                if figure_node_id is not None and selected['node_id'] != figure_node_id:
                    continue
                figure = figures[selected['node_id']]
                snippets,sources = [],[]
                if section['section_summary']:
                    snippets.append(section['section_summary'])
                    sources.append({'section_id':section['section_id'],'field':'section_summary'})
                for chunk in section['chunks']:
                    if chunk['text']:
                        snippets.append(chunk['text'])
                        sources.append({'section_id':section['section_id'],'chunk_id':chunk['node_id'],'field':'text'})
                return figure,snippets[:10],sources[:10]
    raise ValueError('No retained figure is available in the final context')

def crop_from_bundle(bundle,graph_dir,model_path,output_dir,window_limit=360,figure_node_id=None):
    from demo_support import dump,relative_paths
    figure,snippets,sources = retained_figure_context(bundle,graph_dir,figure_node_id)
    image_path = (Path(graph_dir)/figure['image_path']).resolve()
    image = Image.open(image_path).convert('RGB')
    content_box = detect_content_box(image)
    boxes = generate_candidate_windows(image,content_box,window_limit)
    scored = score_windows(image,boxes,bundle['query'],snippets,load_clip(model_path))
    selected,fallback = select_with_paper_fallback(scored)
    union = (min(r['box'][0] for r in selected),min(r['box'][1] for r in selected),
             max(r['box'][2] for r in selected),max(r['box'][3] for r in selected))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    for row in selected:
        path = output_dir/('crop_rank_%02d.png'%row['rank'])
        image.crop(row['box']).save(path)
        row['crop_path'] = str(path.resolve())
    union_path = output_dir/'crop_union.png'
    image.crop(union).save(union_path)
    annotated = image.copy()
    drawing = ImageDraw.Draw(annotated)
    for row in selected:
        drawing.rectangle(row['box'],outline='red',width=3)
        drawing.text((row['box'][0]+4,row['box'][1]+4),str(row['rank']),fill='red')
    annotated.save(output_dir/'annotated.png')
    payload = {'query':bundle['query'],'figure_metadata':figure,'image_path':str(image_path),
        'image_size':list(image.size),'content_box':list(content_box),
        'score_formula':{'text_penalty':.75,'area_penalty':.08},
        'selection_budget':{'keep_top_k':3,'nms_iou_threshold':.5,'max_total_area_ratio':.55},
        'fallback_used':fallback,'candidate_count':len(boxes),'selected_windows':selected,
        'union_box':list(union),'union_image':str(union_path.resolve()),
        'union_area_ratio':(union[2]-union[0])*(union[3]-union[1])/(image.width*image.height),
        'render_mode':'union','attached_image_count':1,'text_context_snippets':snippets,
        'text_context_sources':sources,'top_scored_windows':scored,'clip_model':'openai/clip-vit-base-patch32'}
    payload = relative_paths(payload,ROOT)
    dump(output_dir.parent/'crop.json',payload)
    return payload

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph-dir',type=Path,default=ROOT/'outputs/graph')
    parser.add_argument('--context-bundle',type=Path,required=True)
    parser.add_argument('--clip-model-path',type=Path,required=True)
    parser.add_argument('--figure-node-id')
    parser.add_argument('--window-limit',type=int,default=360)
    parser.add_argument('--export-dir',type=Path,default=ROOT/'outputs/crops')
    args = parser.parse_args()
    payload = crop_from_bundle(json.loads(args.context_bundle.read_text(encoding='utf-8')),args.graph_dir,
        args.clip_model_path,args.export_dir,args.window_limit,args.figure_node_id)
    print(json.dumps({'figure':payload['figure_metadata']['node_id'],'windows':len(payload['selected_windows']),
                      'union_image':payload['union_image']},indent=2))

if __name__ == '__main__':
    main()
