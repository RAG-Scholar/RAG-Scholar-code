"""Qwen2-VL image-token accounting and optional answer generation (Appendix E)."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parent


def measure_visual_tokens(image_path, processor=None):
    if processor is None:
        from transformers import Qwen2VLImageProcessor
        processor = Qwen2VLImageProcessor.from_pretrained(ROOT/'configs/answer_processor',local_files_only=True)
    image_processor = getattr(processor,'image_processor',processor)
    with Image.open(image_path) as image:
        features = image_processor(images=[image.convert('RGB')],return_tensors='pt')
    grids = features['image_grid_thw'].tolist()
    merge = image_processor.merge_size
    return {'visual_tokens':sum(int(np.prod(grid))//(merge*merge) for grid in grids),
            'image_grid_thw':grids,'merge_size':merge,'processor':'Qwen2-VL-7B-Instruct'}


def build_messages(bundle):
    candidates = [{'paper_id':p['paper_id'],'paper_title':p['paper_title']} for p in bundle['papers']]
    images = bundle.get('images',[])
    schema = {'predicted_paper_ids':['paper_id'],'predicted_paper_titles':['paper_title'],
              'answer':'two to four sentences' if images else 'at least four sentences',
              'evidence':[{'paper_id':'paper_id','reason':'why it matches'}],
              'structured_answer':{'paper_answers':[{'paper_id':'paper_id','paper_title':'paper_title',
                  'match_summary':'short explanation','supporting_facts':[{'fact_type':'method',
                  'claim':'source-supported claim','support':[{'support_type':'chunk_text','support_value':'exact short source span'}]}]}],
                  'overall_conclusion':'short conclusion','abstained':False}}
    prompt = ('Answer the user query using only the provided paper context bundle. Choose from the candidate papers; do not invent IDs.\n'
              +'User query: '+bundle['query']+'\nCandidate papers: '+json.dumps(candidates,ensure_ascii=False)
              +'\nAttached image manifest: '+json.dumps(images,ensure_ascii=False)
              +'\nContext bundle:\n'+bundle['llm_context_markdown']
              +'\nSelect only papers that satisfy the query. Return an empty list if none answers it. '
              'Use query_match_evidence, matched_entities and cross_paper_neighbors as source evidence. '
              'Write '+('two to four sentences. ' if images else 'at least four sentences. ')
              +'The answer is the main response; do not discuss evaluator logic or verification rules. '
              'The structured answer is an audit trail of the same claims, with 1-3 short source-supported facts per predicted paper. '
              'For multi-constraint queries cover distinct constraints. Copy short exact supporting spans; omit unsupported claims. '
              'Prefer matched_entity, query_match_evidence, section_summary and figure_caption, then chunk_text or source metadata. '
              'Allowed fact types: institution, author, reference, method, dataset, model, task, topic, visual, other. '
              'Allowed support types: paper_title, authors, institutions, query_match_evidence, matched_entity, section, section_summary, '
              'chunk_text, figure_caption, cross_paper_neighbor. '
              +('Use the attached union crop with its caption and retained section text. ' if images else '')
              +'Return JSON only using this schema: '+json.dumps(schema))
    content = [{'type':'image','image':str(ROOT/row['path'])} for row in images]
    content.append({'type':'text','text':prompt})
    return [{'role':'user','content':content}]


class AnswerModel:
    def __init__(self,model_path,device='cuda:0'):
        import torch
        from transformers import AutoProcessor,Qwen2VLForConditionalGeneration
        self.device = device
        self.processor = AutoProcessor.from_pretrained(str(model_path),local_files_only=True)
        dtype = torch.float16 if device.startswith('cuda') else torch.float32
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(str(model_path),local_files_only=True,
                     torch_dtype=dtype).eval().to(device)

    def answer(self,bundle):
        import torch
        messages = build_messages(bundle)
        text = self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        images = [Image.open(ROOT/row['path']).convert('RGB') for row in bundle.get('images',[])]
        inputs = self.processor(text=[text],images=images or None,padding=True,return_tensors='pt').to(self.device)
        with torch.inference_mode():
            output = self.model.generate(**inputs,do_sample=False,max_new_tokens=700)
        answer = self.processor.batch_decode(output[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)[0]
        parsed = json.loads(answer)
        allowed = {p['paper_id'] for p in bundle['papers']}
        if not set(parsed['predicted_paper_ids']) <= allowed:
            raise ValueError('Answer returned a source outside the retrieved candidate set')
        return {'query':bundle['query'],'answer':parsed,'generation':{'model':'Qwen2-VL-7B-Instruct',
                'do_sample':False,'max_new_tokens':700},'context_token_count':bundle['summary']['context_token_count']}
