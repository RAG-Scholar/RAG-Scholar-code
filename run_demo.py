"""Run the final.pdf method on the three-paper example."""
from __future__ import annotations
import argparse
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0,str(ROOT/'src'))
from pdf_vlm_assistant.document_entity_graph import DocumentEntityGraphConfig,build_document_entity_graph_from_segments
from pdf_vlm_assistant.document_entity_retrieval import DocumentEntityRetriever,DocumentEntityRetrieverConfig
from pdf_vlm_assistant.gnn_embeddings import EncoderConfig,TrainingConfig,train_embeddings
from pdf_vlm_assistant.query_context_bundle import QueryContextBundleConfig,build_query_context_bundle
from demo_support import audit_graph,audit_crop,audit_entity_use,audit_retained_evidence,dump,make_graph_portable,relative_paths,require
from model_runtime import BCE_REPO,BCE_REVISION,CLIP_REPO,CLIP_REVISION,resolve_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs')
    parser.add_argument('--bce-model-path',type=Path)
    parser.add_argument('--clip-model-path',type=Path)
    parser.add_argument('--download-model',action='store_true')
    parser.add_argument('--skip-crop',action='store_true')
    parser.add_argument('--reuse-index',action='store_true',help='Reuse a matching trained index; otherwise train 40 epochs.')
    parser.add_argument('--text-budget',type=int,default=800,help='Appendix E retention target b; 0 disables the token target.')
    parser.add_argument('--window-limit',type=int,default=360)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--answer-model-path',type=Path,help='Optional local Qwen2-VL-7B-Instruct checkpoint for actual answers.')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    require(output == ROOT/'outputs' or ROOT in output.parents,'Keep outputs inside this package')
    output.mkdir(parents=True,exist_ok=True)
    dump(output/'validation.json',{'status':'running'})
    started = time.perf_counter()
    print('[1/6] Rebuild the graph from unchanged MinerU materials',flush=True)
    graph_dir = output/'graph'
    stats = build_document_entity_graph_from_segments(ROOT/'data/segments',graph_dir,
                DocumentEntityGraphConfig(enable_cross_paper_edges=True))
    require(stats['paper_count'] == 3 and stats['skipped_paper_count'] == 0,'Incomplete graph')
    make_graph_portable(graph_dir)
    audit = audit_graph(graph_dir)
    dump(output/'cross_paper_evidence.json',audit)
    print('[2/6] Frozen BCE features and 40-epoch relation-aware GNN',flush=True)
    bce = resolve_model(BCE_REPO,BCE_REVISION,args.bce_model_path,args.download_model)
    index_dir = output/'gnn'
    if not args.reuse_index:
        train_embeddings(graph_dir,index_dir,EncoderConfig(),TrainingConfig(device=args.device),encoder_model=str(bce),progress=lambda message: print(message,flush=True))
    retriever = DocumentEntityRetriever(DocumentEntityRetrieverConfig(
        graph_dir=graph_dir,gnn_index_dir=index_dir,retrieval_backend='hybrid',gnn_device=args.device,
        gnn_encoder_model=str(bce),top_k_papers=5,top_k_sections=2,top_k_chunks=2,top_k_figures=1,top_k_entities=5,
        preview_chars=340))
    queries = json.loads((ROOT/'data/queries.json').read_text(encoding='utf-8'))
    print(f'[3/6] {len(queries)} queries: entity lookup, explanation and 0.7/0.3 rank fusion',flush=True)
    checks,payloads,bundles = [],{},{}
    for case in queries:
        payload = retriever.retrieve(case['query'])
        returned = [p['paper_id'] for p in payload['results']]
        if payload.get('hybrid_constraint_mode'):
            require(set(returned) == set(case['expected_papers']),'Wrong constrained set: '+case['id'])
        else:
            require(set(case['expected_papers']) <= set(returned),'Missing expected source: '+case['id'])
        for paper in payload['results']:
            require(paper['top_sections'] and len(paper['top_sections']) <= 2,'Invalid section limit')
            require(paper['hybrid_sources']['gnn'] is not None,'GNN branch unused')
            for section in paper['top_sections']:
                require(len(section['top_chunks']) <= 2 and len(section['top_figures']) <= 1,'Evidence limit exceeded')
        payloads[case['id']] = payload
        dump(output/'retrieval'/(case['id']+'.json'),relative_paths(payload,ROOT))
        config = QueryContextBundleConfig(graph_dir=graph_dir,max_total_tokens=args.text_budget or None)
        bundle = build_query_context_bundle(case['query'],config,retriever)
        # The serializer uses portable source IDs; token counts measure the text actually delivered.
        import tiktoken
        bundle['summary']['context_token_count'] = len(tiktoken.get_encoding('cl100k_base').encode(bundle['llm_context_markdown']))
        bundle['summary']['tokenizer'] = 'cl100k_base'
        bundles[case['id']] = bundle
        dump(output/'contexts'/(case['id']+'.json'),relative_paths(bundle,ROOT))
        evidence = audit_retained_evidence(case,bundle)
        require(evidence['status'] == 'passed','Missing retained explanation evidence: '+case['id'])
        checks.append({'id':case['id'],'returned_papers':returned,'status':'passed','context':bundle['summary'],
                       'retained_evidence':evidence})
        print(' ',case['id'],returned,flush=True)
    require(any(p['cross_paper_support'] for payload in payloads.values() for p in payload['results']),'Cross-paper relations unused across all demo queries')
    print('[4/6] Check entity messages and entity-channel recall contributions',flush=True)
    entity_audit = audit_entity_use(retriever,payloads)
    dump(output/'entity_usage.json',entity_audit)
    print('[5/6] CLIP scoring conditioned on final retained text; render one union image',flush=True)
    context = bundles['figure']
    context['images'] = []
    crop_check = {'status':'skipped'}
    if not args.skip_crop:
        from scripts.select_pixel_crops_by_query import crop_from_bundle
        clip = resolve_model(CLIP_REPO,CLIP_REVISION,args.clip_model_path,args.download_model)
        crop = crop_from_bundle(context,graph_dir,clip,output/'crops',args.window_limit)
        crop_check = audit_crop(crop)
        context['images'] = [{'path':crop['union_image'],'box':crop['union_box'],
                              'figure_node_id':crop['figure_metadata']['node_id'],'render_mode':'union'}]
        # Actual processor accounting uses the bundled official image-processor configuration.
        from answer_model import measure_visual_tokens
        context['visual_tokens'] = measure_visual_tokens(ROOT/crop['union_image'])
        crop_check['visual_tokens'] = context['visual_tokens']['visual_tokens']
    dump(output/'context_bundle.json',relative_paths(context,ROOT))
    print('[6/6] Save evidence and optional Qwen2-VL answers',flush=True)
    answer_check = {'status':'not_requested'}
    if args.answer_model_path:
        from answer_model import AnswerModel
        model = AnswerModel(args.answer_model_path,args.device)
        for case in queries:
            bundle = context if case['id'] == 'figure' else bundles[case['id']]
            dump(output/'answers'/(case['id']+'.json'),model.answer(bundle))
        answer_check = {'status':'passed','count':len(queries)}
    metadata = retriever.gnn.index.metadata
    validation = {'status':'text_pipeline_passed' if args.skip_crop else 'retrieval_pipeline_passed',
        'python':platform.python_version(),'platform':platform.system(),'elapsed_seconds':round(time.perf_counter()-started,3),
        'graph':{k:audit[k] for k in ['node_count','edge_count','node_types','cross_paper_edge_count']},
        'gnn':{'index_reused':args.reuse_index,'epochs':metadata['training']['epochs'],'embedding_dim':metadata['embedding_dim'],
               'trainable_parameters':metadata['trainable_parameters'],'initial_loss':metadata['history'][0]['loss'],
               'final_loss':metadata['history'][-1]['loss']},
        'query_count':len(queries),'retrieval_check_scope':'paper_ids_hierarchy_and_retained_evidence',
        'retrieval':checks,'entity_usage':entity_audit['status'],'crop':crop_check,'answer_generation':answer_check}
    dump(output/'validation.json',validation)
    print(json.dumps(validation,ensure_ascii=False,indent=2))

if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        # Report failure without leaving a previous successful status.
        args = sys.argv
        output = Path(args[args.index('--output-dir')+1]).resolve() if '--output-dir' in args else ROOT/'outputs'
        if output == ROOT/'outputs' or ROOT in output.parents:
            dump(output/'validation.json',{'status':'failed','error_type':type(exc).__name__})
        raise
