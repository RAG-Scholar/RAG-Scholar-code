"""Equation and routing regressions for the three-paper example."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from pdf_vlm_assistant.gnn_embeddings import tensorize_graph,DocumentGraphEncoder,RelationLayer,TrainingConfig
from pdf_vlm_assistant.gnn_paper_recall import GraphPaperRecall
from pdf_vlm_assistant.hybrid_retrieval import fuse_ranked_rows
from pdf_vlm_assistant.document_entity_retrieval import figure_labels
from pdf_vlm_assistant.document_entity_graph import harmonic_mean_confidence
from pdf_vlm_assistant.query_context_bundle import estimate_token_count,render_context_bundle_markdown
from scripts.select_pixel_crops_by_query import (select_windows_budgeted,select_with_paper_fallback,
    generate_candidate_windows,retained_figure_context,SCALES,ANCHORS)
from PIL import Image

class PaperMethodTests(unittest.TestCase):
    def test_relation_normalization_equalizes_groups_not_edge_counts(self):
        nodes=[{'node_id':n,'node_type':'paper'} for n in ['a','b','c','d']]
        edges=[{'source_id':'a','target_id':'d','edge_type':'r','confidence':.2},
               {'source_id':'b','target_id':'d','edge_type':'r','confidence':.8},
               {'source_id':'c','target_id':'d','edge_type':'s','confidence':.5}]
        graph=tensorize_graph(nodes,edges)
        self.assertEqual(len(graph.relation_names),4)
        weights=graph.weight[graph.target==3]
        self.assertTrue(torch.allclose(weights,torch.tensor([.1,.4,.5])))
        self.assertAlmostEqual(float(weights.sum()),1.)

    def test_relation_layer_matches_equation_seven(self):
        graph=tensorize_graph([{'node_id':str(i),'node_type':'chunk'} for i in range(2)],
            [{'source_id':'0','target_id':'1','edge_type':'link','confidence':1.}])
        layer=RelationLayer(3,len(graph.relation_names))
        with torch.no_grad():
            layer.self_projection.weight.copy_(torch.eye(3));layer.self_projection.bias.zero_()
            layer.neighbor_projection.weight.copy_(2*torch.eye(3));layer.relation_gate.weight.fill_(.5)
        h=torch.tensor([[1.,2.,3.],[3.,-1.,2.]])
        expected=F.layer_norm(h+F.gelu(h+h.flip(0)),(3,))
        self.assertTrue(torch.allclose(layer(h,graph),expected,atol=1e-6))
        self.assertIsNone(layer.neighbor_projection.bias)

    def test_masked_residual_cannot_copy_target_and_export_stays_text_anchored(self):
        nodes=[{'node_id':str(i),'node_type':'chunk'} for i in range(2)]
        graph=tensorize_graph(nodes,[])
        model=DocumentGraphEncoder(3,4,1,0)
        with torch.no_grad():
            for parameter in model.parameters(): parameter.zero_()
            model.output_projection.bias.copy_(torch.tensor([0.,1.,0.]))
        x=torch.tensor([[1.,0.,0.],[0.,0.,1.]])
        expected=F.normalize(x+.25*torch.tensor([0.,1.,0.]),dim=-1)
        self.assertTrue(torch.allclose(model(x,graph),expected))
        noisy=x.clone();noisy[0]=0
        self.assertTrue(torch.allclose(model(noisy,graph)[0],torch.tensor([0.,1.,0.])))

    def test_parameter_count_for_paper_full_schema(self):
        model=DocumentGraphEncoder(768,128,13,70)
        self.assertEqual(sum(p.numel() for p in model.parameters()),283392)
        for layer in model.layers:
            self.assertTrue(torch.equal(layer.relation_gate.weight,torch.ones_like(layer.relation_gate.weight)))

    def test_outer_rrf_and_constraint_intersection(self):
        left=[{'id':'a','rank_score':100},{'id':'b','rank_score':1}]
        right=[{'id':'c','rank_score':.99},{'id':'b','rank_score':.7},{'id':'a','rank_score':.1}]
        rows=fuse_ranked_rows(left,right,'id',.3,3,allowed={'a','b'})
        self.assertEqual({r['id'] for r in rows},{'a','b'})
        a=next(r for r in rows if r['id']=='a')
        self.assertAlmostEqual(a['rank_score'],.7+.3*61/63)
        c=fuse_ranked_rows(left,right,'id',.3,3)[-1]
        self.assertEqual(c['id'],'c');self.assertAlmostEqual(c['rank_score'],.3)

    def test_entity_channel_ties_survive_rank_limit(self):
        papers=[{'node_id':'paper:'+s,'paper_id':s,'node_type':'paper'} for s in ['a','b']]
        entity={'node_id':'dataset:x','node_type':'dataset'}
        nodes=papers+[entity]
        edges=[{'source_id':p['node_id'],'target_id':entity['node_id'],'edge_type':'uses','confidence':1} for p in papers]
        recall=GraphPaperRecall(nodes,edges,papers,[n['node_id'] for n in nodes],candidate_limit=1)
        ranked=recall.rank(np.array([.2,.1,.8]),2,['dataset'])
        for _,_,evidence in ranked:
            e=next(e for e in evidence if e['node_type']=='dataset')
            self.assertEqual(e['channel_rank'],1)
            self.assertAlmostEqual(e['rrf_contribution'],3/61)

    def test_repeated_content_uses_max_per_paper(self):
        papers=[{'node_id':'paper:a','paper_id':'a','node_type':'paper'}]
        chunks=[{'node_id':'chunk:'+str(i),'paper_id':'a','node_type':'chunk'} for i in range(4)]
        first=GraphPaperRecall(papers+chunks[:1],[],papers,['paper:a','chunk:0']).rank([.3,.8],1)
        duplicate=GraphPaperRecall(papers+chunks,[],papers,['paper:a']+[c['node_id'] for c in chunks]).rank([.3,.8,.8,.8,.8],1)
        self.assertAlmostEqual(first[0][1],duplicate[0][1])

    def test_figure_labels_use_complete_identifiers(self):
        self.assertEqual(figure_labels('Fig. 3 architecture'),{('figure','3')})
        self.assertFalse(figure_labels('Figure 3') & figure_labels('Figure 13: ablation'))

    def test_crop_uses_score_order_without_extra_efficiency_or_positive_cutoff(self):
        rows=[{'box':[0,0,40,40],'area_ratio':.16,'retain_score':-.1,'efficiency_score':0},
              {'box':[50,50,80,80],'area_ratio':.09,'retain_score':-.2,'efficiency_score':100}]
        selected=select_windows_budgeted(rows)
        self.assertEqual([r['retain_score'] for r in selected],[-.1,-.2])
        big={'box':[0,0,100,100],'area_ratio':1.,'retain_score':.3}
        selected,fallback=select_with_paper_fallback([big])
        self.assertTrue(fallback);self.assertEqual(selected[0]['box'],big['box'])

    def test_window_candidates_preserve_full_and_content_fallbacks(self):
        image=Image.new('RGB',(300,180),'white')
        content=(8,8,285,170)
        boxes=generate_candidate_windows(image,content)
        self.assertEqual(len(SCALES),28);self.assertEqual(len(ANCHORS),9)
        self.assertEqual(len(boxes),360);self.assertIn((0,0,300,180),boxes);self.assertIn(content,boxes)
        self.assertEqual(len(boxes),len(set(boxes)))

    def test_equation_15_zero_confidence(self):
        self.assertEqual(harmonic_mean_confidence({'confidence':0},{'confidence':.7}),0.)
        self.assertAlmostEqual(harmonic_mean_confidence({'confidence':.4},{'confidence':.8}),2*.4*.8/1.2)

    def test_equation_26_token_estimator(self):
        # Exercise both token classes in Eq. (26) using Unicode code points.
        sample = 'ABC 123 ' + chr(0x4E00) + chr(0x4E01) + ' !'
        self.assertEqual(estimate_token_count(sample),5)

    def test_crop_conditioning_has_only_final_retained_section_text(self):
        figure = {'node_id':'figure:p:1','section_id':'p:kept','caption':['Figure 3: Method.']}
        bundle = {'papers':[{'sections':[
            {'section_id':'p:other','section_summary':'Unrelated section text.',
             'chunks':[],'figures':[]},
            {'section_id':'p:kept','section_summary':'Retained summary.',
             'chunks':[{'node_id':'chunk:p:1','text':'Retained chunk text.'}],
             'figures':[{'node_id':'figure:p:1'}]}
        ]}], 'llm_context_markdown':'Retained summary. Retained chunk text.'}
        with tempfile.TemporaryDirectory(prefix='.test-graph-',dir=ROOT) as directory:
            graph = Path(directory).resolve()
            self.assertEqual(graph.parent,ROOT)
            (graph/'figures.jsonl').write_text(json.dumps(figure)+'\n',encoding='utf-8')
            selected,snippets,sources=retained_figure_context(bundle,graph)
        self.assertEqual(selected['node_id'],figure['node_id'])
        self.assertEqual(snippets,['Retained summary.','Retained chunk text.'])
        self.assertTrue(all(row['section_id']==figure['section_id'] for row in sources))
        self.assertTrue(all(text in bundle['llm_context_markdown'] for text in snippets))

if __name__=='__main__':
    unittest.main()
