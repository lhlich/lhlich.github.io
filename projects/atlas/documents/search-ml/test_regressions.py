"""Regression and independent-oracle tests for the consolidated contracts."""
import unittest
import numpy as np
import search_drills as D

class RegressionTests(unittest.TestCase):
    def test_metric_duplicates_and_negative_cutoff(self):
        with self.assertRaises(ValueError): D.ndcg_at_k(['a','a'], {'a':1}, 2)
        for f in [lambda: D.dcg_at_k([1],-1), lambda: D.recall_at_k(['a'],{'a'},-1), lambda: D.top_k_indices([1],-1)]:
            with self.assertRaises(ValueError): f()
        self.assertIsNone(D.ndcg_at_k(['a'],{},1))
        self.assertIsNone(D.recall_at_k([],set(),1))
    def test_topk_against_full_stable_sort(self):
        rng=np.random.default_rng(7)
        for n in [0,1,10,200]:
            scores=rng.integers(-3,4,n).astype(float)
            for k in [0,1,5,n,n+1]:
                gold=sorted(range(n),key=lambda i:(-scores[i],i))[:k]
                self.assertEqual(D.top_k_indices(scores,k).tolist(),gold)
        self.assertEqual(D.top_k_indices(np.ones(10000),3).tolist(),[0,1,2])
    def test_context_requires_counter_and_counts_serialization(self):
        with self.assertRaises(ValueError): D.pack_context([('x','文'*100)],20)
        # Character count is an explicit synthetic fixture, not model tokenization.
        seen=[]
        def count(s): seen.append(s); return len(s)
        self.assertEqual(D.pack_context([('a','文'),('b','字')],10,count_tokens=count),['a'])
        self.assertEqual(seen[-1],'[a]\n文\n\n[b]\n字')
        self.assertEqual(D.pack_context([('a','long text'),('b','x')],5,count_tokens=len),['b'])
    def test_quantized_block_scores_against_full_oracle(self):
        rng=np.random.default_rng(5)
        X=rng.normal(size=(103,13)).astype('float32'); q=rng.normal(size=13).astype('float32')
        c,s=D.quantize_int8(X); qc,qs=D.quantize_int8(q)
        gold=(c.astype('int64')@qc[0].astype('int64')).astype(float)*s.astype(float)*float(qs[0])
        for block in [1,7,200]:
            ids,scores=D.int8_search(q,c,s,11,block_rows=block)
            np.testing.assert_allclose(scores,gold,rtol=1e-12)
            self.assertEqual(ids.tolist(), sorted(range(len(X)),key=lambda i:(-gold[i],i))[:11])
    def test_quantized_ranking_can_differ(self):
        rng=np.random.default_rng(20); mismatch=False
        for _ in range(500):
            X=rng.normal(size=(20,8)).astype('float32'); q=rng.normal(size=8).astype('float32')
            c,s=D.quantize_int8(X); ids,_=D.int8_search(q,c,s,3)
            if ids.tolist()!=D.top_k_indices(X@q,3).tolist(): mismatch=True; break
        self.assertTrue(mismatch)
    def test_empty_scan_and_mmr(self):
        ids,scores=D.int8_search([1,2],np.empty((0,2),dtype='int8'),np.empty(0),3)
        self.assertEqual(len(ids),0);self.assertEqual(len(scores),0)
        self.assertEqual(D.mmr([1,0],[],3),[])
    def test_levenshtein_independent_full_dp(self):
        def gold(a,b):
            table=[[0]*(len(b)+1) for _ in range(len(a)+1)]
            for i in range(len(a)+1):table[i][0]=i
            for j in range(len(b)+1):table[0][j]=j
            for i in range(1,len(a)+1):
                for j in range(1,len(b)+1):
                    table[i][j]=min(table[i-1][j]+1,table[i][j-1]+1,table[i-1][j-1]+(a[i-1]!=b[j-1]))
            return table[-1][-1]
        words=['','a','ab','ba','aba','kitten','sitting','文書']
        for a in words:
            for b in words: self.assertEqual(D.levenshtein(a,b),gold(a,b))
    def test_mmr_shape_contract(self):
        with self.assertRaises(ValueError): D.mmr([[1,0]],[[1,0],[0,1]],1)
        with self.assertRaises(ValueError): D.mmr([1,0],[[1,0,0]],1)
    def test_memory_and_adapter_arithmetic(self):
        self.assertEqual(2*32*4096*8*128*2,512*1024**2)
        self.assertEqual(8*(4096+4096),65536)
        self.assertEqual(4096**2//65536,256)

if __name__=='__main__': unittest.main()
