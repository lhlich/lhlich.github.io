"""Independent numerical and invariant checks. Run: python test_exercises.py"""
import unittest
import numpy as np
from exercises import MatrixSum, top_k, softmax_ce, linear_step, in_batch_loss, self_attention


def finite_diff(f, a, eps=1e-6):
    g = np.zeros_like(a, dtype=float)
    for index in np.ndindex(a.shape):
        hi, lo = a.copy(), a.copy()
        hi[index] += eps
        lo[index] -= eps
        g[index] = (f(hi) - f(lo)) / (2 * eps)
    return g


class StudyTests(unittest.TestCase):
    def test_all_rectangles_against_direct_sum(self):
        rng = np.random.default_rng(42)
        for rows, cols in [(1, 1), (1, 5), (5, 1), (3, 5), (5, 3)]:
            a = rng.integers(-9, 10, size=(rows, cols))
            model = MatrixSum(a.tolist())
            for r1 in range(rows):
                for r2 in range(r1, rows):
                    for c1 in range(cols):
                        for c2 in range(c1, cols):
                            self.assertEqual(model.sum(r1,c1,r2,c2),
                                             a[r1:r2+1,c1:c2+1].sum())

    def test_empty_and_invalid_matrix_contract(self):
        for a in [[], [[]]]:
            with self.assertRaises(ValueError):
                MatrixSum(a).sum(0,0,0,0)
        with self.assertRaises(ValueError):
            MatrixSum([[1], [1, 2]])
        for rect in [(-1,0,0,0), (0,0,1,0), (0,1,0,0)]:
            with self.assertRaises(ValueError):
                MatrixSum([[1]]).sum(*rect)

    def test_matrix_snapshot_and_example(self):
        a = [[1,-2,3], [4,5,6]]
        m = MatrixSum(a)
        a[0][0] = 999
        self.assertEqual(m.sum(0,0,1,2),17)
        self.assertEqual(m.sum(0,1,1,2),12)

    def test_heap_against_stable_full_sort(self):
        rng = np.random.default_rng(43)
        for n in [0,1,2,20,100]:
            items = [(object(), int(x)) for x in rng.integers(-3,4,n)]
            for k in [0,1,2,9,120]:
                self.assertEqual(top_k(iter(items),k),
                                 sorted(items,key=lambda a:a[1],reverse=True)[:k])

    def test_heap_invalid_input(self):
        with self.assertRaises(ValueError):
            top_k([], -1)
        for x in [float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                top_k([('a',x)],1)

    def test_ce_known_numbers(self):
        loss, grad = softmax_ce([[2.,1.,0.]], [0])
        self.assertAlmostEqual(loss,0.4076059644,places=9)
        np.testing.assert_allclose(grad, [[-.33475904,.24472847,.09003057]], atol=1e-8)

    def test_ce_extreme_logits(self):
        loss, grad = softmax_ce([[1000,0,-1000]],[2])
        self.assertEqual(loss,2000)
        np.testing.assert_allclose(grad,[[1,0,-1]])

    def test_ce_shift_invariance(self):
        z=np.array([[2.,1.,0.],[.3,-.1,.5]])
        l,g=softmax_ce(z,[0,2]); l2,g2=softmax_ce(z+1000,[0,2])
        self.assertAlmostEqual(l,l2)
        np.testing.assert_allclose(g,g2)
        np.testing.assert_allclose(g.sum(axis=1),0,atol=1e-15)

    def test_ce_gradient(self):
        z=np.array([[.1,.3,-.5],[2.,1.,.4]])
        _,g=softmax_ce(z,[1,0])
        np.testing.assert_allclose(g,finite_diff(lambda a:softmax_ce(a,[1,0])[0],z),atol=1e-9)

    def test_ce_rejects_invalid_shape_and_labels(self):
        for z,y in [([],[]), ([[1,2]],[2]), ([[1,2]],[0.0]), ([[np.inf,0]],[0])]:
            with self.assertRaises(ValueError):
                softmax_ce(z,y)

    def test_linear_step_decreases_objective(self):
        x=np.array([[1.,0.],[0.,1.],[1.,1.]])
        w=np.array([[.2,-.1],[-.3,.4]]); b=np.zeros(2); y=np.array([0,1,0])
        initial,nw,nb=linear_step(x,y,w,b,lr=.05,l2=.1)
        final=softmax_ce(x@nw+nb,y)[0]+.05*np.sum(nw*nw)
        self.assertLess(final,initial)

    def test_embedding_gradients(self):
        q=np.array([[.2,.4],[-.1,.3],[.6,-.2]])
        d=np.array([[.4,-.3],[.7,.1],[-.2,.5]])
        _,gq,gd=in_batch_loss(q,d,.7)
        np.testing.assert_allclose(gq,finite_diff(lambda a:in_batch_loss(a,d,.7)[0],q),atol=1e-9)
        np.testing.assert_allclose(gd,finite_diff(lambda a:in_batch_loss(q,a,.7)[0],d),atol=1e-9)

    def test_attention_mask_and_row_sums(self):
        q=np.eye(2); v=np.array([[10.,0.],[0.,20.]])
        out,weights=self_attention(q,q,v,np.tril(np.ones((2,2),dtype=bool)))
        np.testing.assert_allclose(out[0],[10,0])
        np.testing.assert_allclose(weights.sum(axis=1),1)
        self.assertEqual(weights[0,1],0)
        np.testing.assert_allclose(out[1],[3.30238451,13.39523099],atol=1e-7)

    def test_attention_rejects_all_masked_row(self):
        with self.assertRaises(ValueError):
            self_attention(np.eye(2),np.eye(2),np.eye(2),np.zeros((2,2),dtype=bool))

    def test_guide_arithmetic(self):
        idf=np.log(1+90.5/10.5)
        self.assertAlmostEqual(idf,2.2637452597,places=9)
        self.assertAlmostEqual(4.4/3.2,1.375)
        dcg=lambda r:sum((2**g-1)/np.log2(i+2) for i,g in enumerate(r))
        value=dcg([0,2,1,0,3])/dcg([3,2,1,0,0])
        self.assertAlmostEqual(value,.543,places=3)
        scores={d:0 for d in 'ABCD'}
        for seq in ['ABC','BDA']:
            for i,d in enumerate(seq,1): scores[d]+=1/(60+i)
        self.assertEqual(sorted(scores,key=scores.get,reverse=True),list('BADC'))


if __name__=='__main__':
    unittest.main(verbosity=2)
