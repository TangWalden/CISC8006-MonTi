"""Synthetic invariants; no dataset, checkpoint, model or victim calls."""
import unittest
import numpy as np
import torch
from experiments.local_swap.core import proposal_pool,exchange,assert_exchange,fit,predict,choose

class Checks(unittest.TestCase):
    def test_exhaustive_geometry(self):
        for bits in range(256):
            block=[[int(bool(bits&(1<<(r*4+c)))) for c in range(4)] for r in range(2)]
            a=torch.cat([torch.ones(2,1),torch.tensor(block),torch.zeros(2,2)],1)
            pool=proposal_pool(block,0,19,999)
            expected=sum(sum(row)*(4-sum(row)) for row in block)
            self.assertEqual(len(pool),expected);self.assertEqual(len({tuple(v) for v in pool}),expected)
            for v in pool:assert_exchange(a,exchange(a,1,4,v),1,4,v)
    def test_determinism(self):
        b=[[0,1,0,1],[1,0,0,1]]
        self.assertEqual(proposal_pool(b,2,5),proposal_pool(b,2,5))
    def test_invalid_swap(self):
        a=torch.tensor([[1.,0.,1.,0.]])
        with self.assertRaises(ValueError):exchange(a,1,2,[0,0,1])
    def test_rank_and_antisymmetry(self):
        rng=np.random.RandomState(2);x=rng.normal(size=(200,32));y=x[:,0]*.03
        m=fit(x,y);p=predict(m,x)
        self.assertGreater(np.corrcoef(p,y)[0,1],.99)
        np.testing.assert_allclose(predict(m,-x),-p,rtol=0,atol=0)
    def test_gate_and_matched_count(self):
        x=np.eye(32);m=fit(x,np.ones(32));pool=[[0,0,1],[0,0,2]]
        for xx in [x[:2],-x[:2]]:
            d=choose(m,xx,pool,0,1)
            self.assertEqual(d['learned'] is None,d['matched_random'] is None)
        self.assertIsNone(choose(m,-x[:2],pool,0,1)['learned'])
    def test_empty_pool_denominator(self):
        m=fit(np.eye(32),np.zeros(32));d=choose(m,np.zeros((0,32)),[],1,2)
        self.assertFalse(d['active']);self.assertIsNone(d['always_random'])

if __name__=='__main__':unittest.main()
