import unittest
import itertools
import torch
from experiments.block_swap.core import pool,exchange,check,choose,fit

class Checks(unittest.TestCase):
    def test_all_binary_candidate_blocks(self):
        for bits in itertools.product((0,1),repeat=8):
            b=torch.tensor(bits).reshape(2,4).float();base=torch.cat((torch.ones(2,1),b,torch.zeros(2,2)),dim=1)
            case=dict(ids=torch.tensor([0]),candidate_ids=torch.arange(4),native_adjacency=base,seed=0,group=12)
            actions=pool(case);self.assertEqual(actions,pool(case))
            for action in actions:
                out=exchange(base,1,4,action);check(base,out,1,4,action)
                self.assertEqual(int(out[:,1+action['old']].sum()),0)
            if actions:
                self.assertTrue(all(a['rows']==actions[0]['rows'] and a['old']==actions[0]['old'] for a in actions))
    def test_matched_activation(self):
        actions=[dict(old=0,new=i,rows=[0,1]) for i in (1,2)]
        X=[[1.]*32,[-1.]*32];m=fit(X,[1.,-1.]);d=choose(m,X,actions,0,1)
        self.assertEqual(d['learned']['rows'],d['matched_random']['rows'])
        self.assertEqual(d['learned']['old'],d['matched_random']['old'])
        m['threshold']=100;d=choose(m,X,actions,0,1)
        self.assertIsNone(d['learned']);self.assertIsNone(d['matched_random'])
    def test_empty(self):
        self.assertIsNone(exchange(torch.zeros(1,1),0,0,None).grad_fn)
if __name__=='__main__':unittest.main()
