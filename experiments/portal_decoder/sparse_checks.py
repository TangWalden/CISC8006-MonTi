"""Regression: sparse row indices must be global columns for CSR and CSC."""
import unittest
import torch,numpy as np,scipy.sparse as sp
from types import SimpleNamespace
from geometry_hard import proposals
class Checks(unittest.TestCase):
 def test_csr_csc_same_legal_remote_pool(self):
  adj=sp.csr_matrix((np.ones(4),([0,1,1,2],[1,0,2,1])),shape=(6,6));two=(adj@adj).astype(bool).astype(int);two.setdiag(0);two.eliminate_zeros()
  case=dict(ids=torch.tensor([0]),candidate_ids=torch.tensor([1]),source_ids=[6],native_adjacency=torch.tensor([[1.,1.,0.]]),seed=0,group=1)
  results=[]
  for matrix in (two.tocsr(),two.tocsc()):
   view=SimpleNamespace(adj=adj,adj2=matrix,d=torch.zeros(6,7,1))
   donor,bank=proposals(case,view,np.array([2,3,4,5]));self.assertEqual(donor,0);self.assertNotIn(2,bank);self.assertEqual(set(bank),{3,4,5});results.append(bank)
  self.assertEqual(*results)
if __name__=='__main__':unittest.main()
