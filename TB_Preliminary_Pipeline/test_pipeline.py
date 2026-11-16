"""Software-only checks. Generated fixtures are NOT biological preliminary results."""
import tempfile, unittest
from pathlib import Path
import numpy as np
import pandas as pd
from tb_pipeline import guess_metadata,map_genes,numeric_matrix,threshold_at_sensitivity,safe_cv,parse_matrix

class Checks(unittest.TestCase):
 def test_conservative_labels(self):
  self.assertEqual(guess_metadata({'char_disease_state':'latent tuberculosis'})['label'],'LTBI')
  self.assertEqual(guess_metadata({'char_disease_state':'non-TB pneumonia'})['label'],'OD')
  self.assertEqual(guess_metadata({'char_disease_state':'active tuberculosis'})['label'],'TB')
  self.assertEqual(guess_metadata({'title':'unexplained fever'})['label'],'UNKNOWN')
 def test_ambiguous_probes_rejected(self):
  x=pd.DataFrame({'s':[1.,3.,5.,7.]},index=['a','b','c','d'])
  z=map_genes(x,pd.Series({'a':'G1','b':'G1','c':'G2 /// G3','d':''}))
  self.assertEqual(z.index.tolist(),['G1']);self.assertEqual(z.loc['G1','s'],2.)
 def test_rank_is_sample_local(self):
  x=pd.DataFrame({'a':[1.,2.,3.],'b':[10.,0.,-1.]})
  a=numeric_matrix(x);b=numeric_matrix(x[['a']]);np.testing.assert_allclose(a[0],b[0])
 def test_threshold(self):
  y=np.array([0,0,1,1]);p=np.array([.1,.4,.3,.9]);t=threshold_at_sensitivity(y,p,1.)
  self.assertEqual(t,.3);self.assertEqual(((p>=t)&(y==1)).sum(),2)
 def test_group_separation(self):
  y=np.tile([0,1],20);g=np.repeat(np.arange(20),2)
  for t,v in safe_cv(y,g,42):self.assertFalse(set(g[t])&set(g[v]))
 def test_geo_parser(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'m.txt';p.write_text('!Sample_geo_accession\t"GSM1"\n!Sample_title\t"sample"\n!Sample_characteristics_ch1\t"disease state: active tuberculosis"\n!Sample_platform_id\t"GPL1"\n!series_matrix_table_begin\n"ID_REF"\t"GSM1"\n"p1"\t2.0\n!series_matrix_table_end\n')
   x,m=parse_matrix(p);self.assertEqual(x.loc['p1','GSM1'],2.);self.assertEqual(m.iloc[0].char_disease_state,'active tuberculosis')
if __name__=='__main__':unittest.main()
