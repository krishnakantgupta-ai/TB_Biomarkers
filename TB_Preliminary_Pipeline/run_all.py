#!/usr/bin/env python3
"""Run the source-checked GSE37250 pilot. No generated/artificial research data."""
import argparse
from pathlib import Path
import subprocess
import sys
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--work',default='tb_work')
a=p.parse_args()
script=Path(__file__).resolve().with_name('tb_pipeline.py')
for task in [['fetch','--series','GSE37250'],['prepare'],['curate37250'],['analyze']]:
 print('\nRUNNING:',task[0],flush=True)
 subprocess.run([sys.executable,str(script),'--work',a.work]+task,check=True)
print('\nFinished. Open:',Path(a.work).resolve()/'results/PRELIMINARY_REPORT.md')
