#!/bin/bash

resp_dirs='10312021 11092021'
fnames='10312021 11092021'

python -m hypothesis_testing --resp-dirs ${resp_dirs}\
  --fnames ${fnames} --what normal-check --qid Q10.20

python -m hypothesis_testing --resp-dirs ${resp_dirs}\
  --fnames ${fnames} --what normal-check --qid Q201 -xy x

python -m hypothesis_testing --resp-dirs ${resp_dirs}\
  --fnames ${fnames} --what normal-check --qid Q201 -xy y

echo "##################### Risk Perceptions #################"

for qid in IFPI IFNI SFPI SFNI BFNI BFPI; do
  python -m hypothesis_testing --resp-dirs ${resp_dirs}\
    --fnames ${fnames} --what normal-check --qid ${qid}
done