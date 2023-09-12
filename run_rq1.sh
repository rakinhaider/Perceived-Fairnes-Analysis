#!/bin/bash

resp_dirs='10312021 11092021'
fnames='10312021 11092021'

python -m hypothesis_testing --resp-dirs ${resp_dirs} --fnames ${fnames} \
  -rq 1 --what anova --qid Q10.20 --criteria scenario

python -m hypothesis_testing --resp-dirs ${resp_dirs} --fnames ${fnames} \
  -rq 1 --what anova --qid Q10.20 --criteria scenario group

python -m hypothesis_testing --resp-dirs ${resp_dirs} --fnames ${fnames} \
  -rq 1 --what anova --qid Q10.20 --criteria scenario Ethnicity

python -m hypothesis_testing --resp-dirs ${resp_dirs} --fnames ${fnames} \
  -rq 1 --what anova --qid Q10.20 --criteria scenario PGM


python -m hypothesis_testing --resp-dirs ${resp_dirs} --fnames ${fnames} \
  -rq 1 --what tukey --qid Q10.20 --criteria scenario

python -m hypothesis_testing --resp-dirs ${resp_dirs} --fnames ${fnames} \
  -rq 1 --what tukey --qid Q10.20 --criteria scenario group

python -m hypothesis_testing --resp-dirs ${resp_dirs} --fnames ${fnames} \
  -rq 1 --what tukey --qid Q10.20 --criteria scenario Ethnicity

python -m hypothesis_testing --resp-dirs ${resp_dirs} --fnames ${fnames} \
  -rq 1 --what tukey --qid Q10.20 --criteria scenario PGM