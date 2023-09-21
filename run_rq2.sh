#!/bin/bash

resp_dirs='10312021 11092021'
fnames='10312021 11092021'

python -m hypothesis_testing --resp-dirs ${resp_dirs} --fnames ${fnames}\
  -rq 2 --what anova --qid Q201

python -m hypothesis_testing --resp-dirs ${resp_dirs} --fnames ${fnames}\
  -rq 2 --what tukey --qid Q201


python -m box_plots --resp-dirs ${resp_dirs} --fnames ${fnames} --qid IFPI
python -m box_plots --resp-dirs ${resp_dirs} --fnames ${fnames} --qid IFNI
python -m box_plots --resp-dirs ${resp_dirs} --fnames ${fnames} --qid SFPI
python -m box_plots --resp-dirs ${resp_dirs} --fnames ${fnames} --qid SFNI
python -m box_plots --resp-dirs ${resp_dirs} --fnames ${fnames} --qid BFPI
python -m box_plots --resp-dirs ${resp_dirs} --fnames ${fnames} --qid BFNI
