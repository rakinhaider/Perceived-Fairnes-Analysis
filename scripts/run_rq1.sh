#!/bin/bash

resp_dirs='10312021 11092021'
fnames='10312021 11092021'

# echo '############################################################'
# echo 'ANOVA Test Statistics'
# python -m hypothesis_testing --resp-dirs ${resp_dirs} --fnames ${fnames}\
#  -rq 1 --what anova
# python -m hypothesis_testing --resp-dirs ${resp_dirs} --fnames ${fnames}\
#   -rq 1 --what tukey

echo '############################################################'
echo 'Paired t-Test Statistics'
python -m hypothesis_testing --resp-dirs ${resp_dirs} --fnames ${fnames}\
  -rq 1 --what pairedt
