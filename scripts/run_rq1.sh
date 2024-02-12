#!/bin/bash

resp_dirs='10312021 11092021'
fnames='10312021 11092021'

echo '############################################################'
echo 'Wilcoxon Signed Rank Test Statistics'
python -m hypothesis_testing --resp-dirs ${resp_dirs} --fnames ${fnames}\
  -rq 1 --what wsr
