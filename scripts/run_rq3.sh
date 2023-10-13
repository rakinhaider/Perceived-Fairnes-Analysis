#!/bin/bash

echo '############################################################'
echo 'Pearson Correlation Statistics on between
risk perceptions and fairness preferences'

python -m hypothesis_testing --resp-dirs 10312021 11092021\
  --fnames 10312021 11092021 -rq 3 --what pearson

for criteria in group Ethnicity PGM; do
  python -m hypothesis_testing --resp-dirs 10312021 11092021\
    --fnames 10312021 11092021 -rq 3 --what pearson --criteria ${criteria}
done

