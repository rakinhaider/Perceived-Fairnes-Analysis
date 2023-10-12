#!/bin/bash

resp_dirs='10312021 11092021'
fnames='10312021 11092021'

function run_tests() {
  test_name=$1
  qid=$2
  x_or_y=$3
  python -m hypothesis_testing --resp-dirs ${resp_dirs} --fnames ${fnames}\
    -rq 2 --what ${test_name} --qid ${qid} ${x_or_y}
  for criteria in group Ethnicity PGM; do
    python -m hypothesis_testing --resp-dirs ${resp_dirs} --fnames ${fnames}\
    -rq 2 --what ${test_name} --criteria ${criteria} --qid ${qid} ${x_or_y}
  done
}

echo '############################################################'
echo 'ANOVA Test Statistics on XY Preferences'
# python -m hypothesis_testing --resp-dirs ${resp_dirs} --fnames ${fnames}\
#   -rq 2 --what anova --qid 10.20
# for criteria in group Ethnicity PGM; do
#   python -m hypothesis_testing --resp-dirs ${resp_dirs} --fnames ${fnames}\
#   -rq 2 --what anova --criteria ${criteria} --qid Q10.20
# done
run_tests 'anova' 'Q10.20'

echo '############################################################'
echo 'Paired Tukey Test Significant Statistics on XY Preferences'
run_tests 'tukey' 'Q10.20'

python -m bar_plots_mean_preferences --resp-dirs ${resp_dirs}\
  --fnames ${fnames} --qid Q10.20

echo '############################################################'
echo 'ANOVA Test Statistics on  XZ Preferences'
run_tests 'anova' 'Q201' '--x-or-y X'

echo '############################################################'
echo 'Paired Tukey Test Significant Statistics on XZ Preferences'
run_tests 'tukey' 'Q201' '--x-or-y X'

python -m bar_plots_mean_preferences --resp-dirs ${resp_dirs}\
  --fnames ${fnames} --qid Q201 --x-or-y X

echo '############################################################'
echo 'ANOVA Test Statistics on  YZ Preferences'
run_tests 'anova' 'Q201' '--x-or-y Y'

echo '############################################################'
echo 'Paired Tukey Test Significant Statistics on YZ Preferences'
run_tests 'tukey' 'Q201' '--x-or-y Y'

python -m bar_plots_mean_preferences --resp-dirs ${resp_dirs}\
  --fnames ${fnames} --qid Q201 --x-or-y Y
