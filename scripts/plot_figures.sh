#!/bin/bash

resp_dirs='10312021 11092021'
fnames='10312021 11092021'

echo '############################################################'
python -m plotters.bar_plots_mean_preferences --resp-dirs ${resp_dirs}\
    --fnames ${fnames} --qid Q10.20

python -m plotters.bar_plots_mean_preferences --resp-dirs ${resp_dirs}\
    --fnames ${fnames} --qid Q201 -xy X

python -m plotters.bar_plots_mean_preferences --resp-dirs ${resp_dirs}\
    --fnames ${fnames} --qid Q201 -xy Y
