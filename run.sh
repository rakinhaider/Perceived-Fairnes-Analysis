#!/bin/bash

# resp_dirs='09202021 10082021 10312021 11092021'
# resp_dirs='09202021 10082021'
resp_dirs='10312021 11092021'
# fnames='Pilot21_v2.0_10082021 10082021 10312021 11092021'
# fnames='Pilot21_v2.0_10082021 10082021'
fnames='10312021 11092021'
qid='Q10.20'
xz_qid='Q201'
criteria='scenario Ethnicity Sex'
what='choice'
drop_males='--drop-males'

#python -m survey_response_aggregator --resp-dirs ${resp_dirs}\
#	--fnames ${fnames}

#echo "#######################################################"

#python -m probability --qid ${qid} --xz-qid ${xz_qid} --criteria ${criteria}

#echo "#######################################################"

#python -m hypothesis_testing --qid ${qid} --criteria ${criteria} --what ${what}

#echo "#######################################################"

#python -m plotter --qid ${qid} --criteria ${criteria} --resp-dirs ${resp_dirs}

python -m combine --qid ${qid} --xz-qid ${xz_qid} --criteria ${criteria}\
	--resp-dirs ${resp_dirs} --fnames ${fnames} --what ${what} ${drop_males}

# python -m combine --criteria ${criteria} --resp-dirs ${resp_dirs}\
#  	--fnames ${fnames} --what 'model_fair'

# python -m combine --criteria ${criteria} --resp-dirs ${resp_dirs}\
# 	--fnames ${fnames} --what 'model_bias'

# python -m combine --criteria ${criteria} --resp-dirs ${resp_dirs}\
#  	--fnames ${fnames} --what 'cd'

# python -m hypothesis_testing --resp-dirs 09202021 10082021 10312021 11092021
# --fnames Pilot21_v2.0_10082021 10082021 10312021 11092021 --what kendall
