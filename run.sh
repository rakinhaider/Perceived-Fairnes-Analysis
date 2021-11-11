#!/bin/bash

data_dirs='10082021'
fnames='10082021'
qid='Q10.14'
xz_qid='Q199'
criteria='scenario Ethnicity'
what='choice'

#python -m survey_response_aggregator --data-dirs ${data_dirs}\
#	--fnames ${fnames}

#echo "#######################################################"

#python -m probability --qid ${qid} --xz-qid ${xz_qid} --criteria ${criteria}

#echo "#######################################################"

#python -m hypothesis_testing --qid ${qid} --criteria ${criteria} --what ${what}

#echo "#######################################################"

#python -m plotter --qid ${qid} --criteria ${criteria} --data-dirs ${data_dirs}

python -m combine --qid ${qid} --xz-qid ${xz_qid} --criteria ${criteria}\
	--data-dirs ${data_dirs} --fnames ${fnames} --what ${what}