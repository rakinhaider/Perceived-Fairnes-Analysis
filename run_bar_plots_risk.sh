#!/bin/bash

for type in IFPI SFPI BFPI IFNI SFNI BFNI; do
  python -m bar_plots_risk_perceptions --resp-dirs 10312021 11092021\
    --fnames 10312021 11092021 --qid ${type}
done

