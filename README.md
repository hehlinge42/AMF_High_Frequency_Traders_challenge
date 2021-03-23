# Who are the high-frequency traders? 
by Autorité des Marchés Financiers

This repository provides the code related to our participation to the data challenge "Who are the high-frequency traders" issued by the French Authority of Financial Markets. The aim of this challenge is to build a model able to automatically assign traders to one of the three classes of player: High-Frequency Trader (robot), Non High_Frequency-Trader (human) or MIX (using both techniques).

More information about this challenge on: https://challengedata.ens.fr/participants/challenges/50/

## Contents

The repository contains the following files:
* ```AMF_High_Frequency_Traders_report.pdf```: A report explaining our methodology step by step.
* ```data/*```: the source data provided by the AMF and our final submission
* ```src/*```: the ```.py``` source files 
* ```notebooks/*```: an Exploratory Data Analysis notebook providing some data vizualization
* ```requirements.txt```: the list of packages required to execute the program

## Setup

Our final submission to this challenge can be recovered by launching the following commands in the repository on a UNIX terminal:
```
git clone https://github.com/hehlinge42/AMF_High_Frequency_Traders_challenge.git
cd AMF_High_Frequency_Traders_challenge
pip install -r requirements.txt
python3 src/amf_classifier.py --directory [path] --submission [submission_name] --loop [nloops]
```

The program accepts the following options:
* -d or –directory [directory]: path to submission directory.
* -s or –submission [submission name]: base name of submission. The program will output the file directory/submission name.csv, formatted to be submitted, and the file directory/submission name full.csv which outputs the probability of each class for each trader.
* -l or –loop [nloops]: the maximum number of iterations of the pseudo-labeller. If set to -1, the model will be retrained until the pseudo-labeller finds no trader in the testing set with a sufficient confidence level.

## Contributors

Project realized by @MaximeRedstone, @NicolasMB1996 and @hehlinge42
