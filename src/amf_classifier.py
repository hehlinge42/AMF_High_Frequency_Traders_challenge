import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import argparse
import os

import xgboost as xgb

import logging
import logzero
from logzero import logger

from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder

from preprocessor import *
from submit import submit
from pseudo_labelling import aggregate_traders, get_sure_traders, get_pseudo_labelled_data
from voting_classifier import voting_classifier

import warnings
warnings.filterwarnings("ignore")

def unpack_y(X, y):

    Xy = X.merge(y, how="inner", on=["Trader"])
    y = Xy['type']
    return y

def drop_duplicates_x(X):

    X_transformed = X.drop_duplicates()
    X_transformed.set_index(['Trader', 'Share', 'Day'], inplace=True)
    return X_transformed


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Cleaner and tokenizer of raw text stored as json file')
    parser.add_argument('-d', '--directory', type=str, help='path to the data directory')
    parser.add_argument('-s', '--submission', type=str, help='name of the submission')
    parser.add_argument('-l', '--loop', type=int, default=-1, help='number of loops')
    args = parser.parse_args()

    X_train = pd.read_csv(os.path.join(args.directory, "AMF_train_X.csv"), index_col=['Index'])
    X_test = pd.read_csv(os.path.join(args.directory, "AMF_test_X.csv"), index_col=['Index'])
    y_train = pd.read_csv(os.path.join(args.directory, "AMF_train_y.csv"))

    X_train = drop_duplicates_x(X_train)
    X_test = drop_duplicates_x(X_test)
    y_train = unpack_y(X_train, y_train)

    org_X_test = X_test.copy()
    org_X_train = X_train.copy()

    end_condition = False
    loop_counter = 1
    while end_condition is not True:

        preprocessor = FunctionTransformer(preprocess)

        model = xgb.XGBClassifier(objective='multi:softprob', colsample_bytree=0.7,
                                    learning_rate=0.05, n_estimators=100, max_depth=10,
                                    min_child_weight=3, subsample=0.8759, booster='gbtree')

        pipe = make_pipeline(preprocessor, model)
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        Xy_test = aggregate_traders(X_test, y_pred)

        sure_traders = get_sure_traders(Xy_test)
        X_train, y_train, X_test = get_pseudo_labelled_data(sure_traders, X_train, y_train, X_test)

        if args.loop != -1 and loop_counter == args.loop:
            end_condition = True
        elif args.loop == -1 and len(sure_traders) == 0:
            end_condition = True

        loop_counter += 1


    path_to_submission = os.path.join('..', 'data', args.submission + '.csv')
    path_to_full_submission = os.path.join('..', 'data', args.submission + '_full.csv')

    y_pred = pipe.predict(org_X_test)
    y_train_pred = pipe.predict(org_X_train)

    Xy_train = aggregate_traders(org_X_train, y_train_pred)
    Xy_test = aggregate_traders(org_X_test, y_pred)

    Xy_test_voting, Xy_test_voting_full = voting_classifier(Xy_train, Xy_test)
    logger.info(f"Writing file {path_to_full_submission}")
    Xy_test.to_csv(path_to_full_submission, index=True)

    logger.info(f"Writing file {path_to_submission}")
    Xy_test[['type']].to_csv(path_to_submission, index=True)

    # submit(Xy_test, path_to_submission, path_to_full_submission)    