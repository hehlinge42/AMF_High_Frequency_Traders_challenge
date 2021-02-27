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
    args = parser.parse_args()

    X_train = pd.read_csv(os.path.join(args.directory, "AMF_train_X.csv"), index_col=['Index'])
    X_test = pd.read_csv(os.path.join(args.directory, "AMF_test_X.csv"), index_col=['Index'])
    y_train = pd.read_csv(os.path.join(args.directory, "AMF_train_y.csv"))

    X_train = drop_duplicates_x(X_train)
    X_test = drop_duplicates_x(X_test)

    y_train = unpack_y(X_train, y_train)

    preprocessor = FunctionTransformer(preprocess)

    model = xgb.XGBClassifier(objective='multi:softprob', colsample_bytree=0.7,
                                learning_rate=0.05, n_estimators=10, max_depth=10,
                                min_child_weight=3, subsample=0.8759, booster='gbtree')

    pipe = make_pipeline(preprocessor, model)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    X_test.reset_index(inplace=True)
    X_test['Prediction'] = y_pred

    Xy_test = X_test[['Prediction']]
    Xy_test = pd.get_dummies(Xy_test.Prediction)
    Xy_test['Trader'] = X_test['Trader']

    Xy_test = Xy_test.groupby(['Trader']).sum()
    Xy_test['Sum'] = Xy_test['MIX'] + Xy_test['NON HFT'] + Xy_test['HFT']
    Xy_test['MIX'] /= Xy_test['Sum']
    Xy_test['NON HFT'] /= Xy_test['Sum']
    Xy_test['HFT'] /= Xy_test['Sum']
    Xy_test['type'] = np.nan

    mask_mxt = Xy_test['MIX'] > 0.5
    mask_hft = Xy_test['HFT'] > 0.85
    Xy_test.loc[mask_mxt, 'type'] = 'MIX'
    Xy_test.loc[mask_hft, 'type'] = 'HFT'
    Xy_test.fillna('NON HFT', inplace=True)
    logger.info(f"\n{Xy_test.head()}")

    # Xy_test = pd.DataFrame(Xy_test, columns=['type'], index=Xy_test.index)
    Xy_test[['type']].to_csv("../data/claqué_au_sol2.csv", index=True)
    