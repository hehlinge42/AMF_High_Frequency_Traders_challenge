import numpy as np
import pandas as pd

import logging
import logzero
from logzero import logger

from sklearn.linear_model import LogisticRegression
import tensorflow as tf

def voting_classifier(X_train, y_train_pred, X_test, y_pred, y_train_pack):

    X_train.reset_index(inplace=True)
    X_test.reset_index(inplace=True)
    X_train = pd.concat([X_train[['Trader']], pd.DataFrame(y_train_pred, columns=['HFT', 'MIX', 'NON HFT'])], axis=1)
    X_test = pd.concat([X_test[['Trader']], pd.DataFrame(y_pred, columns=['HFT', 'MIX', 'NON HFT'])], axis=1)

    X_train = X_train.groupby(by='Trader').mean()
    count = X_test.groupby(by='Trader').count()
    count.rename({'HFT': 'nb_observations'}, axis=1, inplace=True)
    X_test = X_test.groupby(by='Trader').mean()

    Xy_train = X_train.merge(y_train_pack, left_index=True, right_index=True)

    Xy_train.loc[Xy_train['type'] == 'HFT', 'type'] = 0
    Xy_train.loc[Xy_train['type'] == 'MIX', 'type'] = 1
    Xy_train.loc[Xy_train['type'] == 'NON HFT', 'type'] = 2

    y_train_cat = tf.keras.utils.to_categorical(Xy_train['type'])
    voting_classifier = train_voting_classifier(Xy_train[['HFT', 'MIX', 'NON HFT']], Xy_train[['type']].astype(int))
    
    y_pred_cat = voting_classifier.predict_proba(X_test[['HFT', 'MIX', 'NON HFT']])
    y_pred = voting_classifier.predict(X_test[['HFT', 'MIX', 'NON HFT']])

    Xy_test_voting = pd.DataFrame(data=y_pred, index=X_test.index, columns=['type'])
    Xy_test_voting.loc[Xy_test_voting['type'] == 0] = 'HFT'
    Xy_test_voting.loc[Xy_test_voting['type'] == 1] = 'MIX'
    Xy_test_voting.loc[Xy_test_voting['type'] == 2] = 'NON HFT'

    Xy_test_voting_full = pd.DataFrame(data=y_pred_cat, index=X_test.index, columns=['HFT', 'MIX', 'NON HFT'])
    Xy_test_voting_full = Xy_test_voting_full.merge(count[['nb_observations']], left_index=True, right_index=True)
    Xy_test_voting_full['type'] = Xy_test_voting_full[['HFT', 'MIX', 'NON HFT']].idxmax(axis=1)

    return Xy_test_voting, Xy_test_voting_full


def train_voting_classifier(X_train, y_train):

    clf = LogisticRegression(random_state=0, multi_class='multinomial')
    clf.fit(X_train.to_numpy(), y_train.to_numpy())
    return clf