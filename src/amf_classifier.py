import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import argparse
import os

import xgboost as xgb
import tensorflow as tf
import tensorflow_addons as tfa

import logging
import logzero
from logzero import logger

from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder

from tensorflow import keras
from tensorflow.keras import layers
from official.nlp import optimization  # to create AdamW optmizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

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

def feature_engineer(X):

    X['Nb_trade'] = X['NbSecondWithAtLeatOneTrade'] * X['MeanNbTradesBySecond']
    X['min_vs_max_two_events'] = X['max_time_two_events'] / X['min_time_two_events']
    X['10_vs_90_two_events'] = X['90_p_time_two_events'] / X['10_p_time_two_events']
    X['25_vs_75_two_events'] = X['75_p_time_two_events'] / X['25_p_time_two_events']
    X['min_vs_max_cancel'] = X['max_lifetime_cancel'] / X['min_lifetime_cancel']
    X['10_vs_90_two_cancel'] = X['90_p_lifetime_cancel'] / X['10_p_lifetime_cancel']
    X['25_vs_75_two_cancel'] = X['75_p_lifetime_cancel'] / X['25_p_lifetime_cancel']
    return X


def create_model():

    classifier_model = keras.Sequential(
    [
        layers.Dense(20, input_dim=24, activation="relu", name="layer1"),
        layers.Dense(3, activation='softmax', name="layer2"),
    ])

    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    metrics = tfa.metrics.F1Score(num_classes=3)
    
    classifier_model.compile(optimizer='adam',
                         loss=loss,
                         metrics=metrics)

    return classifier_model



def train_model(X_train, y_train_cat, epochs=15):

    def build_classifier_model():
        inputs = tf.keras.layers.Input(shape=())
        x = tf.keras.layers.Dense(20, input_dim=X_train.shape[1], activation='relu')
        # x = tf.keras.layers.Dropout(0.1)(x)
        outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
        return tf.keras.Model(inputs, outputs)

    classifier_model = keras.Sequential(
    [
        layers.Dense(20, input_dim=X_train.shape[1], activation="relu", name="layer1"),
        layers.Dense(3, activation='softmax', name="layer2"),
    ])

    # classifier_model = build_classifier_model()
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    metrics = tfa.metrics.F1Score(num_classes=3)

    steps_per_epoch = len(X_train)
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1 * num_train_steps)

    init_lr = 3e-5
    optimizer = optimization.create_optimizer(init_lr=init_lr,
                                            num_train_steps=num_train_steps,
                                            num_warmup_steps=num_warmup_steps,
                                            optimizer_type='adamw')
    
    classifier_model.compile(optimizer=optimizer,
                         loss=loss,
                         metrics=metrics)

    # history = classifier_model.fit(X_train, y_train_cat, 32, epochs=epochs, validation_split=0.20)

    return classifier_model, optimizer



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

    X_train = feature_engineer(X_train)
    X_test = feature_engineer(X_test)
    logger.info(f"X_train shape {X_train.shape}, X_test.shape = {X_test.shape}")

    org_X_test = X_test.copy()
    org_X_train = X_train.copy()

    end_condition = False
    loop_counter = 1
    while end_condition is not True:

        preprocessor = FunctionTransformer(preprocess)

        model = xgb.XGBClassifier(objective='multi:softprob', colsample_bytree=0.2,
                                    learning_rate=0.05, n_estimators=1000, max_depth=10,
                                    min_child_weight=3, subsample=0.8759, booster='gbtree')

        pipe = make_pipeline(preprocessor, model)
        pipe.fit(X_train, y_train)
        logger.info(f"X_train.head()\n{X_train.head()}")

        y_pred = pipe.predict_proba(X_test)
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

    y_pred = pipe.predict_proba(org_X_test)
    y_train_pred = pipe.predict_proba(org_X_train)

    Xy_train = aggregate_traders(org_X_train, y_train_pred)
    Xy_test = aggregate_traders(org_X_test, y_pred)

    Xy_test_voting, Xy_test_voting_full = voting_classifier(Xy_train, Xy_test)
    logger.info(f"Writing file {path_to_full_submission}")
    Xy_test.to_csv(path_to_full_submission, index=True)

    logger.info(f"Writing file {path_to_submission}")
    Xy_test[['type']].to_csv(path_to_submission, index=True)

    # submit(Xy_test, path_to_submission, path_to_full_submission)    