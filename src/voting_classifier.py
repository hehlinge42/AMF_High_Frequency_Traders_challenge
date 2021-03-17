import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import pandas as pd

import logging
import logzero
from logzero import logger

from sklearn.linear_model import LogisticRegression

def voting_classifier(Xy_train, Xy_test):

    Xy_train.loc[Xy_train['type'] == 'HFT', 'type'] = 0
    Xy_train.loc[Xy_train['type'] == 'MIX', 'type'] = 1
    Xy_train.loc[Xy_train['type'] == 'NON HFT', 'type'] = 2

    y_train_cat = tf.keras.utils.to_categorical(Xy_train['type'])
    # voting_classifier = train_voting_classifier(Xy_train[['HFT', 'MIX', 'NON HFT']], y_train_cat, epochs=15)
    voting_classifier = train_voting_classifier(Xy_train[['HFT', 'MIX', 'NON HFT']], Xy_train[['type']].astype(int))
    
    # y_pred_cat = voting_classifier.predict(Xy_test[['HFT', 'MIX', 'NON HFT']])
    y_pred_cat = voting_classifier.predict_proba(Xy_test[['HFT', 'MIX', 'NON HFT']])
    # y_pred = np.argmax(y_pred_cat, axis=1)
    y_pred = voting_classifier.predict(Xy_test[['HFT', 'MIX', 'NON HFT']])

    Xy_test_voting = pd.DataFrame(data=y_pred, index=Xy_test.index, columns=['type'])
    Xy_test_voting_full =  pd.DataFrame(data=y_pred_cat, index=Xy_test.index, columns=['HFT', 'MIX', 'NON HFT'])
    Xy_test_voting_full['nb_observations'] = Xy_test['Sum']
    Xy_test_voting_full['type'] = Xy_test_voting_full[['HFT', 'MIX', 'NON HFT']].idxmax(axis=1)

    return Xy_test_voting, Xy_test_voting_full


def train_voting_classifier(X_train, y_train):

    clf = LogisticRegression(random_state=0, multi_class='multinomial')
    clf.fit(X_train.to_numpy(), y_train.to_numpy())
    return clf


# def train_voting_classifier(X_train, y_train, epochs):

#     def build_voting_classifier_model():
#         inputs = tf.keras.Input(shape=(3,))
#         x = tf.keras.layers.Dense(16, activation=tf.nn.relu)(inputs)
#         outputs = tf.keras.layers.Dense(3, activation=tf.nn.softmax)(x)
#         model = tf.keras.Model(inputs=inputs, outputs=outputs)
#         return model
    
#     voting_classifier_model = build_voting_classifier_model()
#     loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
#     metrics = tf.keras.metrics.CategoricalAccuracy()
  
#     voting_classifier_model.compile(optimizer='adam',
#                          loss=loss,
#                          metrics=metrics)

#     logger.info(f"X_train.shape = {X_train.shape}")
#     logger.info(f"y_train.shape = {y_train.shape}")
#     history = voting_classifier_model.fit(X_train, y_train, 32, epochs=epochs, validation_split=0.20)

#     return voting_classifier_model