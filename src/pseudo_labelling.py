import pandas as pd

import logging
import logzero
from logzero import logger


def get_pseudo_labelled_data(sure_traders, X_train, y_train, X_test):

    X_test.set_index(['Trader', 'Share', 'Day'], inplace=True)
    sure_traders_line = X_test.loc[sure_traders.keys()]
    logger.info(f"sure_traders_line.head()\n{sure_traders_line.head()}")

    logger.info(f"Pseudo labelling {len(sure_traders_line)} lines out of {len(X_test)}")
    X_test.drop(sure_traders_line.index, axis=0, inplace=True)

    X_train = pd.concat([X_train, sure_traders_line.drop(['Prediction'], axis=1)])
    y_train = pd.concat([y_train, sure_traders_line['Prediction']])

    logger.info(f"sure_traders_line['Prediction']] = \n{sure_traders_line['Prediction'].head()}")
    logger.info(f"sure_traders_line['Prediction']] = \n{sure_traders_line['Prediction'].unique()}")
    logger.info(y_train.unique())

    X_test.drop(['Prediction'], axis=1, inplace=True)

    return X_train, y_train, X_test




def aggregate_traders(X_test, y_pred):

    logger.info(f"y_pred\n{y_pred[0:100]}")
    X_test.reset_index(inplace=True)
    y_pred = pd.DataFrame(y_pred, columns=['HFT', 'MIX', 'NON HFT'])
    # logger.info(f"y_pred\n{y_pred.head()}")
    # y_pred.reset_index(inplace=True)
    X_test['Prediction'] = y_pred.idxmax(axis=1)

    Xy_test = X_test[['Trader']].merge(y_pred, left_index=True, right_index=True)
    # Xy_test = pd.get_dummies(Xy_test.Prediction)
    # Xy_test['Trader'] = X_test['Trader']

    count = Xy_test.groupby(['Trader']).count()
    logger.info(count.head())
    Xy_test = Xy_test.groupby(['Trader']).mean()    
    Xy_test['Sum'] = count['HFT']
    # Xy_test['Sum'] = Xy_test['MIX'] + Xy_test['NON HFT'] + Xy_test['HFT']
    # Xy_test['MIX'] /= Xy_test['Sum']
    # Xy_test['NON HFT'] /= Xy_test['Sum']
    # Xy_test['HFT'] /= Xy_test['Sum']

    Xy_test['type'] = Xy_test[['NON HFT', 'MIX', 'HFT']].idxmax(axis=1)

    return Xy_test


def get_sure_traders(Xy_test, min_observations=10, threshold=0.9):

    # logger.info(f"Xy_test\n{Xy_test.head()}")
    sure_traders = Xy_test.loc[((Xy_test['Sum'] >= min_observations)) & ((Xy_test['MIX'] >= threshold) | (Xy_test['HFT'] >= threshold) | (Xy_test['NON HFT'] >= threshold)), :]
    sure_traders = pd.Series(sure_traders.type.values, index=sure_traders.index.values).to_dict()
    logger.info(f"Found {len(sure_traders)} sure traders")

    return sure_traders
