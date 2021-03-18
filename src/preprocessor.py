import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import FunctionTransformer
import numpy as np
import matplotlib.pyplot as plt

import logging
import logzero
from logzero import logger

def normalize_events(X):
    
    X_transformed = X.copy()
    mask_OTR = X_transformed['OTR'].isna()
    mask_OCR = X_transformed['OCR'].isna()
    mask_OMR = X_transformed['OMR'].isna()
    X_transformed.loc[mask_OTR, 'OTR'] = 0
    X_transformed.loc[mask_OCR, 'OCR'] = 0
    X_transformed.loc[mask_OMR, 'OMR'] = 0
    X_transformed.loc[~mask_OTR, 'OTR'] = 1 / X_transformed['OTR']
    X_transformed.loc[~mask_OCR, 'OCR'] = 1 / X_transformed['OCR']
    X_transformed.loc[~mask_OMR, 'OMR'] = 1 / X_transformed['OMR']
    X_transformed['total'] = X_transformed['OTR'] + X_transformed['OCR'] + X_transformed['OMR']
    X_transformed['OTR'] /=  X_transformed['total']
    X_transformed['OCR'] /=  X_transformed['total']
    X_transformed['OMR'] /=  X_transformed['total']
    X_transformed.drop(['total'], inplace=True, axis=1)
    return X_transformed


def manage_na(X, epsilon=0.01):

    na_columns = X.columns[X.isna().any()].tolist()
    X_transformed = X.drop(na_columns[3:], axis=1, inplace=False)
    for col in na_columns[0:3]:
        X_transformed[col].fillna(value=max(X_transformed[col]) + epsilon, inplace=True)
    return X_transformed


def perform_pca(X, n_components, plot=False):

    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(X)
    per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)

    labels = []
    for i in range(len(per_var)):
        labels.append('PC' + str(i + 1))
        
    pca_df = pd.DataFrame(pca_data, index=X.index, columns=labels)

    if plot is True:

        plt.plot(range(1, len(per_var) + 1), per_var)
        plt.xlabel("Principal Component")
        plt.ylabel("Variance expliquée")
        plt.show()
        plt.scatter(pca_df[labels[0]], pca_df[labels[1]])
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
        plt.show()
    
    return pca_df


def preprocess(X):

    X_transformed = manage_na(X)
    X_transformed = normalize_events(X_transformed)

    return X_transformed