import pandas as pd

import logging
import logzero
from logzero import logger


X_test = pd.read_csv("../data/AMF_test_X.csv")
trader_names = X_test["Trader"].unique()
y_test = pd.DataFrame(columns = ["Trader", "type"])

def predict_trader(idx, sub_df):

    if idx % 3 == 0:
        return "HFT"
    elif idx % 2 == 0:
        return "MIX"
    else:
        return "NON HFT"


for idx, trader in enumerate(trader_names):

    sub_df = X_test.loc[X_test["Trader"] == trader]
    logger.warn(f"Nb records for trader {trader}: {len(sub_df)}")
    label = predict_trader(idx, sub_df)
    y_test.loc[idx] = [trader, label]

y_test.to_csv("../data/claqué_au_sol.csv", index=False)