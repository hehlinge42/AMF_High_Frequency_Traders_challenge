import pandas as pd

import logging
import logzero
from logzero import logger
import numpy as np

from pseudo_labelling import aggregate_traders

def submit(Xy_test, path_to_submission, path_to_full_submission):

    Xy_test['type'] = np.nan

    mask_mxt = Xy_test['MIX'] > 0.5
    mask_hft = Xy_test['HFT'] > 0.85
    Xy_test.loc[mask_mxt, 'type'] = 'MIX'
    Xy_test.loc[mask_hft, 'type'] = 'HFT'
    Xy_test.fillna('NON HFT', inplace=True)

    logger.info(f"Writing file {path_to_full_submission}")
    Xy_test.to_csv(path_to_full_submission, index=True)

    logger.info(f"Writing file {path_to_submission}")
    Xy_test[['type']].to_csv(path_to_submission, index=True)