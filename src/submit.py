import pandas as pd

import logging
import logzero
from logzero import logger
import numpy as np

def submit(Xy_test_voting, Xy_test_voting_full, path_to_submission, path_to_full_submission):

    mask_min_obs = Xy_test_voting_full[Xy_test_voting_full['nb_observations'] <= 2]
    Xy_test_voting.loc[mask_min_obs.index, 'type'] = 'NON HFT'
    Xy_test_voting_full.loc[mask_min_obs.index, 'type'] = 'NON HFT'

    logger.info(f"Writing file {path_to_full_submission}")
    Xy_test_voting_full.to_csv(path_to_full_submission, index=True)

    logger.info(f"Writing file {path_to_submission}")
    Xy_test_voting[['type']].to_csv(path_to_submission, index=True)