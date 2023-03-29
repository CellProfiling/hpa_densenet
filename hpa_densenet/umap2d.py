"""This module generates data related to plot 2d UMAP for
HPA Densenet model.

Author: Frederic Ballllosera Navarro
"""
import numpy as np
import pandas as pd
import logging
from hpa_densenet import constants


def generateCSV(f_red: str, f_meta: str, dst: str):
    """Generates a CSV file containing image ids from the meta-information file and
    X, Y from the dimensionality reduction file

    Args:
        f_red (str): Path to the input reduction file.

        f_meta (str): Path to the input metainformation file.

        dst (str): Path to the output CSV file
    """
    logger = logging.getLogger(constants.LOGGER_NAME)
    logger.info("Loading data for dimensionality reduction and meta-information")
    try:
        reduced = np.load(f_red)['components']
        image_ids = np.load(f_meta)['image_ids']
    except:
        logger.error("Error when loading input data")

    df = pd.DataFrame()
    df = pd.concat([df, pd.DataFrame(image_ids)], axis=1)
    df = pd.concat([df, pd.DataFrame(reduced)], axis=1)
    df.columns = ['Id', 'X', 'Y']
    df.to_csv(dst, index=False)

