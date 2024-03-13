"""This module generates data related to plot Nd UMAP for
HPA Densenet model.

Author: Frederic Ballllosera Navarro
"""
import numpy as np
import pandas as pd
import logging
import os
import plotly.express as px
from hpa_densenet import constants

dimension_names = ["x", "y", "z", "t"]


def generateCSV(f_red: str, f_meta: str, f_prob: str, dst: str, n_dim: int):
    """Generates a CSV file containing image ids from the meta-information file,
    the predicted value from the probabilities files and
    X, Y from the dimensionality reduction file

    Args:
        f_red (str): Path to the input reduction file.

        f_meta (str): Path to the input metainformation file.

        f_prob (str): Path to the input probabilities file.

        dst (str): Path to the output CSV file
    """
    logger = logging.getLogger(constants.LOGGER_NAME)
    logger.info("Loading data for dimensionality reduction, meta-information and probabilities")
    try:
        reduced = np.load(f_red)["components"]
        image_ids = np.load(f_meta)["image_ids"]
        probs = np.load(f_prob)['probs']
        max_prob = pd.DataFrame(probs).idxmax(axis=1)
    except Exception as e:
        print(e)
        logger.error("Error when loading input data")

    df = pd.DataFrame()
    df = pd.concat([df, pd.DataFrame(image_ids)], axis=1)
    df = pd.concat([df, pd.DataFrame(max_prob)], axis=1)
    final_dim = []
    for i in range(n_dim):
        df = pd.concat([df, pd.DataFrame(reduced[:, i])], axis=1)
        final_dim.append(
            dimension_names[i] if i < len(dimension_names) else "n" + str(i)
        )

    df.columns = ["id", "class"] + final_dim
    df.to_csv(dst, index=False)

    df['Class'] = df.apply(lambda x: constants.CLASS2NAME[x['class']], axis=1)
    list_class = df['Class'].unique()
    list_class.sort()
    color_discrete_map = {}
    for cat_class in df['class'].unique():
        color_discrete_map[constants.CLASS2NAME[cat_class]] = constants.CLASS2COLOR[cat_class]

    if n_dim == 2:
        fig = px.scatter(df, x='x', y='y', color='Class',
                         category_orders={"Class": list_class}, color_discrete_map=color_discrete_map, hover_name="id")
        fig.write_html(os.path.splitext(dst)[0] + '.html')
    elif n_dim == 3:
        fig = px.scatter_3d(df, x='x', y='y', z='z', color='Class',
                            category_orders={"Class": list_class}, color_discrete_map=color_discrete_map, hover_name="id")
        fig.write_html(os.path.splitext(dst)[0] + '.html')
