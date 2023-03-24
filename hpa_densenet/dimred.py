"""This module performs dimensionality reduction functions on predictions from the
HPA Densenet model.

Author: Casper Winsnes
"""
import datetime
import logging
import os
from typing import Optional

import numpy as np
import umap
from numpy.typing import NDArray
from hpa_densenet import constants


def _umap_dimred(
    input_data: NDArray,
    dimensions: int,
    n_neighbors: int = 2,
) -> NDArray:
    """Perform dimensionality reduction using Uniform Manifold Approximation
    and Projection (UMAP).

    Args:
        input_data (NDArray): Input features to reduce dimensions on.

        dimensions (int): Number of dimensions to reduce to.

        n_neighbors (int, optional): Number of neighbors for UMAP locality.
        Defaults to 15.

    Returns:
        NDArray: UMAP embedding of the input data as an NDArray of size
        [num_samples, num_dimensions].
    """
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=0.1,
        n_components=dimensions,
        metric="euclidean",
        random_state=33,
    )

    return reducer.fit_transform(input_data)


def store_dimred(reduced_data: NDArray, filename: Optional[str] = None):
    """Store the reduced dimensions in a compressed numpy format.

    Args:
        reduced_data (NDArray): The data to store.

        filename (Optional[str], optional): Filename to store data in. If None,
        store the data in a timestamped file in the current folder.
        Defaults to None.
    """
    if filename is None:
        curr_time = datetime.date.today().strftime(f"%Y-%m-%d-{time.time()}")
        filename = f"dimred_features-{curr_time}.npz"
    else:
        folder = os.path.dirname(filename)
        if folder:
            os.makedirs(folder, exist_ok=True)
    np.savez_compressed(filename, components=reduced_data)


def dimred(input_data: str, dimensions: int = 2, method: str = "umap") -> NDArray:
    """Perform dimensionality reduction using the specified method.

    Args:
        input_data (str): Path to the input feature file. This file is required
        to be in the same output format as the HPA Densenet prediction function:
        an NDArray loadable using `np.load` and containing the key "feats".

        dimensions (int, optional): Number of dimensions to reduce the input to.
        Defaults to 2.

        method (str, optional): Which dimensionality reduction method to use.
                                Valid options are: "umap".
                                Defaults to "umap".

    Returns:
        The embedding as a Numpy array of size [num_samples, num_dimensions].
    """
    logger = logging.getLogger(constants.LOGGER_NAME)
    logger.info("Loading data for dimensionality reduction")
    try:
        input_features = np.load(input_data)["feats"]
    except:
        logger.error("Error when loading input data")
        raise ValueError(f"Incorrect input data file ({input_data})")
    match method:
        case "umap":
            logger.info("Running UMAP dimensionality reduction")
            return _umap_dimred(input_features, dimensions)
        case _:
            logger.error("No valid dimensionality method chosen.")
            raise ValueError("Need a valid dimensionality method.")
