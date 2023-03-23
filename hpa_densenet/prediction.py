""" This module runs and stores the predictions from the HPA Densenet model.

Author: Casper Winsnes
"""
import logging
import os
import time
from typing import Optional

import numpy as np
import torch
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
from tqdm import tqdm

import hpa_densenet.constants as constants
from hpa_densenet.dataset import ProteinDataset
from hpa_densenet.models import class_densenet121_large_dropout

MANUAL_SEED = 0


def _predict_items(
    dataloader: DataLoader, model: torch.nn.Module, gpu: bool, out_dir=Optional[str]
):
    """
    Perform model prediction on all the items from the dataloader.

    Arguments:
        dataloader: Dataloader for data to be predicted on.
        model: The model to run predictions.
               The model should output logits and features as a tuple.
               A sigmoid will be run on the model logits, equivalent to:
               sigmoid(model(input)[0]).
        gpu: Whether or not to move input images to the gpu.
    """
    logger = logging.getLogger(constants.LOGGER_NAME)
    model.eval()
    result_features = []
    result_probabilities = []

    np.random.seed(MANUAL_SEED)
    torch.manual_seed(MANUAL_SEED)

    iterator = dataloader
    if logger.isEnabledFor(logging.INFO):
        iterator = tqdm(iterator, total=len(dataloader))

    for images, idx in iterator:
        try:
            with torch.no_grad():
                if gpu:
                    images = torch.autograd.Variable(images.cuda())
                else:
                    images = torch.autograd.Variable(images)

                logits, features = model(images)
                probabilities = torch.sigmoid(logits)
                probabilities = probabilities.cpu().data.numpy().tolist()

                result_probabilities += probabilities
                result_features += features.cpu().data.numpy().tolist()
        except:
            logger.error(f"Not all images in batch are the same size")

    return result_probabilities, result_features


def _store_outputs(probabilities, features, out_dir):
    curr_time = time.time()
    feature_target = os.path.join(out_dir, f"features_{curr_time}.npz")
    probabilities_target = os.path.join(out_dir, f"probabilities_{curr_time}.npz")
    np.savez_compressed(feature_target, feats=np.array(features))
    np.savez_compressed(probabilities_target, probs=np.array(probabilities))


def d121_predict(
    src_dir: str,
    out_dir: str,
    size: int = 1536,
    num_workers: int = 2,
    batch_size: int = 32,
    gpus: Optional[str] = None,
) -> None:
    """
    Perform cell predictions using the winning Densenet121 model from the
    HPA kaggle challenge.

    Arguments:
        src_dir: Path to a directory where resized images are located.
        out_dir: Path to a directory in which results are stored.
        size: The size of the images in `src_dir`. Each image should be
              `size x size` large. Defaults to 1536.
        num_workers: The number of parallel workers for dataloading.
                     Defaults to 2.
        batch_size: The size of each batch for the dataloaders.
                    Defaults to 32.
        gpus: A string describing the gpus to be used for predictions.
              Follows the same format as the environment variable
              CUDA_VISIBLE_DEVICES. If `gpus` is None,
              CUDA_VISIBLE_DEVICES will be used instead.
              The string can also be 'cpu', indicating that no gpu
              should be used.
              Defaults to None.
    """
    logger = logging.getLogger(constants.LOGGER_NAME)

    if gpus and gpus != "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    elif gpus is None:
        gpus = os.environ["CUDA_VISIBLE_DEVICES"]

    logger.info(f"Using GPUS: {gpus}")
    logger.info("Loading model")

    model = class_densenet121_large_dropout(
        num_classes=constants.NUM_PREDICTOR_CLASSES,
        in_channels=constants.NUM_IN_CHANNELS,
        pretrained=constants.DEFAULT_MODEL,
    )
    gpu = False
    if len(gpus.split(",")) > 1:
        model = DataParallel(model)
    if gpus != "cpu":
        gpu = True
        model = model.cuda()

    logger.info("Loading dataloaders")
    dataset = ProteinDataset(
        src_dir,
        image_size=size,
        crop_size=constants.CROP_SIZE,
        in_channels=constants.NUM_IN_CHANNELS,
        suffix="jpg",
    )

    dataloader = DataLoader(
        dataset,
        sampler=SequentialSampler(dataset),
        batch_size=batch_size,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    logger.info(f"Performing predictions.")
    os.makedirs(out_dir, exist_ok=True)
    probs, feats = _predict_items(dataloader, model, gpu)

    # TODO: return values instead of storing here.
    #       let the caller use the values as they want.
    logger.info(f"Storing output in {out_dir}.")
    _store_outputs(probs, feats, out_dir)
