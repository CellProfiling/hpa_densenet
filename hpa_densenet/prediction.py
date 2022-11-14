import logging
import os

import constants
import numpy as np
from torch.nn import DataParallel
from torch.utils.data import DataLoader

from hpa_densenet.dataset import ProteinDataset
from models import class_densenet121_large_dropout


def _predict_items(dataloader, model, out_dir):
    model.eval()
    results = []

    np.random.seed(0)
    

def d121_predict(
    src_dir: str, out_dir: str, size: int = 1536, gpus: str = None
) -> None:
    logger = logging.getLogger(constants.LOGGER_NAME)

    gpus = gpus if gpus else os.environ["CUDA_VISIBLE_DEVICES"]
    logger.info(f"Using GPUS: {gpus}")

    logger.info("Loading model")
    model = class_densenet121_large_dropout(
        num_classes=constants.NUM_PREDICTOR_CLASSES,
        in_channels=constants.NUM_IN_CHANNELS,
        pretrained=constants.DEFAULT_MODEL,
    )
    if len(gpus.split(",")) > 1:
        model = DataParallel(model)
    if gpus != "cpu":
        model = model.cuda()

    logger.info("Loading dataloaders")
    dataset = ProteinDataset(
        src_dir,
        image_size=size,
        crop_size=constants.CROP_SIZE,
        in_channels=constants.NUM_IN_CHANNELS,
        suffix=".png",
    )
    dataloader = DataLoader(dataset)
    _predict_items(dataloader, model, out_dir)
