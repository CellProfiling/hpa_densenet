import logging
import os

import cv2
import mlcrate
import numpy as np
from PIL import Image

from hpa_densenet import constants

COLOR_INDEX = {
    "red": 0,
    "green": 1,
    "blue": 2,
    "yellow": 0,
}


def _resize_images_help(param: tuple) -> None:
    logger = logging.getLogger(name=constants.LOGGER_NAME)
    src, fname, dst, size = param
    # eg. 44741_1177_B2_3_red.jpg -> red
    color = fname[fname.rfind("_") + 1 : fname.rfind(".")]
    try:
        image = np.array(Image.open(os.path.join(src, fname)))[
            :, :, COLOR_INDEX.get(color)
        ]
    except Exception as e:
        logger.info(f"Bad image: {fname}, falling back on cv2")
        logger.info(f"Error message: {e}")
        try:
            image = cv2.imread(os.path.join(src, fname))[:, :, -1::-1][
                :, :, COLOR_INDEX.get(color)
            ]
        except Exception as e:
            logger.error(f'Cannot read {fname} at all')
            logger.error(e)
            return

    h, w = image.shape[:2]
    if h != size or w != size:
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    logger.info(f"Writing {fname}")
    cv2.imwrite(os.path.join(dst, fname), image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])


def resize_images(
    src_dir: str, dst_dir: str, size: int = 1536, num_workers: int = 10, cont=False
):
    """
    Resize all images in a directory to a similar size.

    Args:
    src_dir: Path to a directory containing all images to be resized.
    dst_dir: Path to the directory in which to store the resized images.
    size: Size of the resized images.
          The resulting image is square, `size` x `size`.
          Defaults to 1536.
    num_workers: Number of parallel processes to use for resizing. Defaults to 10.
    cont: Continue froma a previously aborted run. Should only be used when
          `src_dir` has been unchanged between runs.
    """
    logger = logging.getLogger(name=constants.LOGGER_NAME)

    fnames = np.sort(os.listdir(src_dir))
    os.makedirs(dst_dir, exist_ok=True)

    if cont:
        start_num = max(0, len(os.listdir(dst_dir)))
        fnames = fnames[start_num:]
    params = [(src_dir, fname, dst_dir, size) for fname in fnames]

    logger.info(f"Spawning {num_workers} to resize images")
    pool = mlcrate.SuperPool(num_workers)
    pool.map(_resize_images_help, params, description="resize image")
    logger.info(f"All images resized")
