import argparse
import logging
import sys

from hpa_densenet import constants, prediction, preprocess


def _build_preprocessing_subcommand(preprocessing: argparse.ArgumentParser) -> None:
    preprocessing.set_defaults(command="preprocess")
    preprocessing.add_argument(
        "-s",
        "--src-dir",
        type=str,
        default=None,
        help="source directory",
        required=True,
    )
    preprocessing.add_argument(
        "-d",
        "--dst-dir",
        type=str,
        default=None,
        help="destination directory",
        required=True,
    )
    preprocessing.add_argument("--size", type=int, default=1536, help="image size")
    preprocessing.add_argument(
        "-w",
        "--num-workers",
        type=int,
        default=10,
        help="The number of multiprocessing workers to perform the resizing",
    )
    preprocessing.add_argument(
        "--continue",
        dest="cont",
        action="store_true",
        help="Continue from a previously aborted run.",
    )


def _build_prediction_subcommand(prediction: argparse.ArgumentParser) -> None:
    prediction.set_defaults(command="predict")
    prediction.add_argument(
        "-s",
        "--src-dir",
        type=str,
        default=None,
        help="src image directory (preprocessed)",
        required=True,
    )
    prediction.add_argument(
        "-d",
        "--dst-dir",
        type=str,
        default=None,
        help="output directory",
        required=True,
    )
    prediction.add_argument("--size", type=int, default=1536, help="image size")
    prediction.add_argument(
        "--gpu",
        type=str,
        default=None,
        help=(
            "Which gpus to use for prediction. Any string valid for `CUDA_VISIBLE_DEVICES`"
            "is valid for this. If cpu calculations ONLY is desired, a value of 'cpu' is "
            "also allowed."
        ),
    )


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PyTorch Protein Classification")
    parser.add_argument("-v", "--verbose", action="store_true")

    subparsers = parser.add_subparsers(title="Commands", help="Valid subcommands")

    preprocessing = subparsers.add_parser(
        "preprocess", help="Preprocess images for Densenet use"
    )
    _build_preprocessing_subcommand(preprocessing)

    prediction = subparsers.add_parser("predict", help="Run Densenet for prediction")
    _build_prediction_subcommand(prediction)

    return parser


def main():
    parser = _build_argparser()
    args = parser.parse_args()

    logger = logging.getLogger(name=constants.LOGGER_NAME)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    if args.verbose:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    match args.command:
        case "preprocess":
            logger.info(f"Preprocessing images from {args.src_dir} to {args.dst_dir}")
            preprocess.resize_images(
                args.src_dir, args.dst_dir, size=args.size, cont=args.cont
            )
        case "predict":
            logger.info(
                f"Running prediction for images from {args.src_dir} to {args.dst_dir}"
            )
            prediction.d121_predict(
                args.src_dir, args.dst_dir, args.size, gpus=args.gpu
            )


if __name__ == "__main__":
    main()
