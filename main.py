import logging
import argparse
from hpa_densenet import preprocess


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PyTorch Protein Classification")
    parser.add_argument("-v", "--verbose", action="store_true")

    subparsers = parser.add_subparsers(title="Commands", help="Valid subcommands")

    preprocessing = subparsers.add_parser(
        "preprocess", help="Preprocess images for Densenet use"
    )
    preprocessing.add_argument(
        "-s", "--src-dir", type=str, default=None, help="source directory"
    )
    preprocessing.add_argument(
        "-d", "--dst-dir", type=str, default=None, help="destination directory"
    )
    preprocessing.add_argument("--size", type=int, default=1536, help="size")
    preprocessing.add_argument(
        "-w",
        "--num-workers",
        type=int,
        default=10,
        help="The number of multiprocessing workers to perform the resizing",
    )
    preprocessing.add_argument(
        "--continue",
        dest='cont',
        type=bool,
        action="store_true",
        help="Continue from a previously aborted run.",
    )
    preprocessing.set_defaults(command="preprocess")

    return parser


def main():
    parser = _build_argparser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    match args.command:
        case "preprocess":
            logging.info(f"Preprocessing images from {args.src_dir} to {args.dst_dir}")
            preprocess.resize_images(args.src_dir, args.dst_dir, size=args.size, cont=args.cont)


if __name__ == "__main__":
    main()
