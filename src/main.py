import os
import sys
import time
import logging

import torch

from model.models import ResNet50, DenseNet121
from model.CNN import CNN
from utils.options import args_parser
from utils.kaggle import download_dataset
from utils.utils import get_data_generator, save_results, get_num_of_models


def logging_setup(args):
    # create log folder
    os.makedirs("../logs", exist_ok=True)

    # add a line end of the log file
    with open(f"../logs/{(args.model).lower()}_log.txt", "a") as f:
        f.write(f"\n{'='*100}\n")

    # set logging format
    logging.basicConfig(
        filename=f"../logs/{(args.model).lower()}_log.txt",
        level=logging.INFO,
        format="(%(asctime)s.%(msecs)03d %(levelname)s) %(module)s - %(funcName)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # set logging to stdout
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


def main():
    args = args_parser()
    download_dataset()

    args.device = torch.device(
        f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu"
    )

    logging_setup(args)
    logging.info(f"Arguments: {args}")

    train_generator, validation_generator, test_generator = get_data_generator(
        args.batch_size
    )

    if (args.model).lower() == "cnn":
        model = CNN(args.num_classes)
    elif (args.model).lower() == "resnet":
        model = ResNet50(args.num_classes)
    elif (args.model).lower() == "densenet":
        model = DenseNet121(args.num_classes)
    else:
        raise ValueError(f"Model {args.model} not found!")

    # save_results(args, history)


if __name__ == "__main__":
    main()
