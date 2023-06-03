import argparse


def args_parser():
    parser = argparse.ArgumentParser(description="Flower Classification")

    parser.add_argument("--batch_size", "-bs", type=int, default=50, help="batch size")
    parser.add_argument(
        "--epochs", "-e", type=int, default=100, help="number of epochs of training"
    )
    parser.add_argument(
        "--learning_rate", "-lr", type=float, default=0.01, help="learning rate"
    )
    parser.add_argument(
        "--num_classes", "-nc", type=int, default=10, help="number of classes"
    )
    parser.add_argument(
        "--device_id", "-d", type=int, default=0, help="device id to use"
    )
    parser.add_argument(
        "--device", "-dev", type=str, default="cuda", help="device to use"
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="cnn",
        help="model to use",
        choices=["CNN", "cnn", "ResNet", "resnet", "DenseNet", "densenet"],
    )

    args = parser.parse_args()
    return args
