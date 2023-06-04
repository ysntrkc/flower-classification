import os
import time
import math
import logging

import torch
from torch import nn, optim

from model.models import ResNet18, DenseNet121, EfficientNetB0
from model.CNN import CNN
from utils.options import args_parser
from utils.kaggle import download_dataset
from utils.utils import (
    get_data_data_loader,
    save_results,
    get_num_of_models,
    logging_setup,
    evaluate_model,
)


def train(
    args,
    model,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    lr_scheduler,
    optimizer,
    loss_fn,
):
    os.makedirs(f"../models/{(args.model).lower()}", exist_ok=True)
    model_num = get_num_of_models((args.model).lower())
    model_path = (
        f"../models/{(args.model).lower()}/{(args.model).lower()}_model_{model_num}.pt"
    )
    model.to(args.device)
    torch.save(model.state_dict(), model_path)

    # initialize variables
    train_loss, train_acc = [0] * args.epochs, [0] * args.epochs
    test_loss, test_acc = [0] * args.epochs, [0] * args.epochs
    val_loss, val_acc = [0] * args.epochs, [0] * args.epochs

    # initialize the min loss
    min_loss = math.inf

    # start training
    for epoch in range(args.epochs):
        # start timer
        start = time.time()

        # train model
        model.train()
        for x_batch, y_batch in train_dataloader:
            x_batch, y_batch = x_batch.to(args.device), y_batch.to(args.device)
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss[epoch] += loss.item() * y_batch.size(0)
            is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
            train_acc[epoch] += torch.sum(is_correct).item()
        train_loss[epoch] /= len(train_dataloader.dataset)
        train_acc[epoch] /= len(train_dataloader.dataset)

        # validate model
        model.eval()
        with torch.no_grad():
            for x_batch, y_batch in val_dataloader:
                x_batch, y_batch = x_batch.to(args.device), y_batch.to(args.device)
                pred = model(x_batch)
                loss = loss_fn(pred, y_batch)
                val_loss[epoch] += loss.item() * y_batch.size(0)
                is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
                val_acc[epoch] += torch.sum(is_correct).item()
        val_loss[epoch] /= len(val_dataloader.dataset)
        val_acc[epoch] /= len(val_dataloader.dataset)
        lr_scheduler.step()

        if val_loss[epoch] < min_loss:
            torch.save(model.state_dict(), model_path)

        # test model
        test_loss[epoch], test_acc[epoch] = evaluate_model(
            model, test_dataloader, loss_fn, args.device
        )

        # end timer
        end = time.time()

        # print results
        logging.info(
            f"{f'{epoch + 1:<4}/{args.epochs:>4}':>9} | "
            f"{train_loss[epoch]:>10.6f} | "
            f"{train_acc[epoch]:>14.6f} | "
            f"{val_loss[epoch]:>15.6f} | "
            f"{val_acc[epoch]:>19.6f} | "
            f"{test_loss[epoch]:>9.6f} | "
            f"{test_acc[epoch]:>13.6f} | "
            f"{end - start:>14.2f}"
        )

    return {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
    }


def main():
    args = args_parser()
    logging_setup(args)
    download_dataset()

    args.device = torch.device(
        f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu"
    )

    logging.info(f"Arguments: {args}")

    # get data data_loaders
    train_data_loader, validation_data_loader, test_data_loader = get_data_data_loader(
        args.batch_size
    )

    # initialize model
    if (args.model).lower() == "cnn":
        model = CNN(args.num_classes)
    elif (args.model).lower() == "resnet":
        model = ResNet18(args.num_classes)
    elif (args.model).lower() == "densenet":
        model = DenseNet121(args.num_classes)
    elif (args.model).lower() == "efficientnet":
        model = EfficientNetB0(args.num_classes)
    else:
        raise ValueError(f"Model {args.model} not found!")

    # initialize loss function, optimizer and learning rate scheduler
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # print header row
    logging.info(
        f"{'Epoch':>9} | "
        f"{'Train loss'} | "
        f"{'Train accuracy'} | "
        f"{'Validation loss'} | "
        f"{'Validation accuracy'} | "
        f"{'Test loss'} | "
        f"{'Test accuracy'} | "
        f"{'Time (seconds)'}"
    )

    # start training
    history = train(
        args,
        model,
        train_data_loader,
        validation_data_loader,
        test_data_loader,
        lr_scheduler,
        optimizer,
        loss_fn,
    )

    # save results
    save_results(args, history)

    # add a line end of the log file
    with open(f"../logs/{(args.model).lower()}.log", "a") as f:
        f.write(f"\n{'='*100}\n")


if __name__ == "__main__":
    main()
