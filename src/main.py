import os
import time
import math
import logging

import torch
import numpy as np
from torch import nn, optim

from model.models import ResNet50, DenseNet121
from model.CNN import CNN
from utils.options import args_parser
from utils.kaggle import download_dataset
from utils.utils import get_data_generator, save_results, get_num_of_models, logging_setup


def test(model, test_generator, loss_fn, device):
    model.eval()
    with torch.no_grad():
        loss, accuracy = np.zeros(1), np.zeros(1)
        for x_batch, y_batch in test_generator:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            prediction = model(x_batch)
            loss += loss_fn(prediction, y_batch).item() * y_batch.size(0)
            is_correct = (torch.argmax(prediction, dim=1) == y_batch).float()
            accuracy += is_correct.sum().item()
        loss /= len(test_generator.dataset)
        accuracy /= len(test_generator.dataset)
    return loss, accuracy


def train(
    args,
    model,
    train_generator,
    validation_generator,
    test_generator,
    lr_scheduler,
    optimizer,
    loss_fn,
):
    os.makedirs(f"../models/{(args.model).lower()}", exist_ok=True)
    model_num = get_num_of_models((args.model).lower())
    model_path = f"../models/{(args.model).lower()}/{(args.model).lower()}_model_{model_num}.pt"
    model.to(args.device)
    torch.save(model.state_dict(), model_path)

    # initialize variables
    min_loss = math.inf
    train_loss, train_accuracy = np.zeros(args.epochs), np.zeros(args.epochs)
    val_loss, val_accuracy = np.zeros(args.epochs), np.zeros(args.epochs)
    test_loss, test_accuracy = np.zeros(args.epochs), np.zeros(args.epochs)
    count = 0

    # start training
    for epoch in range(args.epochs):
        # control if the model has not improved for 5 epochs
        if count == 5:
            break
        count += 1

        # start timer
        start = time.time()
        
        # train
        model.train()
        for x_batch, y_batch in train_generator:
            x_batch, y_batch = x_batch.to(args.device), y_batch.to(args.device)
            prediction = model(x_batch)
            loss = loss_fn(prediction, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss[epoch] += loss.item() * y_batch.size(0)
            is_correct = (torch.argmax(prediction, dim=1) == y_batch).float()
            train_accuracy[epoch] += is_correct.sum().item()
        train_loss[epoch] /= len(train_generator.dataset)
        train_accuracy[epoch] /= len(train_generator.dataset)

        # validation
        model.eval()
        with torch.no_grad():
            for x_batch, y_batch in validation_generator:
                x_batch, y_batch = x_batch.to(args.device), y_batch.to(args.device)
                prediction = model(x_batch)
                loss = loss_fn(prediction, y_batch)
                val_loss[epoch] += loss.item() * y_batch.size(0)
                is_correct = (torch.argmax(prediction, dim=1) == y_batch).float()
                val_accuracy[epoch] += is_correct.sum().item()
        val_loss[epoch] /= len(validation_generator.dataset)
        val_accuracy[epoch] /= len(validation_generator.dataset)
        lr_scheduler.step()

        if val_loss[epoch] < min_loss:
            min_loss = val_loss[epoch]
            count = 0
            torch.save(model.state_dict(), model_path)

        # test
        test_loss[epoch], test_accuracy[epoch] = test(
            model, test_generator, loss_fn, args.device
        )

        # end timer
        end = time.time()

        # print results
        logging.info(
            f"{f'{epoch + 1:<4}/{args.epochs:>4}':>9} | "
            f"{train_loss[epoch]:>10.6f} | "
            f"{train_accuracy[epoch]:>14.6f} | "
            f"{val_loss[epoch]:>15.6f} | "
            f"{val_accuracy[epoch]:>19.6f} | "
            f"{test_loss[epoch]:>9.6f} | "
            f"{test_accuracy[epoch]:>13.6f} | "
            f"{end - start:>14.2f}"
        )

    # return results
    return {
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
    }
    


def main():
    args = args_parser()
    logging_setup(args)
    download_dataset()

    args.device = torch.device(
        f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu"
    )

    logging.info(f"Arguments: {args}")

    # get data generators
    train_generator, validation_generator, test_generator = get_data_generator(
        args.batch_size
    )

    # initialize model
    if (args.model).lower() == "cnn":
        model = CNN(args.num_classes)
    elif (args.model).lower() == "resnet":
        model = ResNet50(args.num_classes)
    elif (args.model).lower() == "densenet":
        model = DenseNet121(args.num_classes)
    else:
        raise ValueError(f"Model {args.model} not found!")

    # initialize loss function, optimizer and learning rate scheduler
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

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
        train_generator,
        validation_generator,
        test_generator,
        lr_scheduler,
        optimizer,
        loss_fn,
    )

    # save results
    save_results(args, history)

    # add a line end of the log file
    with open(f"../logs/{(args.model).lower()}_log.txt", "a") as f:
        f.write(f"\n{'='*100}\n")


if __name__ == "__main__":
    main()
