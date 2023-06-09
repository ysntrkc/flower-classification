import os
import sys
import h5py
import torch
import logging
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


def logging_setup(args):
    # create log folder
    os.makedirs("../logs", exist_ok=True)

    # set logging format
    logging.basicConfig(
        filename=f"../logs/{(args.model).lower()}.log",
        level=logging.INFO,
        format="(%(asctime)s.%(msecs)03d %(levelname)s) - %(funcName)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # set logging to stdout
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


def evaluate_model(model, test_dataloader, loss_fn, device):
    model.eval()
    with torch.no_grad():
        test_loss = 0
        test_acc = 0
        for x_batch, y_batch in test_dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            test_loss += loss.item() * y_batch.size(0)
            is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
            test_acc += torch.sum(is_correct).item()
        test_loss /= len(test_dataloader.dataset)
        test_acc /= len(test_dataloader.dataset)
    return test_loss, test_acc


def get_data_loader(batch_size=50) -> tuple:
    train_dir = "../data/Flower Classification/Training Data"
    test_dir = "../data/Flower Classification/Testing Data"
    validation_dir = "../data/Flower Classification/Validation Data"

    # define transform for train data
    train_transform = transforms.Compose(
        [
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
            transforms.Resize((150, 150)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # define transform for test and validation data
    test_transform = transforms.Compose(
        [
            transforms.Resize((150, 150)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # define datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(validation_dir, transform=test_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

    # define dataloaders
    train_data_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    validation_data_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    test_data_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )

    logging.info(
        f"Train data: {len(train_dataset)} -- Validation data: {len(val_dataset)} -- Test data: {len(test_dataset)}"
    )

    return train_data_loader, validation_data_loader, test_data_loader


def save_results(args, results):
    os.makedirs(f"../results/{args.model}", exist_ok=True)

    filename = f"../results/{args.model}/{args.model}_bs{args.batch_size}_lr{args.learning_rate}_ep{args.epochs}.h5"

    with h5py.File(filename, "w") as f:
        for key, value in results.items():
            f.create_dataset(key, data=value)

    logging.info(f"Results saved to {filename}")


def get_num_of_models(model_name):
    models = os.listdir(f"../models/{model_name}")
    return len(models)
