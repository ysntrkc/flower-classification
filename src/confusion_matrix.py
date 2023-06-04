import os
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from utils.utils import get_data_loader

from model.CNN import CNN
import model.models as models

# create confusion matrix for each model and save it as a .npy file in ./files
if __name__ == "__main__":
    # create directory for files
    os.makedirs("../files", exist_ok=True)

    # define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define models
    cnn_model = CNN(num_classes=10).to(device)
    resnet_model = models.ResNet18(num_classes=10).to(device)
    densenet_model = models.DenseNet121(num_classes=10).to(device)
    efficientnet_model = models.EfficientNetB0(num_classes=10).to(device)

    # load models
    cnn_model.load_state_dict(torch.load("../models/cnn.pt"))
    resnet_model.load_state_dict(torch.load("../models/resnet.pt"))
    densenet_model.load_state_dict(torch.load("../models/densenet.pt"))
    efficientnet_model.load_state_dict(torch.load("../models/efficientnet.pt"))

    # define data loaders
    _, _, test_data_loader = get_data_loader(batch_size=50)

    # define models
    models = {
        "CNN": cnn_model,
        "ResNet": resnet_model,
        "DenseNet": densenet_model,
        "EfficientNet": efficientnet_model,
    }

    # define confusion matrix for each model
    for model_name, model in models.items():
        print(f"Creating confusion matrix for {model_name}...")
        y_true = []
        y_pred = []

        # get predictions for each batch
        for images, labels in test_data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

        # create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        np.save(f"../files/{model_name.lower()}_conf_matrix.npy", cm)
        print(f"Confusion matrix for {model_name} created!")