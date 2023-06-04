import torch
from torchvision import transforms

import streamlit as st
from PIL import Image

from src.model.CNN import CNN
from src.model.models import DenseNet121, ResNet50

classes = [
    'Aster',
    'Daisy',
    'Iris',
    'Lavender',
    'Lily',
    'Marigold',
    'Orchid',
    'Poppy',
    'Rose',
    'Sunflower',
]


def load_image(image):
    image = transforms.Compose(
        [
            transforms.Resize((150, 150)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )(image)
    image = image.unsqueeze(0)
    return image


def make_prediction(model, image):
    image = load_image(image)
    model.eval()
    with torch.no_grad():
        pred = model(image)
        pred = torch.argmax(pred, dim=1)
    return pred.item()


st.set_page_config(page_title="Vegetable Classifier", page_icon="🥦", layout="centered")
st.title("Vegetable Classifier")

st.write("---")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

cnn_model = CNN(len(classes))
resnet_model = ResNet50(len(classes))
densenet_model = DenseNet121(len(classes))

cnn_model.load_state_dict(
    torch.load("./models/cnn.pt", map_location=torch.device("cpu"))
)
resnet_model.load_state_dict(
    torch.load("./models/resnet.pt", map_location=torch.device("cpu"))
)
densenet_model.load_state_dict(
    torch.load("./models/densenet.pt", map_location=torch.device("cpu"))
)


if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("")

    cnn_result = make_prediction(cnn_model, image)
    resnet_result = make_prediction(resnet_model, image)
    densenet_result = make_prediction(densenet_model, image)

    st.write(f"**CNN:** {classes[cnn_result]}")
    st.write(f"**ResNet:** {classes[resnet_result]}")
    st.write(f"**DenseNet:** {classes[densenet_result]}")