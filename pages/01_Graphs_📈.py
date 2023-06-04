import h5py
import numpy as np
import streamlit as st
import plotly.express as px

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


def plot(model_name, model_results, metric):
    fig = px.line(
        x=range(1, len(model_results) + 1),
        y=model_results,
        title=f"{model_name} Model {metric}",
    )
    fig.update_yaxes(title_text=metric)
    fig.update_xaxes(title_text="Epochs")
    st.plotly_chart(fig, use_container_width=True)


st.set_page_config(page_title="Graphs", page_icon="📈", layout="centered")
st.title("Model Results")

st.write("---")

cnn_model_results = h5py.File("results/cnn.h5", "r")
resnet_model_results = h5py.File("results/resnet.h5", "r")
densenet_model_results = h5py.File("results/densenet.h5", "r")
efficientnet_model_results = h5py.File("results/efficientnet.h5", "r")

test_loss_cnn = cnn_model_results["test_loss"][cnn_model_results["test_loss"][:] != 0]
test_acc_cnn = cnn_model_results["test_acc"][cnn_model_results["test_acc"][:] != 0]

test_loss_resnet = resnet_model_results["test_loss"][
    resnet_model_results["test_loss"][:] != 0
]
test_acc_resnet = resnet_model_results["test_acc"][
    resnet_model_results["test_acc"][:] != 0
]

test_loss_densenet = densenet_model_results["test_loss"][
    densenet_model_results["test_loss"][:] != 0
]
test_acc_densenet = densenet_model_results["test_acc"][
    densenet_model_results["test_acc"][:] != 0
]

test_loss_efficientnet = efficientnet_model_results["test_loss"][
    efficientnet_model_results["test_loss"][:] != 0
]
test_acc_efficientnet = efficientnet_model_results["test_acc"][
    efficientnet_model_results["test_acc"][:] != 0
]

st.subheader("CNN Model Results")
st.write(
    """
	The CNN model was trained with a batch size of 50, learning rate of 0.001 and 200 epochs. But the model was converge faster. So, the program stopped the training at 49th epoch.
    \nSome CNN implementations on Kaggle have 95.5% test accuracy with the dataset's first version that have 5 classes. But we got only 67.3% test accuracy. We tried to increase the test accuracy by changing the parameters and the model architecture. But we couldn't increase the test accuracy.
	"""
)

# plot cnn model loss
plot("CNN", test_loss_cnn, "Loss")

# plot cnn model accuracy
plot("CNN", test_acc_cnn, "Accuracy")

conf_matrix = np.load("files/cnn_conf_matrix.npy")
fig = px.imshow(
    conf_matrix,
    labels=dict(x="Predicted", y="Actual", color="Count"),
    x=classes,
    y=classes,
    title="CNN Confusion Matrix",
)
st.plotly_chart(fig, use_container_width=True)

st.write("---")

st.subheader("ResNet Model Results")
st.write(
    """
    We used ResNet18 as a pretrained model. We didn't train it. We just changed the last layer of the model and finetuned it with our data.
	\nThe ResNet18 model was trained with a batch size of 50, learning rate of 0.001 and 200 epochs. But the model was converge faster. So, the program stopped the training at 26th epoch.
    \nWe got 88.1% test accuracy with ResNet18 model. It's a great result.
	"""
)

# plot resnet model loss
plot("ResNet", test_loss_resnet, "Loss")

# plot resnet model accuracy
plot("ResNet", test_acc_resnet, "Accuracy")

conf_matrix = np.load("files/resnet_conf_matrix.npy")
fig = px.imshow(
    conf_matrix,
    labels=dict(x="Predicted", y="Actual", color="Count"),
    x=classes,
    y=classes,
    title="ResNet Confusion Matrix",
)
st.plotly_chart(fig, use_container_width=True)

st.write("---")

st.subheader("DenseNet Model Results")
st.write(
    """
    We used DenseNet121 as a pretrained model. We didn't train it. We just changed the last layer of the model and finetuned it with our data.
    \nThe DenseNet121 model was trained with a batch size of 50, learning rate of 0.001 and 200 epochs. But the model was converge faster. So, the program stopped the training at 26th epoch.
    \nWe got 89.5% test accuracy with DenseNet121 model.
    """
)

# plot densenet model loss
plot("DenseNet", test_loss_densenet, "Loss")

# plot densenet model accuracy
plot("DenseNet", test_acc_densenet, "Accuracy")

conf_matrix = np.load("files/densenet_conf_matrix.npy")
fig = px.imshow(
    conf_matrix,
    labels=dict(x="Predicted", y="Actual", color="Count"),
    x=classes,
    y=classes,
    title="DenseNet Confusion Matrix",
)
st.plotly_chart(fig, use_container_width=True)

st.write("---")

st.subheader("EfficientNet Model Results")
st.write(
    """
    We used EfficientNetB0 as a pretrained model. We didn't train it. We just changed the last layer of the model and finetuned it with our data.
    \nThe EfficientNetB0 model was trained with a batch size of 50, learning rate of 0.001 and 200 epochs. But the model was converge faster. So, the program stopped the training at 17th epoch.
    \nWe got 91.1% test accuracy with EfficientNetB0 model.
    """
)

# plot densenet model loss
plot("EfficientNet", test_loss_efficientnet, "Loss")

# plot densenet model accuracy
plot("EfficientNet", test_acc_efficientnet, "Accuracy")

conf_matrix = np.load("files/efficientnet_conf_matrix.npy")
fig = px.imshow(
    conf_matrix,
    labels=dict(x="Predicted", y="Actual", color="Count"),
    x=classes,
    y=classes,
    title="EfficientNet Confusion Matrix",
)
st.plotly_chart(fig, use_container_width=True)