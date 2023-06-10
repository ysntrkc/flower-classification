import h5py
import numpy as np
import pandas as pd
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

def plot_graphs(model_name):
    plot(model_name, "Loss")
    plot(model_name, "Accuracy")
    conf_matrix(model_name)

def plot(model_name, metric):
    if metric == "Loss":
        metric_postfix = "loss"
    elif metric == "Accuracy":
        metric_postfix = "acc"

    read_file = h5py.File(f"results/{model_name.lower()}.h5", "r")
    model_results = pd.melt(pd.DataFrame({
        "id": range(1, len(read_file[f"train_{metric_postfix}"][read_file[f"train_{metric_postfix}"][:] != 0]) + 1),
        "Train": read_file[f"train_{metric_postfix}"][read_file[f"train_{metric_postfix}"][:] != 0],
        "Test": read_file[f"test_{metric_postfix}"][read_file[f"test_{metric_postfix}"][:] != 0],
        "Validation": read_file[f"val_{metric_postfix}"][read_file[f"val_{metric_postfix}"][:] != 0]
    }), id_vars=["id"], var_name="Dataset", value_name="value")

    fig = px.line(
        data_frame=model_results,
        x='id',
        y="value",
        color="Dataset",
        title=f"{model_name} Model {metric}",
    )
    fig.update_yaxes(title_text=metric)
    fig.update_xaxes(title_text="Epochs")
    st.plotly_chart(fig, use_container_width=True)

def conf_matrix(model_name):
    cm = np.load(f"files/{model_name.lower()}_conf_matrix.npy")
    cm = pd.DataFrame(cm, columns=classes, index=classes)
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        title=f"{model_name} Model Confusion Matrix (Test Data)",
        color_continuous_scale="blues",
    )
    st.plotly_chart(fig, use_container_width=True)


st.set_page_config(page_title="Graphs", page_icon="📈", layout="centered")
st.title("Model Results")

st.write("---")

st.subheader("CNN Model Results")
st.write(
    """
	The CNN model was trained with a batch size of 50, learning rate of 0.001 and 200 epochs. But the model was converge faster. So, the program stopped the training at 49th epoch.
    \nSome CNN implementations on Kaggle have 95.5% test accuracy with the dataset's first version that have 5 classes. But we got only 67.3% test accuracy. We tried to increase the test accuracy by changing the parameters and the model architecture. But we couldn't increase the test accuracy.
	"""
)

# plot graphs for cnn model loss
plot_graphs("CNN")

st.write("---")

st.subheader("ResNet Model Results")
st.write(
    """
    We used ResNet18 as a pretrained model. We didn't train it. We just changed the last layer of the model and finetuned it with our data.
	\nThe ResNet18 model was trained with a batch size of 50, learning rate of 0.001 and 200 epochs. But the model was converge faster. So, the program stopped the training at 26th epoch.
    \nWe got 88.1% test accuracy with ResNet18 model. It's a great result.
	"""
)

# plot graphs for resnet model loss
plot_graphs("ResNet")

st.write("---")

st.subheader("DenseNet Model Results")
st.write(
    """
    We used DenseNet121 as a pretrained model. We didn't train it. We just changed the last layer of the model and finetuned it with our data.
    \nThe DenseNet121 model was trained with a batch size of 50, learning rate of 0.001 and 200 epochs. But the model was converge faster. So, the program stopped the training at 26th epoch.
    \nWe got 89.5% test accuracy with DenseNet121 model.
    """
)

# plot graphs for densenet model loss
plot_graphs("DenseNet")

st.write("---")

st.subheader("EfficientNet Model Results")
st.write(
    """
    We used EfficientNetB0 as a pretrained model. We didn't train it. We just changed the last layer of the model and finetuned it with our data.
    \nThe EfficientNetB0 model was trained with a batch size of 50, learning rate of 0.001 and 200 epochs. But the model was converge faster. So, the program stopped the training at 17th epoch.
    \nWe got 91.1% test accuracy with EfficientNetB0 model.
    """
)

# plot graphs for efficientnet model loss
plot_graphs("EfficientNet")