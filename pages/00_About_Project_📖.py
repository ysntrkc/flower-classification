import streamlit as st

st.set_page_config("About Project", "📖", "centered")

st.title("About Project")

st.write("---")

st.subheader("Problem Description")
st.write(
    "Flowers may need some classification using images as input. For instance some people could want to observe which types of flowers are present in a picture to label them for marketing or just research purposes. Solution to these possible requests we generated three different machine learning models to predict the type of the flower in a picture into 10 different classes and compared the performances of the models to see which one performs the best for our case."
)

st.subheader("Solution Approach")
st.markdown(
    """
    We are going to classify the images using CNN, RESNET-18, DenseNet-121 and EfficientNet-B0 structures. CNN makes image processing easier by providing us with some useful functions to extract information about images.
* We can apply convolution operation to detect patterns in images such as edges, shapes etc.
* We can apply pooling operations to reduce the image size significantly to make calculations faster.
* Lastly, we can pass the parameters through a fully connected layer for classification.
    \nResidual networks are used for complex image classification tasks. Unlike CNN it doesn’t have to pass through all the layers contained in the neural network. It can skip some of the layers and that gives us a faster training process since we don’t have to make all the calculations in all of the layers. In the project we used a pretrained version of RESNET-18. It is trained by a dataset called ImageNet which is a very large image dataset. Compared to our CNN model the accuracy of RESNET-18 is really high.
    \nThe key idea behind DenseNet is to address the vanishing gradient problem and encourage feature reuse by introducing dense connections between layers. In traditional CNN architectures, each layer takes the output of its preceding layer as input. However, in DenseNet, each layer receives inputs not only from its preceding layer but also from all the preceding layers. This dense connectivity pattern allows for the direct flow of information across different depths of the network, facilitating gradient flow and promoting feature reuse.
    \nThe key idea behind EfficientNet-B0 is compound scaling, which involves scaling the network's depth, width, and resolution simultaneously. Traditional scaling methods typically increase only one of these dimensions, resulting in either overfitting or underfitting the data. EfficientNet-B0 addresses this issue by introducing a new scaling method that balances these dimensions.
            """
)

st.subheader("Data Preparation")
st.write(
    """
	The dataset consists of 22355 images, the size of each image is varying and in .jpg, .jpeg or .png format. It contains 3 folders for each class as train, test and validation. Dataset is already divided into subfolders for different usage purposes of data. In total we have 10 different classes such as: Aster, daisy, iris, lavender, lily, marigold, orchid, poppy, rose, sunflower.\n
	Since the size varies, in preprocessing step we resize all of the images to 150x150. We also applied some augmentation methods to improve our model with different versions of the images. We applied rotation, horizontal and vertical flip, blur."""
)
st.subheader("References")
st.write(
    """
    He, K. (2015, December 10). Deep Residual Learning for Image Recognition. arXiv.org. [arxiv](https://arxiv.org/abs/1512.03385)\n
    Dataset: [here](https://www.kaggle.com/datasets/utkarshsaxenadn/flower-classification-5-classes-roselilyetc)
	"""
)
