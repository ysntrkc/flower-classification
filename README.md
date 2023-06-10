# Flower Classification with PyTorch

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://flowerclassification.streamlit.app/)

## Project Overview

This project is part of the our University's course "Introduction to Deep Learning". The goal of this project is to classify flowers using CNN and 3 different pretrained models (ResNet, DenseNet, EfficientNet). The dataset used is the [Flower Classification | 10 Classes |](https://www.kaggle.com/datasets/utkarshsaxenadn/flower-classification-5-classes-roselilyetc) dataset from Kaggle.
* You can find more information about the project in the [About Project](https://flowerclassification.streamlit.app/About_Project_%F0%9F%93%96) page.

## Project Usage (only streamlit page)

### 1. Clone the repository

```bash
git clone <repo-url>
```

### 2. Install the requirements

```bash
pip install -r requirements.txt
```

### 3. Run the streamlit page

```bash
streamlit run Main_Page_🏡.py
```

## Project Usage (training)

### 1. Clone the repository

```bash
git clone <repo-url>
```

### 2. Get the kaggle key

* Go to your kaggle account and download the `kaggle.json` file. Then, move it to the `src` folder. You can also follow the instructions [here](https://www.kaggle.com/docs/api#authentication)

### 3. Go to the `src` folder

```bash
cd src
```

### 4. Create the conda environment

* If you don't have conda installed, you can follow the instructions [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/). I recommend using miniconda.

```bash
conda env create -f environment.yml
```

### 5. Activate the conda environment

```bash
conda activate torch
```

### 6. Run the training script with the desired options

* You can find different training options in the `src/utils/options.py` file
* You don't have to specify all the options and also don't have to download the dataset. If you place `kaggle.json` in the `src` folder, the script will download the dataset automatically.

```bash
python main.py --option1 value1 --option2 value2 ...
```


## Contributors

* Eren Akgül - [GitHub](https://github.com/akguleren) | [LinkedIn](https://www.linkedin.com/in/akguleren) | [email](mailto:akguleren53@gmail.com)
* Yasin Tarakçı - [GitHub](https://github.com/ysntrkc) | [LinkedIn](https://www.linkedin.com/in/yasintarakci) | [email](mailto:yasintarakci42@gmailcom)