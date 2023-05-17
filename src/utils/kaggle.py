import os
import json
import glob
import shutil

os.environ["KAGGLE_USERNAME"] = json.load(open("kaggle.json"))["username"]
os.environ["KAGGLE_KEY"] = json.load(open("kaggle.json"))["key"]

from kaggle.api.kaggle_api_extended import KaggleApi


def download_dataset():
    os.makedirs("../data", exist_ok=True)

    # download dataset if not exists
    if not os.path.exists("../data/Flower Classification V2"):
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(
            "utkarshsaxenadn/flower-classification-5-classes-roselilyetc",
            path="../data",
            unzip=True,
        )

        # move folders in ..data/Flower Classification/V2 to ..data/Flower Classification
        for folder in glob.glob("../data/Flower Classification V2/V2/*"):
            shutil.move(folder, "../data/Flower Classification V2")

        # remove unnecessary files
        shutil.rmtree("../data/Flower Classification")
        shutil.rmtree("../data/Flower Classification V2/TFRecords")
        os.rename("../data/Flower Classification V2", "../data/Flower Classification")
        shutil.rmtree("../data/Flower Classification/V2")
