# d100_d400_project
# Project description

This project aims to predict restaurant rating based on business-level features from the Yelp dataset.

## Get Data

1. Download the Yelp data archive (TAR file: 4.35 GB) from https://business.yelp.com/data/resources/open-dataset/
2. Unzip the archive, move the yelp_academic_dataset_business.json and yelp_academic_dataset_review.json files to the data folder

## Installation

1. cd project-root-dir
2. conda env create -f environment.yml
3. conda activate yelp_predict
4. pip install -e .

## Usage

### For data cleaning and EDA:
Open and execute the eda_cleaning.ipynb file in the scripts folder, using yelp_predict environment.

### For model training and evaluation:
1. cd project-root-dir
2. conda activate yelp_predict
3. python scripts/model_training.py
