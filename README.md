# Wafer Defect Classification (LSWMD Dataset)

This project explores how machine learning can be applied to semiconductor wafer map data to classify defect patterns.

The main idea is to convert wafer maps into numerical features and train a model that can identify different types of defect distributions.

Instead of using deep learning, this project focuses on classical machine learning and feature engineering to better understand how defect patterns can be represented in structured data.


## Project Goal

Wafer maps contain spatial information about defective dies on a semiconductor wafer.  
Different manufacturing issues often create recognizable patterns such as edge rings, scratches, or center clusters.

The goal of this project is to:

- preprocess wafer map datasets
- extract useful spatial features
- train a machine learning model to classify defect patterns
- evaluate model performance


## Dataset

The project uses the **LSWMD (Large Scale Wafer Map Dataset)**.

Each wafer map contains:

- pass / fail information of dies
- spatial distribution of defects
- a defect pattern label

Before training, the dataset needs preprocessing because the raw format contains nested structures and inconsistent labels.


## Project Structure
Wafer_defect_test
│
├── fix_pickle.py
│ Fix compatibility issues with older dataset pickle files
|
├── data_prepare.py
│ Clean labels and prepare wafer map data for feature extraction
│
├── feature.py
│ Extract spatial features from wafer maps
|
├── train.py
│ Train Random Forest model and evaluate classification performance

