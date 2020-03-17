from typing import List
from collections import Counter
import random

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
import matplotlib.pyplot as plt


def read_csv(filename: str) -> pd.DataFrame:
    return pd.read_csv(filename, index_col=0, encoding='utf-8', engine='python')


def preprocess_demographics(df: pd.DataFrame):
    if 'severity' in df.columns:
        df.loc[:, 'severity'] = 0.2 * df.loc[:, 'severity'].values - 1
    if 'age' in df.columns:
        df.loc[:, 'age'] = 0.02 * df.loc[:, 'age'].values - 1


def read_demographics(filename: str) -> (pd.DataFrame, np.ndarray, pd.DataFrame):
    csv_data = read_csv(filename)
    _csv_data = csv_data.copy(deep=True)
    preprocess_demographics(_csv_data)
    features = _csv_data.iloc[:, 1:].sort_index(axis=1)
    classes = _csv_data.iloc[:, 0].values
    return features, classes, _csv_data


def read_feature_appearance(filename: str, threshold=None) -> List:
    appearance_data = read_csv(filename)
    appearance_data.drop('(Intercept)', inplace=True)
    if threshold is not None:
        appearance_data.where(appearance_data >= threshold, inplace=True)
        appearance_data.dropna(inplace=True)
    return appearance_data.index.tolist()


def drop_subjects_by_classes(features: pd.DataFrame, classes: np.ndarray, classes_to_drop: List) -> (pd.DataFrame, np.ndarray):
    indices = np.setdiff1d(np.arange(0, classes.shape[0]), np.where(np.isin(classes, classes_to_drop)))
    return features.iloc[indices, :], classes[indices]


def get_evaluation_score(y_true, y_pred, save_dict=None):
    accuracy = np.mean(y_pred == y_true)
    conf_mat = confusion_matrix(y_true, y_pred)
    min_sensitivity = 1
    min_specificity = 1
    for i in range(conf_mat.shape[0]):
        sensitivity = confmat[i, i] / np.sum(confmat[:, i])
        min_sensitivity = min(sensitivity, min_sensitivity)
        specificity = confmat[i, i] / np.sum(confmat[i, :])
        min_specificity = min(specificity, min_specificity)
    # model_score is used to find best threshold during feature selection process.
    model_score = min(min_sensitivity, min_specificity)
    if isinstance(save_dict, dict):
        save_dict['accuracy'] = accuracy
        save_dict['confusion_matrix'] = conf_mat
        save_dict['sensitivity'] = min_sensitivity
        save_dict['specificity'] = min_specificity
        save_dict['score'] = model_score

