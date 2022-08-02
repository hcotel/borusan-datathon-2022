import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight


def round_probabilities(probabilities, threshold):

    """
    Round probabilities to labels based on the given threshold
    Parameters
    ----------
    probabilities (np.ndarray of shape (n_samples)): Predicted probabilities
    threshold (float): Rounding threshold
    Returns
    -------
    labels (numpy.ndarray of shape (n_samples)): Labels
    """

    labels = np.zeros_like(probabilities, dtype=np.uint8)
    labels[probabilities >= threshold] = 1

    return labels


def specificity_score(y_true, y_pred):

    """
    Calculate specificity (true-negative rate) of predicted labels
    Parameters
    ----------
    y_true (numpy.ndarray of shape (n_samples)): Ground truth labels
    y_pred (numpy.ndarray of shape (n_samples)): Predicted labels
    Returns
    -------
    (float): Specificity score
    """

    tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)


def classification_scores(y_true, y_pred, threshold=0.5):

    """
    Calculate binary classification metrics on predicted probabilities and labels
    Parameters
    ----------
    y_true (numpy.ndarray of shape (n_samples)): Ground truth labels
    y_pred (numpy.ndarray of shape (n_samples)): Predicted probabilities
    threshold (float): Rounding threshold
    Returns
    -------
    scores (dict): Dictionary of calculated scores
    """

    y_pred_labels = round_probabilities(y_pred, threshold=threshold)
    scores = {
        'accuracy': accuracy_score(y_true, y_pred_labels),
        'roc_auc': roc_auc_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred_labels),
        'recall': recall_score(y_true, y_pred_labels),
        'specificity': specificity_score(y_true, y_pred_labels),
        'f1': f1_score(y_true, y_pred_labels),
        'pos_rate': f"{y_pred_labels.sum()}/{y_pred_labels.shape[0]}"
    }

    return scores


def calculate_weights(y):

    """
    Calculate sample and class weights from given labels
    Parameters
    ----------
    y (numpy.ndarray of shape (n_samples)): Ground truth labels
    Returns
    -------
    sample_weights (numpy.ndarray of shape (n_samples)): Sample weights of the given labels
    class_weights (numpy.ndarray of shape (n_classes)): Class weights of the given labels
    """

    sample_weights = compute_sample_weight(class_weight='balanced', y=y)
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)

    return sample_weights, class_weights

def xgb_f1(t, y, threshold=0.5):
    y_bin = (y > threshold).astype(int) # works for both type(y) == <class 'numpy.ndarray'> and type(y) == <class 'pandas.core.series.Series'>
    return 1 - f1_score(t, y_bin)

def lgb_f1_score(y_true, y_pred):
    y_pred = (y_pred > 0.5).astype(int)
    return 'f1', f1_score(y_true, y_pred), True

