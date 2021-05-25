from collections import defaultdict

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from constants import *

def threshold_metrics(yhat_raw, y, thresholds):
    names = ["acc", "prec", "rec", "f1"]
    metric_threshs = defaultdict(list)
    for thresh in thresholds:
        yhat = (yhat_raw[:,1] >= thresh).astype(int)
        acc = accuracy_score(y, yhat)
        prec = precision_score(y, yhat)
        rec = recall_score(y, yhat)
        f1 = f1_score(y, yhat)
        for name, val in zip(names, [acc, prec, rec, f1]):
            metric_threshs[name].append(val)
    return metric_threshs

def balanced_f1(yhat_raw, y):
    thresholds = np.arange(0,1,.01)[1:]
    metric_threshs = threshold_metrics(yhat_raw, y, thresholds)
    best_ix = np.nanargmax(metric_threshs['f1'])
    best_thresh = thresholds[best_ix]
    best_thresh_metrics = {name: vals[best_ix] for name, vals in metric_threshs.items()}
    try:
        best_thresh_metrics['auc'] = roc_auc_score(y, yhat_raw[:,1])
    except:
        print("not enough samples for AUC")
        best_thresh_metrics['auc'] = 0
    return best_thresh, best_thresh_metrics

def precision_at_fixed_recall(yhat_raw, y, fixed_metric):
    tmp_yhat = yhat_raw[:,1] * y
    yhat_true_ys = tmp_yhat[tmp_yhat > 0]
    # pick threshold such that fixed_metric amt of true labels will be predicted
    thresh = sorted(yhat_true_ys)[round(len(yhat_true_ys) * (1 - fixed_metric)) - 1]

    yhat = (yhat_raw[:,1] >= thresh).astype(int)
    prec = precision_score(y, yhat)
    return thresh, prec

def recall_at_fixed_precision(yhat_raw, y, fixed_metric):
    sort_ixs = np.argsort(yhat_raw[:,1])[::-1]
    top_pred_labels = np.array(y)[sort_ixs]
    
    # start by choosing threshold to predict maximally (ie everything is a 1)
    # then reduce until we see precision rise above fixed_metric
    # it's possible this fails, in which case return -1
    num_to_pred = len(yhat_raw)
    thresh = yhat_raw[sort_ixs[num_to_pred - 1],1]
    pred_ys = np.array(y)[sort_ixs][:num_to_pred]
    prec = sum(pred_ys) / len(pred_ys)
    while prec < fixed_metric and num_to_pred > 1:
        num_to_pred -= 1
        thresh = yhat_raw[sort_ixs[num_to_pred - 1],1]
        pred_ys = np.array(y)[sort_ixs][:num_to_pred]
        prec = sum(pred_ys) / len(pred_ys)
    if num_to_pred == 0:
        return -1
    yhat = (yhat_raw[:,1] >= thresh).astype(int)
    rec = recall_score(y, yhat)
    return thresh, rec
    

