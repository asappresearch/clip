"""
    This file contains evaluation methods that take in a set of predicted labels 
        and a set of ground truth labels and calculate precision, recall, accuracy, f1, and metrics @k
"""
from collections import defaultdict
import csv
import json
import numpy as np
import os
import sys

from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from tqdm import tqdm

import binary_eval
from constants import *

def all_metrics(yhat, y, yhat_raw=None, calc_auc=True, label_order=[]):
    """
        Inputs:
            yhat: binary predictions matrix 
            y: binary ground truth matrix
            yhat_raw: prediction scores matrix (floats)
        Outputs:
            dict holding relevant metrics
    """
    names = ["acc", "prec", "rec", "f1"]

    metrics = {}
    for ix,label in enumerate(label_order):
        metrics[f"{label2abbrev[label]}-f1"] = f1_score(y[:,ix], yhat[:,ix])

    #macro
    print("GETTING ALL MACRO")
    macro = all_macro(yhat, y)

    #micro
    print("GETTING ALL MICRO")
    ymic = y.ravel()
    yhatmic = yhat.ravel()
    micro = all_micro(yhatmic, ymic)

    metrics.update({names[i] + "_macro": macro[i] for i in range(len(macro))})
    metrics.update({names[i] + "_micro": micro[i] for i in range(len(micro))})

    #AUC
    print("AUC")
    if yhat_raw is not None and calc_auc:
        roc_auc = auc_metrics(yhat_raw, y, ymic)
        metrics.update(roc_auc)

    return metrics

def balanced_f1(yhat_raw, y):
    names = ["acc", "prec", "rec", "f1"]
    thresholds = np.arange(0,1,.01)[1:]
    metric_threshs = defaultdict(list)
    for thresh in thresholds:
        yhat = (yhat_raw > thresh).astype(int)
        ymic = y.ravel()
        yhatmic = yhat.ravel()
        micro = all_micro(yhatmic, ymic)
        macro = all_macro(yhat, y)
        for name, val in zip(names, micro):
            metric_threshs[f'{name}_micro'].append(val)
        for name, val in zip(names, macro):
            metric_threshs[f'{name}_macro'].append(val)
    if np.all(np.isnan(metric_threshs['f1_micro'])):
        return None, None
    best_ix = np.nanargmax(metric_threshs['f1_micro'])
    best_thresh = thresholds[best_ix]
    best_thresh_metrics = {name: vals[best_ix] for name, vals in metric_threshs.items()}
    return best_thresh, best_thresh_metrics

def macro_metrics_given_thresholds(yhat_raw, y, label_threshs):
    metrics = {}
    get_auc = True
    for ix, label in enumerate(LABEL_TYPES):
        lname = label2abbrev[label]
        yhat = yhat_raw[:,ix] > label_threshs[ix]
        for name, scorer in zip(['acc', 'prec', 'rec', 'f1'], [accuracy_score, precision_score, recall_score, f1_score]):
            metrics[f'balanced-{lname}-{name}'] = scorer(y[:,ix], yhat)
        try:
            metrics[f'balanced-{lname}-auc'] = roc_auc_score(y[:,ix], yhat_raw[:,ix])
        except:
            get_auc = False
            print("couldn't get auc since all examples negative")
    met_names =['acc', 'prec', 'rec', 'f1'] 
    if get_auc:
        met_names.append('auc')
    for metric in met_names:
        macro_metric = np.mean([metrics[f'balanced-{label2abbrev[label]}-{metric}'] for label in LABEL_TYPES])
        metrics[f"balanced-macro-{metric}"] = macro_metric
        metrics[f"{metric}_macro"] = macro_metric
    return metrics


def balance_each_label(yhat_raw, y, get_high_prec_thresh=False):
    label_balanced_threshs = []
    label_high_prec_threshs = []
    label_high_prec_recs = []
    metrics = {}
    label_rec_values = {}
    for ix,label in enumerate(LABEL_TYPES):
        binary_yhat_raw = np.stack((1 - yhat_raw[:,ix], yhat_raw[:,ix]), axis=1)
        if get_high_prec_thresh:
            print("getting cutoff thresholds for high precision...")
            thresh, rec = binary_eval.recall_at_fixed_precision(binary_yhat_raw, y[:,ix], 0.7)
            label_high_prec_threshs.append(thresh)
            label_high_prec_recs.append(rec)
        balanced_thresh, balanced_metrics = binary_eval.balanced_f1(binary_yhat_raw, y[:,ix])
        thresholds = np.arange(0,1,.01)[1:]
        thresh_metrics = binary_eval.threshold_metrics(binary_yhat_raw, y[:,ix], thresholds) 
        label_rec_values[label] = thresh_metrics['rec']

        label_balanced_threshs.append(balanced_thresh)
        for metric, val in balanced_metrics.items():
            metrics[f"balanced-{label2abbrev[label]}-{metric}"] = val
    for metric in ['acc', 'prec', 'rec', 'f1', 'auc']:
        macro_metric = np.mean([metrics[f'balanced-{label2abbrev[label]}-{metric}'] for label in LABEL_TYPES])
        metrics[f"balanced-macro-{metric}"] = macro_metric
    return metrics, label_balanced_threshs, label_high_prec_threshs, label_high_prec_recs, label_rec_values

def balanced_f1_multilabel(yhat_raw, yy):
    names = ["prec", "rec", "f1"]
    metric_threshs = {label2abbrev[label]: defaultdict(list) for label in LABEL_TYPES}
    per_label_metrics = {label2abbrev[label]: defaultdict(float) for label in LABEL_TYPES}
    thresholds = np.arange(0,1,.01)[1:]
    rec_90_threshs = {}
    rec_75_threshs = {}
    prec_90_threshs = {}
    prec_75_threshs = {}
    for ix,label in enumerate(LABEL_TYPES):
        lname = label2abbrev[label]
        for thresh in thresholds:
            yhat = (yhat_raw[:,ix] > thresh).astype(int)
            y = yy[:,ix]
            prec = precision_score(y, yhat)
            rec = recall_score(y, yhat)
            f1 = f1_score(y, yhat)
            for name, val in zip(names, [prec, rec, f1]):
                metric_threshs[lname][name].append(val)
        best_ix = np.nanargmax(metric_threshs[lname]['f1'])
        best_thresh = thresholds[best_ix]
        best_thresh_metrics = {f'{name}_maxf1_thresh': vals[best_ix] for name, vals in metric_threshs[lname].items()}
        try:
            best_thresh_metrics['auc'] = roc_auc_score(yy[:,ix], yhat_raw[:,ix])
        except ValueError:
            best_thresh_metrics['auc'] = 0

        rec_75_ixs = np.where(np.array(metric_threshs[lname]['rec']) > 0.75)[0]
        if len(rec_75_ixs) > 0:
            rec_75_ix = rec_75_ixs[-1]
            rec_75_thresh = thresholds[rec_75_ix]
            best_thresh_metrics['prec@rec=75'] = metric_threshs[lname]['prec'][rec_75_ix]
        else:
            rec_75_ix = 0
            rec_75_thresh = thresholds[rec_75_ix]
            highest_rec = metric_threshs[lname]['rec'][rec_75_ix]
            best_thresh_metrics['prec@rec=75'] = metric_threshs[lname]['prec'][rec_75_ix]

        rec_90_ixs = np.where(np.array(metric_threshs[lname]['rec']) > 0.90)[0]
        if len(rec_90_ixs) > 0:
            rec_90_ix = rec_90_ixs[-1]
            rec_90_thresh = thresholds[rec_90_ix]
            best_thresh_metrics['prec@rec=90'] = metric_threshs[lname]['prec'][rec_90_ix]
        else:
            rec_90_ix = 0
            rec_90_thresh = thresholds[rec_90_ix]
            highest_rec = metric_threshs[lname]['rec'][rec_90_ix]
            best_thresh_metrics['prec@rec=90'] = metric_threshs[lname]['prec'][rec_90_ix]

        prec_75_ixs = np.where(np.array(metric_threshs[lname]['prec']) > 0.75)[0]
        if len(prec_75_ixs) > 0:
            prec_75_ix = prec_75_ixs[0]
            prec_75_thresh = thresholds[prec_75_ix]
            best_thresh_metrics['rec@prec=75'] = metric_threshs[lname]['rec'][prec_75_ix]
        else:
            prec_75_ix = -1
            prec_75_thresh = thresholds[prec_75_ix]
            highest_prec = metric_threshs[lname]['prec'][prec_75_ix]
            best_thresh_metrics['rec@prec=75'] = metric_threshs[lname]['rec'][prec_75_ix]

        prec_90_ixs = np.where(np.array(metric_threshs[lname]['prec']) > 0.90)[0]
        if len(prec_90_ixs) > 0:
            prec_90_ix = prec_90_ixs[0]
            prec_90_thresh = thresholds[prec_90_ix]
            best_thresh_metrics['rec@prec=90'] = metric_threshs[lname]['rec'][prec_90_ix]
        else:
            prec_90_ix = -1
            prec_90_thresh = thresholds[prec_90_ix]
            highest_prec = metric_threshs[lname]['prec'][prec_90_ix]
            best_thresh_metrics['rec@prec=90'] = metric_threshs[lname]['rec'][prec_90_ix]

        per_label_metrics[lname].update(best_thresh_metrics)
        rec_90_threshs[lname] = rec_90_thresh
        rec_75_threshs[lname] = rec_75_thresh
        prec_90_threshs[lname] = prec_90_thresh
        prec_75_threshs[lname] = prec_75_thresh
    return per_label_metrics, rec_90_threshs, prec_90_threshs, rec_75_threshs, prec_75_threshs

def f1_per_label_type(yhat_raw, y, label_order, thresh):
    yhat = (yhat_raw > thresh).astype(int)
    metrics = {}
    for ix,label in enumerate(label_order):
        metrics[f"{label2abbrev[label]}-f1"] = f1_score(y[:,ix], yhat[:,ix])
    return metrics

def all_macro(yhat, y):
    return macro_accuracy(yhat, y), macro_precision(yhat, y), macro_recall(yhat, y), macro_f1(yhat, y)

def all_micro(yhatmic, ymic):
    return micro_accuracy(yhatmic, ymic), micro_precision(yhatmic, ymic), micro_recall(yhatmic, ymic), micro_f1(yhatmic, ymic)

#########################################################################
#MACRO METRICS: calculate metric for each label and average across labels
#########################################################################

def macro_accuracy(yhat, y):
    num = intersect_size(yhat, y, 0) / (union_size(yhat, y, 0) + 1e-10)
    return np.mean(num)

def macro_precision(yhat, y):
    num = intersect_size(yhat, y, 0) / (yhat.sum(axis=0) + 1e-10)
    return np.mean(num)

def macro_recall(yhat, y):
    num = intersect_size(yhat, y, 0) / (y.sum(axis=0) + 1e-10)
    return np.mean(num)

def macro_f1(yhat, y):
    prec = macro_precision(yhat, y)
    rec = macro_recall(yhat, y)
    if prec + rec == 0:
        f1 = 0.
    else:
        f1 = 2*(prec*rec)/(prec+rec)
    return f1

##########################################################################
#MICRO METRICS: treat every prediction as an individual binary prediction
##########################################################################

def micro_accuracy(yhatmic, ymic):
    return intersect_size(yhatmic, ymic, 0) / union_size(yhatmic, ymic, 0)

def micro_precision(yhatmic, ymic):
    return intersect_size(yhatmic, ymic, 0) / yhatmic.sum(axis=0)

def micro_recall(yhatmic, ymic):
    return intersect_size(yhatmic, ymic, 0) / ymic.sum(axis=0)

def micro_f1(yhatmic, ymic):
    prec = micro_precision(yhatmic, ymic)
    rec = micro_recall(yhatmic, ymic)
    if prec + rec == 0:
        f1 = 0.
    else:
        f1 = 2*(prec*rec)/(prec+rec)
    return f1

def auc_metrics(yhat_raw, y, ymic):
    if yhat_raw.shape[0] <= 1:
        return
    fpr = {}
    tpr = {}
    roc_auc = {}
    #get AUC for each label individually
    relevant_labels = []
    auc_labels = {}
    for i in range(y.shape[1]):
        #only if there are true positives for this label
        if y[:,i].sum() > 0:
            fpr[i], tpr[i], _ = roc_curve(y[:,i], yhat_raw[:,i])
            if len(fpr[i]) > 1 and len(tpr[i]) > 1:
                auc_score = auc(fpr[i], tpr[i])
                if not np.isnan(auc_score): 
                    auc_labels["auc_%d" % i] = auc_score
                    relevant_labels.append(i)

    #macro-AUC: just average the auc scores
    aucs = []
    for i in relevant_labels:
        aucs.append(auc_labels['auc_%d' % i])
    roc_auc['auc_macro'] = np.mean(aucs)

    #micro-AUC: just look at each individual prediction
    yhatmic = yhat_raw.ravel()
    fpr["micro"], tpr["micro"], _ = roc_curve(ymic, yhatmic) 
    roc_auc["auc_micro"] = auc(fpr["micro"], tpr["micro"])

    return roc_auc

def union_size(yhat, y, axis):
    #axis=0 for label-level union (macro). axis=1 for instance-level
    return np.logical_or(yhat, y).sum(axis=axis).astype(float)

def intersect_size(yhat, y, axis):
    #axis=0 for label-level union (macro). axis=1 for instance-level
    return np.logical_and(yhat, y).sum(axis=axis).astype(float)

def print_per_label_metrics(metrics):
    print(f"Image, appt, medication, procedure, lab, case, other")
    print("%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f" % (metrics['imaging'], metrics['appointment'], metrics['medication'], metrics['procedure'], metrics['lab'], metrics['case'], metrics['other']))

def print_metrics(metrics, per_label=True, balanced=True):
    print()
    base_metrics = ['acc', 'prec', 'rec', 'f1', 'auc']
    which_metrics = []
    headers = []
    if "auc_macro" in metrics.keys():
        which_metrics.append([f'{met}_macro' for met in base_metrics])
        headers.append("[MACRO] accuracy, precision, recall, f-measure, AUC")
    if balanced and "balanced-macro-f1" in metrics.keys():
        which_metrics.append([f'balanced-macro-{met}' for met in base_metrics[:-1]] + ['auc_macro'])
        headers.append("[BALANCED MACRO] accuracy, precision, recall, f-measure, AUC")
        #print("%.4f, %.4f, %.4f, %.4f, %.4f" % (metrics["acc_macro"], metrics["prec_macro"], metrics["rec_macro"], metrics["f1_macro"], metrics["auc_macro"]))

    if "auc_micro" in metrics.keys():
        which_metrics.append([f'{met}_micro' for met in base_metrics])
        headers.append("[MICRO] accuracy, precision, recall, f-measure, AUC")
    if balanced and "balanced_f1_micro" in metrics.keys():
        which_metrics.append([f'balanced_{met}_micro' for met in base_metrics[:-1]] + ['auc_micro'])
        headers.append("[BALANCED MICRO] accuracy, precision, recall, f-measure, AUC")

    if per_label:
        if 'balanced-Imaging-f1' in metrics:
            which_metrics.append([f'balanced-{label2abbrev[label]}-f1' for label in LABEL_TYPES])
            headers.append(f"[BALANCED] Image, appt, medication, procedure, lab, case, other")
        if 'Imaging-f1' in metrics:
            which_metrics.append([f'{label2abbrev[label]}-f1' for label in LABEL_TYPES])
            headers.append(f"Image, appt, medication, procedure, lab, case, other")

    binary_metrics = {k:v for k, v in metrics.items() if 'binary' in k}
    if len(binary_metrics) > 0:
        which_metrics.append(['binary_prec', 'binary_rec', 'binary_f1'])
        headers.append('[BALANCED BINARY] prec, rec, f1')


    for header, metric_set in zip(headers, which_metrics):
        print(header)
        values = []
        for k in metric_set:
            if k in metrics:
                values.append(f'{metrics[k]:.4f}')
            else:
                values.append('na')
        print(",".join(values))

    print()

