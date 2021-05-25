import argparse
from collections import defaultdict
import csv
import json
import os
import pickle
import random
import time

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier

from constants import *
import binary_eval
import multilabel_eval
from neural_baselines import all_metrics

def build_data_and_label_matrices(fname, cvec, tvec=None, lvec=None, fit=False):
    corpus = []
    labels = []
    with open(fname) as f:
        r = csv.DictReader(f)
        for row in r:
            corpus.append(' '.join(eval(row['sentence'])))
            if lvec:
                labels.append(';'.join(eval(row['labels'])))
            else:
                labels.append(1 if eval(row['labels']) else 0)

    if fit:
        cX = cvec.fit_transform(corpus)
        if tvec:
            tX = tvec.fit_transform(corpus)
        # create multi-label matrix
        if lvec:
            yy = lvec.fit_transform(labels).todense()
    else:
        cX = cvec.transform(corpus)
        if tvec:
            tX = tvec.transform(corpus)
        if lvec:
            yy = lvec.transform(labels).todense()
    if lvec is None:
        yy = labels

    #needed to make parallel training work due to bug in scipy https://github.com/scikit-learn/scikit-learn/issues/6614#issuecomment-209922294
    cX.sort_indices()
    if tvec:
        tX.sort_indices()
        return (cX, tX), yy
    else:
        return cX, yy

def high_rec_false_negatives(X_dv, yhat_raw, y, rec_90_thresh, out_dir, fname, label_name):
    preds = yhat_raw > rec_90_thresh
    fns = np.array(y).astype(bool) & ~preds
    fn_ixs = set(np.where(fns == True)[0])
    fn_sents = []
    with open(fname) as f:
        r = csv.DictReader(f)
        for ix, row in enumerate(r):
            if ix in fn_ixs:
                fn_sents.append(' '.join(eval(row['sentence'])))
    with open(f'{out_dir}/{label_name}_rec_90_fns.txt', 'w') as of:
        for sent in fn_sents:
            of.write(sent + "\n")
    if label_name != 'binary':
        with open(f'{out_dir}/typed_rec_90_fns.txt', 'a') as of:
            of.write(f"#### {label_name} ####\n")
            for sent in fn_sents[:5]:
                of.write(sent + " [[END]]\n")
            of.write("\n")
    return fn_sents

def high_prec_false_positives(X_dv, yhat_raw, y, prec_90_thresh, out_dir, fname, label_name):
    preds = yhat_raw > prec_90_thresh
    fps = ~np.array(y).astype(bool) & preds
    fp_ixs = set(np.where(fps == True)[0])
    fp_sents = []
    with open(fname) as f:
        r = csv.DictReader(f)
        for ix, row in enumerate(r):
            if ix in fp_ixs:
                fp_sents.append(' '.join(eval(row['sentence'])))
    with open(f'{out_dir}/{label_name}_prec_90_fps.txt', 'w') as of:
        for sent in fp_sents:
            of.write(sent + "\n")
    if label_name != 'binary':
        with open(f'{out_dir}/typed_prec_90_fps.txt', 'a') as of:
            of.write(f"#### {label_name} ####\n")
            for sent in fp_sents[:5]:
                of.write(sent + " [[END]]\n")
            of.write("\n")
    return fp_sents


def main(args):
    cvec = CountVectorizer()
    tvec = TfidfVectorizer()
    lvec = None
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.task == 'multilabel':
        lvec = CountVectorizer(tokenizer=lambda x: x.split(';'), lowercase=False, stop_words=[''], vocabulary=LABEL_TYPES)

    print("building train matrices")
    print(args.train_fname)
    Xs, yy = build_data_and_label_matrices(args.train_fname, cvec, tvec=tvec, lvec=lvec, fit=True)
    cX, tX = Xs
    print("building dev matrices")
    dev_fname = args.train_fname.replace('train', 'val')
    print(dev_fname)
    X_dvs, yy_dv = build_data_and_label_matrices(dev_fname, cvec, tvec=tvec, lvec=lvec)
    cX_dv, tX_dv = X_dvs
    print("building test matrices")
    test_fname = args.train_fname.replace('train', 'test')
    print(test_fname)
    X_tes, yy_te = build_data_and_label_matrices(test_fname, cvec, tvec=tvec, lvec=lvec)
    cX_te, tX_te = X_tes

    if args.model_fname:
        clf = pickle.load(open(args.model_fname, 'rb'))
    else:
        solver = 'sag' if args.penalty == 'l2' else 'saga'
        #solver = 'saga'
        #solver = 'liblinear'
        print(f"iterations: {args.max_iter}")
        print(f"solver: {solver}")
        lr_clf = LogisticRegression(
                C=args.C,
                max_iter=args.max_iter,
                penalty=args.penalty,
                l1_ratio=args.l1_ratio,
                solver=solver,
                random_state=args.seed,
                )
        if args.task == 'multilabel':
            clf = OneVsRestClassifier(lr_clf, n_jobs=7)
        else:
            clf = lr_clf

    X = tX if args.feature_type == 'tfidf' else cX
    X_dv = tX_dv if args.feature_type == 'tfidf' else cX_dv
    X_te = tX_te if args.feature_type == 'tfidf' else cX_te
    vec = tvec if args.feature_type == 'tfidf' else cvec

    if not args.model_fname:
        print(f"training {args.feature_type} BOW")
        clf.fit(X, yy)
    yhat = clf.predict(X_dv)
    yhat_raw = clf.predict_proba(X_dv)

    timestamp = time.strftime('%b_%d_%H:%M:%S', time.localtime())
    out_dir = f'results/LR_{timestamp}'

    metrics, label_threshs, micro_thresh, _, _, _ = all_metrics(yhat_raw, yhat, np.array(yy_dv), args.task)

    yhat_raw_te = clf.predict_proba(X_te)
    yhat_te = clf.predict(X_te)
    te_metrics, _, _, _, _, _ = all_metrics(yhat_raw_te, yhat_te, np.array(yy_te), args.task, label_threshs=label_threshs, micro_thresh=micro_thresh)

    # save args
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    with open(f'{out_dir}/model.pkl', 'wb') as of:
        pickle.dump(clf, of)
    with open(f'{out_dir}/vocab.json', 'w') as of:
        v = {w:int(i) for w,i in cvec.vocabulary_.items()}
        json.dump(v, of)
    with open(f'{out_dir}/args.json', 'w') as of:
        of.write(json.dumps(args.__dict__, indent=2) + "\n")
    # save metrics
    with open(f'{out_dir}/metrics.json', 'w') as of:
        of.write(json.dumps(metrics, indent=2) + "\n")

    print(f"Finished! Results at {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_fname", type=str)
    parser.add_argument("--task", choices=['binary', 'multilabel'], default='multilabel')
    parser.add_argument("--C", type=float, default=1.0, help="inverse of regularization strength")
    parser.add_argument("--penalty", choices=['l1', 'l2', 'elasticnet', 'none'],  default='l1', help="type of regularization to use")
    parser.add_argument("--l1_ratio", type=float, default=0.5, help="(for elasticnet only) relative strength of l1 reg term")
    parser.add_argument("--max_iter", type=float, default=5000, help="max number of iterations taken for solvers to converge")
    parser.add_argument("--seed", type=int, default=30024, help="random seed")
    parser.add_argument("--feature_type", choices=['plain', 'tfidf'], default='plain', help="which features to use - tfidf weighted ('tfidf') or not ('plain')")
    parser.add_argument("--model_fname", type=str, help="path to trained model to evaluate")
    parser.add_argument("--print_feats", action="store_true")
    args = parser.parse_args()

    main(args)
