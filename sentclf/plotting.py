from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import recall_score, precision_recall_curve
from tqdm import tqdm

from constants import *
import multilabel_eval
from neural_baselines import SentDataset, SentEvalDataset

def plot_label_rec_values(label_rec_values, out_dir, split):
    # plot recall vs. threshold for each label
    print("per-label recall vs thresholds")
    thresholds = np.arange(0,1,.01)[1:]
    plt.figure()
    for label, recs in label_rec_values.items():
        plt.plot(thresholds, recs, label=label)
    plt.grid(True)
    plt.legend()
    plt.xlabel('threshold value')
    plt.ylabel('recall')
    plt.title('recall vs thresholds')
    plt.savefig(f'{out_dir}/recall_v_thresholds_{split}.png')
    plt.close()

def plot_binarized_recall_vs_thresh(yhat_raw, y, out_dir, split):
    print("binarized recall vs thresholds")
    thresholds = np.concatenate([np.logspace(-5,-1,10)[:-1], np.arange(0,1,.01)[1:]])
    recs = []
    binary_y = np.any(y, axis=1)
    for thresh in thresholds:
        yhat = yhat_raw >= thresh
        binary_yhat = np.any(yhat, axis=1)
        rec = recall_score(binary_y, binary_yhat)
        recs.append(rec)
    plt.plot(thresholds, recs)
    plt.grid(True)
    plt.xlabel('threshold')
    plt.ylabel('recall (binary)')
    plt.title('binary recall, vs threshold')
    plt.savefig(f'{out_dir}/binary_recall_v_thresholds_{split}.png')
    plt.close()

def plot_pct_labeled_vs_thresh(yhat_raw, y, out_dir, split):
    print("pct labeled vs thresh")
    thresholds = np.concatenate([np.logspace(-5,-1,10)[:-1], np.arange(0,1,.01)[1:]])
    pcts = []
    recalls = []
    for thresh in thresholds:
        yhat = yhat_raw >= thresh
        binary_yhat = np.any(yhat, axis=1)
        binary_y = np.any(y, axis=1)
        pct_labeled = sum(binary_yhat) / len(binary_yhat)
        pcts.append(pct_labeled)
        recalls.append(recall_score(binary_y, binary_yhat))
    plt.plot(thresholds, pcts)
    plt.grid(True)
    plt.xlabel('threshold')
    plt.ylabel('% sentences labeled')
    plt.title('% of sentences labeled by model, vs threshold')
    plt.savefig(f'{out_dir}/pct_labeled_v_thresholds_{split}.png')
    plt.close()
    # also save raw values for more involved plotting as needed
    np.save(f'{out_dir}/pcts_labeled_{split}.npy', pcts)
    np.save(f'{out_dir}/thresholds_{split}.npy', thresholds)
    np.save(f'{out_dir}/recalls_{split}.npy', recalls)

def plot_precision_recall_per_label(yhat_raw, y, out_dir, split):
    print("pr curve per label...")
    for ix, label in enumerate(LABEL_TYPES):
        lname = label2abbrev[label]
        yhr_i = yhat_raw[:,ix]
        y_i = y[:,ix]
        p, r, threshs = precision_recall_curve(y_i, yhr_i)
        plt.plot(p, r)
        plt.xlabel('precision')
        plt.ylabel('recall')
        plt.title(lname)
        plt.savefig(f'{out_dir}/pr_curve_{lname}_{split}.png')
        np.save(f'{out_dir}/pr_values_{lname}_{split}.npy', (p, r))

def plot_precision_recall_binary(yhat_raw, y, out_dir, split):
    print("pr curve binary...")
    yhat_raw = np.max(yhat_raw, axis=1)
    y = np.any(y, axis=1)
    p, r, threshs = precision_recall_curve(y, yhat_raw)
    plt.plot(p, r)
    plt.xlabel('precision')
    plt.ylabel('recall')
    plt.title('Binary reduction')
    plt.savefig(f'{out_dir}/pr_curve_binary_{split}.png')
    np.save(f'{out_dir}/pr_values_binary_{split}.npy', (p, r))
    pass

def plot_perf_vs_sent_length(yhat_raw, y, out_dir, loader, tokenizer, label_threshs, micro_thresh, split):
    print("performance vs. sentence length")
    sent_lens = []
    for ix, x in tqdm(enumerate(loader)):
        if isinstance(loader.dataset, SentEvalDataset):
            if isinstance(x, dict):
                toks = [x['sentences'][0].split('[SEP]')[loader.dataset.n_context_sentences].strip().split()]
            else:
                _, _, _, toks, _, _, _ = x
        elif isinstance(loader.dataset, SentDataset):
            toks = [x['sentences'][0].split()]
        else:
            _, _, _, toks = x
        tokd = tokenizer(' '.join(toks[0]), padding=True, max_length = 512, truncation=True)
        sent_lens.append(len(tokd['input_ids']))
    hist_counts, hist_bounds, _ = plt.hist(np.log(sent_lens), bins=10)
    bin_ixs = defaultdict(list)
    bin_yhats = defaultdict(list)
    bin_ys = defaultdict(list)
    for sent_len, yhr_i, y_i in zip(sent_lens, yhat_raw, y):
        bin_ix = 0
        for ix, bound in enumerate(hist_bounds[:-1]):
            if np.log(sent_len) >= bound and np.log(sent_len) <= hist_bounds[ix+1]:
                bin_ix = ix
        bin_yhats[bin_ix].append(yhr_i)
        bin_ys[bin_ix].append(y_i)
    bin_macro_f1s = []
    bin_micro_f1s = []
    for bin_ix in bin_yhats.keys():
        bin_yhat = np.array(bin_yhats[bin_ix])
        bin_y = np.array(bin_ys[bin_ix])
        metrics = multilabel_eval.macro_metrics_given_thresholds(bin_yhat, bin_y, label_threshs)
        yhat = (bin_yhat > micro_thresh).astype(int)
        micro = multilabel_eval.all_micro(yhat.ravel(), bin_y.ravel())

        bin_micro_f1s.append(micro[3])
        bin_macro_f1s.append(metrics['f1_macro'])
    # overlay plots
    bin_micro_f1s = np.nan_to_num(np.array(bin_micro_f1s))
    bin_macro_f1s = np.nan_to_num(np.array(bin_macro_f1s))
    midpoints = (hist_bounds[1:] + hist_bounds[:-1]) / 2
    plt.plot(midpoints, bin_micro_f1s, label='micro F1')
    plt.plot(midpoints, bin_macro_f1s, label='macro F1')
    plt.xlabel('log(sentence length)')
    plt.ylabel('F1')
    plt.legend()
    plt.savefig(f'{out_dir}/perf_vs_sent_len_{split}.png')

    np.save(f'{out_dir}/hist_counts_{split}.npy', hist_counts)
    np.save(f'{out_dir}/hist_bounds_{split}.npy', hist_bounds)
    np.save(f'{out_dir}/bin_macro_f1s_{split}.npy', bin_macro_f1s)
    np.save(f'{out_dir}/bin_micro_f1s_{split}.npy', bin_micro_f1s)
