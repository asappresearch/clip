import argparse
import csv
from collections import Counter, defaultdict
from datetime import date
from dataclasses import dataclass
import glob
import json
import os
import pickle
import random
import sys
import time

from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
from nltk import word_tokenize
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import binary_eval
import multilabel_eval

from constants import *

class MultiCNN(nn.Module):

    def __init__(self, pretrained_embs, embed_size, task, vocab_size, filter_sizes, num_filter_maps=100):
        super(MultiCNN, self).__init__()
        if pretrained_embs:
            embs = torch.Tensor(pretrained_embs)
            self.embed = nn.Embedding.from_pretrained(embs, freeze=False)
        else:
            self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        convs = []
        for ix, fsz in enumerate(filter_sizes):
            conv = nn.Conv1d(embed_size, num_filter_maps, kernel_size=fsz, padding=round(fsz/2))
            nn.init.xavier_uniform_(conv.weight)
            convs.append(conv)
        self.convs = nn.ModuleList(convs)

        self.task = task
        if self.task == 'multilabel':
            self.fc = nn.Linear(num_filter_maps*len(filter_sizes), len(LABEL_TYPES))
        else:
            self.fc = nn.Linear(num_filter_maps*len(filter_sizes), 2)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x, target):
        #embed
        x = self.embed(x)
        x = x.transpose(1,2)
        # conv/max-pooling
        xs = []
        for conv in self.convs:
            xi = conv(x)
            xi = F.max_pool1d(torch.tanh(xi), kernel_size=xi.size()[2])
            xi = xi.squeeze(dim=2)
            xs.append(xi)
        x = torch.cat(xs, dim=1)
        #linear output
        x = self.fc(x)
        #sigmoid to get predictions
        if self.task == 'multilabel':
            loss = F.binary_cross_entropy_with_logits(x, target)
            yhat = torch.sigmoid(x)
        else:
            loss = F.cross_entropy(x, target)
            yhat = torch.softmax(x, dim=1)
        return yhat, loss


class CNN(nn.Module):

    def __init__(self, pretrained_embs, embed_size, task, vocab_size, num_filter_maps=100, filter_size=4):
        super(CNN, self).__init__()
        if pretrained_embs:
            embs = torch.Tensor(pretrained_embs)
            self.embed = nn.Embedding.from_pretrained(embs, freeze=False)
        else:
            self.embed = nn.Embedding(vocab_size+2, embed_size, padding_idx=0)
        self.conv = nn.Conv1d(embed_size, num_filter_maps, kernel_size=filter_size, padding=round(filter_size/2))
        nn.init.xavier_uniform_(self.conv.weight)

        self.task = task
        if (self.task == 'multilabel') or (self.task == 'cnn'):
            self.fc = nn.Linear(num_filter_maps, len(LABEL_TYPES))
        else:
            self.fc = nn.Linear(num_filter_maps, 2)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x, pl_flags=None, target=None):
        #embed
        x = self.embed(x)
        x = x.transpose(1,2)
        # conv/max-pooling
        x = self.conv(x)
        x = F.max_pool1d(torch.tanh(x), kernel_size=x.size()[2])
        x = x.squeeze(dim=2)
        #linear output
        x = self.fc(x)
        #sigmoid to get predictions
        if self.task == 'multilabel':
            yhat = torch.sigmoid(x)
            if pl_flags:
                target[pl_flags] = yhat[pl_flags].round()
            loss = F.binary_cross_entropy_with_logits(x, target)
        elif self.task == 'cnn':
            loss = None 
            yhat = torch.sigmoid(x)
        else:
            loss = F.cross_entropy(x, target)
            yhat = torch.softmax(x, dim=1)
        return yhat, loss

class SentDataset(Dataset):
    def __init__(self, fname, task, word2ix=None, df=None):
        if df is None:
            self.sents = pd.read_csv(fname)
        else:
            self.sents = df
        # reset index necessary for getting context sentences when using unlabeled data
        self.sents = self.sents.dropna(subset=['sentence']).reset_index()
        self.tok_cnt = Counter()
        self.task = task
        if word2ix is not None:
            self.word2ix = word2ix
            self.ix2word = {ix:word for word,ix in self.word2ix.items()}
        else:
            self.word2ix = {'<PAD>': 0}
            self.ix2word = {}
            self.min_cnt = 0

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        row = self.sents.iloc[idx]
        if row.sentence.startswith('[') and row.sentence.endswith(']'):
            try:
                sentence = eval(row.sentence)
            except:
                sentence = word_tokenize(row.sentence)
        else:
            assert isinstance(row.sentence, str)
            sentence = word_tokenize(row.sentence)
        sent = [x.lower() for x in sentence]
        if self.task == 'multilabel':
            if 'labels' in row:
                labels = eval(row.labels)
            else:
                labels = []
        else:
            labels = row.labels
        doc_id = row.doc_id
        sent_ix = None
        if 'sent_index' in row:
            sent_ix = row.sent_index
        elif 'sent_ix' in row:
            sent_ix = row.sent_ix
        return sent, labels, doc_id, sent_ix

    def build_vocab(self):
        print("building vocab...")
        for ix in tqdm(range(len(self))):
            sent, labels, doc_id = self[ix]
            self.tok_cnt.update(sent)
        # add 1 for pad token
        self.word2ix.update({word: ix+1 for ix,(word,count) in enumerate(sorted(self.tok_cnt.items(), key=lambda x: x[0])) if count > self.min_cnt})
        # add UNK to the end
        self.word2ix[UNK] = len(self.word2ix)
        self.ix2word = {ix:word for word,ix in self.word2ix.items()}

    def set_vocab(self, vocab_file):
        # add 1 for pad token
        self.word2ix.update({row.strip(): ix+1 for ix,row in enumerate(open(vocab_file))})
        # add UNK to the end
        self.word2ix[UNK] = len(self.word2ix)
        self.ix2word = {ix:word for word,ix in self.word2ix.items()}

class SentEvalDataset(SentDataset):
    def __init__(self, fname, task, word2ix=None, n_context_sentences=0, doc_position=False, df=None):
        super().__init__(fname, task, word2ix, df=df)
        self.n_context_sentences = n_context_sentences
        self.doc_position = doc_position

    def __getitem__(self, idx):
        row = self.sents.iloc[idx]
        try:
            #sent = [x.lower() for x in eval(row.sentence)]
            sent = eval(row.sentence)
        except:
            assert isinstance(row.sentence, str)
            #sent = word_tokenize(row.sentence.lower())
            sent = word_tokenize(row.sentence)
        try:
            if self.task == 'multilabel':
                labels = eval(row.labels)
            else:
                labels = row.labels
        except:
            labels = []
        doc_id = row.doc_id
        contexts = []
        num_padding = 0
        ctx_sents = []

        if 'sent_index' in row:
            sent_ix = 'sent_index'
        elif 'sent_ix' in row:
            sent_ix = 'sent_ix'

        for offset in range(-self.n_context_sentences, self.n_context_sentences+1):
            if offset == 0:
                continue
            ctx_ix = row[sent_ix] + offset
            if ctx_ix < 0:
                ctx_sents.append(['<DOC_START>'])
            elif idx + offset >= len(self.sents) or self.sents.iloc[idx + offset][sent_ix] != row[sent_ix] + offset:
                ctx_sents.append(['<DOC_END>'])
            else:
                try:
                    #ctx_sent = [x.lower() for x in eval(self.sents.iloc[idx+offset].sentence)]
                    ctx_sent = [str(x) for x in eval(self.sents.iloc[idx+offset].sentence)]
                except:
                    assert isinstance(self.sents.iloc[idx+offset].sentence, str)
                    #ctx_sent = word_tokenize(self.sents.iloc[idx+offset].sentence.lower())
                    ctx_sent = word_tokenize(self.sents.iloc[idx+offset].sentence)
                ctx_sents.append(ctx_sent)

        contexts = tuple(' '.join(ctx_sent) for ctx_sent in ctx_sents)
        annotator = row.annotator if 'annotator' in row else ''
        if self.doc_position:
            n_sents = self.sents[self.sents['doc_id'] == doc_id].iloc[-1][sent_ix] + 1
            doc_position = row[sent_ix] / n_sents
            return sent, row[sent_ix], annotator, labels, doc_id, contexts, doc_position
        else:
            return sent, row[sent_ix], annotator, labels, doc_id, contexts

def sugg_collate(batch, word2ix, task):
    sents, labels, doc_ids, toks, sent_ixs, annotators, contexts = [], [], [], [], [], [], []
    # sort by decreasing length
    batch = sorted(batch, key=lambda x: -len(x[0]))
    max_length = len(batch[0][0])
    for sent, sent_ix, annotator, label, doc_id, context in batch:
        toks.append(sent)
        sent = [word2ix.get(w, word2ix[UNK]) for w in sent]
        sent.extend([0 for ix in range(len(sent), max_length)])
        sents.append(sent)
        if task == 'multilabel':
            label_ixs = [LABEL_TYPES.index(l) for l in label]
            label = np.zeros(len(LABEL_TYPES))
            label[label_ixs] = 1
            labels.append(label)
        else:
            labels.append(int(label))
        doc_ids.append(doc_id)
        sent_ixs.append(sent_ix)
        annotators.append(annotator)
        contexts.append(context)
    if task == 'multilabel':
        labels = torch.Tensor(labels)
    else:
        labels = torch.LongTensor(labels)
    return torch.LongTensor(sents), labels, doc_ids, toks, sent_ixs, annotators, contexts

def collate(batch, word2ix, task):
    sents, labels, doc_ids, toks = [], [], [], []
    # sort by decreasing length
    batch = sorted(batch, key=lambda x: -len(x[0]))
    max_length = len(batch[0][0])
    for sent, label, doc_id in batch:
        toks.append(sent)
        sent = [word2ix.get(w, word2ix[UNK]) for w in sent]
        sent.extend([0 for ix in range(len(sent), max_length)])
        sents.append(sent)
        if task == 'multilabel':
            label_ixs = [LABEL_TYPES.index(l) for l in label]
            label = np.zeros(len(LABEL_TYPES))
            label[label_ixs] = 1
            labels.append(label)
        else:
            labels.append(int(label))
        doc_ids.append(doc_id)
    if task == 'multilabel':
        labels = torch.Tensor(labels)
    else:
        labels = torch.LongTensor(labels)
    return torch.LongTensor(sents), labels, doc_ids, toks

def check_best_model_and_save(model, metrics_hist, criterion, out_dir):
    is_best = False
    if criterion == 'loss':
        if np.isnan(metrics_hist[criterion]).all() or np.nanargmin(metrics_hist[criterion]) == len(metrics_hist[criterion]) - 1:
            # save model
            sd = model.state_dict()
            torch.save(sd, f'{out_dir}/model_best_{criterion}.pth')
            is_best = True
    else:
        if np.isnan(metrics_hist[criterion]).all() or np.nanargmax(metrics_hist[criterion]) == len(metrics_hist[criterion]) - 1:
            # save model
            sd = model.state_dict()
            torch.save(sd, f'{out_dir}/model_best_{criterion}.pth')
            is_best = True
    return is_best

def save_metrics(metrics_hist, out_dir):
    # save predictions
    if out_dir is not None and not os.path.exists(out_dir):
        os.mkdir(out_dir)
    with open(f'{out_dir}/metrics.json', 'w') as of:
        json.dump(metrics_hist, of, indent=1)

def early_stop(metrics_hist, criterion, patience):
    if len(metrics_hist[criterion]) >= patience:
        if criterion == 'loss':
            return np.nanargmin(metrics_hist[criterion]) < len(metrics_hist[criterion]) - patience
        else:
            return np.nanargmax(metrics_hist[criterion]) < len(metrics_hist[criterion]) - patience
    else:
        return False

def select_all_info_ixs(loader, ixs):
    doc_ids, labels, sents, sent_ixs, annotators, contexts = [], [], [], [], [], []
    for ix, x in tqdm(enumerate(loader)):
        _, label, doc_id, toks, sent_ix, annotator, context = x
        if ix in ixs:
            sents.append(' '.join(toks[0]))
            sent_ixs.append(sent_ix[0])
            annotators.append(annotator[0])
            doc_ids.append(doc_id[0])
            lbls = np.array(LABEL_TYPES)[np.where(label[0].numpy())[0]]
            labels.append(lbls)
            contexts.append(context[0])
    return doc_ids, sent_ixs, annotators, sents, labels, contexts

def multilabel_labeler(label):
    label_ixs = [LABEL_TYPES.index(l) for l in label]
    label = np.zeros(len(LABEL_TYPES))
    label[label_ixs] = 1
    return label

def generate_suggestions(loader, yhat_raw, y, rec_60_thresh, out_dir, label_name, fun_sent_selection, wsinfo, sugg_type):
    # iterate over examples in loader and write examples to XLSX of high confidence false predictions using rec_60_thresh
    preds = yhat_raw > rec_60_thresh
    if sugg_type == 'neg':
        fs = np.array(y).astype(bool) & ~preds
    else:
        fs = ~np.array(y).astype(bool) & preds
    f_ixs = set(np.where(fs == True)[0])
    f_sents = fun_sent_selection(loader, f_ixs)
    doc_ids, sent_ixs, annotators, sents, labels, contexts = f_sents
    label_order = [label2abbrev[l] for l in LABEL_TYPES]
    if label_name != 'binary':
        with open(f'{out_dir}/train_{sugg_type}_sugg_60.csv', 'a') as of:
            w = csv.writer(of)
            w.writerow(['doc_id', 'sent_ix', 'annotator', 'sentence', 'labels', 'keep?'])
            if wsinfo.row_num == 0:
                wsinfo.worksheet.write_row('A1', ('doc_id', 'sent_ix', 'annotator', 'sentence') + tuple(label_order))
                wsinfo.row_num = 2
            for doc_id, sent_ix, annotator, sent, lbls, context in zip(doc_ids, sent_ixs, annotators, sents, labels, contexts):
                if len(lbls) > 0:
                    w.writerow([doc_id, sent_ix, annotator, sent, str(lbls), ''])
                    try:
                        annotator = '' if np.isnan(annotator) else annotator
                    except:
                        annotator = ''
                    wsinfo.worksheet.write_row(f'A{wsinfo.row_num}', (doc_id, sent_ix, annotator))
                    wsinfo.worksheet.write_rich_string(f'D{wsinfo.row_num}', context[0], context[1], wsinfo.bold, sent, context[2], context[3])
                    label_ixs = multilabel_labeler(lbls)
                    wsinfo.worksheet.write_row(f'E{wsinfo.row_num}', tuple(label_ixs))
                    wsinfo.row_num += 1
    return wsinfo

def high_thresh_suggestions(loader, yhat_raw, y, thresh, out_dir, label_name, split, sugg_type):
    # write examples to a text file, from high precision threshold
    print(f"saving false positives for {label_name} label")
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    preds = yhat_raw > thresh
    if sugg_type == 'neg':
        fs = np.array(y).astype(bool) & ~preds
        typ = 'fns'
    elif sugg_type == 'pos':
        fs = ~np.array(y).astype(bool) & preds
        typ = 'fps'
    elif sugg_type == 'true_pos':
        fs = np.array(y).astype(bool) & preds
        typ = 'tps'
    MAX_SENTS = 20
    f_ixs = set(np.where(fs == True)[0])
    f_sents = []
    for ix, x in tqdm(enumerate(loader)):
        if isinstance(loader.dataset, SentEvalDataset):
            if isinstance(x, dict):
                toks = [x['sentences'][0].split('[SEP]')[loader.dataset.n_context_sentences].strip().split()]
            else:
                _, _, _, toks, _, _, _ = x
        else:
            _, _, _, toks = x
        if ix in f_ixs:
            f_sents.append(' '.join(toks[0]))
            if len(f_sents) >= MAX_SENTS:
                break
    with open(f'{out_dir}/{label_name}_thresh_{typ}_{split}.txt', 'w') as of:
        for sent in f_sents:
            of.write(sent + "\n")
    if label_name != 'binary':
        with open(f'{out_dir}/typed_thresh_{typ}_{split}.txt', 'a') as of:
            of.write(f"#### {label_name} ####\n")
            for sent in f_sents[:5]:
                of.write(sent + " [[END]]\n")
            of.write("\n")
    return f_sents 

def all_metrics(yhat_raw, yhat, y, task, label_threshs=None, micro_thresh=None, rec_90_thresh=None, rec_95_thresh=None, rec_99_thresh=None, do_print=True):
    metrics = {}
    if task == 'multilabel':
        if label_threshs is not None:
            # macro (and label-specific) first
            if do_print:
                print(f"label thresholds: {label_threshs}")
            metrics = multilabel_eval.macro_metrics_given_thresholds(yhat_raw, y, label_threshs)
            #then micro
            if do_print:
                print(f"micro threshold: {micro_thresh}")
            yhat = (yhat_raw > micro_thresh).astype(int)
            micro = multilabel_eval.all_micro(yhat.ravel(), y.ravel())
            metrics.update({f'{name}_micro': val for name, val in zip(['acc', 'prec', 'rec', 'f1'], micro)})
            metrics['auc_micro'] = roc_auc_score(y.ravel(),yhat_raw.ravel())

        else:
            metrics = multilabel_eval.all_metrics(yhat, y, yhat_raw=yhat_raw, calc_auc=True, label_order=LABEL_TYPES)

            label_metrics, label_threshs, label_hp_threshs, label_hp_recs, _ = multilabel_eval.balance_each_label(yhat_raw, y)
            metrics.update(label_metrics)

            micro_thresh, balanced_metrics = multilabel_eval.balanced_f1(yhat_raw, y)
            if balanced_metrics is not None:
                print(f"threshold = {micro_thresh}")
                metrics.update({f'balanced_{key}': val for key, val in balanced_metrics.items()})

                #label_type_metrics = multilabel_eval.f1_per_label_type(yhat_raw, y, LABEL_TYPES, micro_thresh)
                #metrics.update(label_type_metrics)

        binary_yhat = np.any(yhat_raw >= np.array(label_threshs), axis=1)
        binary_y = np.any(y, axis=1)
        metrics['binary_prec'] = precision_score(binary_y, binary_yhat)
        metrics['binary_rec'] = recall_score(binary_y, binary_yhat)
        metrics['binary_f1'] = f1_score(binary_y, binary_yhat)

        binary_yhat_raw = np.max(yhat_raw, axis=1)
        binary_yhat_raw = np.stack((1 - binary_yhat_raw, binary_yhat_raw), axis=1)
        if do_print:
            print("getting recall at fixed thresholds...")
        rec_60_thresh, prec_at_rec_60 = binary_eval.precision_at_fixed_recall(binary_yhat_raw, binary_y, 0.60)
        metrics['prec@rec=60'] = prec_at_rec_60
        if rec_90_thresh is None:
            rec_90_thresh, prec_at_rec_90 = binary_eval.precision_at_fixed_recall(binary_yhat_raw, binary_y, 0.90)
            metrics['prec@rec=90'] = prec_at_rec_90
        else:
            binary_yhat = (binary_yhat_raw[:,1] >= rec_90_thresh).astype(int)
            metrics['prec@rec=90'] = precision_score(binary_y, binary_yhat)
            metrics['rec@rec=90'] = recall_score(binary_y, binary_yhat)
        if rec_95_thresh is None:
            rec_95_thresh, prec_at_rec_95 = binary_eval.precision_at_fixed_recall(binary_yhat_raw, binary_y, 0.95)
            metrics['prec@rec=95'] = prec_at_rec_95
        else:
            binary_yhat = (binary_yhat_raw[:,1] >= rec_95_thresh).astype(int)
            metrics['prec@rec=95'] = precision_score(binary_y, binary_yhat)
            metrics['rec@rec=95'] = recall_score(binary_y, binary_yhat)
        if rec_99_thresh is None:
            rec_99_thresh, prec_at_rec_99 = binary_eval.precision_at_fixed_recall(binary_yhat_raw, binary_y, 0.99)
            metrics['prec@rec=99'] = prec_at_rec_99
        else:
            binary_yhat = (binary_yhat_raw[:,1] >= rec_99_thresh).astype(int)
            metrics['prec@rec=99'] = precision_score(binary_y, binary_yhat)
            metrics['rec@rec=99'] = recall_score(binary_y, binary_yhat)

        prec_at_rec_vals = ['prec@rec=90', 'prec@rec=95', 'prec@rec=99']
        if rec_90_thresh is not None and 'rec@rec=90' in metrics:
            prec_at_rec_vals.append('rec@rec=90')
        if rec_95_thresh is not None and 'rec@rec=95' in metrics:
            prec_at_rec_vals.append('rec@rec=95')
        if rec_99_thresh is not None and 'rec@rec=99' in metrics:
            prec_at_rec_vals.append('rec@rec=99')
        header_str = ','.join(prec_at_rec_vals)
        values_str = ','.join([f"{metrics[val]:.4f}" for val in prec_at_rec_vals])
        if do_print:
            print(header_str)
            print(values_str)

    else:
        micro_thresh, balanced_metrics = binary_eval.balanced_f1(yhat_raw, y)
        acc, prec, rec, f1, auc = balanced_metrics['acc'], balanced_metrics['prec'], balanced_metrics['rec'], balanced_metrics['f1'], balanced_metrics['auc'], 
        metrics.update({f'balanced_{key}': val for key, val in balanced_metrics.items()})
        print(f"threshold={micro_thresh}")

        prec_90_thresh, rec_at_prec_90 = binary_eval.recall_at_fixed_precision(yhat_raw, y, 0.90)
        metrics['rec@prec=90'] = rec_at_prec_90
        rec_60_thresh, prec_at_rec_60 = binary_eval.precision_at_fixed_recall(yhat_raw, y, 0.60)
        metrics['prec@rec=60'] = prec_at_rec_60
        rec_90_thresh, prec_at_rec_90 = binary_eval.precision_at_fixed_recall(yhat_raw, y, 0.90)
        metrics['prec@rec=90'] = prec_at_rec_90
        rec_95_thresh, prec_at_rec_95 = binary_eval.precision_at_fixed_recall(yhat_raw, y, 0.95)
        metrics['prec@rec=95'] = prec_at_rec_95
        rec_99_thresh, prec_at_rec_99 = binary_eval.precision_at_fixed_recall(yhat_raw, y, 0.99)
        metrics['prec@rec=99'] = prec_at_rec_99

    if do_print:
        print_metrics(metrics, task)
    return metrics, label_threshs, micro_thresh, rec_90_thresh, rec_95_thresh, rec_99_thresh

def run_on_test(model, loader, task, out_dir, label_threshs, micro_thresh):
    with torch.no_grad():
        model.eval()
        yhat_raw = []
        yhat = []
        y = []
        print("EVALUATING MODEL ON TEST DATASET...")
        for ix, x in tqdm(enumerate(loader)):
            sent, label, doc_id, *_ = x
            pred, _ = model(sent.to(DEVICE), target=label.to(DEVICE))
            pred = pred.cpu().numpy()[0]
            yhat_raw.append(pred)
            if task == 'multilabel':
                yhat.append(np.round(pred))
            else:
                yhat.append(np.argmax(pred))
            y.append(label.cpu().numpy()[0])
        yhat = np.array(yhat)
        yhat_raw = np.array(yhat_raw)
        y = np.array(y)
        metrics, label_threshs, micro_thresh, _, _, _ = all_metrics(yhat_raw, yhat, y, task, label_threshs=label_threshs, micro_thresh=micro_thresh)
    return metrics

def eval_loop(model, loader, task, out_dir, fp_fn_analysis=False, suggestions_fold="", get_high_prec_thresh=False, return_thresholds=False):
    label_threshs = []
    with torch.no_grad():
        model.eval()
        yhat_raw = []
        yhat = []
        y = []
        print("evaluating model on dataset...")
        for ix, x in tqdm(enumerate(loader)):
            sent, label, doc_id, *_ = x
            pred, _ = model(sent.to(DEVICE), target=label.to(DEVICE))
            pred = pred.cpu().numpy()[0]
            yhat_raw.append(pred)
            if task == 'multilabel':
                yhat.append(np.round(pred))
            else:
                yhat.append(np.argmax(pred))
            y.append(label.cpu().numpy()[0])
        yhat = np.array(yhat)
        yhat_raw = np.array(yhat_raw)
        y = np.array(y)
        metrics = {}
        if task == 'multilabel':
            metrics = multilabel_eval.all_metrics(yhat, y, yhat_raw=yhat_raw, calc_auc=True, label_order=LABEL_TYPES)

            label_metrics, label_threshs, label_hp_threshs, label_hp_recs, _ = multilabel_eval.balance_each_label(yhat_raw, y, get_high_prec_thresh=get_high_prec_thresh)
            metrics.update(label_metrics)
            if len(label_hp_threshs):
                print("label cutoffs for high precision:")
                for ix,label in enumerate(LABEL_TYPES):
                    print(f"{label2abbrev[label]}: {label_hp_threshs[ix]} (recall: {label_hp_recs[ix]})")

            if not suggestions_fold:
                # choose threshold to optimize micro f1, and print per-label performance
                micro_thresh, balanced_metrics = multilabel_eval.balanced_f1(yhat_raw, y)
                print(f"threshold = {micro_thresh}")
                metrics.update({f'balanced_{key}': val for key, val in balanced_metrics.items()})

                label_type_metrics = multilabel_eval.f1_per_label_type(yhat_raw, y, LABEL_TYPES, micro_thresh)
                metrics.update(label_type_metrics)

            binary_yhat = np.any(yhat_raw >= np.array(label_threshs), axis=1)
            binary_y = np.any(y, axis=1)
            metrics['binary_prec'] = precision_score(binary_y, binary_yhat)
            metrics['binary_rec'] = recall_score(binary_y, binary_yhat)
            metrics['binary_f1'] = f1_score(binary_y, binary_yhat)

            if not suggestions_fold and fp_fn_analysis:
                thresh_metrics, rec_90_threshs, prec_90_threshs, rec_60_threshs, prec_60_threshs = multilabel_eval.balanced_f1_multilabel(yhat_raw, y)
                for ix, label in enumerate(LABEL_TYPES):
                    lname = label2abbrev[label]
                    high_prec_fps = high_thresh_suggestions(loader, yhat_raw[:,ix], y[:,ix], prec_90_threshs[lname], out_dir, label2abbrev[label], 'val', 'pos')
                    high_rec_fns = high_thresh_suggestions(loader, yhat_raw[:,ix], y[:,ix], rec_90_threshs[lname], out_dir, label2abbrev[label], 'val', 'neg')
            elif suggestions_fold != '':
                import xlsxwriter
                @dataclass
                class WorksheetInfo:
                    worksheet: xlsxwriter.worksheet.Worksheet
                    row_num: int
                    bold: xlsxwriter.format.Format

                workbook = xlsxwriter.Workbook(f'{out_dir}/annotation_suggestions_{suggestions_fold}.xlsx')
                worksheet_n = workbook.add_worksheet('Possible negatives')
                bold = workbook.add_format({'bold': True})
                row_num_n = 0
                wsinfo_n = WorksheetInfo(worksheet_n, row_num_n, bold)

                worksheet_p = workbook.add_worksheet('Possible positives')
                row_num_p = 0
                wsinfo_p = WorksheetInfo(worksheet_p, row_num_p, bold)
                for ix, label in enumerate(LABEL_TYPES):
                    lname = label2abbrev[label]
                    print(f"suggesting for label {lname}")
                    wsinfo_n = generate_suggestions(loader, yhat_raw[:,ix], y[:,ix], 0.5, out_dir, lname, select_all_info_ixs, wsinfo_n, 'neg')
                    wsinfo_p = generate_suggestions(loader, yhat_raw[:,ix], y[:,ix], 0.5, out_dir, lname, select_all_info_ixs, wsinfo_p, 'pos')
                workbook.close()
        else:
            micro_thresh, balanced_metrics = binary_eval.balanced_f1(yhat_raw, y)
            metrics.update({f'balanced_{key}': val for key, val in balanced_metrics.items()})
            print(f"threshold={micro_thresh}")

            prec_90_thresh, rec_at_prec_90 = binary_eval.recall_at_fixed_precision(yhat_raw, y, 0.90)
            metrics['rec@prec=90'] = rec_at_prec_90
            rec_60_thresh, prec_at_rec_60 = binary_eval.precision_at_fixed_recall(yhat_raw, y, 0.60)
            metrics['prec@rec=60'] = prec_at_rec_60
            rec_90_thresh, prec_at_rec_90 = binary_eval.precision_at_fixed_recall(yhat_raw, y, 0.90)
            metrics['prec@rec=90'] = prec_at_rec_90
            rec_95_thresh, prec_at_rec_95 = binary_eval.precision_at_fixed_recall(yhat_raw, y, 0.95)
            metrics['prec@rec=95'] = prec_at_rec_95
            rec_99_thresh, prec_at_rec_99 = binary_eval.precision_at_fixed_recall(yhat_raw, y, 0.99)
            metrics['prec@rec=99'] = prec_at_rec_99
            

            if not suggestions_fold and fp_fn_analysis:
                high_prec_fps = high_thresh_suggestions(loader, yhat_raw[:,1], y, prec_90_thresh, out_dir, 'binary', 'val', 'neg')
                high_rec_fns = high_thresh_suggestions(loader, yhat_raw[:,1], y, rec_90_thresh, out_dir, 'binary', 'val', 'pos')
            else:
                pass
        print_metrics(metrics, task)
    if return_thresholds:
        return metrics, label_threshs, micro_thresh
    else:
        return metrics

def print_metrics(metrics, task):
    if task == 'binary':
        acc, prec, rec, f1, auc = metrics['balanced_acc'], metrics['balanced_prec'], metrics['balanced_rec'], metrics['balanced_f1'], metrics['balanced_auc'], 
        print("accuracy, precision, recall, f1, AUROC")
        print(f"{acc:.4f},{prec:.4f},{rec:.4f},{f1:.4f},{auc:.4f}")
        prec_at_rec_vals = ['prec@rec=90', 'prec@rec=95', 'prec@rec=99']
        header_str = ','.join(prec_at_rec_vals)
        values_str = ','.join([f"{metrics[val]:.4f}" for val in prec_at_rec_vals])
        print(header_str)
        print(values_str)
    else:
        multilabel_eval.print_metrics(metrics, True, True)


def get_embeddings(type_, path, id_to_token):
    if type_ == "GLOVE":
        pretrained = {}
        for line in open(path):
            parts = line.strip().split()
            word = parts[0]
            vector = [float(v) for v in parts[1:]]
            pretrained[word] = vector
    else:
        pretrained = KeyedVectors.load_word2vec_format(
            os.path.join(path, "BioWordVec_PubMed_MIMICIII_d200.vec.bin"), binary=True
        )
    pretrained_list = []
    scale = np.sqrt(3.0 / DIM_EMBEDDING)
    for word in id_to_token:
        # apply lower() because all GloVe vectors are for lowercase words
        if word.lower() in pretrained:
            pretrained_list.append(np.array(pretrained[word.lower()]))
        else:
            if word == '__PAD__':
                random_vector = np.zeros(DIM_EMBEDDING)
            else:
                random_vector = np.random.uniform(-scale, scale, [DIM_EMBEDDING])
            pretrained_list.append(random_vector)
    return pretrained_list


def main(args):
    dev_fname = args.train_fname.replace('train', 'val')
    test_fname = args.train_fname.replace('train', 'test')
    tr_data = SentDataset(args.train_fname, args.task)
    collate_fn = lambda batch: collate(batch, tr_data.word2ix, args.task)
    if not args.vocab_file:
        tr_data.build_vocab()
        date_str = date.today().strftime('%Y%m%d')
        with open(f'cnn_vocab_{date_str}.txt', 'w') as of:
            for word, _ in sorted(tr_data.word2ix.items(), key=lambda x: x[1]):
                of.write(word + "\n")
    else:
        tr_data.set_vocab(args.vocab_file)
    tr_loader = DataLoader(tr_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    dv_data = SentDataset(dev_fname, args.task, tr_data.word2ix)
    dv_loader = DataLoader(dv_data, batch_size=1, shuffle=False, collate_fn=lambda batch: collate(batch, tr_data.word2ix, args.task))

    timestamp = time.strftime('%b_%d_%H:%M:%S', time.localtime())

    # load pre-trained embeddings
    if args.embed_file is None or len(glob.glob(args.embed_file)) == 0:
        word_list = [word for ix,word in sorted(tr_data.ix2word.items(), key=lambda x: x[0])]
        pretrained_embs = get_embeddings("BioWord", "../tagger/embeddings/", word_list)
        emb_out_fname = f'cnn_embs_{timestamp}.pkl'
        pickle.dump(pretrained_embs, open(emb_out_fname, 'wb'))
    else:
        pretrained_embs = pickle.load(open(args.embed_file, 'rb'))


    if args.model == 'cnn':
        # add one for UNK
        model = CNN(pretrained_embs, args.embed_size, args.task, len(tr_data.word2ix)+1, args.num_filter_maps, args.filter_size)
    elif args.model == 'multicnn':
        model = MultiCNN(pretrained_embs, args.embed_size, args.task, len(tr_data.word2ix)+1, args.filter_size, args.num_filter_maps)
    if args.local_weights:
        print(f"Loading local weights")
        sd = torch.load(args.local_weights)
        model.load_state_dict(sd)
    model.to(DEVICE)

    optimizer = optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)

    if args.eval_model:
        out_dir = '/'.join(args.local_weights.split('/')[:-1])
    else:
        out_dir = f'results/{args.model}_{timestamp}'
        print(f"will put results in {out_dir}")

    losses = []
    metrics_hist = defaultdict(list)
    best_epoch = 0
    best_iter = 0
    model.train()
    stop_training = False
    for epoch in range(args.max_epochs):
        if args.eval_model:
            break
        for batch_ix, batch in tqdm(enumerate(tr_loader)):
            if batch_ix > args.max_iter:
                break
            optimizer.zero_grad()
            sents, labels, doc_ids, _ = batch
            yhat, loss = model(sents.to(DEVICE), target=labels.to(DEVICE))
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

            if args.eval_iter and (batch_ix + 1) % args.eval_iter == 0:
                metrics = eval_loop(model, dv_loader, args.task, out_dir)
                for name, metric in metrics.items():
                    metrics_hist[name].append(metric)

                # save best model, creating results dir if needed
                if not os.path.exists(out_dir):
                    os.mkdir(out_dir)
                save_metrics(metrics_hist, out_dir)
                is_best = check_best_model_and_save(model, metrics_hist, args.criterion, out_dir)
                if is_best:
                    best_epoch = epoch
                    best_iter = batch_ix

                if early_stop(metrics_hist, args.criterion, args.patience):
                    print(f"{args.criterion} hasn't improved in {args.patience} evaluations, early stopping...")
                    stop_training = True
                    break
         
        if stop_training:
            break

        # eval
        metrics = eval_loop(model, dv_loader, args.task, out_dir)
        for name, metric in metrics.items():
            metrics_hist[name].append(metric)

        # save best model, creating results dir if needed
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        save_metrics(metrics_hist, out_dir)
        is_best = check_best_model_and_save(model, metrics_hist, args.criterion, out_dir)
        if is_best:
            best_epoch = epoch

        if early_stop(metrics_hist, args.criterion, args.patience):
            print(f"{args.criterion} hasn't improved in {args.patience} evaluations, early stopping...")
            stop_training = True
            break
 
        if stop_training:
            break

    # save args
    if not args.eval_model:
        with open(f'{out_dir}/args.json', 'w') as of:
            of.write(json.dumps(args.__dict__, indent=2) + "\n")

    if args.max_epochs > 0 and not args.eval_model:
        # save the model at the end
        sd = model.state_dict()
        torch.save(sd, out_dir + "/model.pth")

        # reload the best model
        print(f"\nReloading and evaluating model with best {args.criterion} (epoch {best_epoch} iter {best_iter})")
        sd = torch.load(f'{out_dir}/model_best_{args.criterion}.pth')
        model.load_state_dict(sd)

    # save suggestions, for revision purposes
    if args.task == 'multilabel' and args.make_suggestions:
        print("making suggestions...")
        dv_sugg_data = SentEvalDataset(dev_fname, args.task, tr_data.word2ix, n_context_sentences=2)
        dv_sugg_loader = DataLoader(dv_sugg_data, batch_size=1, shuffle=False, collate_fn=lambda batch: sugg_collate(batch, tr_data.word2ix, args.task))
        print("Making suggestions on valid data (first calculating metrics)")
        eval_loop(model, dv_sugg_loader, args.task, out_dir, suggestions_fold="valid")

        te_sugg_data = SentEvalDataset(test_fname, args.task, tr_data.word2ix, n_context_sentences=2)
        te_sugg_loader = DataLoader(te_sugg_data, batch_size=1, shuffle=False, collate_fn=lambda batch: sugg_collate(batch, tr_data.word2ix, args.task))
        print("Making suggestions on test data (first calculating metrics)")
        eval_loop(model, te_sugg_loader, args.task, out_dir, suggestions_fold="test")

    # eval on dev at end
    metrics, label_threshs, micro_thresh = eval_loop(model, dv_loader, args.task, out_dir, get_high_prec_thresh=args.high_prec_thresholds, return_thresholds=True)

    if args.run_test or args.eval_model:
        te_data = SentDataset(test_fname, args.task, tr_data.word2ix)
        te_loader = DataLoader(te_data, batch_size=1, shuffle=False, collate_fn=lambda batch: collate(batch, tr_data.word2ix, args.task))
        test_metrics = run_on_test(model, te_loader, args.task, out_dir, label_threshs, micro_thresh)
    print(f"done! results at {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_fname", type=str)
    parser.add_argument("model", choices=['cnn', 'lstm', 'multicnn'])
    parser.add_argument("--embed_file", type=str, help="path to a file holding pre-trained token embeddings", default='embs.pkl')
    parser.add_argument("--task", choices=['binary', 'multilabel'], default='multilabel')
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--max_iter", type=int, default=1e10, help="max iterations (batches) to train on - use for debugging")
    parser.add_argument("--criterion", type=str, default="auc_micro", required=False, help="metric to use for early stopping")
    parser.add_argument("--patience", type=int, default=5, required=False, help="epochs to wait for improved criterion before early stopping (default 5)")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--eval_iter", type=int, help="set to evaluate every x batches, in addition to at the end of each epoch.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--embed_size", type=int, default=200)
    parser.add_argument("--num_filter_maps", type=int, default=100)
    parser.add_argument("--filter_size", type=str, default="4", help="filter size for CNN. optionally multiple when using multicnn - input separated by commas e.g. 2,4,6")
    parser.add_argument("--vocab_file", type=str, help="path to precomputed vocab")
    parser.add_argument("--seed", type=int, default=11, help="random seed")
    parser.add_argument("--run_test", action="store_true", help="set to run on test too after running on dev at the end")
    parser.add_argument("--eval_model", action="store_true", help="set to run only on test, for use with local_weights to test saved model")
    parser.add_argument("--make_suggestions", action='store_true')
    parser.add_argument("--high_prec_thresholds", action='store_true')
    parser.add_argument("--local_weights", type=str, required=False, help="optionally point to a file with local weights corresponding to given model type")
    args = parser.parse_args()

    # change default criterion to f1 for binary
    if args.task == 'binary' and args.criterion == 'f1_micro':
        args.criterion = 'f1'
    if ',' in args.filter_size:
        if args.model != 'multicnn':
            print(f"provided multiple filter sizes but model was {args.model} instead of multicnn.")
            import sys; sys.exit(0)
        args.filter_size = [int(fsz) for fsz in args.filter_size.split(',')]
    else:
        if args.model == 'cnn':
            args.filter_size = int(args.filter_size)
        elif args.model == 'multicnn':
            args.filter_size = [int(args.filter_size)]

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(args)
    main(args)
