""" Finetuning library models for sequence classification on MIMIC discharge summaries """

import argparse
from collections import defaultdict
import csv
import json
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification
from transformers import (
    Trainer,
    TrainingArguments,
    set_seed,
)

from bert_model import (
        BertForSequenceMultilabelClassification,
        BertSequenceMultilabelClassificationContext,
        BertCNNContextMultilabel,
        )
import binary_eval
from constants import *
import multilabel_eval
from neural_baselines import all_metrics, print_metrics, check_best_model_and_save, early_stop, high_thresh_suggestions, save_metrics
from neural_baselines import SentDataset, SentEvalDataset
from plotting import *

def multilabel_labeler(label):
    # convert list of label names to a multi-hot array
    if isinstance(label, str):
        label = eval(label)
    label_ixs = [LABEL_TYPES.index(l) for l in label]
    label = np.zeros(len(LABEL_TYPES))
    label[label_ixs] = 1
    return label

def collator(batch, task, tokenizer, eval=False, doc_position=False):
    # standard bert collator that applies the right label for the training task
    sents = []
    labels = []
    doc_poses = []
    doc_ids = []
    sent_ixs = []
    for sent, label, doc_id, sent_ix in batch:
        sents.append(' '.join(sent))
        label = multilabel_labeler(label) if task == 'multilabel' else int(label)
        labels.append(label)
        doc_ids.append(doc_id)
        sent_ixs.append(sent_ix)
    tokd = tokenizer(sents, padding=True, max_length = 512, truncation=True)
    input_ids, token_type_ids, attention_mask = tokd['input_ids'], tokd['token_type_ids'], tokd['attention_mask']
    toks = torch.LongTensor(input_ids)
    mask = torch.LongTensor(attention_mask)
    labels = torch.Tensor(labels)
    # for whatever reason, pytorch's cross entropy requires long labels but BCE doesn't
    if task == 'binary':
        labels = labels.long()
    if not eval:
        return {'input_ids': toks, 'attention_mask': mask, 'labels': labels, 'doc_ids': doc_ids, 'sent_ixs': sent_ixs}
    else:
        return {'input_ids': toks, 'attention_mask': mask, 'labels': labels, 'sentences': sents, 'doc_ids': doc_ids, 'sent_ixs': sent_ixs}

def ctx_collator(batch, task, tokenizer, eval=False, doc_position=False, abc=False):
    """
        Collator that handles attaching n sentences of context to either side of the focus sentence. 

        So here, we have to do some tricks to make sure the input fits into a size 512 vector.
        We have to use the tokenizer to check if each sentence w/ context is too long.
        If it is, we first try to remove context sentences on either end.
        When we do this we have to make sure we have the right number of [SEP] tokens,
        because the model relies on them to extract the representation for the right sentence
        Sometimes the sentence itself is too long - this isn't a problem for BERT without context,
        because it can just truncate, but again, we need the [SEP] tokens so we do some special handling.
        Seems annoying/inefficient to run the tokenizer so many times, but BERT computation dominates,
        so I don't think this affects speed too terribly much.
    """
    sents = []
    labels = []
    replace_ixs = []
    replace_toks = []
    doc_ids = []
    sent_ixs = []
    doc_poses = []
    for ix, btch in enumerate(batch):
        if not doc_position:
            (toks, sent_ix, annotator, label, doc_id, context) = btch
        else:
            (toks, sent_ix, annotator, label, doc_id, context, doc_pos) = btch
            doc_poses.append(doc_pos)
        doc_ids.append(doc_id)
        sent_ixs.append(sent_ix)
        n_ctx_sents = len(context)//2
        sent = ' '.join(toks).lower()

        context = tuple(ctx.lower() for ctx in context)
        before_ctx, after_ctx = context[:n_ctx_sents], context[n_ctx_sents:]
        sent_w_context = ' [SEP] '.join(before_ctx + (sent,) + after_ctx)
        test = tokenizer(sent_w_context, padding=True, max_length=512, truncation=True)
        test = test['input_ids']

        # iteratively remove context until we're below 512 tokens
        # if first context sentence is longer, start by removing that one
        trim_start = len(context[0]) > len(context[1])
        start_ix = 1 if len(context[0]) > len(context[1]) else 0
        end_ix = 0 if len(context[0]) > len(context[1]) else 1
        printed = False
        while len(test) >= 512:
            # split around context
            before_ctx, after_ctx = context[start_ix:n_ctx_sents], context[n_ctx_sents:len(context)-end_ix]
            sent_w_context = ' [SEP] '.join(('',) * start_ix + before_ctx + (sent,) + after_ctx + ('',) * end_ix)
            test = tokenizer(sent_w_context, padding=True, max_length=512, truncation=True)
            test = test['input_ids']
            if len(test) >= 512 and (end_ix > n_ctx_sents or start_ix > n_ctx_sents):
                # this must mean the sentence is itself too long already, so split it in half
                split_toks = sent_w_context.split()
                # at this point the last [sep] is extraneous for some reason so don't include it, hence the :-1
                sent_w_context = ' '.join(split_toks[:n_ctx_sents] + split_toks[len(split_toks)//2:-1])
                test = tokenizer(sent_w_context, padding=True, max_length=512, truncation=True)
                test = test['input_ids']
                if len(test) >= 512:
                    # sometimes you have to split twice...
                    # at this point just back fill sep tokens to make it fit (extra one at the end b/c that's how tokenizer ends a sentence)
                    for i in range(n_ctx_sents+1):
                        test[len(test)-1-i] = tokenizer.sep_token_id
                    replace_ixs.append(ix)
                    replace_toks.append(torch.LongTensor(test))
                # catch stuff that falls through the cracks - fill in sep tokens to make it fit
                if len(np.where(np.array(test) == tokenizer.sep_token_id)[0]) < n_ctx_sents * 2 + 1:
                    for i in range(n_ctx_sents+1):
                        test[len(test)-1-i] = tokenizer.sep_token_id
                    replace_ixs.append(ix)
                    replace_toks.append(torch.LongTensor(test))
                break
            trim_start = not trim_start
            if trim_start:
                start_ix += 1
            else:
                end_ix += 1

        sents.append(sent_w_context)
        label = multilabel_labeler(label) if task == 'multilabel' else int(label)
        labels.append(label)

    # final tokenization
    tokd = tokenizer(sents, padding=True, max_length = 512, truncation=True)
    input_ids, token_type_ids, attention_mask = tokd['input_ids'], tokd['token_type_ids'], tokd['attention_mask']
    toks = torch.LongTensor(input_ids)

    replace_masks = []
    if len(replace_ixs) > 0:
        for replace_ix, replace_tok in zip(replace_ixs, replace_toks):
            replace_mask = torch.LongTensor([1] * len(replace_toks) + [0] * (toks.size(1) - len(replace_toks)))
            replace_tok = torch.cat((replace_tok, torch.LongTensor([tokenizer.pad_token_id] * (toks.size(1) - len(replace_tok)))))
            toks[replace_ix] = replace_tok

    mask = torch.LongTensor(attention_mask)
    for replace_ix, replace_mask in zip(replace_ixs, replace_masks):
        mask[replace_ix] = replace_mask

    # fill in token type ids = 0 for focus, 1 for context. If abc option, 2 for left context
    tok_type_ids = torch.zeros(toks.shape).long()
    seps = torch.where(toks == tokenizer.sep_token_id)[1].reshape(-1,2*n_ctx_sents+1)
    for ttid, sep in zip(tok_type_ids, seps):
        # give context tokens type 1
        if abc:
            ttid[:sep[n_ctx_sents-1]] = 2
        else:
            ttid[:sep[n_ctx_sents-1]] = 1
        ttid[sep[n_ctx_sents]:] = 1

    labels = torch.Tensor(labels)
    if task == 'binary':
        labels = labels.long()

    if doc_position:
        doc_poses = torch.Tensor(doc_poses)

    if eval:
        return {'input_ids': toks, 'attention_mask': mask, 'labels': labels, 'token_type_ids': tok_type_ids, 'sentences': sents, 'doc_positions': doc_poses, 'doc_ids': doc_ids, 'sent_ixs': sent_ixs}
    else:
        return {'input_ids': toks, 'attention_mask': mask, 'labels': labels, 'token_type_ids': tok_type_ids, 'doc_positions': doc_poses, 'doc_ids': doc_ids, 'sent_ixs': sent_ixs}


def select_inputs(x, args):
    inputs = {'input_ids': x['input_ids'], 'attention_mask': x['attention_mask'], 'labels': x['labels']}
    if 'token_type_ids' in x:
        inputs['token_type_ids'] = x['token_type_ids']
    if args.doc_position:
        inputs['doc_positions'] = x['doc_positions']
    if args.use_penultimate_repr or args.sum_last_four:
        inputs['output_hidden_states'] = True
    if args.use_penultimate_repr:
        inputs['layer_repr'] = 'penultimate'
    elif args.sum_last_four:
        inputs['layer_repr'] = 'sum_last_four'
    return inputs

def gather_predictions(args, model, loader, task, device, doc_position=False, select_inputs=select_inputs, save_preds=False, max_preds=1e9):
    # run through data loader and get model predictions
    with torch.no_grad():
        model.eval()
        yhat_raw = []
        yhat = []
        y = []
        sentences = []
        doc_ids = []
        sent_ixs = []
        for ix, x in tqdm(enumerate(loader)):
            if ix >= max_preds:
                break
            for k, v in x.items():
                if isinstance(v, torch.Tensor):
                    x[k] = v.to(device)
            sentences.extend(x['sentences'])
            doc_ids.extend(x['doc_ids'])
            sent_ixs.extend(x['sent_ixs'])
            inputs = select_inputs(x, args)
            loss, pred = model(**inputs)
            if task == 'multilabel':
                pred = torch.sigmoid(pred)
            else:
                pred = torch.softmax(pred, dim=0)
            pred = pred.cpu().numpy()[0]
            yhat_raw.append(pred)
            if task == 'multilabel':
                yhat.append(np.round(pred))
            else:
                yhat.append(np.argmax(pred))
            y.append(x['labels'].cpu().numpy()[0])
        yhat = np.array(yhat)
        yhat_raw = np.array(yhat_raw)
        y = np.array(y)
    return yhat_raw, yhat, y, sentences, doc_ids, sent_ixs

def run_on_test(args, model, loader, task, device, tokenizer, out_dir, label_threshs, micro_thresh, doc_position, save_fps, rec_90_thresh, rec_95_thresh, rec_99_thresh):
    # apply model to test set and compute metrics, plots, save examples
    print("EVALUATING MODEL ON TEST DATASET...")
    yhat_raw, yhat, y, sentences, doc_ids, sent_ixs = gather_predictions(args, model, loader, task, device, doc_position, max_preds=100 if args.debug else 1e9)
    metrics, _, _, _, _, _ = all_metrics(yhat_raw, yhat, y, task, label_threshs=label_threshs, micro_thresh=micro_thresh, rec_90_thresh=rec_90_thresh, rec_95_thresh=rec_95_thresh, rec_99_thresh=rec_99_thresh)

    if task == 'multilabel':
        plot_pct_labeled_vs_thresh(yhat_raw, y, out_dir, 'test')
        plot_binarized_recall_vs_thresh(yhat_raw, y, out_dir, 'test')

        plot_precision_recall_per_label(yhat_raw, y, out_dir, 'test')
        plot_precision_recall_binary(yhat_raw, y, out_dir, 'test')
        plot_perf_vs_sent_length(yhat_raw, y, out_dir, loader, tokenizer, label_threshs, micro_thresh, 'test')

    if save_fps:
        # get the thresholds we'll use to save examples - high confidence low
        thresh_metrics, rec_90_threshs, prec_90_threshs, rec_75_threshs, prec_75_threshs = multilabel_eval.balanced_f1_multilabel(yhat_raw, y)
        for ix, label in enumerate(LABEL_TYPES):
            lname = label2abbrev[label]
            high_prec_fps = high_thresh_suggestions(loader, yhat_raw[:,ix], y[:,ix], prec_75_threshs[lname], out_dir, label2abbrev[label], 'test', 'neg')
            high_rec_fns = high_thresh_suggestions(loader, yhat_raw[:,ix], y[:,ix], rec_75_threshs[lname], out_dir, label2abbrev[label], 'test', 'pos')

            # also get unfiltered error examples by putting in default 0.5 threshold
            high_prec_fps = high_thresh_suggestions(loader, yhat_raw[:,ix], y[:,ix], 0.5, out_dir, label2abbrev[label], 'test_50', 'neg')
            high_rec_fns = high_thresh_suggestions(loader, yhat_raw[:,ix], y[:,ix], 0.5, out_dir, label2abbrev[label], 'test_50', 'pos')

    # save raw predictions
    with open(f'{out_dir}/raw_preds_test.csv', 'w') as of:
        w = csv.writer(of)
        for y_i, yhr_i, sentence, doc_id, sent_ix in zip(y, yhat_raw, sentences, doc_ids, sent_ixs):
            w.writerow([doc_id, sent_ix, sentence] + y_i.tolist() + yhr_i.tolist())

    # save thresholds
    #np.save(f'{out_dir}/label_threshs.npy', np.array(label_threshs))
    #np.save(f'{out_dir}/micro_thresh.npy', np.array([micro_thresh]))

    return metrics


def evaluate(args, model, dv_loader, task, device, tokenizer, save_fps=False, out_dir="", return_thresholds=False, doc_position=False, select_input_fn=select_inputs):
    # apply model to validation set and compute metrics, including identifying best thresholds
    yhat_raw, yhat, y, sentences, doc_ids, sent_ixs = gather_predictions(args, model, dv_loader, task, device, doc_position, select_inputs=select_input_fn, max_preds=100 if args.debug else 1e9)

    # save raw predictions
    print("SAVING RAW PREDICTIONS")
    with open(f'{out_dir}/raw_preds_val.csv', 'w') as of:
        w = csv.writer(of)
        for y_i, yhr_i, sentence, doc_id, sent_ix in zip(y, yhat_raw, sentences, doc_ids, sent_ixs):
            w.writerow([doc_id, sent_ix, sentence] + y_i.tolist() + yhr_i.tolist())

    if task == 'multilabel':
        # unbalanced metrics
        metrics = multilabel_eval.all_metrics(yhat, y, yhat_raw=yhat_raw, calc_auc=True, label_order=LABEL_TYPES)

        # balance each label individually (macro)
        label_metrics, label_threshs, label_hp_threshs, label_hp_recs, label_rec_values = multilabel_eval.balance_each_label(yhat_raw, y, get_high_prec_thresh=False)
        metrics.update(label_metrics)

        plot_label_rec_values(label_rec_values, out_dir, 'val')

        # balance via micro f1
        balanced_thresh, balanced_metrics = multilabel_eval.balanced_f1(yhat_raw, y)
        print(f"threshold = {balanced_thresh}")
        metrics.update({f'balanced_{key}': val for key, val in balanced_metrics.items()})

        # binary metrics - transform predictions and true labels to binary
        binary_yhat = np.any(yhat_raw >= np.array(label_threshs), axis=1)
        binary_y = np.any(y, axis=1)
        metrics['binary_prec'] = precision_score(binary_y, binary_yhat)
        metrics['binary_rec'] = recall_score(binary_y, binary_yhat)
        metrics['binary_f1'] = f1_score(binary_y, binary_yhat)

        # binary precision at given thresholds
        print("getting recall at fixed thresholds...")
        binary_yhat_raw = np.max(yhat_raw, axis=1)
        binary_yhat_raw = np.stack((1 - binary_yhat_raw, binary_yhat_raw), axis=1)
        rec_90_thresh, prec_at_rec_90 = binary_eval.precision_at_fixed_recall(binary_yhat_raw, binary_y, 0.90)
        metrics['prec@rec=90'] = prec_at_rec_90
        rec_95_thresh, prec_at_rec_95 = binary_eval.precision_at_fixed_recall(binary_yhat_raw, binary_y, 0.95)
        metrics['prec@rec=95'] = prec_at_rec_95
        rec_99_thresh, prec_at_rec_99 = binary_eval.precision_at_fixed_recall(binary_yhat_raw, binary_y, 0.99)
        metrics['prec@rec=99'] = prec_at_rec_99

        if save_fps:
            thresh_metrics, rec_90_threshs, prec_90_threshs, rec_75_threshs, prec_75_threshs = multilabel_eval.balanced_f1_multilabel(yhat_raw, y)
            for ix, label in enumerate(LABEL_TYPES):
                lname = label2abbrev[label]
                #high_prec_fps = high_thresh_suggestions(dv_loader, yhat_raw[:,ix], y[:,ix], prec_90_threshs[lname], out_dir, label2abbrev[label], 'val', 'neg')
                #high_rec_fns = high_thresh_suggestions(dv_loader, yhat_raw[:,ix], y[:,ix], rec_90_threshs[lname], out_dir, label2abbrev[label], 'val', 'pos')

                print("GATHERING FALSE NEGATIVES")
                high_prec_fps = high_thresh_suggestions(dv_loader, yhat_raw[:,ix], y[:,ix], 0.5, out_dir, label2abbrev[label], 'val_50', 'neg')
                print("GATHERING FALSE POSITIVES")
                high_rec_fns = high_thresh_suggestions(dv_loader, yhat_raw[:,ix], y[:,ix], 0.5, out_dir, label2abbrev[label], 'val_50', 'pos')
                # true positive examples, why not
                print("GATHERING TRUE POSITIVES")
                tps = high_thresh_suggestions(dv_loader, yhat_raw[:,ix], y[:,ix], 0.5, out_dir, label2abbrev[label], 'val_50', 'true_pos')

    else:
        # balance metrics
        balanced_thresh, balanced_metrics = binary_eval.balanced_f1(yhat_raw, y)
        acc, prec, rec, f1, auc = balanced_metrics['acc'], balanced_metrics['prec'], balanced_metrics['rec'], balanced_metrics['f1'], balanced_metrics['auc'], 
        metrics.update({f'balanced_{key}': val for key, val in balanced_metrics.items()})
        print(f"threshold={balanced_thresh}")

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
        
        if save_fps:
            high_prec_fps = high_thresh_suggestions(dv_loader, yhat_raw[:,1], y, prec_90_thresh, out_dir, 'binary', 'val', 'neg')
            high_rec_fns = high_thresh_suggestions(dv_loader, yhat_raw[:,1], y, rec_90_thresh, out_dir, 'binary', 'val', 'pos')

    print_metrics(metrics, task)

    if return_thresholds:
        return metrics, label_threshs, balanced_thresh, rec_90_thresh, rec_95_thresh, rec_99_thresh
    else:
        return metrics

def args_to_model(n_context_sentences, cnn_on_top):
    if cnn_on_top:
        if n_context_sentences == 0:
            print("not a valid combination: cnn with no context")
            sys.exit(0)
        else:
            return BertCNNContextMultilabel
    else:
        if n_context_sentences == 0:
            return BertForSequenceMultilabelClassification
        else:
            return BertSequenceMultilabelClassificationContext
    return None

def load_pretrained_local(model, args):
    print(f"\nLoading local weights")
    sd = torch.load(args.local_weights)
    if args.eval_model:
        model.load_state_dict(sd)
    else:
        if args.cnn_on_top:
            sd_conv = {
                    'weight': sd['conv.weight'],
                    'bias': sd['conv.bias'],
                    }
            model.conv.load_state_dict(sd_conv)
        sd_bert = {k[len('bert.'):] : v for k, v in sd.items() if 'bert' in k}
        model.bert.load_state_dict(sd_bert)

    if args.freeze_bert:
        for param in model.bert.parameters():
            param.requires_grad = False
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_fname", type=str)
    parser.add_argument("model", choices=['bert', 'clinicalbert', 'clinicalbert_disch'])
    parser.add_argument("--n_context_sentences", type=int, default=0, help="set to >0 to use N context sentences on both sides")
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--criterion", type=str, default="auc_macro", required=False, help="metric to use for early stopping")
    parser.add_argument("--task", choices=['binary', 'multilabel'], default='multilabel')
    parser.add_argument("--patience", type=int, default=3, required=False, help="number of evaluations to wait for improved criterion before early stopping (default 3)")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=11, help="random seed")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="num batches to wait before updating weights")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm (for gradient clipping).")
    parser.add_argument("--max_steps", type=int, default=-1, help="put a positive number to limit number of training steps for debugging")
    parser.add_argument("--eval_steps", type=int, default=3000, help="number of steps between evaluations during training")
    parser.add_argument("--print_every", type=int, default=1e3, help="how often (in batches) to print avg'd loss")
    parser.add_argument("--cnn_on_top", action="store_true", help="set to use CNN on top of bert embedded tokens")
    parser.add_argument("--abc", action="store_true", help="set to use two types of context embedddings to distinguish left vs. right-context")
    parser.add_argument("--doc_position", action="store_true", help="set to use relative document position feature")
    parser.add_argument("--run_test", action="store_true", help="set to run on test too after running on dev at the end")
    parser.add_argument("--freeze_bert", action="store_true", help="set to not update bert parameters")
    parser.add_argument("--save_fps", action="store_true", help="set to find and save false positives / false negative examples")
    parser.add_argument("--local_weights", type=str, required=False, help="optionally point to a file with local weights corresponding to given model type")
    parser.add_argument("--eval_model", action="store_true", help="when using local_weights, set this to load all weights, including final layer, and just run evaluation")
    parser.add_argument("--use_penultimate_repr", action="store_true", help="flag to use penultimate BERT layer for training/prediction")
    parser.add_argument("--sum_last_four", action="store_true", help="flag to sum last four BERT layers as base repr for training/prediction")
    parser.add_argument("--debug", action="store_true", help="flag to debug on small data")
    parser.add_argument("--bert_oov_file", type=str, help="path to file of BERT OOV's to use for bert tokenizer")
    args = parser.parse_args()

    print(args.criterion)

    if args.eval_model:
        assert args.max_epochs == 0, "max epochs must be 0 when evaluating a model"

    if args.model == 'bert':
        args.model = 'bert-base-uncased'
    elif args.model == 'clinicalbert':
        args.model = 'emilyalsentzer/Bio_ClinicalBERT'
    elif args.model == 'clinicalbert_disch':
        args.model = 'emilyalsentzer/Bio_Discharge_Summary_BERT'

    # Set seed
    set_seed(args.seed)

    label_set = LABEL_TYPES if args.task == 'multilabel' else ['Non-followup', 'Followup']
    num_labels = len(label_set) 

    # Load pretrained tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.model)

    # load config
    label2id = {label:ix for ix,label in enumerate(label_set)}
    id2label = {ix:label for label,ix in label2id.items()}
    config = BertConfig.from_pretrained(
        args.model,
        num_labels=num_labels,
        finetuning_task="text_classification",
        label2id=label2id,
        id2label=id2label,
    )

    # load model
    model_class = args_to_model(args.n_context_sentences, args.cnn_on_top)
    model = model_class.from_pretrained(args.model, config=config)

    # customize model via args
    model.set_task(args.task)
    if args.n_context_sentences > 0:
        model.set_sep_token_id(tokenizer.sep_token_id)
        model.set_n_context_sentences(args.n_context_sentences)
        if args.abc:
            model.update_tok_type_embeddings()
    if args.doc_position:
        model.add_doc_position_feature(config)
    if args.bert_oov_file:
        bert_oovs = [line.strip() for line in open(args.bert_oov_file)]
        tokenizer.add_tokens(bert_oovs)
        model.expand_vocab_by_num(len(bert_oovs))

    if args.local_weights:
        model = load_pretrained_local(model, args)

    # Get datasets
    #train
    dev_fname = args.train_fname.replace('train', 'val')
    test_fname = args.train_fname.replace('train', 'test')
    if args.n_context_sentences == 0:
        train_dataset = SentDataset(args.train_fname, args.task)
        eval_dataset = SentDataset(dev_fname, args.task)
        test_dataset = SentDataset(test_fname, args.task)
    else:
        train_dataset = SentEvalDataset(args.train_fname, args.task, n_context_sentences=args.n_context_sentences, doc_position=args.doc_position)
        eval_dataset = SentEvalDataset(dev_fname, args.task, n_context_sentences=args.n_context_sentences, doc_position=args.doc_position)
        test_dataset = SentEvalDataset(test_fname, args.task, n_context_sentences=args.n_context_sentences, doc_position=args.doc_position)

    # setup data collator
    if args.n_context_sentences == 0:
        data_collator = lambda x: collator(x, args.task, tokenizer)
        eval_collate_fn = lambda x: collator(x, args.task, tokenizer, eval=True)
    else:
        data_collator = lambda x: ctx_collator(x, args.task, tokenizer, doc_position=args.doc_position, abc=args.abc)
        eval_collate_fn = lambda x: ctx_collator(x, args.task, tokenizer, eval=True, doc_position=args.doc_position, abc=args.abc)

    # set up experiment directory
    timestamp = time.strftime('%b_%d_%H:%M:%S', time.localtime())
    if not args.eval_model:
        out_dir = f"results/{args.model}_{timestamp}"
    else:
        out_dir = '/'.join(args.local_weights.split('/')[:-1])
    print(f"will put results in {out_dir}")

    # just create this Trainer object to get the scheduler/optimizer
    training_args = TrainingArguments(
            output_dir = out_dir,
            do_train=True,
            do_eval=True,
            do_predict=args.run_test,
            evaluate_during_training=True,
            max_steps=args.max_steps,
            learning_rate=args.lr,
            num_train_epochs=args.max_epochs,
            save_total_limit=10,
            eval_steps=args.eval_steps,
            seed=args.seed,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    tr_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=data_collator)
    dv_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, collate_fn=eval_collate_fn)

    if args.max_steps > 0:
        t_total = args.max_steps
    else:
        t_total = int(len(tr_loader) // args.gradient_accumulation_steps * args.max_epochs)
    optimizer, scheduler = trainer.get_optimizers(t_total)


    # Training loop
    tr_loss = 0.0
    model.zero_grad()
    model.train()
    metrics_hist = defaultdict(list)
    best_epoch = 0
    best_step = 0
    step = 0
    losses = []
    for epoch in range(args.max_epochs):
        for x in tqdm(tr_loader):
            if args.max_steps > -1 and step > args.max_steps:
                break
            # transfer to gpu
            for k, v in x.items():
                if isinstance(v, torch.Tensor):
                    x[k] = v.to(trainer.args.device)
            inputs = select_inputs(x, args)
            #outputs = model(**inputs)
            loss, pred = model(**inputs)

            # do update
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0 or (
                # last step in epoch but step is always smaller than gradient_accumulation_steps
                len(tr_loader) <= args.gradient_accumulation_steps
                and (step + 1) == len(epoch_iterator)
            ):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                losses.append(tr_loss)
                tr_loss = 0.0

            # periodic evaluation
            if (step + 1) % args.eval_steps == 0:
                metrics = evaluate(args, model, dv_loader, args.task, trainer.args.device, tokenizer, out_dir=out_dir, doc_position=args.doc_position)
                for name, metric in metrics.items():
                    metrics_hist[name].append(metric)
                if not os.path.exists(out_dir):
                    os.mkdir(out_dir)

                save_metrics(metrics_hist, out_dir)
                is_best = check_best_model_and_save(model, metrics_hist, args.criterion, out_dir)
                if is_best:
                    best_epoch = epoch
                    best_step = step

                if early_stop(metrics_hist, args.criterion, args.patience):
                    print(f"{args.criterion} hasn't improved in {args.patience} epochs, early stopping...")
                    stop_training = True
                    break

            # poor man's tensorboard
            if (step + 1) % args.print_every == 0:
                print(f"loss: {np.mean(losses[-10:])}")

            step += 1
            
    # save args
    if not args.eval_model:
        with open(f'{out_dir}/args.json', 'w') as of:
            of.write(json.dumps(args.__dict__, indent=2) + "\n")

    # Evaluation
    eval_results = {}

    if args.max_epochs > 0 and not args.eval_model:
        # save the model at the end
        sd = model.state_dict()
        torch.save(sd, out_dir + "/model_final.pth")

        # reload the best model
        best_model_fname = f'{out_dir}/model_best_{args.criterion}.pth'
        if os.path.exists(best_model_fname):
            print(f"\nReloading model with best {args.criterion} (epoch {best_epoch})")
            sd = torch.load(best_model_fname)
            model.load_state_dict(sd)

    # run evaluation loop again to get false positive examples, compute specialized metrics, etc
    print("RUNNING EVALUATION LOOP")
    metrics, label_threshs, micro_thresh, rec_90_thresh, rec_95_thresh, rec_99_thresh = evaluate(args, model, dv_loader, args.task, trainer.args.device, tokenizer, save_fps=args.save_fps, out_dir=out_dir, return_thresholds=True, doc_position=args.doc_position)
    print("SAVING LABEL THRESHOLDS")
    np.save(f'{out_dir}/label_threshs.npy', np.array(label_threshs))
    np.save(f'{out_dir}/micro_thresh.npy', np.array([micro_thresh]))

    # run test if applicable
    if args.run_test:
        print("RUNNING TEST")
        te_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=eval_collate_fn)
        test_metrics = run_on_test(args, model, te_loader, args.task, trainer.args.device, tokenizer, out_dir, label_threshs, micro_thresh, args.doc_position, args.save_fps, rec_90_thresh=rec_90_thresh, rec_95_thresh=rec_95_thresh, rec_99_thresh=rec_99_thresh)

    print(f"done! results in {out_dir}")
    return eval_results


if __name__ == "__main__":
    main()
