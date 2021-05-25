""" run a finetuned bert model on unlabeled data to get its predictions """
import argparse
import csv
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification
from transformers import ReformerTokenizer

from bert import *
from constants import *
from neural_baselines import SentDataset, SentEvalDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("unlabeled_fname", type=str)
    parser.add_argument("model", choices=['bert', 'clinicalbert', 'clinicalbert_disch', 'reformer'])
    parser.add_argument("local_weights", type=str, help="point to a file with local weights corresponding to given model type to run on unlabeled data")
    parser.add_argument("--threshold", type=float, default=0.05, help="threshold to use for selecting sentences to pretrain on")
    parser.add_argument("--n_context_sentences", type=int, default=0, help="set to >0 to use N context sentences on both sides")
    parser.add_argument("--task", choices=['binary', 'multilabel'], default='multilabel')
    parser.add_argument("--cache_dir",default="",type=str,help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--cnn_on_top", action="store_true", help="set to use CNN on top of bert embedded tokens")
    args = parser.parse_args()

    if args.model == 'bert':
        args.model = 'bert-base-uncased'
    elif args.model == 'clinicalbert':
        args.model = 'emilyalsentzer/Bio_ClinicalBERT'
    elif args.model == 'clinicalbert_disch':
        args.model = 'emilyalsentzer/Bio_Discharge_Summary_BERT'
    elif args.model == 'reformer':
        #args.model = 'google/reformer-enwik8'
        args.model = 'google/reformer-crime-and-punishment'

    label_set = LABEL_TYPES if args.task == 'multilabel' else ['Non-followup', 'Followup']
    num_labels = len(label_set) 

    # Load pretrained model and tokenizer
    if 'reformer' not in args.model:
        tokenizer = BertTokenizer.from_pretrained(
            args.model,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
    else:
        tokenizer = ReformerTokenizer.from_pretrained(
            args.model,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
    label2id = {label:ix for ix,label in enumerate(label_set)}
    id2label = {ix:label for label,ix in label2id.items()}
    config = BertConfig.from_pretrained(
        args.model,
        num_labels=num_labels,
        finetuning_task="text_classification",
        cache_dir=args.cache_dir if args.cache_dir else None,
        label2id=label2id,
        id2label=id2label,
    )

    model_class = args_to_model(args.n_context_sentences, args.cnn_on_top)
    model = model_class.from_pretrained(
        args.model,
        from_tf=bool(".ckpt" in args.model),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')
    model.to(device)
    if args.n_context_sentences > 0:
        model.set_sep_token_id(tokenizer.sep_token_id)
        model.set_n_context_sentences(args.n_context_sentences)
    if args.local_weights:
        print(f"\nLoading local weights")
        sd = torch.load(args.local_weights, map_location=device)
        model.load_state_dict(sd)
    model.set_task(args.task)

    # Get dataset
    if args.n_context_sentences == 0:
        dataset = SentDataset(args.unlabeled_fname, args.task)
        data_collator = lambda x: collator(x, args.task, tokenizer)
    else:
        dataset = SentEvalDataset(args.unlabeled_fname, args.task, n_context_sentences=args.n_context_sentences)
        data_collator = lambda x: ctx_collator(x, args.task, tokenizer, get_doc_id_sent_ix=True)

    out_dir = '/'.join(args.local_weights.split('/')[:-1])

    loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=data_collator)

    date = datetime.strftime(datetime.today(), '%Y%m%d')
    num_examples = 0
    num_selected = 0
    with open(f'{out_dir}/pretrain_doc_sents_{date}_{args.threshold}.csv', 'w') as of:
        w = csv.writer(of)
        with torch.no_grad():
            model.eval()
            for ix, x in tqdm(enumerate(loader), total=len(dataset)//8):
                for k, v in x.items():
                    if isinstance(v, torch.Tensor):
                        x[k] = v.to(device)
                num_examples += len(x['input_ids'])
                inputs = {'input_ids': x['input_ids'], 'attention_mask': x['attention_mask'], 'labels': x['labels']}
                doc_ids, sent_ixs = x['doc_ids'], x['sent_ixs']
                if 'token_type_ids' in x:
                    inputs['token_type_ids'] = x['token_type_ids']
                loss, pred = model(**inputs)
                if args.task == 'multilabel':
                    pred = torch.sigmoid(pred)
                else:
                    pred = torch.softmax(pred, dim=0)
                yhat_raw = pred.cpu().numpy()
                if args.task == 'multilabel':
                    yhat = np.any(yhat_raw > args.threshold, axis=1)
                else:
                    yhat = yhat_raw > args.threshold
                if np.any(yhat):
                    ixs = np.where(yhat)[0]
                    for ix in ixs:
                        num_selected += 1
                        w.writerow([x['doc_ids'][ix], x['sent_ixs'][ix], yhat_raw[ix].max()])
                # if ix % 10 == 0 and num_examples > 0:
                #     print(f"selected fraction: {num_selected/num_examples}")


    print(f"done! results in {out_dir}")
    return eval_results


if __name__ == "__main__":
    main()
