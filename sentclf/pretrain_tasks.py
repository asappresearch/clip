import argparse
import json
import os
import random
import time

from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertConfig, BertForPreTraining, BertForMaskedLM, BertTokenizer
from transformers import Trainer, TrainingArguments
from tqdm import tqdm

from bert_model import BertContextMLMSwitch, BertContextCNNMLMSwitch
from neural_baselines import SentDataset

class PrecomputedDataset(Dataset):
    def __init__(self, fname):
        self.insts = []
        print("reading precomputed instances...")
        with open(fname) as f:
            for line in tqdm(f):
                inst = json.loads(line.strip())
                self.insts.append(inst)

    def __len__(self):
        return len(self.insts)

    def __getitem__(self, idx):
        return self.insts[idx]

class SentContextDataset(SentDataset):
    def __init__(self, fname, focus_fname, task, n_context_sentences, switch_prob=0.0, same_doc = True, doc_position=False, start_at=0):
        super().__init__(fname, task)
        self.doc_sents = pd.read_csv(focus_fname)
        print("num focus sentences: ", len(self.doc_sents))
        self.n_context_sentences = n_context_sentences
        self.switch_prob = switch_prob
        self.same_doc = same_doc
        self.doc_position = doc_position
        self.start_at = start_at

    def tokenize_sentence(self, row):
        sent = row.sentence
        if not isinstance(sent, str):
            return ['']
        else:
            return [x.lower() for x in word_tokenize(sent)]

    def __len__(self):
        return len(self.doc_sents)


    def __getitem__(self, idx):
        doc_sent = self.doc_sents.iloc[idx+self.start_at]
        row = self.sents[(self.sents.doc_id == doc_sent.doc_id) & (self.sents.sent_ix == doc_sent.sent_ix)]
        try:
            idx = row.index[0]
        except:
            print(f"issue getting index from idx {idx}: row: {row}")
            return None
        row = row.iloc[0]
        sent_ix = row.sent_ix
        doc_id = row.doc_id
        if random.random() < self.switch_prob:
            if self.same_doc:
                idxs = set(self.sents[self.sents['doc_id'] == doc_id].index)
                idxs.remove(idx)
                switch_ix = random.choice(list(idxs))
            else:
                switch_ix = random.choice(range(len(self.sents)))
            sent = self.tokenize_sentence(self.sents.iloc[switch_ix])
            label = 1
        else:
            sent = self.tokenize_sentence(row)
            label = 0
        contexts = []
        ctx_sents = []
        sent_ix = row.sent_ix
        for offset in range(-self.n_context_sentences, self.n_context_sentences+1):
            if offset == 0:
                continue
            ctx_ix = sent_ix + offset
            if ctx_ix < 0:
                ctx_sents.append(['<DOC_START>'])
            elif idx + offset >= len(self.sents) or self.sents.iloc[idx + offset].sent_ix != sent_ix + offset:
                ctx_sents.append(['<DOC_END>'])
            else:
                ctx_sents.append(self.tokenize_sentence(self.sents.iloc[idx+offset]))
        contexts = tuple(' '.join(ctx_sent) for ctx_sent in ctx_sents)
        if self.doc_position:
            n_sents = self.sents[self.sents['doc_id'] == doc_id].iloc[-1].sent_ix + 1
            doc_position = row.sent_ix / n_sents
            return sent, label, row.sent_ix, doc_id, contexts, doc_position
        else:
            return sent, label, row.sent_ix, doc_id, contexts

def get_padded_length(lst, el):
    try:
        return lst.index(el)
    except ValueError:
        return len(lst)

def simple_collate(batch):
    input_ids = []
    sentences = []
    attention_mask = []
    token_type_ids = []
    mlm_labels = []
    batch_len = max([get_padded_length(inst['input_ids'], 0) for inst in batch])
    for inst in batch:
        inst_len = get_padded_length(inst['input_ids'], 0)
        padded = inst['input_ids'][:inst_len] + [0] * (batch_len - inst_len)
        input_ids.append(padded)
        sentences.append(inst['sentences'])
        inst_attention_mask = [1] * inst_len + [0] * (batch_len - inst_len)
        attention_mask.append(inst_attention_mask)
        inst_token_type_ids = inst['token_type_ids'][:inst_len] + [1] * (batch_len - inst_len)
        token_type_ids.append(inst_token_type_ids)
        inst_mlm_labels = inst['mlm_labels'][:inst_len] + [-100] * (batch_len - inst_len)
        mlm_labels.append(inst_mlm_labels)
    input_ids = torch.LongTensor(input_ids)
    attention_mask = torch.LongTensor(attention_mask)
    token_type_ids = torch.LongTensor(token_type_ids)
    mlm_labels = torch.LongTensor(mlm_labels)
    # subselect 25% of UMLS tokens
    new_mlm_labels = mlm_labels.clone()
    prob_mask = torch.zeros(new_mlm_labels.size())
    prob_mask.masked_fill_(new_mlm_labels != -100, 0.25)
    new_mask = torch.bernoulli(prob_mask).bool()
    new_mlm_labels[~new_mask] = -100
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'sentences': sentences, 'token_type_ids': token_type_ids, 'mlm_labels': new_mlm_labels}

def switch_mlm_collate(batch, tokenizer, mlm_probability=0, doc_position=False, abc=False):
    """
        adapted from bert.py
    """
    sents = []
    labels = []
    replace_ixs = []
    replace_toks = []
    mlm_labels = []
    doc_poses = []
    for ix, inst in enumerate(batch):
        if inst is None:
            continue
        else:
            if not doc_position:
                (toks, label, sent_ix, doc_id, context) = inst
            else:
                (toks, label, sent_ix, doc_id, context, doc_pos) = inst
                doc_poses.append(doc_pos)
        n_ctx_sents = len(context)//2
        sent = ' '.join(toks)
        before_ctx, after_ctx = context[:n_ctx_sents], context[n_ctx_sents:]
        sent_w_context = ' [SEP] '.join(before_ctx + (sent,) + after_ctx)
        test = tokenizer(sent_w_context, padding=True, max_length=512, truncation=True)
        test = test['input_ids']
        # iteratively remove context until we're below 512 tokens
        # if first context sentence is longer, start by rmoving that one
        if len(context) >= 2:
            trim_start = len(context[0]) > len(context[1])
            start_ix = 1 if len(context[0]) > len(context[1]) else 0
            end_ix = 0 if len(context[0]) > len(context[1]) else 1
        else:
            trim_start = False
            start_ix = 0
            end_ix = 0
        while len(test) >= 512:
            before_ctx, after_ctx = context[start_ix:n_ctx_sents], context[n_ctx_sents:len(context)-end_ix]
            sent_w_context = ' [SEP] '.join(('',) * start_ix + before_ctx + (sent,) + after_ctx + ('',) * end_ix)
            test = tokenizer(sent_w_context, padding=True, max_length=512, truncation=True)
            test = test['input_ids']
            if len(test) >= 512 and (end_ix > n_ctx_sents or start_ix > n_ctx_sents):
                # this must mean the sentence is itself too long already, so split it in half
                split_toks = sent_w_context.split()
                # at this point the last [sep] is extraneous for some reason so don't include it
                sent_w_context = ' '.join(split_toks[:n_ctx_sents] + split_toks[len(split_toks)//2:-1])
                test = tokenizer(sent_w_context, padding=True, max_length=512, truncation=True)
                test = test['input_ids']
                if len(test) >= 512:
                    # at this point just back fill sep tokens to make it fit (extra one at the end b/c that's how tokenizer ends a sentence
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
        labels.append(label)
    if len(sents) == 0:
        return None
    tokd = tokenizer(sents, padding=True, max_length = 512, truncation=True)
    input_ids, token_type_ids, attention_mask = tokd['input_ids'], tokd['token_type_ids'], tokd['attention_mask']
    toks = torch.LongTensor(input_ids)

    #segment embeddings
    tok_type_ids = torch.zeros(toks.shape).long()
    try:
        seps = torch.where(toks == tokenizer.sep_token_id)[1].reshape(-1, 2*n_ctx_sents+1)
    except RuntimeError as e:
        return None
    for ttid, sep in zip(tok_type_ids, seps):
        if abc:
            ttid[:sep[n_ctx_sents-1]] = 2
        else:
            ttid[:sep[n_ctx_sents-1]] = 1
        ttid[sep[n_ctx_sents]:] = 1

    # do mlm masking
    if mlm_probability > 0:
        mlm_labels = toks.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(mlm_labels.shape, mlm_probability)
        special_tokens_mask = [
            tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in mlm_labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if tokenizer._pad_token is not None:
            padding_mask = mlm_labels.eq(tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        if n_ctx_sents > 0:
            # finally, only apply MLM to the focus sentence, so mask out context sentences
            context_mask = ~tok_type_ids.eq(0)
            probability_matrix.masked_fill_(context_mask, value=0.0)

        #now draw probabilities w/ bernoulli
        masked_indices = torch.bernoulli(probability_matrix).bool()
        mlm_labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(mlm_labels.shape, 0.8)).bool() & masked_indices
        toks[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(mlm_labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(tokenizer), mlm_labels.shape, dtype=torch.long)
        toks[indices_random] = random_words[indices_random]
    # The rest of the time (10% of the time) we keep the masked input tokens unchanged

    replace_masks = []
    if len(replace_ixs) > 0:
        for replace_ix, replace_tok in zip(replace_ixs, replace_toks):
            if len(replace_toks) > toks.size(1):
                print("replace toks too long... ")
                return None
            elif len(replace_toks) < toks.size(1):
                replace_mask = torch.LongTensor([1] * len(replace_toks) + [0] * (toks.size(1) - len(replace_toks)))
                replace_tok = torch.cat((replace_tok, torch.LongTensor([tokenizer.pad_token_id] * (toks.size(1) - len(replace_tok)))))
            toks[replace_ix] = replace_tok
    mask = torch.LongTensor(attention_mask)
    for replace_ix, replace_mask in zip(replace_ixs, replace_masks):
        mask[replace_ix] = replace_mask

    labels = torch.Tensor(labels)
    labels = labels.long()

    if doc_position:
        doc_poses = torch.Tensor(doc_poses)

    output = {'input_ids': toks, 'attention_mask': mask, 'labels': labels, 'token_type_ids': tok_type_ids, 'doc_positions': doc_poses}
    if mlm_probability > 0:
        output['mlm_labels'] = mlm_labels
    if eval:
        output['sentences'] = sents
    return output

def args_to_model(task, n_context_sentences, cnn_on_top):
    if n_context_sentences == 0:
        if task == 'mlm':
            return BertForMaskedLM
        else:
            return BertForPreTraining
    else:
        if cnn_on_top:
            return BertContextCNNMLMSwitch
        else:
            return BertContextMLMSwitch

def select_inputs(batch, task, n_context_sentences, doc_position):
    # pull out the right elements from the batch and put with the right keys for feeding into models
    if args.task == 'mlm':
        if n_context_sentences == 0:
            inputs = {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask'], 'labels': batch['mlm_labels']}
        else:
            inputs = {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask'], 'mlm_labels': batch['mlm_labels']}
    else:
        inputs = {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask'], 'labels': batch['labels'], 'mlm_labels': batch['mlm_labels']}
    if args.doc_position:
        inputs['doc_positions'] = batch['doc_positions']
    if 'token_type_ids' in batch:
        inputs['token_type_ids'] = batch['token_type_ids']
    return inputs

def evaluate(model, loader, task, device, n_context_sentences, doc_position=False):
    total_loss = 0.0
    with torch.no_grad():
        model.eval()
        for ix, x in tqdm(enumerate(loader)):
            if ix >= 1000:
                break
            if x is None:
                continue
            for k, v in x.items():
                if isinstance(v, torch.Tensor):
                    x[k] = v.to(device)
            inputs = select_inputs(x, task, n_context_sentences, doc_position)
            del x
            try:
                output = model(**inputs)
            except Exception as exp:
                print(f"error putting example through model: {exp}")
                continue
            if output is None:
                continue
            loss, _ = output
            total_loss += loss.item()
    return total_loss

def main(args):
    tokenizer = BertTokenizer.from_pretrained(
        args.model,
    )
    if args.focus_fname:
        tr_data = SentContextDataset(args.fname, args.focus_fname, args.task, n_context_sentences=args.n_context_sentences, switch_prob=args.switch_prob, doc_position=args.doc_position)
        vl_fname = args.focus_fname.replace('train.csv', 'test.csv')
        vl_data = SentContextDataset(args.fname, vl_fname, args.task, n_context_sentences=args.n_context_sentences, switch_prob=args.switch_prob, doc_position=args.doc_position)
    elif args.precomputed_insts_fname:
        tr_data = PrecomputedDataset(args.precomputed_insts_fname)
        vl_fname = args.precomputed_insts_fname.replace('_train', '_valid')
        print(f"reading validation set {vl_fname}")
        vl_data = PrecomputedDataset(vl_fname)

    label2id = {label:ix for ix,label in enumerate(["non-switched", "switched"])}
    id2label = {ix:label for label,ix in label2id.items()}
    num_labels = len(label2id) 
    config = BertConfig.from_pretrained(
        args.model,
        num_labels=num_labels,
        finetuning_task="text_classification",
        label2id=label2id,
        id2label=id2label,
    )
    model_class = args_to_model(args.task, args.n_context_sentences, args.cnn_on_top)
    model = model_class.from_pretrained(
        args.model,
        config=config,
    )
    if args.n_context_sentences > 0:
        model.set_sep_token_id(tokenizer.sep_token_id)
        model.set_n_context_sentences(args.n_context_sentences)
        if args.abc:
            model.update_tok_type_embeddings()
        if args.doc_position:
            model.add_doc_position_feature(config)
    if args.bert_oov_file:
        print("reading bert OOVs and expanding vocab")
        bert_oovs = [line.strip() for line in open(args.bert_oov_file)]
        tokenizer.add_tokens(bert_oovs)
        model.expand_vocab_by_num(len(bert_oovs))

    timestamp = time.strftime('%b_%d_%H:%M:%S', time.localtime())
    out_dir = f"results/{args.model}_{timestamp}"

    # just create this to get the optimizer
    trainer = Trainer(model=model, args=TrainingArguments(output_dir=out_dir), train_dataset=tr_data)
    optimizer, _ = trainer.get_optimizers(11111)

    if args.focus_fname:
        collate_fn = lambda batch: switch_mlm_collate(batch, tokenizer, mlm_probability=args.mlm_probability, doc_position=args.doc_position, abc=args.abc)
    elif args.precomputed_insts_fname:
        collate_fn = simple_collate

    tr_loader = DataLoader(tr_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    vl_loader = DataLoader(vl_data, batch_size=1, shuffle=True, collate_fn=collate_fn)
    tr_loss = 0.0
    model.zero_grad()
    model.train()
    step = 0
    losses = []
    vl_losses = []
    stop_training = False
    for epoch in range(args.max_epochs):
        for batch_ix, batch in tqdm(enumerate(tr_loader)):
            if args.max_iter > -1 and step > args.max_iter:
                stop_training = True
                break
            # skip errors in data processing
            if batch is None:
                continue
            # put everything on gpu if available
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(trainer.args.device)
            inputs = select_inputs(batch, args.task, args.n_context_sentences, args.doc_position)
            del batch
            try:
                output = model(**inputs)
            except Exception as e:
                print(f"exception when feeding in batch: {e}")
                continue
            # skip errors in data processing that only show up at model time (e.g. too many SEP tokens)
            if output is None:
                continue
            loss, _ = output
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            tr_loss += loss.item()
            # do update when we hit grad accum steps
            if (step + 1) % args.gradient_accumulation_steps == 0 or (
                # last step in epoch but step is always smaller than gradient_accumulation_steps
                len(tr_loader) <= args.gradient_accumulation_steps
                and (step + 1) == len(epoch_iterator)
            ):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                model.zero_grad()
                losses.append(tr_loss)
                tr_loss = 0.0

            # poor man's tensorboard
            if (step + 1) % args.print_every == 0:
                print(f"loss: {np.mean(losses[-10:])}")

            if (step + 1) % args.eval_steps == 0:
                total_loss = evaluate(model, vl_loader, args.task, trainer.args.device, args.n_context_sentences, doc_position=args.doc_position)
                vl_losses.append(total_loss)
                print(f'validation losses: {vl_losses}')
                # save model if this is the best val loss so far
                if np.nanargmin(vl_losses) == len(vl_losses) - 1:
                    if not os.path.exists(out_dir):
                        os.mkdir(out_dir)
                    print("new best, saving model...")
                    torch.save(model.state_dict(), f'{out_dir}/model_{args.task}_best_vl_loss.pth')
                # stop training if we've done at least [patience] evaluations, and best result was [patience] or more ago
                if len(vl_losses) >= args.patience:
                    stop_training = np.nanargmin(vl_losses) < len(vl_losses) - args.patience
                    if stop_training:
                        print("early stopping...")

            step += 1
        if stop_training:
            break

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    sd = model.state_dict()
    torch.save(sd, f'{out_dir}/model_{args.task}_{args.max_iter}.pth')
    print("done!")
    print(f"saved to {out_dir}")
    with open(f'{out_dir}/args.json', 'w') as of:
        of.write(json.dumps(args.__dict__, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fname", type=str, help="point to unlabeled data to pretrain on")
    parser.add_argument("model", choices=['cnn', 'bert', 'clinicalbert', 'clinicalbert_disch'], default='bert')
    parser.add_argument('--focus_fname', type=str, help="path to file of doc id's/sent ix's to focus prtraining on")
    parser.add_argument('--precomputed_insts_fname', type=str, help="path to file of precomputed instances to batch up and serve to model")
    parser.add_argument("--task", choices=['mlm', 'mlm_switch'], default='mlm')
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--mlm_probability", type=float, default=0.15)
    parser.add_argument("--n_context_sentences", type=int, default=2)
    parser.add_argument("--switch_prob", type=float, default=0.25)
    parser.add_argument("--max_iter", type=int, default=1e10, help="max iterations (batches) to train on - use for debugging")
    parser.add_argument("--patience", type=int, default=5, required=False, help="num evaluations to wait for improved validation loss before early stopping (default 5)")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=11, help="random seed")
    parser.add_argument("--print_every", type=int, default=100, help="how often (in batches) to print avg'd loss")
    parser.add_argument("--cnn_on_top", action="store_true", help="set to use CNN on top of bert embedded tokens")
    parser.add_argument("--abc", action="store_true", help="set to use two types of context embedddings to distinguish left vs. right-context")
    parser.add_argument("--doc_position", action="store_true", help="set to use relative document position feature")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm (for gradient clipping).")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="num batches to wait before updating weights")
    parser.add_argument("--eval_steps", type=int, default=3000, help="number of steps between evaluations during training")
    parser.add_argument("--bert_oov_file", type=str, help="path to file of BERT OOV's to use for bert tokenizer")
    args = parser.parse_args()

    if args.model == 'bert':
        args.model = 'bert-base-uncased'
    elif args.model == 'clinicalbert':
        args.model = 'emilyalsentzer/Bio_ClinicalBERT'
    elif args.model == 'clinicalbert_disch':
        args.model = 'emilyalsentzer/Bio_Discharge_Summary_BERT'

    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
