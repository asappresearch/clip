from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import sys

sys.path.append('sentclf')
from neural_baselines import SentEvalDataset
from bert_model import BertSequenceMultilabelClassificationContext
from bert import ctx_collator, select_inputs
from constants import LABEL_TYPES

N_CONTEXT_SENTENCES = 2
TASK = 'multilabel'

label2id = {label:ix for ix,label in enumerate(LABEL_TYPES)}
id2label = {ix:label for label,ix in label2id.items()}

model = BertSequenceMultilabelClassificationContext.from_pretrained('jamesmullenbach/CLIP_DNote_BERT_Context')
tokenizer = AutoTokenizer.from_pretrained('jamesmullenbach/CLIP_DNote_BERT_Context')

model.set_sep_token_id(tokenizer.sep_token_id)
model.set_n_context_sentences(N_CONTEXT_SENTENCES)
model.set_task(TASK)

# load model thresholds
label_threshs = np.load('clip_dnote_label_threshs.npy')

dataset = SentEvalDataset('dummy_test_data.csv', TASK, n_context_sentences=N_CONTEXT_SENTENCES)
loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=partial(ctx_collator, task=TASK, tokenizer=tokenizer))

for x in loader:
    inputs = select_inputs(x)
    loss, pred = model(**inputs)
    pred = torch.sigmoid(pred)
    pred_label_ids = np.where(pred.data.squeeze().numpy() > label_threshs)[0]
    pred_labels = [id2label[lid] for lid in pred_label_ids]
    print(f"pred_labels: {pred_labels}")
