# this should basically hopefully replace data_generation.py
# take recon tool output (doc_id.json / doc_id.pkl) and write out MIMIC_{split}_{binary,finegrained}.sentclf.csv

import json
import os
import pandas as pd
from tqdm import tqdm
from nltk import word_tokenize

from utils import section_tokenize
from agreement import Span, get_sentence_level_labels

base_dir = 'all_revised_data'
doc_jsons = [f for f in os.listdir(base_dir) if f.endswith('.json')]
df_data = []
for jf in tqdm(doc_jsons):
    doc = json.load(open(os.path.join(base_dir, jf)))
    doc_anno = doc['users'][-1]['spans']
    
    text = doc['text']
    sentences, _ = section_tokenize(text, ['wboag'])

    # pull out followup sentences
    followup_sents = []
    add_to_followup = False
    for ix, sent in enumerate(sentences):
        if add_to_followup:
            followup_sents.append(ix)
        if sent.endswith(':'):
            if 'discharge instructions' in sent.lower() or 'followup instructions' in sent.lower():
                if not add_to_followup:
                    followup_sents.append(ix)
                add_to_followup = True
            else:
                if not ('date' in sent.lower() or 'time' in sent.lower() or 'phone' in sent.lower() or 'provider' in sent.lower()):
                    add_to_followup = False

    labels = get_sentence_level_labels(text, sentences, doc_anno)

    # add PT instructions labels based on extracted instructions
    PT_LABEL = 'Case-specific instructions for patient'
    sent_labels = []
    for sent_ix, lbls in enumerate(labels):
        # remove extraneous stuff i dont wanna debug rn
        for char in 'NOT':
            if char in lbls:
                lbls.remove(char)

        span_types = set(lbls)
        if sent_ix in followup_sents:
            span_types.add(PT_LABEL)
        if len(span_types) > 0:
            sent_labels.append(sorted(span_types))
        else:
            sent_labels.append([])

    doc_id = doc['id']
    for sent_ix, (sent, label) in enumerate(zip(sentences, sent_labels)):
        df_data.append([doc_id, sent_ix, '', word_tokenize(sent), [f'I-{lbl}' for lbl in label]])

df = pd.DataFrame(df_data, columns=['doc_id', 'sent_index', 'annotator', 'sentence', 'labels'])

out_file = 'processed_wboag_nltk_20201210/all_revised.csv'
df.to_csv(out_file, index=False)
