"""
The character offset is in the form
    MIMIC_row_id: Label type -> char offsets.
"""
from collections import Counter
import os
import json
import csv
import sys
from datetime import datetime

sys.path.append("../")
import pandas as pd


from tqdm.auto import tqdm
from typing import List
from models import Span
from utils import section_tokenize, sentence_to_token
from i2b2 import documents as i2b2_document
from CLIP import list_of_label_types
from os.path import isfile, join
from os import listdir

source_dir = "MIMIC/source"
# raw_notes = pd.read_csv(os.path.join(source_dir, "NOTEEVENTS.csv"))
test_document_ids = json.load(open("test_document_ids", "r"))


def get_i2b2_files():
    current_dir = "i2b2/source/concept_assertion_relation_training_data/"
    onlyfiles_beth = [
        os.path.join(current_dir, "beth/txt", f)
        for f in listdir(os.path.join(current_dir, "beth/txt"))
        if isfile(os.path.join(current_dir, "beth/txt", f))
    ]
    onlylabels_beth = [
        os.path.join(current_dir, "beth/concept", f)
        for f in listdir(os.path.join(current_dir, "beth/concept"))
        if isfile(os.path.join(current_dir, "beth/concept", f))
    ]
    onlyfiles_beth = [x for x in onlyfiles_beth if "-" in x]
    onlylabels_beth = [x for x in onlylabels_beth if "-" in x]
    onlyfiles_beth = sorted(
        onlyfiles_beth, key=lambda x: int(x.split("-")[1].split(".")[0])
    )
    onlylabels_beth = sorted(
        onlylabels_beth, key=lambda x: int(x.split("-")[1].split(".")[0])
    )
    onlyfiles_partners = [
        os.path.join(current_dir, "partners/txt", f)
        for f in listdir(os.path.join(current_dir, "partners/txt"))
        if isfile(os.path.join(current_dir, "partners/txt", f))
    ]
    onlyfiles_partners_unann = [
        os.path.join(current_dir, "partners/unannotated", f)
        for f in listdir(os.path.join(current_dir, "partners/unannotated"))
        if isfile(os.path.join(current_dir, "partners/unannotated", f))
    ]
    onlylabels_partners = [
        os.path.join(current_dir, "partners/concept", f)
        for f in listdir(os.path.join(current_dir, "partners/concept"))
        if isfile(os.path.join(current_dir, "partners/concept", f))
    ]
    copy_label = [onlylabels_partners[0] for i in range(len(onlyfiles_partners_unann))]
    onlyfiles_partners.extend(onlyfiles_partners_unann)
    onlylabels_partners.extend(copy_label)
    onlyfiles_partners = sorted(
        onlyfiles_partners,
        key=lambda x: int(x.split("/")[-1].split(".")[0].split("_")[0]),
    )
    onlylabels_partners = sorted(
        onlylabels_partners,
        key=lambda x: int(x.split("/")[-1].split(".")[0].split("_")[0]),
    )
    onlyfiles_beth = zip(onlyfiles_beth, onlylabels_beth)
    onlyfiles_partners = zip(onlyfiles_partners, onlylabels_partners)
    df = pd.DataFrame(columns=["id", "text", "labels"])
    dataset_mapping = json.load(open("i2b2-test_offset_mapping.jsonl", "r"))
    for dataset in [onlyfiles_beth, onlyfiles_partners]:
        for file, label_file in dataset:
            if file.split(current_dir)[1] not in dataset_mapping:
                # Only include in the df file i2b2 data that is in the CLIP dataset.
                continue
            tokenized_sents, _ = i2b2_document.read_i2b2(file, label_file)
            spans = []
            for tag_name, offsets in dataset_mapping[
                file.split(current_dir)[1]
            ].items():
                for j, offset in enumerate(offsets):
                    spans.append(
                        Span(
                            type=tag_name,
                            id=j,
                            document_id=file.split(current_dir)[1],
                            start=offset["start_offset"],
                            end=offset["end_offset"],
                        )
                    )
            current = pd.DataFrame(
                [[file.split(current_dir)[1], tokenized_sents, spans]],
                columns=["id", "text", "labels"],
            )
            df = df.append(current)
    assert len(df) == len(dataset_mapping)
    return df


def get_dataset_raw_MIMIC_offsets(split="MIMIC_train"):
    if "i2b2" in split:
        dataset = get_i2b2_files()
        return dataset, split
    if split == 'MIMIC_train':
        note_pd = pd.read_csv('mimic-train.csv')
    elif split == 'MIMIC_val':
        note_pd = pd.read_csv('mimic-val.csv')
    elif split == 'MIMIC_test':
        note_pd = pd.read_csv('mimic-test.csv')
    elif split == 'i2b2_test':
        note_pd = pd.read_csv('i2b2-test.csv')
    #dataset_mapping = json.load(open("%s_offset_mapping.jsonl" % split))
    dataset_mapping = json.load(open("v2-%s-mapping.json" % split.lower().replace('_', '-')))
    dataset = []
    for note_id in dataset_mapping.keys():
        #orig_MIMIC_note = raw_notes[raw_notes["ROW_ID"] == int(note_id)]
        orig_MIMIC_note = note_pd[note_pd["id"] == int(note_id)]
        spans = []
        for tag_name, offsets in dataset_mapping[note_id].items():
            for j, offset in enumerate(offsets):
                spans.append(
                    Span(
                        type=tag_name,
                        id=j,
                        document_id=note_id,
                        start=offset["start_offset"],
                        end=offset["end_offset"],
                    )
                )
        # Make sure that the start_offset and end_offset is from token level.
        dataset.append([str(note_id), orig_MIMIC_note["TEXT"].iloc[0], spans])
    return pd.DataFrame(dataset, columns=["id", "text", "labels"]), split


def tags_to_IO_binary(label_sentences):
    bio_labels = []
    for label_sent in label_sentences:
        bio_label_sent = []
        for label in label_sent:
            if len(label) == 1 and label[0] == "NOT":
                bio_label_sent.append("O")
            else:
                # We train with I-O format for binary.
                bio_label_sent.append("I-followup")
        assert len(label_sent) == len(bio_label_sent)
        bio_labels.append(bio_label_sent)
    assert len(bio_labels) == len(label_sentences)
    return bio_labels


def tags_to_IO_finegrained(label_sentences, possible_labels):
    bio_labels = []
    for label_sent in label_sentences:
        bio_label_sent = []
        for label in label_sent:
            if (len(label) == 1 and label[0] == "NOT") or label == 'NOT':
                bio_label_sent.append(["O"])
            else:
                bio_label = []
                bio_label.extend(list(set(["I-" + key for key in label])))
                bio_label_sent.append(bio_label)
        assert len(label_sent) == len(bio_label_sent)
        bio_labels.append(bio_label_sent)
    assert len(bio_labels) == len(label_sentences)
    return bio_labels


def preprocess_dataset(
    documents,
    word_tokenizer_type: List[str],
    sentence_tokenizer_type: str,
    cast: str = "binary",
    name=None,
) -> str:
    PT_LABEL = 'I-Case-specific instructions for patient'
    rows = []
    doc_offsets = []
    doc_labels = []
    doc_ids = []
    headers = Counter()
    docs_with_instructions_sections = 0
    num_labeled_sents_in_instructions_sections = 0
    num_labeled_sents = 0
    num_non_auto_sents_in_instructions_sections = 0
    num_auto_sents = 0
    nonpt_labeled_sents = 0
    nonpt_labeled_sents_in_instructions_sections = 0
    for i in tqdm(range(len(documents)), desc="preprocessing"):
        doc = documents.iloc[i]
        text = doc["text"]
        if "i2b2" in name:
            text = " ".join([x for y in doc["text"] for x in y])
        sentences, sent_offsets = section_tokenize(text, sentence_tokenizer_type)
        #print("----------------------------------------------------")
        #print("------------------ START DOCUMENT -------------------")
        #print("----------------------------------------------------")
        #print(text)
        #print("----------------------------------------------------")
        #print("------------------ END DOCUMENT -------------------")
        #print("----------------------------------------------------")
        #print("----------------------------------------------------")
        #print("------------------ START SENTENCES -------------------")
        #print("----------------------------------------------------")
        followup_sents = []
        add_to_followup = False
        has_instructions_sections = False
        for ix, sent in enumerate(sentences):
            #print(f"-------------- SENTENCE {ix} -----------------------")
            #print(sent)
            if add_to_followup:
                followup_sents.append(ix)
            if sent.endswith(':'):
                headers[sent] += 1
                if 'discharge instructions' in sent.lower() or 'followup instructions' in sent.lower():
                    # add header to followup if didn't already
                    if not add_to_followup:
                        followup_sents.append(ix)
                    add_to_followup = True
                    has_instructions_sections = True
                else:
                    # still add things to followup if it's phone/date/time/provider info
                    if not ('date' in sent.lower() or 'time' in sent.lower() or 'phone' in sent.lower() or 'provider' in sent.lower()):
                        add_to_followup = False
        if has_instructions_sections:
            docs_with_instructions_sections += 1
        #print("----------------------------------------------------")
        #print("------------------ END SENTENCES -------------------")
        #print("----------------------------------------------------")
        spans = doc["labels"]
        tokens, labels = sentence_to_token(text, sentences, word_tokenizer_type, spans)
        list_of_pos_label_types = [
            "Appointment-related followup",
            "Imaging-related followup",
            "Case-specific instructions for patient",
            "Medication-related followups",
            "Other helpful contextual information",
            "Lab-related followup",
            "Procedure-related followup",
        ]
        if cast == "binary":
            labels = tags_to_IO_binary(labels)
        else:
            labels = tags_to_IO_finegrained(labels, list_of_pos_label_types)
        rows.append({"document_id": doc["id"], "tokens": tokens, "labels": labels})
        sent_labels = []
        for sent_ix, (tks, lbls) in enumerate(zip(tokens, labels)):
            span_types = set()
            for ix, (tk, lbl) in enumerate(zip(tks, lbls)):
                span_types.update(set(lbl))
            if sent_ix in followup_sents:
                span_types.add(PT_LABEL)
                num_auto_sents += 1
            if 'O' in span_types:
                span_types.remove('O')
            if len(span_types) > 0:
                has_nonpt_label = len(set([f"I-{lbl}" for lbl in list_of_pos_label_types[:2] + list_of_pos_label_types[3:]]).intersection(span_types)) > 0
                if has_nonpt_label:
                    nonpt_labeled_sents += 1
                sent_labels.append(sorted(span_types))
                num_labeled_sents += 1
                if sent_ix in followup_sents:
                    if len(span_types) > 1:
                        num_non_auto_sents_in_instructions_sections += 1
                        if has_nonpt_label:
                            nonpt_labeled_sents_in_instructions_sections += 1
                    num_labeled_sents_in_instructions_sections += 1
            else:
                sent_labels.append(['O'])
        doc_labels.append(sent_labels)
        doc_offsets.append(sent_offsets)
        doc_ids.append(doc['id'])
    # go back thru to add patient instruction labels to token lvel
    if cast != 'binary':
        new_labels = []
        print("adding patient instruction labels")
        for row, doc_label in tqdm(zip(rows, doc_labels)):
            new_doc_labels = []
            for tok_labels, sent_label in zip(row['labels'], doc_label):
                # modify sentence labels
                new_tok_labels = []
                if PT_LABEL in sent_label:
                    # add case-specific label to every token if it exists at sentence level
                    for tok_label in tok_labels:
                        new_tok_label = set(tok_label)
                        new_tok_label.add(PT_LABEL)
                        if 'O' in new_tok_label:
                            new_tok_label.remove('O')
                        new_tok_labels.append(sorted(new_tok_label))
                else:
                    # new sentence label is same
                    new_tok_labels = tok_labels
                new_doc_labels.append(new_tok_labels)
            new_labels.append(new_doc_labels)
        rows = [{'document_id': row['document_id'], 'tokens': row['tokens'], 'labels': labels} for row, labels in zip(rows, new_labels)]
    return pd.DataFrame(rows), doc_labels, doc_offsets, doc_ids


def make_test_files(dir_name):
    # Merge the MIMIC and i2b2 test set files.
    for cast in ["binary", "finegrained"]:
        mimic_test = pd.read_csv(f"{dir_name}/MIMIC_test_{cast}.csv")
        mimic_test["source"] = "mimic"
        i2b2_test = pd.read_csv(f"{dir_name}/i2b2-test_{cast}.csv")
        i2b2_test["source"] = "i2b2"
        mimic_test = mimic_test.append(i2b2_test)
        current = pd.DataFrame(columns=["document_id", "tokens", "labels", "source"])
        mimic_test["document_id"] = mimic_test["document_id"].apply(lambda x: str(x))
        for doc_id in test_document_ids:
            doc = mimic_test[mimic_test["document_id"] == doc_id].iloc[0]
            current = current.append(doc)
        current.to_csv(f"{dir_name}/test_{cast}.csv")


def main(args):

    print("load raw MIMIC offsets")
    train_pd, tr_name = get_dataset_raw_MIMIC_offsets(("MIMIC_train"))
    i2b2_test_pd, itest_name = get_dataset_raw_MIMIC_offsets(("i2b2-test"))
    val_pd, val_name = get_dataset_raw_MIMIC_offsets(("MIMIC_val"))
    mimic_test_pd, mtest_name = get_dataset_raw_MIMIC_offsets(("MIMIC_test"))
    datasets = {
        #itest_name: i2b2_test_pd,
        #tr_name: train_pd,
        val_name: val_pd,
        mtest_name: mimic_test_pd,
    }
    date = datetime.strftime(datetime.now().date(), '%Y%m%d')
    dir_name = f"processed_{args.sent_tok}_{args.word_tok}_{date}"
    os.makedirs(dir_name, exist_ok=True)
    for name, dataset in datasets.items():
        if 'i2b2' in name:
            continue
        print(f'processing {name}')
        preproc_file = f"{dir_name}/{name}_finegrained.csv"
        preproc_dataset, doc_labels, doc_offsets, doc_ids = preprocess_dataset(
            dataset,
            word_tokenizer_type=args.word_tok,
            sentence_tokenizer_type=[args.sent_tok],
            cast="finegrained",
            name=name,
        )
        # reformat with sentence-level labels for loading into reconciliation tool
        print("reformatting for loading into recon tool")
        data = []
        for surrogate_doc, doc_id, offsets, labels in zip(dataset['text'], doc_ids, doc_offsets, doc_labels):
            spans = []
            for sent_ix, (offset, label) in enumerate(zip(offsets, labels)):
                for lbl in label:
                    if lbl != 'O':
                        span = Span(
                            id=sent_ix,
                            type=lbl[2:], # get rid of I- prefix
                            document_id=doc_id,
                            start=offset[0],
                            end=offset[1]
                            )
                        spans.append(span)
            recon_format_spans = str([[span.id, span.type, span.start, span.end - span.start] for span in spans])
            recon_format_tokens = str({'tokens': [(s, 0) for s in surrogate_doc.split(' ')]})
            # make up user id, empty notes
            data.append(['1234321', '', doc_id, recon_format_spans, recon_format_tokens])

        df = pd.DataFrame(data, columns=['user_id', 'user_notes', 'document_id', 'annotations', 'document'])
        df.to_csv(f'{dir_name}/{name}_finegrained_sent_recon.csv', index=False)
                        #of.write(json.dumps(span.__dict__) + "\n")

        preproc_dataset.to_csv(preproc_file, index=False)
        preproc_file = f"{dir_name}/{name}_binary.csv"
        preproc_dataset, doc_labels, doc_offsets, doc_ids = preprocess_dataset(
            dataset,
            word_tokenizer_type=args.word_tok,
            sentence_tokenizer_type=[args.sent_tok],
            cast="binary",
            name=name,
        )
        preproc_dataset.to_csv(preproc_file, index=False)

    make_test_files(dir_name)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("sent_tok", choices=['stanfordnlp_pretrained', 'stanfordnlp_whitespace', 'nltk', 'nnsplit', 'deepsegment', 'wboag', 'syntok', 'custom'])
    parser.add_argument("word_tok", choices=['stanfordnlp_pretrained', 'stanfordnlp_whitespace', 'nltk', 'syntok', 'nnsplit'])
    args = parser.parse_args()
    print(args)
    main(args)

