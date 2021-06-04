import sys

from dataclasses import dataclass

from nltk import word_tokenize

LABEL_TYPES = ['Imaging-related followup',
 'Appointment-related followup',
 'Medication-related followups',
 'Procedure-related followup',
 'Lab-related followup',
 'Case-specific instructions for patient',
 'Other helpful contextual information',
 ]
LABEL2IX = {label: ix for ix, label in enumerate(LABEL_TYPES)}
label2abbrev = {'Imaging-related followup': 'Imaging',
        'Appointment-related followup': 'Appointment',
        'Medication-related followups': 'Medication',
        'Procedure-related followup': 'Procedure',
        'Lab-related followup': 'Lab',
        'Case-specific instructions for patient': 'Patient instructions',
        'Other helpful contextual information': 'Other',
 }

@dataclass
class Span:
    id: str
    document_id: str
    type: str
    start: int
    end: int

def index_in_span(span: Span, token_start: int, token_end):
    if (
        token_start <= int(span['start']) <= token_end
        or token_start <= int(span['end']) <= token_end
        or int(span['start']) <= token_start <= int(span['end'])
        or int(span['start']) <= token_end <= int(span['end'])
    ):
        return True
    else:
        return False

def flatten(lol):
    if isinstance(lol[0], list):
        return [l for ll in lol for l in ll]
    else:
        return lol

def get_sentence_level_labels(text, sentences, spans):
    result_tokens = []
    labels = []
    cursor = 0
    for sentence in sentences:
        sentence_labels = []
        sentence_tokens = []
        sentence_start = cursor
        sentence_index = text.find(sentence)
        if sentence_index == -1:
            if sentence == "-----" or sentence == "_____":
                labels.append(['NOT'])
                result_tokens.append([sentence])
                continue
            else:
                print("ALERT")
                sys.exit(0)
        tokens = word_tokenize(sentence)
        cursor = 0
        for token in tokens:
            sentence_tokens.append(token)
            token_index = sentence[cursor:].find(token)
            # Avoid wonky things happening with find() where it finds
            # a preivous version of a token that appears in the sentence twice.
            index = sentence_index + token_index + cursor
            curr_label = []
            cursor += token_index + len(token)
            for span in spans:
                if span['start'] is None or span['end'] is None:
                    continue
                if index_in_span(span, index, index + len(token)):
                    curr_label.append(span['type'])
            if len(curr_label) == 0:
                curr_label = ["NOT"]
            sentence_labels.append(curr_label)
        labels.append(sentence_labels)
        result_tokens.append(sentence_tokens) 
    sent_level_labels = []
    for tok_labels in labels:
        sent_label = set()
        f2 = flatten(flatten(tok_labels))
        for label in f2:
            if isinstance(label, list):
                sent_label.update(label)
            else:
                sent_label.add(label)
        if 'NOT' in sent_label:
            sent_label.remove('NOT')
        sent_level_labels.append(sent_label)
    return sent_level_labels
