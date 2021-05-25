import typing as t
from tokenizers import SentenceTokenizer, WordTokenizer
from models import Span


def get_sentences_list(text: str, model_type: str) -> t.List[str]:
    """
    Splits document text into a list of sentences, given some model.
    """
    sentences = []
    sent_offsets = []
    stok = SentenceTokenizer.from_type(model_type)
    if isinstance(text, list):
        sentences, sent_offsets = list(zip(*map(stok.tokenize, text)))
    elif isinstance(text, str):
        sentences, sent_offsets = stok.tokenize(text)
    return sentences, sent_offsets


def section_tokenize(text, st_types):
    sentences = [text]
    for st_type in st_types:
        sentences, sent_offsets = get_sentences_list(sentences, st_type)
        sentences = [sent for tokenized_sent in sentences for sent in tokenized_sent]
        sent_offsets = [so for sent_offset in sent_offsets for so in sent_offset]
    return sentences, sent_offsets


def sentence_to_token(
    text: str, sentences: t.List[str], wt_type: str, spans: t.List[Span]
):
    word_tokenizer = WordTokenizer.from_type(wt_type)

    labels = []
    cursor = 0
    result_tokens = []

    def index_in_span(span: Span, token_start: int, token_end):
        # span starts overlaps with token
        # 2. (the span ends in th emiddl eof the token)
        # 3 and 4. token is in th emiddl eof a span.
        if (
            token_start <= int(span.start) <= token_end
            or token_start <= int(span.end) <= token_end
            or int(span.start) <= token_start <= int(span.end)
            or int(span.start) <= token_end <= int(span.end)
        ):
            return True
        else:
            return False

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
        tokens = word_tokenizer.tokenize(sentence)
        cursor = 0
        for token in tokens:
            sentence_tokens.append(token)
            token_index = sentence[cursor:].find(token)
            # Avoid wonky things ahppening with find() where it finds
            # a preivous version of a token that appears int he sentence twice.
            index = sentence_index + token_index + cursor
            curr_label = []
            cursor += token_index + len(token)
            for span in spans:
                if span.start is None or span.end is None:
                    continue
                if index_in_span(span, index, index + len(token)):
                    curr_label.append(span.type)
            if len(curr_label) == 0:
                curr_label = ["NOT"]
            sentence_labels.append(curr_label)
        labels.append(sentence_labels)
        result_tokens.append(sentence_tokens)
    assert len(sentences) == len(labels)
    return result_tokens, labels
