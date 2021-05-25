from transformers import BertPreTrainedModel, BertModel, BertForMaskedLM
from transformers.modeling_bert import BertLMPredictionHead
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, KLDivLoss
import torch.nn.functional as F

class BertSequenceMultilabelClassificationContext(BertPreTrainedModel):
    """
        BERT with n sentences of context on either side, using linear layer to
        predict as in Cohan et al 2019
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def set_sep_token_id(self, sep_token_id):
        # used for getting focus sentence representation
        self.sep_token_id = sep_token_id

    def set_n_context_sentences(self, n_context_sentences):
        self.n_context_sentences = n_context_sentences

    def set_task(self, task):
        #used for computing loss
        self.task = task

    def update_tok_type_embeddings(self):
        # used because pretrained BertModels have/assume only 2 different segment embeddings
        tok_typ_embeds = torch.zeros(3, 768)
        tok_typ_embeds[:2,:] = self.bert.embeddings.token_type_embeddings.weight.clone()
        # randomly init new embedding for 3rd segment type
        init = nn.Embedding(1, 768).weight.data.clone()
        tok_typ_embeds[2,:] = init
        new_embed = nn.Embedding.from_pretrained(tok_typ_embeds)
        self.bert.embeddings.token_type_embeddings = new_embed
        self.bert.embeddings.token_type_embeddings.weight.requires_grad = True

    def add_doc_position_feature(self, config):
        # need to modify final layer dimension for this feature
        self.classifier = nn.Linear(config.hidden_size+1, config.num_labels)

    def _collect_focus_repr(self, outputs, input_ids, doc_positions):
        # generate indices of end of sentences
        try:
            end_indices = torch.where(input_ids == self.sep_token_id)[1].reshape(-1,self.n_context_sentences*2+1)[:,self.n_context_sentences]
        except RuntimeError as e:
            print(f"error when pulling out end of sentence indices: {e}")
            return None
        end_mask = end_indices.unsqueeze(1).unsqueeze(2).repeat(1,1,outputs[0].shape[2])

        pooled_output = torch.gather(outputs[0], 1, end_mask).squeeze(dim=1)

        pooled_output = self.dropout(pooled_output)

        if doc_positions is not None and len(doc_positions) > 0:
            pooled_output = torch.cat((pooled_output, doc_positions.unsqueeze(1)), dim=1)
        return pooled_output

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        doc_positions=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        pooled_output = self._collect_focus_repr(outputs, input_ids, doc_positions)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.task == 'multilabel':
                loss = F.binary_cross_entropy_with_logits(outputs[0], labels)
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs


class BertContextMLMSwitch(BertSequenceMultilabelClassificationContext):
    """
        BERT with n sentences of context on either side, using linear layer to predict
        as in Cohan et al 2019, but used for pretraining with MLM on focus sentence
    """
    def __init__(self, config):
        super().__init__(config)
        self.mlm_predictions = BertLMPredictionHead(config)
        self.init_weights()

    def expand_vocab_by_num(self, num):
        # used because pretrained BertModels have/assume only 2 different segment embeddings
        cur_vocab_size = self.bert.embeddings.word_embeddings.weight.shape[0]
        new_word_embeds = torch.zeros(cur_vocab_size + num, 768)
        new_word_embeds[:cur_vocab_size,:] = self.bert.embeddings.word_embeddings.weight.clone()
        # randomly init new embedding for 3rd segment type
        init = nn.Embedding(num, 768).weight.data.clone()
        init = init / init.norm(dim=1).unsqueeze(dim=1)
        new_word_embeds[cur_vocab_size:,:] = init
        new_embed = nn.Embedding.from_pretrained(new_word_embeds)
        self.bert.embeddings.word_embeddings = new_embed
        self.bert.embeddings.word_embeddings.weight.requires_grad = True

        # update mlm predictions output as well
        self.mlm_predictions.decoder = nn.Linear(self.mlm_predictions.decoder.weight.shape[1], cur_vocab_size + num, bias=False)
        self.mlm_predictions.bias = nn.Parameter(torch.zeros(cur_vocab_size + num))
        self.mlm_predictions.decoder.bias  = self.mlm_predictions.bias

        # do this so mlm prediction head resizes properly
        self.config.vocab_size += num

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        mlm_labels=None,
        output_attentions=None,
        output_hidden_states=None,
        doc_positions=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        # generate indices of end of sentences
        pooled_output = self._collect_focus_repr(outputs, input_ids, doc_positions)

        mlm_scores = self.mlm_predictions(outputs[0])

        if labels is not None:
            # predict switch label with [SEP] token
            logits = self.classifier(pooled_output)
            outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        else:
            outputs = (outputs[2:],)

        total_loss = None
        if mlm_labels is not None:
            # get MLM loss
            loss_fct = CrossEntropyLoss()
            mlm_loss = loss_fct(mlm_scores.view(-1, self.config.vocab_size), mlm_labels.view(-1))
            total_loss = mlm_loss
            if labels is not None:
                # get switch prediction loss and add
                switch_loss = loss_fct(logits.view(-1, 2), labels.view(-1))
                total_loss += switch_loss
            outputs = (total_loss,) + outputs

        return outputs


class BertForSequenceMultilabelClassification(BertPreTrainedModel):
    """
        simple mod of BERT to accept multilabel or binary task
        there may or may not be a class in huggingface that does this but this is here for historical reasons
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def add_doc_position_feature(self, config):
        self.classifier = nn.Linear(config.hidden_size+1, config.num_labels)

    def set_task(self, task):
        self.task = task

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        doc_positions=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        if doc_positions is not None and len(doc_positions) > 0:
            pooled_output = torch.cat((pooled_output, doc_positions.unsqueeze(1)), dim=1)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.task == 'multilabel':
                loss = F.binary_cross_entropy_with_logits(outputs[0], labels)
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs

