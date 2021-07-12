import logging

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn.modules.loss import BCELoss
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.bert.modeling_bert import BertConfig, BertForSequenceClassification, BertModel, \
    BertPreTrainedModel

# from transformers.models.roberta.modeling_roberta import RobertaConfig, RobertaForSequenceClassification, \
#     RobertaModel, RobertaPreTrainedModel

logger = logging.getLogger(__name__)

_CONFIG_FOR_DOC = "BertaConfig"
_TOKENIZER_FOR_DOC = "BertTokenizer"

# ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST = [
#     "roberta-base",
#     "roberta-large",
#     "roberta-large-mnli",
#     "distilroberta-base",
#     "roberta-base-openai-detector",
#     "roberta-large-openai-detector",
#     # See all RoBERTa models at https://huggingface.co/models?filter=roberta
# ]


class TokenClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, features, **kwargs):
        pass


class ClassificationHead(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, *kwargs):
        x = self.dropout(features)
        x = self.out_proj(x)
        return x


class RepresentationProjectionLayer(nn.Module):
    def __init__(self, config, layer=-1, tokens=False, tagger=False):
        super().__init__()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.layer_to_use = layer
        self.tokens = tokens
        self.tagger = tagger
        self.hidden_size = config.hidden_size
        if tokens and tagger:
            raise Exception(
                'Inconsistent configuration: tokens and tagger cannot both be true'
            )

    def forward(self, features, event_tokens, **kwargs):
        seq_length = features[0].shape[1]
        if self.tokens:
            # grab the average over the tokens of the thing we want to classify
            # probably involved passing in some sub-sequence of interest so we know what tokens to grab,
            # then we average across those tokens.
            token_lens = event_tokens.sum(1)
            expanded_tokens = event_tokens.unsqueeze(2).expand(
                features[0].shape[0], seq_length, self.hidden_size)
            filtered_features = features[self.layer_to_use] * expanded_tokens
            x = filtered_features.sum(1) / token_lens.unsqueeze(1).expand(
                features[0].shape[0], self.hidden_size)
        elif self.tagger:
            x = features[self.layer_to_use]
        else:
            # take <s> token (equiv. to [CLS])
            x = features[self.layer_to_use][:, 0, :]
        # x = self.dropout(x)
        # x = self.dense(x)
        # x = self.activation(x)
        return x


# class CnlpBertForClassification(RobertaPreTrainedModel):
class CnlpBertForClassification(BertPreTrainedModel):
    def __init__(
            self,
            config,
            #  num_labels_list=[3],
            num_labels_list=[2],
            layer=-1,
            freeze=False,
            tokens=False,
            tagger=[False]):

        ###### update paramters "num_labels_list" and "tagger" for different tasks #######
        super().__init__(config)
        self.num_labels = num_labels_list

        # self.roberta = RobertaModel(config)
        # if freeze:
        #     for param in self.roberta.parameters():
        #         param.requires_grad = False

        self.bert = BertModel(config)
        if freeze:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.feature_extractors = nn.ModuleList()
        self.classifiers = nn.ModuleList()

        for task_ind, task_num_labels in enumerate(num_labels_list):
            self.feature_extractors.append(
                RepresentationProjectionLayer(config,
                                              layer=layer,
                                              tokens=tokens,
                                              tagger=tagger[task_ind]))
            # if task_num_labels == 2:
            #     self.classifiers.append(ClassificationHead(config, 1))
            # else:
            self.classifiers.append(ClassificationHead(config,
                                                       task_num_labels))

        # Are we operating as a sequence classifier (1 label per input sequence) or a tagger (1 label per input token in the sequence)
        self.tagger = tagger
        # self.sigmoid = nn.Sigmoid()

        self.init_weights()

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
        event_tokens=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        # outputs = self.roberta(input_ids,
        #                        attention_mask=attention_mask,
        #                        token_type_ids=token_type_ids,
        #                        position_ids=position_ids,
        #                        head_mask=head_mask,
        #                        inputs_embeds=inputs_embeds,
        #                        output_attentions=output_attentions,
        #                        output_hidden_states=True,
        #                        return_dict=True)

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds,
                            output_attentions=output_attentions,
                            output_hidden_states=True,
                            return_dict=True)

        batch_size, seq_len = input_ids.shape

        logits = []

        loss = None
        for task_ind, task_num_labels in enumerate(self.num_labels):
            features = self.feature_extractors[task_ind](outputs.hidden_states,
                                                         event_tokens)

            task_logits = self.classifiers[task_ind](features)

            # if task_num_labels == 2:
            #     task_logits = self.sigmoid(task_logits).squeeze(-1)
            logits.append(task_logits)

            if labels is not None:
                # if task_num_labels == 2:
                #     loss_fct = BCELoss()
                # else:
                weights = [0.1, 50]
                class_weights = torch.FloatTensor(weights).to(
                    task_logits.device)
                loss_fct = CrossEntropyLoss(weight=class_weights)

                # loss_fct = CrossEntropyLoss()
                if self.tagger[task_ind] == True:
                    if attention_mask is not None:
                        active_loss = attention_mask.view(-1) == 1
                        active_logits = task_logits.view(-1, task_num_labels)
                        active_labels = torch.where(
                            active_loss, labels.view(-1),
                            torch.tensor(
                                loss_fct.ignore_index).type_as(labels))
                        task_loss = loss_fct(active_logits, active_labels)
                    else:
                        task_loss = loss_fct(
                            task_logits.view(-1, task_num_labels),
                            labels.view(-1))

                else:
                    # if task_num_labels == 2:

                    #     labels = labels.to(torch.float32)
                    #     task_loss = loss_fct(task_logits, labels.view(-1))

                    # else:
                    task_loss = loss_fct(task_logits.view(-1, task_num_labels),
                                         labels.view(-1))

                if loss is None:
                    loss = task_loss
                else:
                    loss += task_loss


#         if not return_dict:
#             output = (logits,) + outputs[2:]
#             return ((loss,) + output) if loss is not None else output

        if self.training:
            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        else:
            return SequenceClassifierOutput(loss=loss, logits=logits)
