from transformers import BertTokenizer, BertForMaskedLM
import torch
import numpy as np


def load_pretrained_bert(bert_path):
    with torch.no_grad():
        tokenizer = BertTokenizer.from_pretrained(bert_path)
        bert_model = BertForMaskedLM.from_pretrained(bert_path)
        bert_model.eval()
    return tokenizer, bert_model


def ppl_score(sent, tokenizer=None, bert_model=None):
    # tokens = ['[CLS]'] + tokenizer.tokenize(sent) + ['[SEP]']
    tokens = tokenizer.tokenize(sent)
    token_ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokens)])
    sent_loss = 0.
    for i, t in enumerate(tokens):
        tokens[i] = '[MASK]'
        masked_token_ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokens)])
        output = bert_model(masked_token_ids, labels=token_ids)
        sent_loss += output.loss.data.numpy()
    return np.exp(sent_loss / len(tokens))


if __name__ == '__main__':
    tokenizer, bert_model = load_pretrained_bert('bert/base')
    inputs = tokenizer("The capital of France is [MASK].", return_tensors="pt")
    labels = tokenizer("The capital of France is Paris.", return_tensors="pt")["input_ids"]
    outputs = bert_model(**inputs, labels=labels)
    print(outputs.loss)
    sent1 = '我来自湖北！'
    ppl1 = ppl_score(sent1, tokenizer, bert_model)
    sent2 = '我出自湖北！'
    ppl2 = ppl_score(sent2, tokenizer, bert_model)
    print('Sentence PPL:', ppl1, ppl2)