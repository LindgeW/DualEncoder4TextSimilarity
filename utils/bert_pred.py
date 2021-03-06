from transformers import BertTokenizer, BertForMaskedLM
import torch


def load_pretrained_bert(bert_path):
    with torch.no_grad():
        tokenizer = BertTokenizer.from_pretrained(bert_path)
        bert_model = BertForMaskedLM.from_pretrained(bert_path)
        bert_model.eval()
    return tokenizer, bert_model


def masked_pred(sent, tokenizer=None, bert_model=None):
    tokens = ['[CLS]'] + tokenizer.tokenize(sent) + ['[SEP]']
    for i in range(1, len(tokens)-1):
        masked_tokens = tokens[:i] + ['[MASK]'] + tokens[i+1:]
        masked_token_ids = torch.tensor([tokenizer.convert_tokens_to_ids(masked_tokens)])
        segment_ids = torch.tensor([[0]*len(masked_tokens)])
        output = bert_model(masked_token_ids, token_type_ids=segment_ids)
        pred_idx = torch.argmax(output[0][0, i]).item()
        pred_token = tokenizer.convert_ids_to_tokens([pred_idx])[0]
        print(masked_tokens, ': ', pred_token)


if __name__ == '__main__':
    tokenizer, bert_model = load_pretrained_bert('bert/zh_wwm_ext')
    sent = '小明来自天津大学！'
    # sent = '他在阿里巴巴实习'
    masked_pred(sent, tokenizer, bert_model)