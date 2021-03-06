import torch
from model.QAMatcher import BertQAMatcher
from utils.datautil import load_from, batch_variable
from utils.dataset import DataLoader
import sys

MODEL_CKPT = 'model_128.pkl'
VOCAB_PATH = 'vocab.pkl'
BERT_PATH = 'bert/base/'
DEVICE = torch.device('cuda', 1)


def load_ckpt(ckpt_path, vocab_path, bert_path):
    print(f'pretrained model path: {ckpt_path}, vocab path: {vocab_path}, bert path: {bert_path}')
    vocabs = load_from(vocab_path)
    # ckpt = torch.load(ckpt_path, map_location='cpu')
    ckpt = torch.load(ckpt_path)
    args = ckpt['args_settings']
    model = BertQAMatcher(
        bert_embed_dim=args.bert_embed_dim,
        num_bert_layer=args.bert_layer,
        cmp_mode=args.sim_mode,
        dropout=args.dropout,
        bert_model_path=bert_path
    ).to(DEVICE)
    model.load_state_dict(ckpt['model_state'])
    model.zero_grad()
    print('Loading the previous model states ...')
    return model, vocabs


def get_sent_repr(sent, model, vocabs):
    print('processing ...')
    model.eval()
    with torch.no_grad():
        bert_inp = vocabs['bert'].batch_bert2id([sent[:510]])
        pred_score = model.get_repr(bert_inp)
        sent_repr_ = pred_score.data.tolist()[0]
    return sent_repr_


def get_batch_sent_reprs(sents: list, batch_size=8, model=None, vocabs=None):
    print('processing ...')
    if not isinstance(sents, list) and len(sents) == 0:
        return None
    test_loader = DataLoader(sents, batch_size=batch_size)
    model.eval()
    with torch.no_grad():
        for i, batcher in enumerate(test_loader):
            crop_batcher = [s[:510] for s in batcher]
            bert_inp = vocabs['bert'].batch_bert2id(crop_batcher)
            cuda_bert_inp = bert_inp.to(DEVICE)
            pred_score = model.get_repr(cuda_bert_inp)
            sent_reprs = pred_score.data.tolist()
    return sent_reprs


# Run:
# python extractor.py 这是个测试例子！
if __name__ == '__main__':
    sent = sys.argv[1:2]  # sentence
    model, vocabs = load_ckpt(MODEL_CKPT, VOCAB_PATH, BERT_PATH)
    sent_repr = get_sent_repr(sent, model, vocabs)
    print(sent_repr)
    print('Done !!')

