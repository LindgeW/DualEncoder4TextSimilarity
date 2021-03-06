import torch
from model.QAMatcher import BertQAMatcher
from utils.datautil import load_from, batch_variable, dataset_loader
from utils.dataset import DataLoader
import argparse


def load_ckpt(ckpt_path, vocab_path, bert_path):
    vocabs = load_from(vocab_path)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    args = ckpt['args_settings']
    model = BertQAMatcher(
        bert_embed_dim=args.bert_embed_dim,
        num_bert_layer=args.bert_layer,
        sim_mode=args.sim_mode,
        dropout=args.dropout,
        bert_model_path=bert_path
    )
    model.load_state_dict(ckpt['model_state'])
    model.zero_grad()
    print('Loading the previous model states ...')
    return model, vocabs


class FileWriter(object):
    def __init__(self, path, mode='w'):
        self.path = path
        self.fw = open(self.path, mode, encoding='utf-8')

    def write_to_txt(self, sents, preds, split=' '):
        assert len(sents) == len(preds)
        for st, pd in zip(sents, preds):
            sent = f'{pd}{split}{st}\n'
            self.fw.write(sent)

    def write_to_conll(self, golds, preds, split=' '):
        assert len(golds) == len(preds)
        for gt, pt in zip(golds, preds):
            sent = f'_{split}{gt}{split}{pt}\n'
            self.fw.write(sent)
        self.fw.write('\n')

    # def write_to_conll(self, wds, golds, preds, split=' '):
    #     assert len(wds) == len(golds) == len(preds)
    #     for wd, gt, pt in zip(wds, golds, preds):
    #         sent = f'{wd}{split}{gt}{split}{pt}\n'
    #         self.fw.write(sent)

    def close(self):
        if self.fw is not None:
            self.fw.close()


def evaluate(data_loader, model, vocabs, batch_size=1, save_path='output.txt'):
    print('processing ...')
    fw = FileWriter(save_path, 'w')
    for test_data in data_loader:
        test_loader = DataLoader(test_data, batch_size=batch_size)
        model.eval()
        with torch.no_grad():
            for i, batcher in enumerate(test_loader):
                batch = batch_variable(batcher, vocabs)
                pred_score = model(batch.query_bert_inp, batch.doc_bert_inp)
                sents = [inst.query + '\t' + inst.doc for inst in batcher]
                preds = pred_score.data.tolist()
                fw.write_to_txt(sents, preds, '\t')
    fw.close()


# Run:
# python predictor.py --data_path data.test --ckpt_path model.pkl  --vocab_path vocab.pkl --bert_path bert/base/
if __name__ == '__main__':
    parse = argparse.ArgumentParser('Inference')
    parse.add_argument('--data_path', type=str, help='test data path')
    parse.add_argument('--batch_size', type=int, default=32, help='data size')
    parse.add_argument('--ckpt_path', type=str, help='checkpoint path')
    parse.add_argument('--vocab_path', type=str, help='vocab path')
    parse.add_argument('--bert_path', type=str, help='bert path')
    parse.add_argument('--output_path', type=str, default='output.txt', help='output file path')
    args = parse.parse_args()

    model, vocabs = load_ckpt(args.ckpt_path, args.vocab_path, args.bert_path)
    test_data = dataset_loader(args.data_path)
    evaluate(test_data, model, vocabs, args.batch_size, args.output_path)
    print('Done !!')

