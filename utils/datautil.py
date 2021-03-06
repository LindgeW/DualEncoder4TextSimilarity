import os
from utils.vocab import MultiVocab, BERTVocab
import torch
import collections
import pickle
import json
import html
import random
import re
from utils.instance import Instance

SEP_A = ''


def _is_chinese(a_chr):
    return u'\u4e00' <= a_chr <= u'\u9fff'


def clear_line(s):
    s = s.strip()
    s = html.unescape(s)
    s = re.sub(r'&nbsp+', ' ', s)
    new_line = re.sub(r'\s+', ' ', s)
    return new_line


# def get_rand_line_from_file(path, range_=100):
#     fin = open(path, 'r', encoding='utf8', errors='ignore')
#     for _ in range(random.randint(1, range_)):
#         try:
#             next(fin)
#         except StopIteration:
#             fin.close()
#             fin = open(path, 'r', encoding='utf8', errors='ignore')
#     rand_line = next(fin)
#     fin.close()
#     return rand_line


# def read_pair(pos_reader, neg_reader):
#     try:
#         for pos_line in pos_reader:
#             neg_line = neg_reader.readline()
#             if not neg_line:
#                 break
#             pl = clear_line(pos_line)
#             nl = clear_line(neg_line)
#             yield pl, nl
#     except Exception as e:
#         print('exception occur: ', e)
#     yield None, None

def line_reader(reader):
    try:
        for line in reader:
            cl = clear_line(line)
            yield cl
    except Exception as e:
        print('exception occur: ', e)
    yield None


# Version I: creat negative samples manually
def dataset_loader(pos_path, neg_path=None, margin=5000):
    dataset = []
    neg_fp = None
    try:
        if neg_path:
            neg_fp = open(neg_path, 'r', encoding='utf8', errors='ignore')
            neg_reader = line_reader(neg_fp)
        else:
            neg_reader = None

        with open(pos_path, 'r', encoding='utf8', errors='ignore') as pos_reader:
            for pl in line_reader(pos_reader):
                if pl:
                    pitems = pl.strip().split(SEP_A)
                    if len(pitems) > 1:
                        dataset.append(Instance(pitems[0][:510], pitems[1][:510], 1))

                    if neg_reader:
                        nl = next(neg_reader)
                        if not nl:
                            break
                        nitems = nl.strip().split(SEP_A)
                        if len(nitems) > 1:
                            dataset.append(Instance(nitems[0][:510], nitems[1][:510], 0))

                    if len(dataset) >= margin:
                        yield dataset
                        dataset = []
    except Exception as e:
        print('exception occur: ', e)
        pass
    finally:
        if neg_path and neg_fp:
            neg_fp.close()

    if len(dataset) > 0:
        yield dataset


def in_batch_dataset_loader(pos_path, margin=20000):
    dataset = []
    try:
        with open(pos_path, 'r', encoding='utf8', errors='ignore') as pos_reader:
            for pl in line_reader(pos_reader):
                if pl:
                    pitems = pl.strip().split(SEP_A)
                    if len(pitems) > 2 and pitems[0].strip() != '' and pitems[1].strip() != '' and pitems[-1].strip() != '':
                        dataset.append(Instance(pitems[0][:510], pitems[1][:510], category=pitems[-1]))

                    if len(dataset) >= margin:
                        yield dataset
                        dataset = []
    except Exception as e:
        print('exception occur: ', e)
        pass

    if len(dataset) > 0:
        yield dataset


def create_vocab(data_sets, bert_model_path=None, embed_file=None):
    bert_vocab = BERTVocab(bert_model_path)
    # if embed_file is not None:
    #     embed_count = char_vocab.load_embeddings(embed_file)
    #     print("%d word pre-trained embeddings loaded..." % embed_count)

    return MultiVocab(dict(
        bert=bert_vocab
    ))


def batch_variable(batch_data, mVocab):
    batch_size = len(batch_data)
    bert_vocab = mVocab['bert']
    match_ids = torch.zeros((batch_size, ), dtype=torch.long)
    query_batch = []
    doc_batch = []
    for i, inst in enumerate(batch_data):
        query_batch.append(inst.query)
        doc_batch.append(inst.doc)
        if inst.match:
            match_ids[i] = inst.match

    query_bert_inp = bert_vocab.batch_bert2id(query_batch)
    doc_bert_inp = bert_vocab.batch_bert2id(doc_batch)

    return Batch(query_bert_inp=query_bert_inp,
                 doc_bert_inp=doc_bert_inp,
                 match_ids=match_ids)


class Batch:
    def __init__(self, **args):
        for prop, v in args.items():
            setattr(self, prop, v)

    def to_device(self, device):
        for prop, val in self.__dict__.items():
            if torch.is_tensor(val):
                setattr(self, prop, val.to(device))
            elif isinstance(val, collections.abc.Sequence) or isinstance(val, collections.abc.Iterable):
                val_ = [v.to(device) if torch.is_tensor(v) else v for v in val]
                setattr(self, prop, val_)
        return self


def save_to(path, obj):
    if os.path.exists(path):
        return None
    with open(path, 'wb') as fw:
        pickle.dump(obj, fw)
    print('Obj saved!')


def load_from(pkl_file):
    with open(pkl_file, 'rb') as fr:
        obj = pickle.load(fr)
    return obj

