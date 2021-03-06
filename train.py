import os
import time
import torch
import random
import numpy as np
import torch.nn.functional as F
from modules.optimizer import Optimizer, AdamW, WarmupLinearSchedule
from model.QAMatcher import BertQAMatcher
from config.conf import args_config, data_config
from utils.dataset import DataLoader, BucketDataLoader
from utils.datautil import dataset_loader, in_batch_dataset_loader, create_vocab, batch_variable, save_to
import torch.nn.utils as nn_utils
from logger.logger import logger


def bce_loss(pred, targt):
    # pos = torch.eq(targt, 1).float()
    # neg = torch.eq(targt, 0).float()
    # num_pos = torch.sum(pos)
    # num_neg = torch.sum(neg)
    # num_total = num_pos + num_neg
    # alpha_pos = num_neg / num_total
    # alpha_neg = num_pos / num_total
    # weights = alpha_pos * pos + alpha_neg * neg
    targt = targt.reshape(pred.size())
    return F.binary_cross_entropy_with_logits(pred, targt.float())


def cosine_loss(cos_sim, gold):
    '''
    :param cos_sim: cosine similarity  [-1, 1]
    :param gold: 0 / 1 sequence
    :return:
    '''
    # loss = (1 - cos_sim) * gold.float() + cos_sim * (1 - gold).float()
    loss = F.mse_loss(cos_sim, gold.float())
    return loss.sum()


def bin_calc_acc(pred, gold):
    # nb_correct = ((pred.data.sigmoid() > 0.5) == gold.bool()).sum().item()
    nb_correct = ((pred.data > 0.5) == gold.bool()).sum().item()
    return nb_correct, len(gold)


class Trainer(object):
    def __init__(self, args, data_config):
        self.args = args
        self.data_config = data_config
        if self.args.sim_mode == 'multineg':
            self.train_loader = in_batch_dataset_loader(data_config['pos']['train'])
        else:
            self.train_loader = dataset_loader(data_config['pos']['train'], data_config['neg']['train'])

        self.val_loader = dataset_loader(data_config['pos']['dev'], data_config['neg']['dev'])
        self.test_loader = dataset_loader(data_config['pos']['test'], data_config['neg']['test'])

        bert_path = data_config['pretrained']['bert_model']
        self.vocabs = create_vocab(None, bert_path, embed_file=None)
        save_to(args.vocab_chkp, self.vocabs)
        print('save to vocab ', args.vocab_chkp)

        self.model = BertQAMatcher(
            bert_embed_dim=args.bert_embed_dim,
            num_bert_layer=args.bert_layer,
            sim_mode=args.sim_mode,
            dropout=args.dropout,
            bert_model_path=bert_path
        ).to(args.device)

        print(self.model)
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Training %dM trainable parameters..." % (total_params/1e6))

        no_decay = ['bias', 'LayerNorm.weight']
        self.optimizer_bert_parameters = [
            {'params': [p for n, p in self.model.bert_named_params()
                        if not any(nd in n for nd in no_decay) and p.requires_grad],
             'weight_decay': 0.01},
            {'params': [p for n, p in self.model.bert_named_params()
                        if any(nd in n for nd in no_decay) and p.requires_grad],
             'weight_decay': 0.0}
        ]
        self.bert_optimizer = AdamW(self.optimizer_bert_parameters, lr=self.args.bert_lr, eps=1e-8)
        self.bert_scheduler = WarmupLinearSchedule(self.bert_optimizer, warmup_steps=self.args.max_step//20, t_total=self.args.max_step)
        self.base_params = self.model.non_bert_params()
        self.optimizer = Optimizer(self.base_params, self.args)

    def train_steps(self):
        t1 = time.time()
        global_step = 0
        train_loss = 0.
        nb_train_sample = 0
        best_dev_metric, best_test_metric = dict(), dict()
        for ep in range(self.args.epoch):
            print('Epoch:', 1+ep)
            for train_set in self.train_loader:
                logger.info(f'loading {len(train_set)} train data ...')
                # train_batch_loader = DataLoader(train_set, batch_size=self.args.batch_size, shuffle=True)
                # train_batch_loader = BucketDataLoader(train_set, batch_size=self.args.batch_size, key=lambda x: len(x.query), shuffle=True, sort_within_batch=True)
                train_batch_loader = BucketDataLoader(train_set, batch_size=self.args.batch_size, key=lambda x: x.category,
                                                      shuffle=True, sort_within_batch=True)
                for i, batcher in enumerate(train_batch_loader):
                    self.model.train()
                    nb_train_sample += len(batcher)
                    batch = batch_variable(batcher, self.vocabs)
                    batch.to_device(self.args.device)
                    pred_score = self.model(batch.query_bert_inp, batch.doc_bert_inp)
                    if self.args.sim_mode == 'cosine':
                        loss = cosine_loss(pred_score, batch.match_ids)
                    elif self.args.sim_mode == 'multineg':
                        loss = pred_score
                    else:
                        loss = bce_loss(pred_score, batch.match_ids)

                    loss_val = loss.data.item()
                    train_loss += loss_val

                    if self.args.update_step > 1:
                        loss = loss / self.args.update_step

                    loss.backward()

                    if (i + 1) % self.args.update_step == 0 or (i + 1 == len(train_batch_loader)):
                        nn_utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.base_params),
                                                 max_norm=self.args.grad_clip)
                        nn_utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.bert_params()),
                                                 max_norm=self.args.bert_grad_clip)
                        self.optimizer.step()
                        self.bert_optimizer.step()
                        self.bert_scheduler.step()
                        self.model.zero_grad()
                        global_step += 1

                        if global_step % self.args.eval_step == 0:
                            dev_metric = self.evaluate(self.val_loader)
                            if dev_metric['acc'] > best_dev_metric.get('acc', 0):
                                best_dev_metric = dev_metric
                                test_metric = self.evaluate(self.test_loader)
                                if test_metric['acc'] > best_test_metric.get('acc', 0):
                                    best_test_metric = test_metric
                                    # self.save_states(self.args.model_chkp, best_test_metric)

                    logger.info('[Step %d] Iter%d time cost: %.2fs, lr: %.6f, train loss: %.3f, dev_metric: %s, test_metric: %s, %d samples trained.' % (
                        global_step, i, (time.time() - t1), self.optimizer.get_lr(), loss_val, best_dev_metric, best_test_metric, nb_train_sample))
                    print('bert lr:', self.bert_optimizer.param_groups[0]['lr'])

            if self.args.eval_during_training:
                test_metric = self.evaluate(self.test_loader)
                if test_metric['acc'] > best_test_metric.get('acc', 0):
                    best_test_metric = test_metric

            if global_step > self.args.max_step:
                break

        logger.info('Final Dev Metric: %s, Test Metric: %s' % (best_dev_metric, best_test_metric))
        return global_step, train_loss / nb_train_sample

    def evaluate(self, test_data_loader):
        nb_correct, nb_total = 0, 0
        for test_data in test_data_loader:
            test_loader = DataLoader(test_data, batch_size=self.args.test_batch_size)
            self.model.eval()
            with torch.no_grad():
                for i, batcher in enumerate(test_loader):
                    batch = batch_variable(batcher, self.vocabs)
                    batch.to_device(self.args.device)
                    if self.args.sim_mode == 'multineg':
                        pred_score = self.model.get_cos_sim(batch.query_bert_inp, batch.doc_bert_inp)
                    else:
                        pred_score = self.model(batch.query_bert_inp, batch.doc_bert_inp)
                    nb_batch_correct, nb_batch_total = bin_calc_acc(pred_score, batch.match_ids)
                    nb_correct += nb_batch_correct
                    nb_total += nb_batch_total
        acc = nb_correct / nb_total
        return dict(acc=acc)

    def save_states(self, save_path, best_test_metric=None):
        check_point = {'best_metric': best_test_metric,
                       'model_state': self.model.state_dict(),
                       'args_settings': self.args}
        torch.save(check_point, save_path)
        logger.info(f'Saved the current model states to {save_path} ...')


def set_seeds(seed=3349):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    print('cuda available:', torch.cuda.is_available())
    print('cuDNN available:', torch.backends.cudnn.enabled)
    print('gpu numbers:', torch.cuda.device_count())

    args = args_config()
    if torch.cuda.is_available() and args.cuda >= 0:
        args.device = torch.device('cuda', args.cuda)
        torch.cuda.empty_cache()
    else:
        args.device = torch.device('cpu')

    data_path = data_config('./config/data_path.json')

    set_seeds(1357)
    trainer = Trainer(args, data_path)
    final_res = trainer.train_steps()



