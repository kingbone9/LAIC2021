# -*- coding: utf-8 -*-
import json
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import transformers
import sys
import pandas as pd
from transformers import glue_convert_examples_to_features as convert_examples_to_features

sys.path.append('..')
from utils import Road3Dataset, label_prob2labels_idx, clr

from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW
from transformers import BertConfig
from transformers.data.processors.utils import InputExample, DataProcessor
from bert_attention import BertForClassification
from sklearn.exceptions import UndefinedMetricWarning
import warnings

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def split_content(text):
    l_total = []
    l_parcial = []
    if len(text) // 450 > 0:
        n = len(text) // 450
    else:
        n = 1
    for w in range(5):
        if w == n:
            l_parcial = text[w * 450:]
            l_total.append("".join(l_parcial))
        elif w == 0:
            l_parcial = text[:500]
            l_total.append("".join(l_parcial))
        elif w < n:
            l_parcial = text[w * 450:w * 450 + 500]
            l_total.append("".join(l_parcial))
        else:
            l_total.append("")
    return l_total


class transformers_bert_binary_classification(object):
    def __init__(self, writer, save_name, device, model_path='bert-base-chinese', tokenizer_path='bert-base-chinese'):
        self.device_setup(device, model_path, tokenizer_path)
        self.writer = writer
        self.save_name = save_name
        self.model_save_path = ""

    def device_setup(self, device, model_path, tokenizer_path):
        """
        设备配置并加载BERT模型
        :return:
        """
        # TODO 多卡并行
        self.max_len = 512
        self.thre = 0.5
        self.freezeSeed()
        # 使用GPU，通过model.to(device)的方式使用
        self.device = torch.device("cuda:" + str(device) if torch.cuda.is_available() else print('nogpu'))

        MODEL_PATH = model_path
        config_PATH = model_path
        vocab_PATH = tokenizer_path
        # 通过词典导入分词器
        self.tokenizer = transformers.BertTokenizer.from_pretrained(vocab_PATH)
        self.bertconfig = BertConfig.from_pretrained(config_PATH)
        self.bertconfig.num_labels = 148
        self.bertconfig.hidden_dropout_prob = 0.2
        self.bertconfig.attention_probs_dropout_prob = 0.2
        self.bertconfig.problem_type = 'multi_label_classification'
        self.model = BertForClassification.from_pretrained(MODEL_PATH, config=self.bertconfig)
        self.model.to(self.device)
        self.train_loader, self.valid_loader, self.test_loader = self.get_data()

    def model_setup(self):
        weight_decay = 0.01
        learning_rate = 2e-5
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

    def split_content(self, text):
        l_total = []
        l_parcial = []
        if len(text) // 450 > 0:
            n = len(text) // 450
        else:
            n = 1
        for w in range(5):
            if w == n:
                l_parcial = text[w * 450:]
                l_total.append("".join(l_parcial))
            elif w == 0:
                l_parcial = text[:500]
                l_total.append("".join(l_parcial))
            elif w < n:
                l_parcial = text[w * 450:w * 450 + 500]
                l_total.append("".join(l_parcial))
            else:
                l_total.append("")
        return l_total

    # def get_data(self, ):
    #     """
    #     读取数据
    #     :return:
    #     """
    #
    #     train_set_path = "./data/train_dropshort.txt"
    #     valid_set_path = "./data/eval_dropshort.txt"
    #     test_set_path = "./data/test_v1(赛题).txt"
    #     batch_size = 6
    #     # 数据读入
    #     # 加载数据集
    #     train_set = Road3Dataset(train_set_path)
    #     train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    #     valid_set = Road3Dataset(valid_set_path)
    #     valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=2)
    #     test_set = Road3Dataset(test_set_path)
    #     test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    #     self.labellist = train_set.label_list
    #
    #     return train_loader, valid_loader, test_loader

    def get_data(self):
        labels_embedding = np.load("./data/labels_embedding.npy")
        self.labels_embedding = torch.tensor(labels_embedding, dtype=torch.float)

        my_label_list = [x for x in range(148)]
        train_set_path = "./data/train_reducef.txt"
        valid_set_path = "./data/eval_reducef.txt"
        test_set_path = "./data/test_v1(赛题).txt"

        batch_size = 2

        train_set = pd.read_json(train_set_path, lines=True)
        val_set = pd.read_json(valid_set_path, lines=True)
        test_set = pd.read_json(test_set_path, lines=True)
        self.test_id_list = test_set.testid.to_list()
        len_train = len(train_set)
        len_val = len(val_set)
        len_text = len(test_set)

        test_set['features_content'] = test_set['features_content'].apply(lambda x: ''.join(x))

        train_set['text_split'] = train_set['text'].apply(split_content)
        val_set['text_split'] = val_set['text'].apply(split_content)
        test_set['text_split'] = test_set['features_content'].apply(split_content)

        train_l = []
        index_l = []
        for idx, row in train_set.iterrows():
            for l in row['text_split']:
                train_l.append(l)
                index_l.append(idx)

        val_l = []
        val_index_l = []
        for idx, row in val_set.iterrows():
            for l in row['text_split']:
                val_l.append(l)
                val_index_l.append(idx)

        test_l = []
        test_index_l = []
        for idx, row in test_set.iterrows():
            for l in row['text_split']:
                test_l.append(l)
                test_index_l.append(idx)

        train_df = pd.DataFrame({'text': train_l, 'index': index_l})
        val_df = pd.DataFrame({'text': val_l, 'index': val_index_l})
        test_df = pd.DataFrame({'text': test_l, 'index': test_index_l})

        train_Examples = train_df.apply(lambda x: InputExample(guid=None,
                                                               text_a=x['text'],
                                                               text_b=None), axis=1)
        val_Examples = val_df.apply(lambda x: InputExample(guid=None,
                                                           text_a=x['text'],
                                                           text_b=None), axis=1)
        test_Examples = test_df.apply(lambda x: InputExample(guid=None,
                                                             text_a=x['text'],
                                                             text_b=None), axis=1)
        train_features = convert_examples_to_features(train_Examples,
                                                      self.tokenizer,
                                                      label_list=my_label_list,
                                                      output_mode="classification",
                                                      max_length=500)
        val_features = convert_examples_to_features(val_Examples,
                                                    self.tokenizer,
                                                    label_list=my_label_list,
                                                    output_mode="classification",
                                                    max_length=500)
        test_features = convert_examples_to_features(test_Examples,
                                                     self.tokenizer,
                                                     label_list=my_label_list,
                                                     output_mode="classification",
                                                     max_length=500)

        train_input_ids = []
        train_attention_mask = []
        for i in range(len_train):
            train_input_ids.append(
                train_features[5 * i].input_ids + train_features[5 * i + 1].input_ids + train_features[
                    5 * i + 2].input_ids + train_features[5 * i + 3].input_ids + train_features[5 * i + 4].input_ids)
            train_attention_mask.append(
                train_features[5 * i].attention_mask + train_features[5 * i + 1].attention_mask + train_features[
                    5 * i + 2].attention_mask + train_features[5 * i + 3].attention_mask + train_features[
                    5 * i + 4].attention_mask)

        val_input_ids = []
        val_attention_mask = []
        for i in range(len_val):
            val_input_ids.append(
                val_features[5 * i].input_ids + val_features[5 * i + 1].input_ids + val_features[5 * i + 2].input_ids +
                val_features[5 * i + 3].input_ids + val_features[5 * i + 4].input_ids)
            val_attention_mask.append(
                val_features[5 * i].attention_mask + val_features[5 * i + 1].attention_mask + val_features[
                    5 * i + 2].attention_mask + val_features[5 * i + 3].attention_mask + val_features[
                    5 * i + 4].attention_mask)

        test_input_ids = []
        test_attention_mask = []
        for i in range(len_text):
            test_input_ids.append(test_features[5 * i].input_ids + test_features[5 * i + 1].input_ids + test_features[
                5 * i + 2].input_ids + test_features[5 * i + 3].input_ids + test_features[5 * i + 4].input_ids)
            test_attention_mask.append(
                test_features[5 * i].attention_mask + test_features[5 * i + 1].attention_mask + test_features[
                    5 * i + 2].attention_mask + test_features[5 * i + 3].attention_mask + test_features[
                    5 * i + 4].attention_mask)

        train_input_ids = torch.tensor(train_input_ids, dtype=torch.long)
        train_attention_mask = torch.tensor(train_attention_mask, dtype=torch.long)
        train_labels = torch.tensor([i for i in train_set.labels_onehot], dtype=torch.float)
        train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_labels)

        val_input_ids = torch.tensor(val_input_ids, dtype=torch.long)
        val_attention_mask = torch.tensor(val_attention_mask, dtype=torch.long)
        val_labels = torch.tensor([i for i in val_set.labels_onehot], dtype=torch.float)
        val_dataset = TensorDataset(val_input_ids, val_attention_mask, val_labels)

        test_input_ids = torch.tensor(test_input_ids, dtype=torch.long)
        test_attention_mask = torch.tensor(test_attention_mask, dtype=torch.long)
        test_dataset = TensorDataset(test_input_ids, test_attention_mask)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size * 4, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=batch_size * 4, shuffle=False, num_workers=2)

        return train_loader, val_loader, test_loader

    def forward_common(self, batch):
        input_ids = batch[0]
        attention_mask = batch[1]
        labels = batch[2]

        # 迁移到GPU
        input_ids, attention_mask, labels = input_ids.to(self.device), attention_mask.to(self.device), \
                                            labels.to(self.device)
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                            labels_embedding=self.labels_embedding.to(self.device))
        loss = output[0]
        logits = torch.sigmoid(output[1])
        logits = logits.cpu().detach().numpy()
        return loss, logits

    def train_an_epoch(self, iterator, epoch):
        epoch_loss = 0
        epoch_logits = []
        epoch_labels = []
        self.model.train()
        with tqdm(total=len(iterator), ncols=120) as bar:
            for i, batch in enumerate(iterator):
                self.optimizer.zero_grad()
                loss, logits = self.forward_common(batch)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                # tqdm.write("batch_loss: %s"%loss.item())
                [epoch_logits.append(logit.tolist()) for logit in logits]
                [epoch_labels.append(label.tolist()) for label in batch[2]]
                bar.set_postfix(loss='{:^.4f}'.format(loss.item()))
                bar.update()
        clr_epoach = clr(np.array(epoch_labels), np.array(epoch_logits), self.thre)
        return epoch_loss / len(iterator), clr_epoach

    def evaluate(self, iterator):
        self.model.eval()
        epoch_loss = 0
        epoch_logits = []
        epoch_labels = []
        with torch.no_grad():
            for _, batch in enumerate(iterator):
                loss, logits = self.forward_common(batch)
                epoch_loss += loss.item()
                [epoch_logits.append(logit.tolist()) for logit in logits]
                [epoch_labels.append(label.tolist()) for label in batch[2]]
        clr_epoach_5 = clr(np.array(epoch_labels), np.array(epoch_logits), self.thre)
        clr_epoach_6 = clr(np.array(epoch_labels), np.array(epoch_logits), 0.6)
        clr_epoach_4 = clr(np.array(epoch_labels), np.array(epoch_logits), 0.4)
        clr_epoach_3 = clr(np.array(epoch_labels), np.array(epoch_logits), 0.3)
        clr2 = clr(np.array(epoch_labels), np.array(epoch_logits), 0.2)

        return epoch_loss / len(iterator), clr_epoach_5, clr_epoach_6, clr_epoach_4, clr_epoach_3, clr2

    def train(self, epochs):
        # TODO 十折交叉取平均
        self.model_setup()
        best_score = 0
        best_score6 = 0
        best_score4 = 0
        best_score3 = 0
        best_score2 = 0
        for i in range(epochs):
            print('epochs', i + 1)
            train_loss, train_clr = self.train_an_epoch(self.train_loader, i)
            train_f1_macro, train_f1_micro = train_clr['macro avg']['f1-score'], train_clr['micro avg']['f1-score']
            train_score = (train_f1_macro + train_f1_micro) / 2
            print("train loss: ", train_loss, "\t", "train f1 macro:", train_f1_macro, "\t", "train f1 micro:",
                  train_f1_micro, "\t", "train score:", train_score)
            valid_loss, valid_clr, vc6, vc4, vc3, vc2 = self.evaluate(self.valid_loader)
            valid_f1_macro, valid_f1_micro = valid_clr['macro avg']['f1-score'], valid_clr['micro avg']['f1-score']
            vma6, vmi6 = vc6['macro avg']['f1-score'], vc6['micro avg']['f1-score']
            vma4, vmi4 = vc4['macro avg']['f1-score'], vc4['micro avg']['f1-score']
            vma3, vmi3 = vc3['macro avg']['f1-score'], vc3['micro avg']['f1-score']
            vma2, vmi2 = vc2['macro avg']['f1-score'], vc2['micro avg']['f1-score']
            valid_score = (valid_f1_macro + valid_f1_micro) / 2
            vs6 = (vma6 + vmi6) / 2
            vs4 = (vma4 + vmi4) / 2
            vs3 = (vma3 + vmi3) / 2
            vs2 = (vma2 + vmi2) / 2
            print("valid loss: ", valid_loss, "\t", "valid f1 macro:", valid_f1_macro, "\t", "valid f1 micro:",
                  valid_f1_micro, "\t", "valid score:", valid_score)
            print("valid loss: ", valid_loss, "\t", "valid_0.6 macro:", vma6, "\t", "valid_0.6 micro:",
                  vmi6, "\t", "valid score:", vs6)
            print("valid loss: ", valid_loss, "\t", "valid_0.4 macro:", vma4, "\t", "valid_0.4 micro:",
                  vmi4, "\t", "valid score:", vs4)
            print("valid loss: ", valid_loss, "\t", "valid_0.3 macro:", vma3, "\t", "valid_0.3 micro:",
                  vmi3, "\t", "valid score:", vs3)
            print("valid loss: ", valid_loss, "\t", "valid_0.3 macro:", vma3, "\t", "valid_0.3 micro:",
                  vmi3, "\t", "valid score:", vs2)
            self.writer.add_scalars('Loss', {'train': train_loss, 'valid': valid_loss}, i)
            self.writer.add_scalars('macro f1', {'train': train_f1_macro, 'valid': valid_f1_macro, 'valid_0.4': vma4,
                                                 'valid_0.3': vma3, 'valid_0.6': vma6}, i)
            self.writer.add_scalars('micro f1', {'train': train_f1_micro, 'valid': valid_f1_micro, 'valid_0.4': vmi4,
                                                 'valid_0.3': vmi4, 'valid_0.6': vmi6}, i)
            self.writer.add_scalars('score',
                                    {'train': train_score, 'valid': valid_score, 'valid_0.4': vs4, 'valid_0.3': vs3,
                                     'valid_0.6': vs6}, i)
            self.writer.add_scalars('weighted precision', {'train': train_clr['weighted avg']['precision'],
                                                           'valid': valid_clr['weighted avg']['precision'],
                                                           'valid_0.4': vc4['weighted avg']['precision'],
                                                           'valid_0.3': vc3['weighted avg']['precision'],
                                                           'valid_0.6': vc6['weighted avg']['precision'], }, i)
            self.writer.add_scalars('weighted recall', {'train': train_clr['weighted avg']['recall'],
                                                        'valid': valid_clr['weighted avg']['recall'],
                                                        'valid_0.4': vc4['weighted avg']['recall'],
                                                        'valid_0.3': vc3['weighted avg']['recall'],
                                                        'valid_0.6': vc6['weighted avg']['recall']}, i)
            if valid_score > best_score:
                self.save_model()
                best_score = valid_score
                bert_clr = valid_clr
            if vs3 > best_score3:
                self.save_model(thre='0.3')
                best_score3 = vs3
                bert_clr = vc3
            if vs4 > best_score4:
                self.save_model(thre='0.4')
                best_score4 = vs4
                bert_clr = vc4
            if vs6 > best_score6:
                self.save_model(thre='0.6')
                best_score6 = vs6
                bert_clr = vc6
            if vs2 > best_score2:
                self.save_model(thre='0.2')
                best_score2 = vs2
                bert_clr = vc2
        print('best valid score:')
        print(bert_clr)

    def save_model(self, thre=''):
        self.model_save_path = "./result/" + save_name + thre
        self.model.save_pretrained(self.model_save_path)
        print("model saved...")

    def freezeSeed(self):
        seed = 1
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)  # Numpy module.
        random.seed(seed)  # Python random module.
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def test(self, thre=0.5):
        if thre == 0.5:
            self.model_save_path = "./result/" + save_name
        else:
            self.model_save_path = "./result/" + save_name + str(thre)

        self.model = BertForClassification.from_pretrained(self.model_save_path, config=self.bertconfig)
        self.model.to(self.device)
        self.model.eval()
        test_logits = []
        with torch.no_grad():
            for _, batch in enumerate(self.test_loader):
                input_ids = batch[0]
                attention_mask = batch[1]

                input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
                output = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                    labels_embedding=self.labels_embedding.to(self.device))
                logits = torch.sigmoid(output[0])
                logits = logits.cpu().detach().numpy()
                [test_logits.append(logit.tolist()) for logit in logits]
        pred = [label_prob2labels_idx(label_prob, thre) for label_prob in test_logits]
        df_submit = pd.DataFrame({'testid': self.test_id_list, 'labels_index': pred})
        df_submit.to_json('./result/' + save_name + '/test_submit.txt', orient='records', lines=True, force_ascii=False)
        self.get_test_distribution(df_submit)

    def get_test_distribution(self, df_submit):
        df_submit['labels_num'] = df_submit.apply(lambda x: len(x['labels_index']), axis=1)
        print('模型得出的标签数量分布为：')
        print(df_submit['labels_num'].value_counts())
        label_dis_model = [0 for i in range(148)]
        for tup in zip(df_submit['labels_index']):
            for label in tup[0]:
                label_dis_model[label] += 1
        for i, item in enumerate(label_dis_model):
            self.writer.add_scalar('labels_distribution', item, i)


if __name__ == '__main__':
    bert_model = '../bertmodel/ms/'
    tokenizer_path = '../bertmodel/ms/'
    save_name = 'bert_with_attention_use_cleaned_vf'
    writer = SummaryWriter('../log/' + save_name + '/')
    classifier = transformers_bert_binary_classification(writer, save_name, device=1, model_path=bert_model,
                                                         tokenizer_path=tokenizer_path)
    classifier.train(70)
    classifier.test()
    writer.close()
