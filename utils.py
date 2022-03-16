import numpy as np
import jsonlines
import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn import metrics
from sklearn.exceptions import UndefinedMetricWarning


class Road3Dataset(Dataset):
    def __init__(self, path_to_file, type=None):
        self.label_list = label_load('./data/label_index.txt')
        self.type = type
        self.dataset = pd.DataFrame.from_dict(jsonl_load(path_to_file))

        self.traineval_data = True if 'labels_index' in self.dataset.columns.values.tolist() else False
        if self.traineval_data:
            self.dataset['labels'] = self.dataset.apply(lambda x: labels_idx2labels_mutil_hot(x['labels_index']),
                                                        axis=1)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset.loc[idx, "features_content"]
        if isinstance(text, list):
            text = ''.join(text)
        if self.traineval_data:
            labels = self.dataset.loc[idx, "labels"]
            labels = torch.Tensor(labels)
            sample = {"text": text, "labels": labels}
        else:
            sample = {"text": text}
        return sample


def labels_idx2labels_mutil_hot(label):
    one_hot_label = np.zeros(148, dtype=int)
    one_hot_label[label] = 1
    return one_hot_label.tolist()


def label_prob2labels_idx(label_prob, thre):
    label_prob = np.array(label_prob)
    label_idx = []
    thre_filer = label_prob > thre
    if any(thre_filer):
        for idx, item in enumerate(label_prob):
            if item > thre:
                label_idx.append(idx)
    return label_idx


def labels_mutil_hot2labels_idx(model_label, label_prob=None):
    model_label = np.array(model_label)
    label_idx = np.nonzero(model_label)[0]
    if label_prob is not None and len(label_idx) == 0:
        return [np.argmax(label_prob)]
    return label_idx.tolist()


def jsonl_load(filename):
    dic_com_list = []
    with jsonlines.open(filename) as f:
        for i in f:
            dic_com_list.append(i)
    return dic_com_list


def prob2zeroone(label_prob, thre, num_label=148):
    thre_filer = label_prob > thre
    if thre_filer.any():
        return thre_filer.astype(int).tolist()
    else:
        label_idx = np.zeros(num_label, dtype=int)
        return label_idx.tolist()


def convert_text_to_ids(tokenizer, text, max_len=510):
    if isinstance(text, str):
        tokenized_text = tokenizer.encode_plus(text, max_length=max_len, add_special_tokens=True, padding=True,
                                               truncation=True)
        input_ids = tokenized_text["input_ids"]
        token_type_ids = tokenized_text["token_type_ids"]

    elif isinstance(text, list):
        input_ids = []
        token_type_ids = []

        for t in text:
            tokenized_text = tokenizer.encode_plus(t, max_length=max_len, add_special_tokens=True, truncation=True)
            input_ids.append(tokenized_text["input_ids"])
            token_type_ids.append(tokenized_text["token_type_ids"])

    else:
        print("Unexpected input")
    return input_ids, token_type_ids


def seq_padding(tokenizer, X, max_len=400):
    pad_id = tokenizer.convert_tokens_to_ids("[PAD]")
    if len(X) <= 1:
        return torch.tensor(X)
    L = [len(x) for x in X]
    ML = max(L)
    X = torch.Tensor([x + [pad_id] * (ML - len(x)) if len(x) < ML else x for x in X])
    return X


def label_load(file):
    with open("./data/label_index.txt", 'r') as f:
        labels = f.readlines()
        label_list = []
        for line in labels:
            label_list.append(line.split()[0])
    return label_list


def f1_macro(labels, preds, thre):
    preds_get = []
    for pred in preds:
        pred = prob2zeroone(pred, thre)
        preds_get.append(pred)
    preds_get = np.array(preds_get)
    return metrics.f1_score(labels, preds_get, average='macro')


def f1_micro(labels, preds, thre):
    preds_get = []
    for pred in preds:
        pred = prob2zeroone(pred, thre)
        preds_get.append(pred)
    preds_get = np.array(preds_get)
    return metrics.f1_score(labels, preds_get, average='micro')


def clr(labels, preds, thre):
    preds_get = []
    for pred in preds:
        pred = prob2zeroone(pred, thre)
        preds_get.append(pred)
    preds_get = np.array(preds_get)
    return metrics.classification_report(labels, preds_get, output_dict=True)
