import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, precision_score, recall_score
from imblearn.metrics import geometric_mean_score
from typing import Tuple, List, Literal, Dict
import scipy.io
from torch_geometric.data import Data
import torch
import networkx as nx
import pickle
import random
from datetime import datetime
import yaml

def now():
    return datetime.now().strftime('%Y%m%d%H%M')


def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu': 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_yaml(fn):
    with open(fn) as fp:
        config = yaml.safe_load(fp)
    return config

def load_data(
        dataname:Literal['YelpHotel','YelpRes','Amazon','YelpNYC'],
        if_split:bool=False,
        split:Tuple[float]=(0.6,0.2,0.2)
    ) -> Tuple[Data,np.ndarray]:

    dataset = scipy.io.loadmat(f'dataset/{dataname}/{dataname}.mat')
    dataset['Network'].toarray()
    D = nx.DiGraph(dataset['Network'].toarray())
    edges = torch.tensor([[u for (u, v) in D.edges()],[v for (u,v) in D.edges()]], dtype=torch.int64)
    attr = torch.tensor(dataset['Attributes'].toarray(), dtype=torch.float)
    label = torch.tensor(dataset['Label'], dtype=torch.float)
    data = Data(x=attr, edge_index=edges)

    if if_split:
        n = label.shape[0]
        train_mask, val_mask, test_mask = np.zeros(n).astype(bool), np.zeros(n).astype(bool), np.zeros(n).astype(bool)
        indexes = np.arange(n)
        np.random.shuffle(indexes)
        n_train, n_val = int(n*split[0]),int(n*split[1])
        train_mask[indexes[:n_train]] = True
        val_mask[indexes[n_train:(n_train+n_val)]] = True
        test_mask[indexes[(n_train+n_val):]] = True
        train_mask, val_mask, test_mask = torch.tensor(train_mask), torch.tensor(val_mask), torch.tensor(test_mask)
        data.train_mask, data.val_mask, data.test_mask = train_mask, val_mask, test_mask
    return data,label


def eval_all(y_true, score, threshold=0.5, Klist=[50,100,200,300]) -> Dict:
    acc = eval_acc(y_true, score, threshold)
    auc = eval_rocauc(y_true, score)
    ap = eval_average_precision(y_true,score)
    f1 = eval_f1(y_true, score, threshold)
    precision = eval_precision(y_true, score, threshold)
    recall = eval_recall(y_true, score, threshold)
    g_mean = eval_geometric_mean(y_true, score, threshold)

    f1_at_k = []
    precision_at_k = []
    recall_at_k = []
    for K in Klist:
        f1_at_k.append(eval_f1_at_k(y_true, score, K))
        precision_at_k.append(eval_precision_at_k(y_true, score, K))
        recall_at_k.append(eval_recall_at_k(y_true, score, K))

    res = {
        'acc':acc,
        'auc':auc,
        'ap':ap,
        'f1':f1,
        'precision':precision,
        'recall':recall,
        'g_mean':g_mean
    }
    for i in range(len(Klist)):
        res[f'f1_at_{Klist[i]}'] = f1_at_k[i]
        res[f'precision_at_{Klist[i]}'] = precision_at_k[i]
        res[f'recall_at_{Klist[i]}'] = recall_at_k[i]

    return res



@torch.no_grad()
def eval_acc(y_true, score, threshold=0.5) -> float:
    y_true = y_true.detach().cpu().numpy().flatten()
    y_pred = score.detach().cpu().numpy().flatten()
    y_pred = y_pred>threshold
    return (y_true == y_pred).mean()


@torch.no_grad()
def eval_rocauc(y_true, score) -> float:
    y_true = y_true.detach().cpu().numpy().flatten()
    score = score.detach().cpu().numpy().flatten()
    auc = roc_auc_score(y_true, score)
    return auc


@torch.no_grad()
def eval_f1(y_true, score, threshold=0.5) -> float:
    y_true = y_true.detach().cpu().numpy().flatten()
    y_pred = score.detach().cpu().numpy().flatten()
    y_pred = y_pred>threshold
    f1 = f1_score(y_true, y_pred)
    return f1


@torch.no_grad()
def eval_precision(y_true, score, threshold=0.5) -> float:
    y_true = y_true.detach().cpu().numpy().flatten()
    y_pred = score.detach().cpu().numpy().flatten()
    y_pred = y_pred>threshold
    if not y_pred.sum():
        return np.nan
    precision = precision_score(y_true, y_pred)
    return precision


@torch.no_grad()
def eval_recall(y_true, score, threshold=0.5) -> float:
    y_true = y_true.detach().cpu().numpy().flatten()
    y_pred = score.detach().cpu().numpy().flatten()
    y_pred = y_pred>threshold
    recall = recall_score(y_true, y_pred)
    return recall


@torch.no_grad()
def eval_geometric_mean(y_true, score, threshold=0.5) -> float:
    y_true = y_true.detach().cpu().numpy().flatten()
    y_pred = score.detach().cpu().numpy().flatten()
    y_pred = y_pred>threshold
    g = geometric_mean_score(y_true, y_pred)
    return g


@torch.no_grad()
def eval_f1_at_k(y_true, score, K=50):
    """ 
    According to "Deep Anomaly Detection on Attributed Networks", K = 50, 100, 200, 300
    """
    y_true = y_true.detach().cpu().numpy().flatten()
    score = score.detach().cpu().numpy().flatten()

    n = y_true.shape[0]
    idx_topk = np.argsort(score.flatten())[n:-K:-1]
    precision_at_k = y_true[idx_topk].mean()
    recall_at_k = y_true[idx_topk].sum()/y_true.sum()
    f1_at_k = 2*precision_at_k*recall_at_k/(precision_at_k+recall_at_k)
    return f1_at_k


@torch.no_grad()
def eval_precision_at_k(y_true, score, K=50) -> float:
    """ 
    According to "Deep Anomaly Detection on Attributed Networks", K = 50, 100, 200, 300
    """
    y_true = y_true.detach().cpu().numpy().flatten()
    score = score.detach().cpu().numpy().flatten()

    n = y_true.shape[0]
    idx_topk = np.argsort(score.flatten())[n:-K:-1]
    precision_at_k = y_true[idx_topk].mean()
    return precision_at_k


@torch.no_grad()
def eval_recall_at_k(y_true, score, K=50) -> float:
    """ 
    According to "Deep Anomaly Detection on Attributed Networks", K = 50, 100, 200, 300
    """
    y_true = y_true.detach().cpu().numpy().flatten()
    score = score.detach().cpu().numpy().flatten()

    n = y_true.shape[0]
    idx_topk = np.argsort(score.flatten())[n:-K:-1]
    recall_at_k = y_true[idx_topk].sum()/y_true.sum()
    return recall_at_k


@torch.no_grad()
def eval_average_precision(y_true,score) -> float:
    y_true = y_true.detach().cpu().numpy().flatten()
    score = score.detach().cpu().numpy().flatten()
    ap = average_precision_score(y_true, score)
    return ap


def save_model(model,filename):
    with open(filename,'wb') as f:
        pickle.dump(model,f)

def read_model(filename):
    with open(filename,'rb') as f:
        return(pickle.load(f))