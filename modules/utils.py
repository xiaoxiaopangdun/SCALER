import os
import networkx as nx
import scipy.sparse as sp
import dgl
import random
import numpy as np
import scipy.io as sio
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features

def load_mat(dataset):
    """Load .mat dataset."""
    data = sio.loadmat("./datasets/{}.mat".format(dataset))
    label = data['Label'] if ('Label' in data) else data['gnd']
    attr = data['Attributes'] if ('Attributes' in data) else data['X']
    network = data['Network'] if ('Network' in data) else data['A']

    adj = sp.csr_matrix(network)
    feat = sp.lil_matrix(attr)

    ano_labels = np.squeeze(np.array(label, dtype=np.int64))
    if 'str_anomaly_label' in data:
        str_ano_labels = np.squeeze(np.array(data['str_anomaly_label']))
        attr_ano_labels = np.squeeze(np.array(data['attr_anomaly_label']))
    else:
        str_ano_labels = None
        attr_ano_labels = None

    return adj, feat, ano_labels, str_ano_labels, attr_ano_labels

def load_dataset(dataset_name, normalize=True, to_bidirected=True):
    adj, features, ano_label, str_ano_label, attr_ano_label = load_mat(dataset_name)
    features = features.todense()
    nx_graph = nx.from_scipy_sparse_array(adj)
    graph = dgl.from_networkx(nx_graph)

    if normalize:
        features = torch.tensor(preprocess_features(features))
    else:
        features = torch.tensor(features)
    if to_bidirected:
        graph = dgl.to_bidirected(graph)
    features = features.to(torch.float32)
    return graph.long(), features, ano_label

def set_random_seeds(seed):
    dgl.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_all_metrics(pred_scores, true_labels, k=None):
    """
    calculate AUC, AP, PRC-AUC, and Recall@K
    """
    # numpy
    pred_scores = np.array(pred_scores)
    true_labels = np.array(true_labels).astype(int)

    # AUC and AP
    auc_score = roc_auc_score(true_labels, pred_scores)
    ap_score = average_precision_score(true_labels, pred_scores)

    # PRC-AUC
    precision, recall, _ = precision_recall_curve(true_labels, pred_scores)
    prc_auc = auc(recall, precision)

    # Recall@K
    if k is None:
        k = np.sum(true_labels)
        if k == 0:
            raise ValueError("Error")

    sorted_indices = np.argsort(pred_scores)[::-1]
    top_k_pred = sorted_indices[:k]
    true_anomalies = np.where(true_labels == 1)[0]
    correct_predictions = len(set(top_k_pred).intersection(set(true_anomalies)))
    recall_at_k = correct_predictions / k

    return {
        'AUC': auc_score,
        'AP': ap_score,
        'PRC-AUC': prc_auc,
        'Recall@K': recall_at_k
    }
