import warnings
warnings.filterwarnings('ignore')
import argparse
import torch
import os
import dgl
from modules import utils
from modules.utils import calculate_all_metrics
from modules import model
from modules.loss import compute_sim_batch, compute_edge_contrast
import datetime
import numpy as np
from modules.logger import create_logger

# Settings the warnings to be ignored
torch.autograd.set_detect_anomaly(True)
os.environ['CUDA_VISIBLE_DEVICE'] = '0'
os.environ["KMP_DUPLICATE_LnIB_OK"] = "TRUE"
device_ids = 0

# %%
parser = argparse.ArgumentParser(description='SCALER')
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--lr', type=float)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--runs', type=int, default=5)
parser.add_argument('--batch_size_sampling', type=int, default=8192)
parser.add_argument('--hidden', type=int, default=128)
parser.add_argument('--epoch', type=int)
parser.add_argument('--alpha', type=float)
parser.add_argument('--beta', type=float)
parser.add_argument('--gamma', type=float)
parser.add_argument('--neg_num', type=int, default=60)
parser.add_argument('--tau', type=float, default=1)
parser.add_argument('--original_score', action='store_false')

args = parser.parse_args()

current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
log_path = f"./results/{args.dataset}_{current_time}.log"
if not os.path.exists("./results"):
    os.mkdir("./results")
if not os.path.exists("./models"):
    os.mkdir("./models")
logger = create_logger(log_path)
logger.info(args)
args.gamma = 10000 * args.gamma

graph, features, ano_label = utils.load_dataset(args.dataset, normalize=True, to_bidirected=True)
graph = graph.to(device_ids)
features = features.to(device_ids)
graph = graph.remove_self_loop()
graph = graph.add_self_loop()
graph.ndata["features"] = features

aucs = []
aps = []
pres = []
recalls = []
for seed in range(args.runs):
    logger.info(f"-----------------Seed:{seed}------------------")
    # Set the random seed
    utils.set_random_seeds(seed)

    # Batch Training
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    dataloader = dgl.dataloading.DataLoader(
        graph, torch.tensor(list(range(graph.num_nodes()))).to(device_ids), sampler,
        batch_size=args.batch_size_sampling,
        shuffle=True,
        drop_last=False,
        num_workers=0)

    # Init MLP and GNN parameters
    model_GAD = model.SCALER(features.shape[1], args.hidden, features.shape[0]).cuda()
    optimiser = torch.optim.Adam(model_GAD.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model_GAD.train()

    epochs = args.epoch
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logger.info(f"-----------------Train------------------")
        train_total_sec = 0
        b_xent = torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([1]).to(device_ids))
        best = 1e9
        best_t = 0
        for epoch in range(epochs):
            time_train_start = datetime.datetime.now()
            total_loss = 0
            model_GAD.train()
            # Load batch and neighbor subgraph
            for input_nodes, output_nodes, blocks in dataloader:
                optimiser.zero_grad()
                block = blocks[0]
                batch_features = block.srcdata['features']
                emb_mlp, emb_gnn, reg_loss = model_GAD.forward(blocks[0], batch_features, output_nodes.shape[0], True)

                lbl = torch.unsqueeze(torch.cat((torch.ones(output_nodes.shape), torch.zeros(output_nodes.shape))),
                                      1).to(device_ids)

                # node-level contrast
                if args.alpha != 0:
                    score_pred_mlp, score_neg_mlp = compute_sim_batch(emb_mlp, blocks[1], compute_neg=True, corrected_score=args.original_score)
                    score_mlp = torch.cat(tuple([score_pred_mlp, score_neg_mlp])).unsqueeze(1)
                    loss_node_mlp = torch.mean(b_xent(score_mlp, lbl))
                else:
                    loss_node_mlp = 0.0
                score_pred_gnn, score_neg_gnn = compute_sim_batch(emb_gnn, blocks[1], compute_neg=True, corrected_score=args.original_score)
                score_gnn = torch.cat(tuple([score_pred_gnn, score_neg_gnn])).unsqueeze(1)
                loss_node_gnn = torch.mean(b_xent(score_gnn, lbl))
                loss_node = loss_node_gnn + args.alpha * loss_node_mlp

                # edge-level contrast
                if args.beta != 0:
                    loss_edge = compute_edge_contrast(emb_gnn, blocks[1], args.tau, args.neg_num)
                else:
                    loss_edge = 0.0

                # total loss
                loss = loss_node + args.beta * loss_edge + args.gamma * reg_loss

                loss.backward()
                optimiser.step()
                total_loss += loss.detach().cpu().numpy()

            time_train_end = datetime.datetime.now()
            timedelta_train = time_train_end - time_train_start
            train_total_sec += timedelta_train.total_seconds()

            if total_loss < best:
                best = total_loss
                best_t = epoch
                file_name = f'./models/best_model_{args.dataset}.pkl'
                torch.save(model_GAD.state_dict(), file_name)

            logger.info(f"Epoch {epoch}")

        logger.info(f"-----------------Infer------------------")
        logger.info(f"Loading {best_t}-th epoch")
        file_name = f'./models/best_model_{args.dataset}.pkl'
        model_GAD.load_state_dict(torch.load(file_name))
        time_eval_start = datetime.datetime.now()
        with torch.no_grad():
            model_GAD.eval()
            scores = torch.zeros(graph.num_nodes(), dtype=torch.float32).to(graph.device)
            for input_nodes, output_nodes, blocks in dataloader:
                block = blocks[0]
                batch_features = block.srcdata['features']
                emb_mlp, _, _ = model_GAD.forward(blocks[0], batch_features, output_nodes.shape[0], False)
                scores_batch = compute_sim_batch(emb_mlp, blocks[1], compute_neg=False, corrected_score=args.original_score)
                scores[output_nodes] = scores_batch.detach()
            scores = - scores.detach().cpu().numpy()
            dic = calculate_all_metrics(scores, ano_label)
        time_eval_end = datetime.datetime.now()
        timedelta_eval = time_eval_end - time_eval_start

    logger.info(f"-----------------Result------------------")
    logger.info(f"AUC: {dic['AUC']}")
    logger.info(f"AP: {dic['AP']}")
    logger.info(f"PRC-AUC: {dic['PRC-AUC']}")
    logger.info(f"Recall@K: {dic['Recall@K']}")
    logger.info(f"Train time (sec): {train_total_sec}")
    logger.info(f"Test time (sec): {timedelta_eval.total_seconds()}")
    logger.info(f"\n")
    aucs.append(dic['AUC'])
    aps.append(dic['AP'])
    pres.append(dic['PRC-AUC'])
    recalls.append(dic['Recall@K'])

logger.info(f"-----------------Final------------------")
logger.info(f"AUC: {aucs}")
logger.info(f"AP: {aps}")
logger.info(f"PRC-AUC: {pres}")
logger.info(f"Recall@K: {recalls}")
logger.info(f"AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
logger.info(f"AP: {np.mean(aps):.4f} ± {np.std(aps):.4f}")
logger.info(f"PRC-AUC: {np.mean(pres):.4f} ± {np.std(pres):.4f}")
logger.info(f"Recall@K: {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
