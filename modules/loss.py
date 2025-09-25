import torch
from torch_scatter import scatter_add
import torch.nn.functional as F
from torch_scatter import scatter_softmax, scatter

def compute_edge_contrast(feature, block, tau=1, num_neg=10):
    device = feature.device
    src_all, dst_all = block.edges()
    deg = block.in_degrees().to(device)
    E = src_all.numel()

    pos_mask = torch.logical_and(
        src_all >= dst_all,
        torch.logical_and(dst_all < block.num_dst_nodes(), src_all < block.num_dst_nodes())
    )
    pos_src = src_all[pos_mask]
    pos_dst = dst_all[pos_mask]
    if pos_src.numel() == 0:
        return torch.tensor(0.0)

    # edge_emb
    edge_emb_all = (feature[src_all] + feature[dst_all]) * 0.5
    pos_emb = edge_emb_all[pos_mask]

    # —— 1) Pos pool
    node_sum = scatter_add(edge_emb_all, dst_all, dim=0,
                           dim_size=feature.size(0))
    h_ki = (node_sum[pos_src] - pos_emb) / (deg[pos_src] - 1 + 1e-6).unsqueeze(1)
    h_kj = (node_sum[pos_dst] - pos_emb) / (deg[pos_dst] - 1 + 1e-6).unsqueeze(1)

    # —— 2) Pos
    pos_sims = torch.stack([
        F.cosine_similarity(pos_emb, pos_emb, dim=-1),
        F.cosine_similarity(pos_emb, h_ki,  dim=-1),
        F.cosine_similarity(pos_emb, h_kj,  dim=-1)
    ], dim=1)
    pos_exp = torch.exp(pos_sims / tau).sum(dim=1)

    # —— 3) Sample
    if num_neg < pos_src.numel():
        neg_idx = torch.randint(E, (num_neg,), device=device)
        neg_emb = edge_emb_all[neg_idx]
    else:
        neg_emb = edge_emb_all

    neg_sim = F.cosine_similarity(
        pos_emb.unsqueeze(1), neg_emb.unsqueeze(0), dim=-1)
    neg_exp = torch.exp(neg_sim / tau).sum(dim=1)

    # —— 4) InfoNCE loss ——
    loss = -torch.log(pos_exp / (pos_exp + neg_exp))
    norm = deg[pos_src] + deg[pos_dst] - 1
    loss = loss / norm
    return loss.mean()

def compute_sim_batch(feature, block, compute_neg=False, corrected_score=True):
    src, dst = block.edges()
    similarity = torch.zeros((block.num_dst_nodes())).to(block.device)
    h = torch.nn.functional.cosine_similarity(feature[dst], feature[src])

    similarity.index_add_(0, dst, h)
    similarity_pos = similarity / block.in_degrees()
    if not compute_neg:
        if corrected_score:
            mask = src != dst
            h = h[mask]
            dst = dst[mask]
            p_v = scatter_softmax(h, dst, dim=0)
            temp = - scatter(p_v * torch.log(p_v + 1e-8), dst, dim=0, reduce='sum')
            max_entropy = torch.log((block.in_degrees()-1).float() + 1e-8)
            local_entropy = torch.zeros_like(max_entropy)
            local_entropy[:temp.size(0)] = temp
            entropy_norm = local_entropy / (max_entropy + 1e-8)
            score = (similarity_pos - 1) * (1 + (1 - entropy_norm) / 5)
            return score
        else:
            return similarity_pos - 1
    else:
        feature_norm = torch.nn.functional.normalize(feature)
        similarity_neg = torch.mm(feature_norm[block.dstnodes()], feature_norm.T).sum(axis=1)
        similarity_neg = similarity_neg - similarity
        similarity_neg = similarity_neg / (feature.shape[0] - block.in_degrees())
        return similarity_pos, similarity_neg
