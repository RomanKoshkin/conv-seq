import torch
import torch.nn.functional as F


def ProximityLoss(centroids):
    distances_btn_centroids = torch.cdist(centroids.unsqueeze(0), centroids.unsqueeze(0), p=2).squeeze()
    return torch.triu(distances_btn_centroids).mean()


def DimensionSimilarityLoss(x, params):
    loss = torch.tensor([0.0], dtype=torch.float, device=params.device)
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            if i > j:
                loss += F.cosine_similarity(x[i, :].view(1, -1, 1), x[j, :].view(1, -1, 1)).squeeze()
    return loss.abs()


def DispersionLoss(embeds, kmeans, params):
    clust_dispersion_loss = torch.tensor([0.0]).to(params.device)
    for i in range(params.K):
        clust_dispersion_loss += torch.cdist(embeds[kmeans.cluster_ids == 0].unsqueeze(0),
                                             kmeans.centroids[0, :].unsqueeze(0),
                                             p=2).mean()
    return clust_dispersion_loss


def L2_loss(model, params):
    l2_loss = torch.tensor([0.0], dtype=torch.float, device=params.device)
    for p in model.parameters():
        l2_loss += p.pow(2).mean()
    return l2_loss


def SelfSimilarityLoss(embeds):
    sm = F.cosine_similarity(embeds.T.unsqueeze(0), embeds.unsqueeze(2)).triu(diagonal=1)
    sm = sm[sm > 0]
    return -sm.mean()