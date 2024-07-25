import pyro
import torch
import torch.nn.functional as F
from cellavi_data import CellaviData

NSUPP_CELLS = 65_536
N_BATCHES = 256
SUBSMPL = 512


def n_choose_k__01(n: int, k: int):
    rnd__01 = torch.zeros(n, dtype=torch.bool)
    rnd__01[:k] = True
    return rnd__01[torch.randperm(n)]


def gather_ctype_profiles(data: CellaviData):
    # Add 1/2 pseudo-count.
    ctype_profiles = 0.5 * torch.ones(data.C, data.x.shape[-1])
    for chunk_i in data.iterate_by_chunk(SUBSMPL):
        x_i = chunk_i.x
        x_i[~chunk_i.cmask] = 0.0  # Remove cells without known type.
        ctype_profiles += torch.einsum("iG,iC->CG", x_i, chunk_i.one_hot_ctype)
    return F.normalize(ctype_profiles, p=1, dim=-1)


def kmeans(x, K):
    # Here-function to compute WCSS.
    def compute_wcss(X, centroids, assgt):
        wcss = 0.0
        for k in range(K):
            cluster_points = X[assgt == k]
            if cluster_points.size(0) > 0:
                wcss += ((cluster_points - centroids[k]) ** 2).sum()
        return wcss

    best_wcss = float("inf")
    for _ in range(100):
        ctrd = x[torch.randperm(x.shape[0])[:K]]
        for _ in range(35):
            distances = torch.cdist(x, ctrd)
            assgt = torch.argmin(distances, dim=1)
            new_ctrd = torch.stack([x[assgt == k].mean(dim=0) for k in range(K)])
            if torch.all(ctrd == new_ctrd):
                break
            ctrd = new_ctrd
        # Compute WCSS for the current clustering
        wcss = compute_wcss(x, ctrd, assgt)
        if wcss < best_wcss:
            best_wcss = wcss
            best_ctrd = ctrd
            best_assgt = assgt
    return best_ctrd, best_assgt


def initialize_parameters(data: CellaviData, amortizer=None):
    # NOTE: This function does not take care of putting tensors
    # on the correct device. This is typically done in the main
    # script by wrapping it in `with torch.device(...):`.

    # Collect parameters directly to Pyro's parameter store.
    param_store = pyro.get_param_store()

    # 1) Collect up to `NSUPP_CELLS` cells plus all labelled cells.
    smpl__01 = n_choose_k__01(data.ncells, NSUPP_CELLS) | data.smask
    smpl = data[smpl__01]
    smpl_x = smpl.x
    smpl_f = smpl_x / smpl_x.sum(dim=-1, keepdim=True)

    logsum = torch.log(smpl_f.sum(dim=0) + 0.5)  # Add 1/2 pseudo-count.
    global_base = logsum - logsum.mean()
    param_store["autonormal.locs.global_base"] = global_base

    if data.C > 1:
        ctype_profiles = gather_ctype_profiles(data)
        ctype_fx = torch.log(ctype_profiles) - global_base
        ctype_fx -= ctype_fx.mean(dim=-1, keepdim=True)
        param_store["autonormal.locs.ctype_fx"] = ctype_fx
    else:
        ctype_profiles = torch.zeros(1, data.G)

    # 2) Subtract the cell baseline.
    smpl_f -= ctype_profiles[data.ctype[smpl__01]]

    # 3) Keep only top genes.
    topG = smpl_f.std(dim=0).sort(descending=True).indices[:2048]
    smpl_f = smpl_f[:, topG]

    # 4) Perform PCA on the subset.
    smpl_f_mean = smpl_f.mean(dim=0, keepdim=True)
    smpl_f_std = smpl_f.std(dim=0, keepdim=True)
    smpl_f_norm = (smpl_f - smpl_f_mean) / smpl_f_std
    cov = smpl_f_norm.T @ smpl_f_norm / smpl_f_norm.shape[0]
    _, eigenvectors = torch.linalg.eigh(cov)
    smpl_proj = smpl_f_norm @ eigenvectors[:, -data.K :]

    # 5) Remove batch effects.
    smpl_one_hot_batch = smpl.one_hot_batch
    batch_avg = torch.linalg.lstsq(smpl_one_hot_batch, smpl_proj).solution
    smpl_proj -= smpl_one_hot_batch @ batch_avg

    # 6) Use k-means clustering to find the initial centroids.
    ctrd, assgt = kmeans(smpl_proj, data.K)

    # 7) Compute labelled centroids.
    available_topic_labels = torch.unique(data.topic[data.smask])
    labl_ctrd = torch.zeros(data.K, data.K)
    smpl_topic = smpl.topic
    smpl_smask = smpl.smask
    for t in available_topic_labels:
        idx_t = (smpl_topic == t) & smpl_smask
        smpl_proj_t = smpl_proj[idx_t, :]
        labl_ctrd[t] = smpl_proj_t.mean(dim=0, keepdim=True)

    # 8) match labelled centroids with k-means centroids.
    if len(available_topic_labels) > 0:
        cost_matrix = torch.cdist(labl_ctrd[available_topic_labels, :], ctrd)
        from scipy.optimize import linear_sum_assignment

        _, matched = linear_sum_assignment(cost_matrix.cpu())
        unmatched = list(set(range(data.K)).difference(matched))
    else:
        matched = []
        unmatched = range(data.K)

    # 9) Fit a Cauchy distribution to each cluster.
    Cauchy = dict()
    i = j = 0
    for t in range(data.K):
        if t in available_topic_labels:
            ctrd = matched[i]
            i += 1
        else:
            ctrd = unmatched[j]
            j += 1
        idx_t = assgt == ctrd
        smpl_proj_t = smpl_proj[idx_t, :]
        mean_t = smpl_proj_t.mean(dim=0, keepdim=True)
        smpl_proj_t_c = smpl_proj_t - mean_t
        cov_t = smpl_proj_t.T @ smpl_proj_t_c / smpl_proj_t_c.shape[0]
        # Tikhonov regularization.
        cov_t += 1e-6 * torch.eye(data.K)
        Cauchy[int(t)] = pyro.distributions.MultivariateStudentT(
            df=1.0,  # Equivalent to Cauchy distribution.
            loc=mean_t.squeeze(-1),
            scale_tril=torch.linalg.cholesky(cov_t),
        )

    # 10) Assign potential clusters to every cell.
    init_value = list()
    topics = 0.5 * torch.ones(data.K, data.x.shape[-1])
    for chunk_i in data.iterate_by_chunk(SUBSMPL):
        ctype_i = chunk_i.ctype
        one_hot_batch_i = chunk_i.one_hot_batch
        x_i = chunk_i.x
        f_i = x_i / x_i.sum(dim=-1, keepdim=True)
        f_i -= ctype_profiles[ctype_i]
        f_i = f_i[:, topG]
        f_i_norm = (f_i - smpl_f_mean) / smpl_f_std
        proj_i = f_i_norm @ eigenvectors[:, -data.K :]
        proj_i -= one_hot_batch_i @ batch_avg
        # dim(log_p_i): `SUBSMPL` x K
        log_p_i = torch.stack([Cauchy[k].log_prob(proj_i) for k in range(data.K)]).transpose(-1, -2)
        maxval_i = log_p_i.max(dim=-1).values
        log_theta_i_loc = torch.clamp(log_p_i - maxval_i.unsqueeze(-1) + 2.0, min=-2.0, max=2.0)
        wgts_i = log_theta_i_loc.softmax(dim=-1)
        topics += torch.einsum("iG,iK->KG", x_i, wgts_i)
        init_value.append(log_theta_i_loc)
    init_value = torch.vstack(init_value)

    topics /= topics.sum(dim=-1, keepdim=True)
    topics_KR = torch.log(topics) - global_base
    topics_KR -= topics_KR.mean(dim=-1, keepdim=True)
    param_store["autonormal.locs.topics_KR"] = topics_KR

    if amortizer is not None:
        pretrain_amortizer(amortizer, data, init_value)


def pretrain_amortizer(amortizer, data, init_value):
    # Use a simple training loop to optimize the amortizer.
    optimizer = torch.optim.Adam(amortizer.parameters(), lr=0.01)
    # Train for `N_BATCHES` batches.
    for _ in range(N_BATCHES):
        # Sample `SUBSMPL` cells at random.
        idx_i = n_choose_k__01(data.ncells, SUBSMPL)
        data_i = data[idx_i]
        # Prepare input data.
        x_i = data_i.x
        ohb_i = data_i.one_hot_batch
        ohc_i = data_i.one_hot_ctype
        ohg_i = data_i.one_hot_group
        freq_i = x_i / x_i.sum(dim=-1, keepdim=True)
        # Keep only top genes.
        bcgf_i = torch.cat([ohb_i, ohc_i, ohg_i, freq_i], dim=-1)
        optimizer.zero_grad()
        loc, scale = amortizer(bcgf_i)
        # Compute composite MSE loss.
        l2_loss = torch.nn.MSELoss()
        loss = l2_loss(loc, init_value[idx_i]) + l2_loss(scale, 0.1 * torch.ones_like(scale))
        loss.backward()
        optimizer.step()
