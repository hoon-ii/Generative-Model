#%%
"""
Reference:
[1] Synthcity: facilitating innovative use cases of synthetic data in different data modalities
- https://github.com/vanderschaarlab/synthcity/blob/main/src/synthcity/metrics/eval_statistical.py
"""
#%%
import numpy as np
from tqdm import tqdm
import torch

from geomloss import SamplesLoss
from scipy.stats import entropy, ks_2samp
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

from modules import utils
#%%
def KLDivergence(train_dataset, syndata):
    """
    Marginal statistical fidelity: KL-Divergence
    : lower is better
    """
    
    train = train_dataset.raw_data
    num_bins = 10
    
    # get distribution of continuous variables
    cont_freqs = utils.get_frequency(
        train[train_dataset.continuous_features], 
        syndata[train_dataset.continuous_features], 
        n_histogram_bins=num_bins
    )
    
    result = [] 
    for col in train.columns:
        if col in train_dataset.continuous_features:
            gt, syn = cont_freqs[col]
            kl_div = entropy(syn, gt)
        else:
            pmf_p = train[col].value_counts(normalize=True)
            pmf_q = syndata[col].value_counts(normalize=True)
            
            # Ensure that both PMFs cover the same set of categories
            all_categories = pmf_p.index.union(pmf_q.index)
            pmf_p = pmf_p.reindex(all_categories, fill_value=0)
            pmf_q = pmf_q.reindex(all_categories, fill_value=0)
            
            # Avoid division by zero and log(0) by filtering out zero probabilities
            non_zero_mask = (pmf_p > 0) & (pmf_q > 0)
            
            kl_div = np.sum(pmf_q[non_zero_mask] * np.log(pmf_q[non_zero_mask] / pmf_p[non_zero_mask]))
        result.append(kl_div)
    return np.mean(result)
#%%
def GoodnessOfFit(train_dataset, syndata):
    """
    Marginal statistical fidelity: Kolmogorov-Smirnov test & Chi-Squared test
    : lower is better
    """
    
    train = train_dataset.raw_data
    
    result = [] 
    for col in train.columns:
        if col in train_dataset.continuous_features:
            # Compute the Kolmogorov-Smirnov test for goodness of fit.
            statistic, _ = ks_2samp(train[col], syndata[col])
        else:
            pmf_p = train[col].value_counts(normalize=True) # expected
            pmf_q = syndata[col].value_counts(normalize=True) # observed
            
            # Ensure that both PMFs cover the same set of categories
            all_categories = pmf_p.index.union(pmf_q.index)
            pmf_p = pmf_p.reindex(all_categories, fill_value=0)
            pmf_q = pmf_q.reindex(all_categories, fill_value=0)
            
            # Avoid division by zero and log(0) by filtering out zero probabilities
            non_zero_mask = pmf_p > 0
            
            # Compute the Chi-Squared test for goodness of fit.
            statistic = ((pmf_q[non_zero_mask] - pmf_p[non_zero_mask]) ** 2 / pmf_p[non_zero_mask]).sum()
        result.append(statistic)
    return np.mean(result)
#%%
def MaximumMeanDiscrepancy(train_dataset, syndata):
    """
    Joint statistical fidelity: Maximum Mean Discrepancy (MMD)
    : lower is better
    
    - MMD using RBF (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))
    """
    
    train = train_dataset.raw_data
    # only continuous
    train = train[train_dataset.continuous_features]
    syndata = syndata[train_dataset.continuous_features]
    
    scaler = StandardScaler().fit(train)
    train_ = scaler.transform(train)
    syndata_ = scaler.transform(syndata)
    
    XX = metrics.pairwise.rbf_kernel(
        train_.reshape(len(train_), -1),
        train_.reshape(len(train_), -1),
    )
    YY = metrics.pairwise.rbf_kernel(
        syndata_.reshape(len(syndata_), -1),
        syndata_.reshape(len(syndata_), -1),
    )
    XY = metrics.pairwise.rbf_kernel(
        train_.reshape(len(train_), -1),
        syndata_.reshape(len(syndata_), -1),
    )
    MMD = XX.mean() + YY.mean() - 2 * XY.mean()
    return MMD
#%%
def WassersteinDistance(train_dataset, syndata, device):
    """
    Joint statistical fidelity: Wasserstein Distance
    : lower is better
    """
    
    train = train_dataset.raw_data
    # only continuous
    train = train[train_dataset.continuous_features]
    syndata = syndata[train_dataset.continuous_features]
    
    train_ = train.values.reshape(len(train), -1)
    syndata_ = syndata.values.reshape(len(syndata), -1)
    
    # assert len(train_) == len(syndata_)

    scaler = StandardScaler().fit(train_)
    train_ = scaler.transform(train_)
    syndata_ = scaler.transform(syndata_)
    
    train_ = torch.from_numpy(train_).to(device)
    syndata_ = torch.from_numpy(syndata_).to(device)
    
    OT_solver = SamplesLoss(loss="sinkhorn")
    if len(train_) > 4000:
        WD = []
        iter_ = len(train_) // 4000 + 1
        for _ in tqdm(range(iter_), desc="Batch WD..."):
            idx = np.random.choice(range(len(train_)), 4000, replace=False)
            WD.append(OT_solver(train_[idx, :], syndata_[idx, :]).cpu().numpy().item())
        WD = np.mean(WD)
    else:
        WD = OT_solver(train_, syndata_).cpu().numpy().item()
    return WD
#%%
def phi(s, D):
    return (1 + (4 * s) / (2 * D - 3)) ** (-1 / 2)

def CramerWoldDistance(train_dataset, syndata, config, device):
    """
    Joint statistical fidelity: Cramer-Wold Distance
    : lower is better
    """
    
    train = train_dataset.raw_data
    # only continuous
    train = train[train_dataset.continuous_features]
    syndata = syndata[train_dataset.continuous_features]
    if config["dataset"] == "adult": ### OOM
        train = train.sample(frac=0.5, random_state=42)
        syndata = syndata.sample(frac=0.5, random_state=42)
    
    scaler = StandardScaler().fit(train)
    train_ = scaler.transform(train)
    syndata_ = scaler.transform(syndata)
    train_ = torch.from_numpy(train_).to(device)
    syndata_ = torch.from_numpy(syndata_).to(device)
    
    gamma_ = (4 / (3 * train_.size(0))) ** (2 / 5)
    
    cw1 = torch.cdist(train_, train_) ** 2 
    cw2 = torch.cdist(syndata_, syndata_) ** 2 
    cw3 = torch.cdist(train_, syndata_) ** 2 
    cw = phi(cw1 / (4 * gamma_), D=train_.size(1)).sum()
    cw += phi(cw2 / (4 * gamma_), D=train_.size(1)).sum()
    cw += -2 * phi(cw3 / (4 * gamma_), D=train_.size(1)).sum()
    cw /= (2 * train_.size(0) ** 2 * torch.tensor(torch.pi * gamma_).sqrt())
    return cw.cpu().numpy().item()
#%%