# %%
from collections import namedtuple
from modules import metric_stat, metric_MLu, metric_privacy

import warnings
warnings.filterwarnings("ignore", "use_inf_as_na")

Metrics = namedtuple(
    "Metrics",
    [
        "KL",
        "GoF",
        "MMD",
        "CW",
        "base_cls", 
        "syn_cls",
        "model_selection", 
        "feature_selection",
        "Kanon_base",
        "Kanon_syn",
        "KMap",
        "DCR_RS",
        "DCR_RR",
        "DCR_SS",
        "AD",
    ]
)
#%%
def evaluate(syndata, train_dataset, test_dataset, config, device):
    
    print("\n1. Statistical Fidelity: KL-Divergence...")
    KL = metric_stat.KLDivergence(train_dataset, syndata)
    
    print("\n2. Statistical Fidelity: Goodness Of Fit...")
    GoF = metric_stat.GoodnessOfFit(train_dataset, syndata)
    
    print("\n3. Statistical Fidelity: MMD...")
    MMD = metric_stat.MaximumMeanDiscrepancy(train_dataset, syndata)
    
    print("\n4. Statistical Fidelity: Cramer-Wold Distance...")
    CW = metric_stat.CramerWoldDistance(train_dataset, syndata, config, device)
    
    # print("\n5. Machine Learning Utility: Regression...")
    # base_reg, syn_reg = metric_MLu.MLu_reg(train_dataset, test_dataset, syndata)
    
    print("\n5. Machine Learning Utility: Classification...")
    base_cls, syn_cls, model_selection, feature_selection = metric_MLu.MLu_cls(
        train_dataset, test_dataset, syndata)
    
    print("\n6. Privacy: K-anonimity...")
    Kanon_base, Kanon_syn = metric_privacy.kAnonymization(train_dataset, syndata)
    
    print("\n7. Privacy: K-Map...")
    KMap = metric_privacy.kMap(train_dataset, syndata)
    
    print("\n8. Privacy: DCR...")
    DCR_RS, DCR_RR, DCR_SS = metric_privacy.DCR_metric(train_dataset, syndata)
    
    print("\n9. Privacy: Attribute Disclosure...")
    AD = metric_privacy.AttributeDisclosure(train_dataset, syndata)
    
    return Metrics(
        KL, GoF, MMD, CW, 
        base_cls, syn_cls, model_selection, feature_selection,
        Kanon_base, Kanon_syn, KMap, DCR_RS, DCR_RR, DCR_SS, AD
    )
#%%