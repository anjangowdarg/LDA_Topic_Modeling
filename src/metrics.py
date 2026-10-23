import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

def category_metrics(labels,dominant):
    labels=np.asarray(labels); dominant=np.asarray(dominant)
    purity=sum(np.bincount(np.unique(labels[dominant==t],return_inverse=True)[1]).max() for t in np.unique(dominant))/len(labels)
    return {'purity':float(purity),'NMI':float(normalized_mutual_info_score(labels,dominant)),'ARI':float(adjusted_rand_score(labels,dominant))}
