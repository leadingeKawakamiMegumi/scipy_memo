import scipy.stats

X_standard = []
for ie in range(0,Xt.shape[0]):
    X_standard1 = Xt[ie]
    X_standard.append(scipy.stats.zscore(X_standard1))
