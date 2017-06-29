
import numpy as np
import pandas as pd


def InitCorrels(levels, T):
    n_levels = len(levels)
    correls = np.zeros((T, n_levels, n_levels), dtype="<i8")
    responses = np.zeros((T, n_levels))
    
    return correls, responses


def GetCorrels(df, keys, levels, correls, responses):
    n_levels = len(levels)
    idx = pd.Index(levels)
    labels = [np.repeat(np.arange(n_levels), n_levels), np.repeat(np.arange(n_levels).reshape((1, n_levels)), n_levels, axis=0).flatten()]
    midx = pd.MultiIndex(levels=[levels, levels], labels=labels)
    
    df["delta"] = df[keys["price"]].shift(-1) - df[keys["price"]]
    sp = df[keys["sign"]].divide(df[keys["price"]])
    for i in range(correls.shape[0]):
        shifted = df.shift(-i)
        data = pd.DataFrame({"key1": df[keys["action"]],
                             "key2": shifted[keys["action"]].fillna(0).astype("<i8"),
                             "s": df[keys["sign"]] * shifted[keys["sign"]].fillna(0).astype("<i8"),
                             "response": shifted["delta"].multiply(sp)
                            })
        correls[i, :] += data.groupby(["key1", "key2"])["s"].sum().reindex(midx).fillna(0).unstack().astype("<i8")
        
        responses[i, :] += data.groupby("key1")["response"].sum().reindex(idx).fillna(0)


def GetImpact(correls, responses):
    T = correls.shape[0]
    K = correls.shape[1]
    diag = np.diagonal(correls[0])
    
    c = np.zeros((T * K, T * K))
    i = 0
    for tau in range(T):
        for a in range(K):
            j = 0
            for n in range(T):
                if n < tau:
                    c[i, j:j + K] = correls[tau - n, a, :]
                else:
                    c[i, j:j + K] = correls[n - tau, :, a]
                
                if diag[a] > 0:
                    c[i, j:j + K] /= diag[a]
                
                j += K
            i += 1
    resp = (responses / diag).flatten()
    
    impacts = np.linalg.solve(c, resp).reshape((T, K))
    
    return impacts
