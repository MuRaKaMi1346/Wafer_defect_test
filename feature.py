import numpy as np

def extract_features(wmap):

    w = np.array(wmap)
    fail = (w == 2).astype(int)

    total_fail = fail.sum()
    if total_fail == 0:
        return [0]*6

    h, w_ = fail.shape
    cy, cx = h//2, w_//2

    center = fail[cy-5:cy+5, cx-5:cx+5].sum()

    edge = np.concatenate([
        fail[:5,:].ravel(), fail[-5:,:].ravel(),
        fail[:, :5].ravel(), fail[:, -5:].ravel()
    ]).sum()

    ys, xs = np.where(fail==1)
    spread = np.std(ys) + np.std(xs)

    vertical = fail.sum(axis=0).max()
    horizontal = fail.sum(axis=1).max()

    return [total_fail, center, edge, spread, vertical, horizontal]