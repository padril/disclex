from collections import defaultdict
import numpy as np
from scipy.special import logsumexp
import matplotlib.pyplot as plt
import pandas as pd


def mean_nlld(nllds):
    return -(logsumexp(-nllds) - np.log(len(nllds)))

def main(args):
    deltas, *_ = args
    df = pd.read_csv(deltas)
    df = df[df['method'] == 'crp']
    rough = df.groupby('step')['nlld'].agg(mean_nlld)
    smooth = rough.ewm(alpha=1 - 0.99).mean()

    plt.plot(rough, alpha=0.4)
    plt.plot(smooth)
    plt.grid(alpha=0.3)

    plt.show()

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
