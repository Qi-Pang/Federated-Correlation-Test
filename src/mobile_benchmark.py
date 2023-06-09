import time
import numpy as np
from scipy.stats import levy_stable

def encoding_benchmark(row, col, iters=100):

    ob = np.zeros((row, col))
    gt_ex = np.zeros((row, col))

    rv = levy_stable(2.0, 0.0)
    proj_matrix = rv.rvs(size=[50, row * col])

    start = time.time()
    for i in range(iters):
        inter = np.divide(ob - gt_ex / 1000, np.sqrt(gt_ex))
        np.matmul(proj_matrix, inter.flatten())
    end = time.time()

    print(row, ", ", (end-start)/iters)

if __name__ == '__main__':

    for i in range(50):
        encoding_benchmark((i+1)*10, (i+1)*10)
