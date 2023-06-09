from multiprocessing import Pool
import argparse
import numpy as np
from scipy.special import gamma
from scipy.stats import levy_stable
from data import read_data
from scipy.stats.distributions import chi2

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--data', default='synthesize')
    parser.add_argument('--correlation', default='independent')
    parser.add_argument('--test', default='chi2')
    parser.add_argument('--nworker', type=int, default=10)
    parser.add_argument('--row', type=int, default=20)
    parser.add_argument('--col', type=int, default=20)
    parser.add_argument('--samples', type=int, default=50)
    parser.add_argument('--iters', type=int, default=10)
    # implement parallel using pool to accelerate the process
    parser.add_argument('--para_level', type=int, default=1)
    parser.add_argument('--power', action='store_true')
    parser.add_argument('--dropout', type=float, default=0)


    return parser.parse_args()

def global_table(row=20, col=20, clients=10, correlation='independent'):
    def local_table(row=2, col=2, correlation='independent'):
        if correlation == 'independent':
            if row <= 20:
                return np.random.randint(low=0, high=5, size=(row, col))
            elif row < 40 and row > 20:
                return np.random.randint(low=0, high=4, size=(row, col))
            elif row >= 40:
                return np.random.randint(low=0, high=3, size=(row, col))

        elif correlation == 'linear':
            X = np.tile(np.arange(start=0, stop=row), (col, 1)).transpose()
            proj_matrix = np.random.randint(low=0, high=5, size=(row, col))
            noise_matrix = np.random.normal(size=(row, col)).astype(int)
            result = np.multiply(proj_matrix, X) + noise_matrix
            return np.where(result<0, 0, result)
        elif correlation == 'quadratic':
            X = np.tile(np.arange(start=0, stop=row), (col, 1)).transpose()
            X = X**2
            proj_matrix = np.random.randint(low=0, high=5, size=(row, col))
            noise_matrix = np.random.normal(size=(row, col)).astype(int)
            result = np.multiply(proj_matrix, X) + noise_matrix
            return np.where(result<0, 0, result)
        elif correlation == 'logistic':
            X = np.exp(-np.tile(np.arange(start=0, stop=row), (col, 1))).transpose().astype(int)
            proj_matrix = np.random.randint(low=0, high=10, size=(row, col))
            noise_matrix = np.random.normal(size=(row, col)).astype(int)
            result = np.multiply(proj_matrix, X) + noise_matrix
            return np.where(result<0, 0, result)
        elif correlation == 'gaussian':
            cov = np.random.randint(low=-5, high=5, size=(row * col, row * col))
            cov = np.matmul(cov, cov.T)
            mu = np.random.randint(low=0, high=5, size=(row * col))
            result = np.random.multivariate_normal(mu, cov)
            result = result.reshape((row, col))
            return np.where(result<0, 0, result)
    lts = []
    gt = np.zeros((row, col), dtype=float)
    for i in range(clients):
        lt = local_table(row, col, correlation)
        lts.append(lt)
        gt += lt

    return lts, gt 

def geometric_mean(alpha, sketch_size, x):
    return np.prod(np.power(np.abs(x), alpha/sketch_size))/np.power(2*gamma(alpha/sketch_size)*gamma(1-1/sketch_size)*np.sin(np.pi*alpha/2/sketch_size)/np.pi, sketch_size)

def chi2_test(lts, gt, args):

    n = np.sum(gt)
    gt_x = gt.sum(axis=1)
    gt_y = gt.sum(axis=0)
    gt_ex = np.zeros_like(gt, dtype=float)
    for i in range(args.row):
        for j in range(args.col):
            gt_ex[i][j] = 1.0 * gt_x[i] * gt_y[j] / n

    score = 0.0
    for i in range(args.row):
        for j in range(args.col):
            score += 1.0 * (gt[i][j] - gt_ex[i][j]) * (gt[i][j] - gt_ex[i][j]) / gt_ex[i][j]
    
    return score

def fed_chi2_test(lts, gt, args):

    n = np.sum(gt)
    gt_x = gt.sum(axis=1)
    gt_y = gt.sum(axis=0)
    gt_ex = np.zeros_like(gt, dtype=float)
    for i in range(args.row):
        for j in range(args.col):
            gt_ex[i][j] = 1.0 * gt_x[i] * gt_y[j] / n

    rv = levy_stable(2.0, 0.0)
    proj_matrix = rv.rvs(size=[args.samples, args.row * args.col])

    samples = np.zeros(args.samples)
    if args.dropout > 0:
        drop_clients = np.random.choice(np.arange(args.nworker), int(args.dropout * args.nworker))
    for i in range(args.nworker):
        if args.dropout > 0 and i in drop_clients:
            continue
        ob = lts[i]
        inter = np.divide(ob - gt_ex / args.nworker, np.sqrt(gt_ex))
        samples += np.matmul(proj_matrix, inter.flatten())

    return geometric_mean(2.0, args.samples, samples)

def G_test(lts, gt, args):

    n = np.sum(gt)
    gt_x = gt.sum(axis=1)
    gt_y = gt.sum(axis=0)
    gt_ex = np.zeros_like(gt, dtype=float)
    for i in range(args.row):
        for j in range(args.col):
            gt_ex[i][j] = 1.0 * gt_x[i] * gt_y[j] / n

    score = 0.0
    for i in range(args.row):
        for j in range(args.col):
            score += 2.0 * gt[i][j] * np.log(gt[i][j] / gt_ex[i][j])

    return score

def fed_G_test(lts, gt, args, delta=0.05):

    n = np.sum(gt)
    gt_x = gt.sum(axis=1)
    gt_y = gt.sum(axis=0)
    gt_ex = np.zeros_like(gt, dtype=float)
    for i in range(args.row):
        for j in range(args.col):
            gt_ex[i][j] = 1.0 * gt_x[i] * gt_y[j] / n

    rv_upper = levy_stable(1+delta, 0.0)
    proj_matrix_upper = rv_upper.rvs(size=[args.samples, args.row * args.col])
    rv_lower = levy_stable(1-delta, 0.0)
    proj_matrix_lower = rv_lower.rvs(size=[args.samples, args.row * args.col])

    samples_upper = np.zeros(args.samples)
    samples_lower = np.zeros(args.samples)
    for i in range(args.nworker):
        ob = lts[i]
        samples_upper += np.matmul(proj_matrix_upper, ob.flatten())
        samples_lower += np.matmul(proj_matrix_lower, ob.flatten())

    return (geometric_mean(1+delta, args.samples, samples_upper) - geometric_mean(1-delta, args.samples, samples_lower)) / 0.05 - 2.0 * np.sum(np.multiply(gt, np.log(gt_ex)))

if __name__ == '__main__':

    args = parse_args()

    # synthesize data
    if args.data == 'synthesize':
        lts, gt = global_table(args.row, args.col, args.nworker, args.correlation)
    else:
        lts, gt = read_data(args.data, args.nworker)
        args.row = gt.shape[0]
        args.col = gt.shape[1]

    if args.test == 'chi2' and args.power:
        args.samples = 50
        count = 0
        for _ in range(args.iters//10):
            ests = []
            lts, gt = global_table(args.row, args.col, args.nworker, args.correlation)
            score = chi2_test(lts, gt, args)
            dof = gt.shape[0] * gt.shape[1] - 1
            pval_orig = chi2.sf(score, dof)
            print('orig pval: ', pval_orig, score)
            for _ in range(args.iters):
                fedresult = fed_chi2_test(lts, gt, args)
                pval = chi2.sf(fedresult, dof)
                ests.append(np.abs((fedresult-score)/score))
                if args.correlation == 'independent':
                    if pval >= 0.05:
                        count += 1
                else:
                    if pval < 0.05:
                        count += 1
            print(np.mean(ests), np.std(ests))
        print(args.correlation, count/(args.iters**2 // 10))

    elif args.test == 'chi2' and args.dropout > 0:
        args.samples = 50
        count = 0
        for _ in range(args.iters//10):
            ests = []
            lts, gt = global_table(args.row, args.col, args.nworker, args.correlation)
            score = chi2_test(lts, gt, args)
            dof = gt.shape[0] * gt.shape[1] - 1
            pval_orig = chi2.sf(score, dof)
            print('orig pval: ', pval_orig, score)
            for _ in range(args.iters):
                fedresult = fed_chi2_test(lts, gt, args)
                pval = chi2.sf(fedresult, dof)
                ests.append(np.abs((fedresult-score)/score))
                if args.correlation == 'independent':
                    if pval >= 0.05:
                        count += 1
                else:
                    if pval < 0.05:
                        count += 1
            print(np.mean(ests), np.std(ests))
        print(args.correlation, count/(args.iters**2 // 10))

    elif args.test == 'chi2':
        score = chi2_test(lts, gt, args)
        f = open("../results/%s_%d.txt"%(args.data, args.nworker), "w")
        # print(score)
        # f.write("%f\n"%score)
        f.write("x, y, err\n")
        high = max(50, min(200, 100*(args.row*args.col//100)))
        for s in np.linspace(10, high, high//10):
            args.samples = int(s)
            ests = []
            for _ in range(args.iters):
                ests.append(np.abs((fed_chi2_test(lts, gt, args)-score)/score))
            print("%d, %f, %f"%(s, np.mean(ests), np.std(ests)))
            f.write("%d, %f, %f\n"%(s, np.mean(ests), np.std(ests)))
        f.close()
    

    elif args.test == 'G':
        score = G_test(lts, gt, args)
        est = fed_G_test(lts, gt, args)

    # print("score: %f, estimation: %f"%(score, est))
