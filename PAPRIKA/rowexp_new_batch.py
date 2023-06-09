import numpy as np
from toimport import *

from scipy.stats import norm
from scipy.stats import truncexpon
from scipy.stats import bernoulli
from scipy.stats import levy_stable
from scipy.stats.distributions import chi2
from scipy import stats
from scipy.special import gamma
from scipy.special import comb 
from tqdm import tqdm

class rowexp_new_batch:

    def __init__(self, NUMHYP, numdraws, alt_vec, mu0, mu_alt_vec, pi):
        self.numhyp = NUMHYP
        self.alt_vec = alt_vec
        self.mu0 = mu0
        self.mu_vec = mu0*np.ones(NUMHYP) + np.multiply(alt_vec, mu_alt_vec)
        self.pvec = np.zeros(NUMHYP)
        self.numdraws = numdraws
        self.pi=pi
        self.lts=[]
        self.gt=None
        self.sample_size = 10
        self.count_ind = 0
        self.count_gaussian = 0
        '''
        Function drawing p-values: Mixture of two Gaussians
        '''
    def gauss_two_mix(self, mu_gap, sigma = 1, rndsd = 0):

        np.random.seed(rndsd)

        # Draw Z values according to lag
        Z = self.mu_vec + np.random.randn(self.numhyp)*sigma # draw gaussian acc. to hypothesis, if sigma are all same


        # Compute p-values and save
        if mu_gap > 0:
            self.pvec = [(1 - norm.cdf(z)) for z in Z] # one-sided p-values
        else:
            self.pvec = [2*norm.cdf(-abs(z)) for z in Z] # two-sided p-values

    def beta_draws(self, rndsd = 0):
        np.random.seed(rndsd)
        self.pvec = [(np.random.beta(0.5,5,1)*self.alt_vec[i]+np.random.uniform(0,1,1)*(1-self.alt_vec[i])) for i in range(self.numhyp)]

    def bernoulli_draws(self, theta1, theta2, rndsd=1, datasize=1000):
        np.random.seed(rndsd)
        for i in range(self.numhyp):
            if self.alt_vec[i]==0:
                database=bernoulli.rvs(theta1,size=datasize)
            else:
                database=bernoulli.rvs(theta2,size=datasize)
            t=sum(database)
            pval=0
            for j in range(t,datasize+1):
                pval=pval+1/(2**datasize)*comb(datasize,j,exact=True)
            self.pvec[i]=pval
        dirname = './expsettings'
        filename = "P_NH%d_PM%.2f_T1%.2f_T2%.2f" % (self.numhyp, self.pi, theta1, theta2)
        saveres(dirname, filename, self.pvec)

    def chi_draws_global(self, rndsd=0, row=20, col=20, fedmode='centralized', dropout=0):
        def chi2_test(lts, gt, row=20, col=20):
            n = np.sum(gt)
            gt_x = gt.sum(axis=1)
            gt_y = gt.sum(axis=0)
            gt_ex = np.zeros_like(gt, dtype=float)
            for i in range(row):
                for j in range(col):
                    gt_ex[i][j] = 1.0 * gt_x[i] * gt_y[j] / n
            score = 0.0
            for i in range(row):
                for j in range(col):
                    score += 1.0 * (gt[i][j] - gt_ex[i][j]) * (gt[i][j] - gt_ex[i][j]) / gt_ex[i][j]
            return score

        def geometric_mean(alpha, sketch_size, x):
            return np.prod(np.power(np.abs(x), alpha/sketch_size))/np.power(2*gamma(alpha/sketch_size)*gamma(1-1/sketch_size)*np.sin(np.pi*alpha/2/sketch_size)/np.pi, sketch_size)

        def fed_chi2_test(lts, gt, row, col, samples_num, nworker, dropout=dropout):
            if dropout != 0:
                dropout_list = np.random.choice(int(nworker), int(dropout * nworker // 100), replace=False).tolist()
            else:
                dropout_list = []
            # print(dropout_list)
            n = np.sum(gt)
            gt_x = gt.sum(axis=1)
            gt_y = gt.sum(axis=0)
            gt_ex = np.zeros_like(gt, dtype=float)
            for i in range(row):
                for j in range(col):
                    gt_ex[i][j] = 1.0 * gt_x[i] * gt_y[j] / n
            rv = levy_stable(2.0, 0.0)
            proj_matrix = rv.rvs(size=[samples_num, row * col])
            samples = np.zeros(samples_num)
            for i in range(nworker):
                if i in dropout_list:
                    continue
                ob = lts[i]
                inter = np.divide(ob - gt_ex / nworker, np.sqrt(gt_ex))
                samples += np.matmul(proj_matrix, inter.flatten())
            return geometric_mean(2.0, samples_num, samples)

        np.random.seed(rndsd)
        for i in range(self.numhyp):
            if self.alt_vec[i] == 0:
                global_database, local_database = self.get_global_table(correlation='independent')
            else:
                global_database, local_database = self.get_global_table(correlation='gaussian')
            dof = global_database.shape[0] * global_database.shape[1] - 1
            if fedmode == 'centralized':
                chi2_score = chi2_test(local_database, global_database, row=row, col=col)
                # print(chi2_score)
                pval = chi2.sf(chi2_score, dof)
                # pval = 1 - stats.chi2.cdf(chi2_score, dof)
                self.pvec[i] = pval
            elif fedmode == 'federated':
                # FIXME: nworker and samples 
                chi2_score = fed_chi2_test(local_database, global_database, row=row, col=col, samples_num=self.sample_size, nworker=10)
                pval = chi2.sf(chi2_score, dof)
                # pval = 1 - stats.chi2.cdf(chi2_score, dof)
                self.pvec[i] = pval
        dirname = './expsettings'
        filename = "P_NH%d_%s_%d_SEED%d_dropout%d" % (self.numhyp, fedmode, self.sample_size ,rndsd, dropout)
        saveres(dirname, filename, self.pvec)

    def get_global_table(self, row=20, col=20, clients=10, correlation='independent'):
        def local_table(row=2, col=2, correlation='independent'):
            if correlation == 'independent':
                ind_filename = './store_data/ind_' + str(self.count_ind) + '.npy'
                try:
                    result = np.load(ind_filename)
                except:
                    result = np.random.randint(low=0, high=5, size=(row, col))
                    np.save(ind_filename, result)
                self.count_ind += 1
                return result
            elif correlation == 'linear':
                X = np.tile(np.arange(start=0, stop=row), (col, 1)).transpose()
                proj_matrix = np.random.randint(low=1, high=5, size=(row, col))
                noise_matrix = np.random.normal(size=(row, col)).astype(int)
                result = np.multiply(proj_matrix, X) + noise_matrix
                # result /= 500
                result = result.astype(int)
                return np.where(result<0, 0, result)
            elif correlation == 'quadratic':
                X = np.tile(np.arange(start=0, stop=row), (col, 1)).transpose()
                X = X**2
                proj_matrix = np.random.randint(low=0, high=5, size=(row, col))
                noise_matrix = np.random.normal(size=(row, col)).astype(int)
                result = np.multiply(proj_matrix, X) + noise_matrix
                return np.where(result<0, 0, result)
            elif correlation == 'logistic':
                X = np.exp(-np.tile(np.arange(start=0, stop=row) / 20, (col, 1))).transpose().astype(int)
                proj_matrix = np.random.randint(low=0, high=50, size=(row, col))
                noise_matrix = np.random.normal(size=(row, col)).astype(int)
                result = np.multiply(proj_matrix, X) + noise_matrix
                return np.where(result<0, 0, result)
            elif correlation == 'gaussian':
                gaussian_filename = './store_data/gaussian_' + str(self.count_gaussian) + '.npy'
                try:
                    result = np.load(gaussian_filename)
                except:
                    # cov = np.random.randint(low=-1, high=1, size=(row * col, row * col))
                    cov = np.random.rand(row * col, row * col) - 0.5
                    cov = np.matmul(cov, cov.T)
                    # mu = np.random.randint(low=0, high=5, size=(row * col))
                    # print(cov)
                    mu = np.zeros((row * col))
                    result = np.random.multivariate_normal(mu, cov).astype(int)
                    result = result.reshape((row, col))
                    np.save(gaussian_filename, result)
                self.count_gaussian += 1
                return np.where(result<0, 0, result)

        lts = []
        gt = np.zeros((row, col), dtype=float)
        for _ in range(clients):
            lt = local_table(row, col, correlation)
            lts.append(lt)
            gt += lt
        # normalize = np.sum(gt) / 8000
        # gt = np.zeros((row, col), dtype=float)
        # for i in range(len(lts)):
        #     lts[i] = (lts[i] / normalize).astype(int)
        #     gt += lts[i]
        return gt, lts

    def get_chi_draws(self, rndsd=0, row=20, col=20, fedmode='centralized', dropout=0):
        filename_pre = "P_NH%d_%s_%d_SEED%d_dropout%d" % (self.numhyp, fedmode, self.sample_size ,rndsd, dropout)
        p_filename = [filename for filename in os.listdir('./expsettings') if filename.startswith(filename_pre)]
        if len(p_filename) > 0:
        # Just take the first sample
            self.pvec = np.loadtxt('./expsettings/%s' % p_filename[0])
        else:
            #print("Hyp file doesn't exist, thus generating the file now ...")
        # Generate pvec with given setting
            self.chi_draws_global(rndsd=rndsd, row=row, col=col, fedmode=fedmode, dropout=dropout)

    def get_bernoulli_draws(self, rndsd=1, theta1=0.5, theta2=0.75, datasize=1000):
        
        # Read pvalues from file
        filename_pre = "P_NH%d_PM%.2f_T1%.2f_T2%.2f" % (self.numhyp, self.pi, theta1, theta2)
        p_filename = [filename for filename in os.listdir('./expsettings') if filename.startswith(filename_pre)]
        if len(p_filename) > 0:
        # Just take the first sample
            self.pvec = np.loadtxt('./expsettings/%s' % p_filename[0])    
        else:
            #print("Hyp file doesn't exist, thus generating the file now ...")
        # Generate pvec with given setting
            self.bernoulli_draws(theta1, theta2, rndsd, datasize=1000)
    
    def truncexpon_draws(self, lbd_scale, rndsd=0, thresh=1, datasize=1000):
        np.random.seed(rndsd)
        for i in range(self.numhyp):
            if self.alt_vec[i]==0:
                database=truncexpon.rvs(b=thresh, size=datasize)
            else:
                database=truncexpon.rvs(b=thresh, scale=lbd_scale, size=datasize)
            z=sum(database)
            pval=1 - norm.cdf(z,loc=datasize*(1+1/(1-np.exp(1))), scale=datasize*(1-np.exp(1)/(np.exp(1)-1)**2))
            self.pvec[i]=pval 
        dirname = './expsettings'
        filename = "P_NH%d_PM%.2f_lbd%.2f_SEED%d" % (self.numhyp, self.pi, lbd_scale,rndsd)
        saveres(dirname, filename, self.pvec)

        
    def get_truncexpon_draws(self, rndsd=0, lbd_scale=2.00, datasize=1000):
        
        # Read pvalues from file
        filename_pre = "P_NH%d_PM%.2f_lbd%.2f_SEED%d" % (self.numhyp, self.pi, lbd_scale,rndsd)
        p_filename = [filename for filename in os.listdir('./expsettings') if filename.startswith(filename_pre)]
        if len(p_filename) > 0:
        # Just take the first sample
            self.pvec = np.loadtxt('./expsettings/%s' % p_filename[0])    
        else:
            #print("Hyp file doesn't exist, thus generating the file now ...")
        # Generate pvec with given setting
            self.truncexpon_draws(lbd_scale, rndsd, thresh=1, datasize=1000)
