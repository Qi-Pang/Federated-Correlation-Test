# Import Python libraries
import numpy as np
from datetime import datetime
np.set_printoptions(precision = 4)

import os
#import matplotlib.pyplot as plt
import scipy.optimize as optim
from scipy.stats import norm
from scipy.stats import bernoulli

# Import FDR procedures
from LORD_batch import*
from SAFFRON_batch import*
from AlphaInvest_batch import*
from SAFFRON_ALPHA_INV import*
from PAPRIKA_ALPHA_INV import*
from PAPRIKA_batch import*


# Import utilities
from rowexp_new_batch import*
from toimport import*
from settings_util import*
# from tqdm import tqdm
from progressbar import progressbar
  
################ Running entire framework  ####################

def run_single(NUMRUN, NUMHYP, NUMDRAWS, mu_gap,lbd_scale, pi, alpha0, mod_choice, FDR, eps, sensitivity,shift, sigma = 1, verbose = False, TimeString = False, rndseed = 1, startfac = 0.1, dropout=0):

    
    if rndseed == 0:
        TimeString = True

    ##------------- Setting hypotheses, penalty and prior weights -------------## 
    if TimeString:
        time_str = datetime.today().strftime("%m%d%y_%H%M")
    else:
        time_str = '0'

    # Hypo = get_hyp(pi, NUMHYP)
    Hypo = get_hyp_new(NUMHYP)
    Hypo = Hypo.astype(int)
    num_alt = np.sum(Hypo)
       
    #### Set file and dirnames ##########
    dir_name = './dat'
    filename = 'Lbd%.2f_Si%.1f_FDR%d_NH%d_ND%d_MOD%d_EPS%.1f_SE%.1f_SHT%.1f_PM%.2f_NR%d_%s_dropout%d' % (lbd_scale, sigma, FDR, NUMHYP, NUMDRAWS, mod_choice, eps, sensitivity,shift, pi, NUMRUN, time_str, dropout)
    
    # ----------- initialize result vectors and mats ------------- ##
    pval_mat = np.zeros([NUMHYP, NUMRUN])
    rej_mat = np.zeros([NUMHYP, NUMRUN])
    alpha_mat = np.zeros([NUMHYP, NUMRUN])
    wealth_mat = np.zeros([NUMHYP, NUMRUN])
    falrej_vec = np.zeros(NUMRUN)
    correj_vec = np.zeros(NUMRUN)
    totrej_vec = np.zeros(NUMRUN)
    FDR_mat = np.zeros([NUMHYP, NUMRUN])
    TDR_mat = np.zeros([NUMHYP, NUMRUN])
    falrej_mat = np.zeros([NUMHYP, NUMRUN])
    correj_mat = np.zeros([NUMHYP, NUMRUN])

    # ----------------- Run experiments (with same mus) ------------------ # 
    for l in progressbar(range(NUMRUN), redirect_stdout=True):

        #%%%%%%%%  Initialize theta_j, FDR and experiments %%%%%%%%#
        # Some random seed
        if (rndseed == 1):
            rndsd = l
        else:
            rndsd = None

        # Create a vector of gaps
        gap = np.ones(NUMHYP)*mu_gap
        this_exp = rowexp_new_batch(NUMHYP, NUMDRAWS, Hypo, 0, gap,pi)

        #%%%%%%%%% Run experiments: Get sample and p-values etc. %%%%%%%%%%%%%
        # Run random experiments with same random seed for all FDR procedures
        if mod_choice == 1:
            this_exp.gauss_two_mix(mu_gap, sigma, rndsd)
        elif mod_choice == 2:
            this_exp.beta_draws(rndsd)
        elif mod_choice == 3:
            this_exp.get_truncexpon_draws(rndsd,lbd_scale,1000)
        elif mod_choice == 4:
            this_exp.get_bernoulli_draws(rndsd,0.5,0.75,1000)
        elif mod_choice == 5:
            this_exp.get_chi_draws(rndsd, row=20, col=20, fedmode='federated', dropout=dropout)
        pval_mat[:, l] = this_exp.pvec

        # Initialize 
        if FDR == 1:
            proc = SAFFRON_proc_batch(alpha0, NUMHYP, 0.5, 1.6)
        elif FDR == 2:
            proc = LORD_proc_batch(alpha0, NUMHYP, startfac, 0)
        elif FDR == 3:
            proc = ALPHA_proc_batch(alpha0, NUMHYP, startfac)
        elif FDR == 4:
            proc = SAFFRON_ALPHA_INV_proc_batch(alpha0, NUMHYP, 1.4)
        elif FDR == 5:
            proc = PAPRIKA_ALPHA_INV_proc_batch(alpha0,NUMHYP, 1.4, eps, sensitivity, NUMHYP*pi, shift)
        elif FDR == 6:
            proc = PAPRIKA_proc_batch(alpha0,NUMHYP, 0.2, 1.6, eps, sensitivity, NUMHYP*pi, shift)


        #%%%%%%%%%% Run FDR, get rejection and next alpha %%%%%%%%%%%%
        rej_mat[:, l] = proc.run_fdr(this_exp.pvec)
        alpha_mat[:, l] = proc.alpha
        wealth_mat[:, l] = proc.wealth_vec[0:NUMHYP]

        #%%%%%%%%%%  Save results %%%%%%%%%%%%%%
        falrej_singlerun = np.array(rej_mat[:,l])*np.array(1-Hypo)
        correj_singlerun = np.array(rej_mat[:,l])*np.array(Hypo)
        totrej_singlerun = np.array(rej_mat[:,l])
        falrej_vec[l] = np.sum(falrej_singlerun)
        correj_vec[l] = np.sum(correj_singlerun)
        totrej_vec[l] = np.sum(totrej_singlerun)
        falrej_mat[:, l] = falrej_singlerun

        FDR_vec = np.zeros(NUMHYP)
        for j in range(NUMHYP):
            time_vec = np.arange(NUMHYP) < (j+1)
            FDR_num = np.sum(falrej_singlerun * time_vec)
            FDR_denom = np.sum(totrej_singlerun * time_vec)
            if FDR_denom > 0:
                FDR_vec[j] = np.true_divide(FDR_num, max(1, FDR_denom))
            else:
                FDR_vec[j] = 0

        FDR_mat[:,l] = FDR_vec
        
    # -----------------  Compute average quantities we care about ------------- #


    TDR_vec = np.true_divide(correj_vec, num_alt)
    FDR_vec = [FDR_mat[NUMHYP - 1][l] for l in range(NUMRUN)]

    if verbose == 1:
        print("done with computation")

    # Save data
    data = np.r_[FDR_mat, rej_mat, falrej_mat, wealth_mat, pval_mat, alpha_mat, np.expand_dims(TDR_vec, axis=0), np.expand_dims(np.asarray(FDR_vec),axis=0)]
    saveres(dir_name, filename, data)

