# Import Python libraries
import numpy as np
np.set_printoptions(precision = 4)
from scipy.stats import bernoulli

# Import own utilities
from toimport import *


def get_hyp(pi, num_hyp):
        
    # Read hyp from file
    filename_pre = "H_PM%.2f_NH%d" % (pi, num_hyp)
    hypo_filename = [filename for filename in os.listdir('./expsettings') if filename.startswith(filename_pre)]
    if len(hypo_filename) > 0:
        # Just take the first sample
        hyp_mat = np.loadtxt('./expsettings/%s' % hypo_filename[0])    
    else:
        print("Hyp file doesn't exist, thus generating the file now ...")
        # Generate 100 draws of num_hyp hypotheses with given pi_1 setting
        hyp_mat = generate_hyp(pi, num_hyp, 100)

    # Choose some Hypvector could choose a different sample
    Hypo = hyp_mat[0]
    
    return Hypo


def get_hyp_new(num_hyp):
    # Read hyp from file
    filename_pre = "H_PM_NH%d" % (num_hyp)
    hypo_filename = [filename for filename in os.listdir('./expsettings') if filename.startswith(filename_pre)]
    if len(hypo_filename) > 0:
        # Just take the first sample
        hyp_mat = np.loadtxt('./expsettings/%s' % hypo_filename[0])    
    else:
        print("Hyp file doesn't exist, thus generating the file now ...")
        # Generate 100 draws of num_hyp hypotheses with given pi_1 setting
        hyp_mat = generate_hyp_new(num_hyp, 100)

    # Choose some Hypvector could choose a different sample
    Hypo = hyp_mat[0]
    
    return Hypo

def generate_hyp_new(max_hyp, samples):
    # ---- Get pi1 progression ---- #
    pi1_vec = np.ones(max_hyp)*0.5

    # Caculate lengths of constant pieces using pi1_vec and max_hyp
    hyp_steps = 1
    length_vec = [max_hyp]

    hyp_mat = np.zeros([samples, max_hyp])

     # ---- Sample hypotheses vectors using the pi1 progression ------ #
    for i in range(samples):

        Hyp = np.array([])
        for j in range(hyp_steps):
            Hyp = np.concatenate((Hyp, bernoulli.rvs(pi1_vec[j], size=length_vec[j])))

        hyp_mat[i] = Hyp

    # ----- Save sample hypotheses vectors ----- #
    dirname = './expsettings'
    filename = "H_PM_NH%d" % (max_hyp)
    saveres(dirname, filename, hyp_mat)
    return hyp_mat

def generate_hyp(pi, max_hyp, samples):

    # ---- Get pi1 progression ---- #
    pi1_vec = np.ones(max_hyp)*pi

    # Caculate lengths of constant pieces using pi1_vec and max_hyp
    hyp_steps = 1
    length_vec = [max_hyp]

    hyp_mat = np.zeros([samples, max_hyp])

     # ---- Sample hypotheses vectors using the pi1 progression ------ #
    for i in range(samples):

        Hyp = np.array([])
        for j in range(hyp_steps):
            Hyp = np.concatenate((Hyp, bernoulli.rvs(pi1_vec[j], size=length_vec[j])))

        hyp_mat[i] = Hyp

    # ----- Save sample hypotheses vectors ----- #
    dirname = './expsettings'
    filename = "H_PM%.2f_NH%d" % (pi, max_hyp)
    saveres(dirname, filename, hyp_mat)
    return hyp_mat
        
        
        
