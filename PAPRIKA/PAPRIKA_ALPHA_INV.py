# PAPRIKA AI simulation code

import numpy as np


class PAPRIKA_ALPHA_INV_proc_batch:
    def __init__(self, alpha0, numhyp, gamma_vec_exponent, eps, sensitivity, cutoff,shift):
        self.alpha0 = alpha0 # FDR level

        # Compute the discount gamma sequence and make it sum to 1
        tmp = range(1, 10000)
        self.gamma_vec = np.true_divide(np.ones(len(tmp)),
                np.power(tmp, gamma_vec_exponent))
        self.gamma_vec = self.gamma_vec / np.float(sum(self.gamma_vec))

        self.w0 = self.alpha0/2 # initial wealth
        self.wealth_vec = np.zeros(numhyp + 1)  # vector of wealth at every step
        self.wealth_vec[0] = self.w0
        self.alpha = np.zeros(numhyp + 1) # vector of test levels alpha_t at every step
        self.alpha[0:2] = [0, (self.gamma_vec[0] * self.w0)/(1 + 2*self.gamma_vec[0] * self.w0)]
        self.eps = eps
        self.sensitivity= sensitivity
        self.cutoff = cutoff
        self.shift=shift
        self.delta=2.5e-4


    # Computing the number of candidates after each rejection
    def count_candidates(self, last_rej, candidates):
        ret_val = [];
        for j in range(1,len(last_rej)):
            ret_val = np.append(ret_val, sum(candidates[last_rej[j]+1:]))
        return ret_val.astype(int)

    # Running PAPRIKA on pvec
    def run_fdr(self, pvec):
        numhyp = len(pvec)
        last_rej = []
        flag = 0
        rej = np.zeros(numhyp + 1)
        candidates = np.zeros(numhyp + 1)
        Lap_T = np.random.laplace(loc=0.0,scale=2*self.sensitivity*self.cutoff/self.eps)
        totalrej=0
        A=self.shift*self.sensitivity*self.cutoff*np.log(2/(3*min(self.delta,1-((1-self.delta)/np.exp(self.eps))**(1/numhyp))))/self.eps        

        for k in range(0, numhyp):
            
            Lap_p = np.random.laplace(loc=0.0,scale=4*self.sensitivity*self.cutoff/self.eps)

            # Get candidate and rejection indicators
            this_alpha = self.alpha[k + 1]
            candidates[k + 1] = (pvec[k] < 2*this_alpha)
            if candidates[k + 1] == True:
                rej[k + 1] = (np.log(pvec[k])+Lap_p < np.log(this_alpha)+Lap_T-A)
            else:
                rej[k + 1] = False

            # Check rejection
            if (rej[k + 1] == 1):
                last_rej = np.append(last_rej, k + 1).astype(int)
                Lap_T = np.random.laplace(loc=0.0,scale=2*self.sensitivity*self.cutoff/self.eps) #reset noise for threshold after each rejection
                totalrej=totalrej+1
                if totalrej>self.cutoff:
                   break

            if len(last_rej) == 1 and (k + 1) == last_rej[0]:
                flag = 1

            # Update wealth
            wealth = self.wealth_vec[k] - (1 - candidates[k + 1]) * this_alpha + rej[k + 1] * (1 - 2*this_alpha) * (
                self.alpha0) - rej[k + 1] * flag * self.w0
            self.wealth_vec[k + 1] = wealth

            if len(last_rej) == 1 and (k + 1) == last_rej[0]:
                flag = 0


            candidates_total = sum(candidates)
            zero_gam = self.gamma_vec[k + 1 - (int)(candidates_total)]
            # Update alpha_t
            if len(last_rej) > 0:
                if last_rej[0]<= (k+1):
                    candidates_after_first = sum(candidates[last_rej[0]+1:])
                    first_gam = self.gamma_vec[k + 1 - (last_rej[0]) - (int)(candidates_after_first)]
                else:
                    first_gam = 0
                if len(last_rej) >= 2:
                    sum_gam = self.gamma_vec[(k + 1) * np.ones(len(last_rej)-1, dtype=int) - (last_rej[1:]) - self.count_candidates(last_rej, candidates)]
                    sum_gam = sum(sum_gam)
                else:
                    sum_gam = 0
                s = zero_gam * self.w0 + (self.alpha0 - self.w0) * first_gam + self.alpha0 * sum_gam
                next_alpha = s / (1 + 2*s)
            else:
                next_alpha = zero_gam * self.w0 / (1 + 2*zero_gam * self.w0)
            if k < numhyp - 1:
                self.alpha[k + 2] = next_alpha



        rej = rej[1:]
        self.alpha = self.alpha[1:]
        return rej

