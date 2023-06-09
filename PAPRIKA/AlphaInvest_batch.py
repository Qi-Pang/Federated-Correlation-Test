import numpy as np

class ALPHA_proc_batch:
    
    def __init__(self, alpha0, numhyp, startfac):
        self.alpha0 = alpha0
        self.wealth_vec = np.zeros(numhyp+1)
        self.wealth_vec[0] = startfac*self.alpha0
        self.alpha = np.zeros(numhyp + 1)
        self.phi_fac = 0.25
        self.phi_k = self.phi_fac*self.wealth_vec[0]
        self.alpha[0:2] = [0, self.wealth_vec[0]/2]
        self.b0 = self.alpha0 - self.wealth_vec[0]


    def run_fdr(self, pvec):

        numhyp = len(pvec)
        last_rej = 0
        rej = np.zeros(numhyp+1)
  
        for k in range(0, numhyp-1):   

            if self.wealth_vec[k] > 0:
                # Get rejection indicator
                this_alpha = self.alpha[k + 1]
                rej[k + 1] = (pvec[k] < this_alpha)

                # Update wealth
                if (rej[k + 1] == 1):
                    last_rej = k + 1

                # Update wealth
                wealth = self.wealth_vec[k]  - (1-rej[k + 1])*this_alpha/(1-this_alpha)  + rej[k + 1]*self.b0
                self.wealth_vec[k + 1] = wealth
                # Update alpha_k
                next_alpha = min(wealth/(1 + k + 2 - last_rej), ((1-self.phi_fac)*wealth)/((1-self.phi_fac)*wealth+1))
                self.alpha[k + 2] = next_alpha    
            else:
                break

        self.alpha = self.alpha[1:]
        rej = rej[1:]
        return rej
