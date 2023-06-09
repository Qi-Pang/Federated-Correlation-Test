### Import Python libraries
import numpy as np
np.set_printoptions(precision = 4)

import sys


### Import utilities for plotting
from plotting import*
from settings_util import*
from toimport import*

# Plot_styles:
# 0: wealth, alpha_k over time
# 1: power/FDR over pi1
# 2: varying espilon
# 3: varying shift magnitude
# 4: varying signal in the truncated exponential distribution example


def plot_results(plot_style, whichrun, FDRrange, pirange, hyprange, mu_gap,lbd_scale, sigma, NUMHYP, num_runs, mod_choice, eps, epsrange, sensitivity,shift,shiftrange,lbdrange, NUMDRAWS = 1):

    plot_dirname = './plots'
    numrun = 100000

    #%%%%%%%%%%%%%%%%%%%%  PLOTS vs. Hyp (time)  %%%%%%%%%%%%%%%%%%%%%%

    if plot_style == 0:

        numFDR = len(FDRrange)
        
        pi = pirange[3]

        # ----------- LOAD DATA --------
        FDR_mat = [None]*len(FDRrange)
        wealth_mat = [None]*len(FDRrange)
        TDR_mat = [None]*len(FDRrange)
        alpha_mat = [None]*len(FDRrange)

        for FDR_j, FDR in enumerate(FDRrange):


            filename_pre = 'Lbd%.2f_Si%.1f_FDR%d_NH%d_ND%d_MOD%d_EPS%.1f_SE%.1f_SHT%.1f_PM%.2f_NR%d' % (lbd_scale, sigma, FDR, NUMHYP, NUMDRAWS, mod_choice,eps, sensitivity,shift, pi, num_runs)
            all_filenames = [filename for filename in os.listdir('./dat') if filename.startswith(filename_pre)]

            if all_filenames == []:
                print("No files found!")
                print(filename_pre)
                sys.exit()

            # Load results
            result_mat = np.loadtxt('./dat/%s' % all_filenames[0])
            FDR_vec = result_mat[0:NUMHYP, whichrun]
            rej_vec = result_mat[1*NUMHYP:2*NUMHYP, whichrun]
            falrej_vec = result_mat[2*NUMHYP:3*NUMHYP, whichrun]
            wealth_vec = result_mat[3*NUMHYP:4*NUMHYP, whichrun]
            alpha_vec = result_mat[5*NUMHYP:6*NUMHYP, whichrun]

            # Get true Hypo vector
            Hypo = get_hyp(pi, NUMHYP)
            Hypo = Hypo.astype(int)

            # Save to matrix
            FDR_mat[FDR_j] = FDR_vec
            wealth_mat[FDR_j] = wealth_vec
            # TDR_mat[FDR_j] = TDR_vec
            alpha_mat[FDR_j] = alpha_vec
        
        # -------- PLOT ---------------
        # Set x axis
        if len(hyprange) == 1:
            xs = range(NUMHYP)
            hyplen = NUMHYP
        else:
            # Cut the matrices
            xs = hyprange
            hyplen = len(hyprange)
            FDR_mat = np.array(FDR_mat)[:,0:len(hyprange)]
            wealth_mat = np.array(wealth_mat)[:,0:len(hyprange)]
            alpha_mat = np.array(alpha_mat)[:,0:len(hyprange)]
        
        legends_list = np.array(proc_list).take([f - 1 for f in FDRrange])
        #[0:numFDR]


        leg_col = 1

        #### Wealth vs HYP ####
        filename = 'WealthvsHP_Lbd%.2f_Si%.1f_NH%d_ND%d_PM%.2f_HR%d_R%d_MOD%d_EPS%.1f_SE%.1f_SHT%.1f' %  (lbd_scale, sigma, NUMHYP, NUMDRAWS, pi, hyplen, whichrun, mod_choice, eps, sensitivity,shift)
        plot_curves_mat(xs, wealth_mat, legends_list, plot_dirname, filename,  'Hypothesis index', 'Wealth($J$)', 0, leg_col = leg_col)

        #### alpha vs. HYP ####
        filename = 'alphavsHP_Lbd%.2f_Si%.1f_NH%d_ND%d_PM%.2f_HR%d_R%d_MOD%d_EPS%.1f_SE%.1f_SHT%.1f' %  (lbd_scale, sigma, NUMHYP, NUMDRAWS, pi, hyplen, whichrun, mod_choice, eps, sensitivity,shift)
        plot_curves_mat(xs, alpha_mat, legends_list, plot_dirname, filename,  'Hypothesis index', '$alpha(J)$', 0, leg_col = leg_col)

        #%%%%%%%%%%%%%%%%%%%  PLOTS vs. pi1 %%%%%%%%%%%%%%%%%%%%%%%%%%
        
    elif plot_style == 1:

        TDR_av = []
        TDR_std = []
        FDR_av = []
        FDR_std = []
        ind = 0
        
        for index, FDR in enumerate(FDRrange):


            filename_pre = 'Lbd%.2f_Si%.1f_FDR%d_NH%d_ND%d_MOD%d_EPS%.1f_SE%.1f_SHT%.1f' % (lbd_scale, sigma, FDR, NUMHYP, NUMDRAWS,mod_choice, eps, sensitivity,shift)
            all_filenames = [filename for filename in os.listdir('./dat') if filename.startswith(filename_pre)]

            if all_filenames == []:
                print("No file found!")
                print(filename_pre)
                sys.exit()

            # Get different pis
            pos_PM_start = [all_filenames[i].index('PM') for i in range(len(all_filenames))]
            pos_PM_end = [all_filenames[i].index('_NR') for i in range(len(all_filenames))]
            PM_vec = [float(all_filenames[i][pos_PM_start[i] + 2:pos_PM_end[i]]) for i in range(len(all_filenames))]

            order = np.argsort(PM_vec)
            PM_list = sorted(set(np.array(PM_vec)[order]))

            # Initialize result matrices
            TDR_av.append(np.zeros([1, len(PM_list)]))
            TDR_std.append(np.zeros([1, len(PM_list)]))
            FDR_av.append(np.zeros([1, len(PM_list)]))
            FDR_std.append(np.zeros([1, len(PM_list)]))
            TDR_vec = np.zeros(len(PM_list))
            FDR_vec = np.zeros(len(PM_list))
            TDR_vec_std = np.zeros(len(PM_list))
            FDR_vec_std = np.zeros(len(PM_list))

            # Merge everything with the same NA and NH
            for k, PM in enumerate(PM_list):
                indices = np.where(np.array(PM_vec) == PM)[0]
                result_mat = []
                # Load resultmats and append
                for j, idx in enumerate(indices):
                    result_mat_cache = np.loadtxt('./dat/%s' % all_filenames[idx])
                    result_mat_cache = result_mat_cache[6*NUMHYP:6*NUMHYP+2,0:200]
                    if (j == 0):
                        result_mat = result_mat_cache
                    else:
                        result_mat = np.c_[result_mat, result_mat_cache]

                numrun = len(result_mat[0])
                # Get first vector for TDR
                TDR_vec[k] = np.average(result_mat[0])
                TDR_vec_std[k] = np.true_divide(np.std(result_mat[0]),np.sqrt(numrun))
                # FDR
                FDR_vec[k] = np.average(result_mat[1])
                FDR_vec_std[k] = np.true_divide(np.std(result_mat[1]), np.sqrt(numrun))
            TDR_av[ind] = [TDR_vec[k] for k in range(len(PM_list))]
            TDR_std[ind] = [TDR_vec_std[k] for k in range(len(PM_list))]
            FDR_av[ind] = [FDR_vec[k] for k in range(len(PM_list))]
            FDR_std[ind] = [FDR_vec_std[k] for k in range(len(PM_list))]

            ind = ind + 1



        # -------- PLOT ---------------
        xs = PM_list
        x_label = '$\pi_1$'

        # Create legend
        legends_list = np.array(proc_list).take([f - 1 for f in FDRrange])

        ##### FDR vs pi #####

        filename = 'FDRvsPI_Lbd%.2f_Si%.1f_NH%d_ND%d_MOD%d_EPS%.1f_SE%.1f_SHT%.1f' %  (lbd_scale, sigma, NUMHYP, NUMDRAWS, mod_choice, eps, sensitivity,shift)
        plot_errors_mat(xs, FDR_av, FDR_std, legends_list, plot_dirname, filename, x_label, 'FDR')

        ##### TDR vs pi ####
        filename = 'PowervsPI_Lbd%.2f_Si%.1f_NH%d_ND%d_MOD%d_EPS%.1f_SE%.1f_SHT%.1f' %  (lbd_scale, sigma, NUMHYP, NUMDRAWS, mod_choice, eps, sensitivity,shift)
        plot_errors_mat(xs, TDR_av, TDR_std, legends_list, plot_dirname, filename, x_label, 'Power')

    elif plot_style == 2:
        plot_dirname = './plots'
        numrun = 100000
        FDR=FDRrange[0]

        TDR_av = []
        TDR_std = []
        FDR_av = []
        FDR_std = []
        ind = 0
        
        for index, epsilon in enumerate(epsrange):


            filename_pre = 'Lbd%.2f_Si%.1f_FDR%d_NH%d_ND%d_MOD%d_EPS%.1f_SE%.1f_SHT%.1f' % (lbd_scale, sigma, FDR, NUMHYP, NUMDRAWS,mod_choice, epsilon , sensitivity,shift)
            all_filenames = [filename for filename in os.listdir('./dat') if filename.startswith(filename_pre)]

            if all_filenames == []:
                print("No file found!")
                print(filename_pre)
                sys.exit()

            # Get different pis
            pos_PM_start = [all_filenames[i].index('PM') for i in range(len(all_filenames))]
            pos_PM_end = [all_filenames[i].index('_NR') for i in range(len(all_filenames))]
            PM_vec = [float(all_filenames[i][pos_PM_start[i] + 2:pos_PM_end[i]]) for i in range(len(all_filenames))]

            order = np.argsort(PM_vec)
            PM_list = sorted(set(np.array(PM_vec)[order]))

            # Initialize result matrices
            TDR_av.append(np.zeros([1, len(PM_list)]))
            TDR_std.append(np.zeros([1, len(PM_list)]))
            FDR_av.append(np.zeros([1, len(PM_list)]))
            FDR_std.append(np.zeros([1, len(PM_list)]))
            TDR_vec = np.zeros(len(PM_list))
            FDR_vec = np.zeros(len(PM_list))
            TDR_vec_std = np.zeros(len(PM_list))
            FDR_vec_std = np.zeros(len(PM_list))

            # Merge everything with the same NA and NH
            for k, PM in enumerate(PM_list):
                indices = np.where(np.array(PM_vec) == PM)[0]
                result_mat = []
                # Load resultmats and append
                for j, idx in enumerate(indices):
                    result_mat_cache = np.loadtxt('./dat/%s' % all_filenames[idx])
                    result_mat_cache = result_mat_cache[6*NUMHYP:6*NUMHYP+2,0:200]
                    if (j == 0):
                        result_mat = result_mat_cache
                    else:
                        result_mat = np.c_[result_mat, result_mat_cache]

                numrun = len(result_mat[0])
                # Get first vector for TDR
                TDR_vec[k] = np.average(result_mat[0])
                TDR_vec_std[k] = np.true_divide(np.std(result_mat[0]),np.sqrt(numrun))
                # FDR
                FDR_vec[k] = np.average(result_mat[1])
                FDR_vec_std[k] = np.true_divide(np.std(result_mat[1]), np.sqrt(numrun))
            TDR_av[ind] = [TDR_vec[k] for k in range(len(PM_list))]
            TDR_std[ind] = [TDR_vec_std[k] for k in range(len(PM_list))]
            FDR_av[ind] = [FDR_vec[k] for k in range(len(PM_list))]
            FDR_std[ind] = [FDR_vec_std[k] for k in range(len(PM_list))]

            ind = ind + 1



        # -------- PLOT ---------------
        xs = PM_list
        x_label = '$\pi_1$'

        # Create legend
        legends_list = np.array(['EPS = 3', 'EPS = 5', 'EPS = 10'])

        ##### FDR vs pi #####

        filename = 'FDRvsPI_NH%d_ND%d_MOD%d_SHT%.1f_FDR%d' %  (NUMHYP, NUMDRAWS, mod_choice,shift,FDR)
        plot_errors_mat(xs, FDR_av, FDR_std, legends_list, plot_dirname, filename, x_label, 'FDR')

        ##### TDR vs pi ####
        filename = 'PowervsPI_NH%d_ND%d_MOD%d_SHT%.1f_FDR%d' %  (NUMHYP, NUMDRAWS, mod_choice, shift,FDR)
        plot_errors_mat(xs, TDR_av, TDR_std, legends_list, plot_dirname, filename, x_label, 'Power')

    elif plot_style == 3:
        plot_dirname = './plots'
        numrun = 100000
        FDR=FDRrange[0]

        TDR_av = []
        TDR_std = []
        FDR_av = []
        FDR_std = []
        ind = 0
        
        for index, s in enumerate(shiftrange):


            filename_pre = 'Lbd%.2f_Si%.1f_FDR%d_NH%d_ND%d_MOD%d_EPS%.1f_SE%.1f_SHT%.1f' % (lbd_scale, sigma, FDR, NUMHYP, NUMDRAWS, mod_choice, eps , sensitivity, s)
            all_filenames = [filename for filename in os.listdir('./dat') if filename.startswith(filename_pre)]

            if all_filenames == []:
                print("No file found!")
                print(filename_pre)
                sys.exit()

            # Get different pis
            pos_PM_start = [all_filenames[i].index('PM') for i in range(len(all_filenames))]
            pos_PM_end = [all_filenames[i].index('_NR') for i in range(len(all_filenames))]
            PM_vec = [float(all_filenames[i][pos_PM_start[i] + 2:pos_PM_end[i]]) for i in range(len(all_filenames))]

            order = np.argsort(PM_vec)
            PM_list = sorted(set(np.array(PM_vec)[order]))

            # Initialize result matrices
            TDR_av.append(np.zeros([1, len(PM_list)]))
            TDR_std.append(np.zeros([1, len(PM_list)]))
            FDR_av.append(np.zeros([1, len(PM_list)]))
            FDR_std.append(np.zeros([1, len(PM_list)]))
            TDR_vec = np.zeros(len(PM_list))
            FDR_vec = np.zeros(len(PM_list))
            TDR_vec_std = np.zeros(len(PM_list))
            FDR_vec_std = np.zeros(len(PM_list))

            # Merge everything with the same NA and NH
            for k, PM in enumerate(PM_list):
                indices = np.where(np.array(PM_vec) == PM)[0]
                result_mat = []
                # Load resultmats and append
                for j, idx in enumerate(indices):
                    result_mat_cache = np.loadtxt('./dat/%s' % all_filenames[idx])
                    result_mat_cache = result_mat_cache[6*NUMHYP:6*NUMHYP+2,0:200]
                    if (j == 0):
                        result_mat = result_mat_cache
                    else:
                        result_mat = np.c_[result_mat, result_mat_cache]

                numrun = len(result_mat[0])
                # Get first vector for TDR
                TDR_vec[k] = np.average(result_mat[0])
                TDR_vec_std[k] = np.true_divide(np.std(result_mat[0]),np.sqrt(numrun))
                # FDR
                FDR_vec[k] = np.average(result_mat[1])
                FDR_vec_std[k] = np.true_divide(np.std(result_mat[1]), np.sqrt(numrun))
            TDR_av[ind] = [TDR_vec[k] for k in range(len(PM_list))]
            TDR_std[ind] = [TDR_vec_std[k] for k in range(len(PM_list))]
            FDR_av[ind] = [FDR_vec[k] for k in range(len(PM_list))]
            FDR_std[ind] = [FDR_vec_std[k] for k in range(len(PM_list))]

            ind = ind + 1



        # -------- PLOT ---------------
        xs = PM_list
        x_label = '$\pi_1$'

        # Create legend
        legends_list = np.array(['s=0.5','s=1','s=1.5','s=2'])

        ##### FDR vs pi #####

        filename = 'FDRvsPI_NH%d_ND%d_MOD%d_EPS%.1f_FDR%d' %  (NUMHYP, NUMDRAWS, mod_choice,eps,FDR)
        plot_errors_mat(xs, FDR_av, FDR_std, legends_list, plot_dirname, filename, x_label, 'FDR')

        ##### TDR vs pi ####
        filename = 'PowervsPI_NH%d_ND%d_MOD%d_EPS%.1f_FDR%d' %  (NUMHYP, NUMDRAWS, mod_choice, eps,FDR)
        plot_errors_mat(xs, TDR_av, TDR_std, legends_list, plot_dirname, filename, x_label, 'Power')
        
    elif plot_style == 4:
        plot_dirname = './plots'
        numrun = 100000
        FDR=FDRrange[0]

        TDR_av = []
        TDR_std = []
        FDR_av = []
        FDR_std = []
        ind = 0
        
        for index, lbdscale in enumerate(lbdrange):


            filename_pre = 'Lbd%.2f_Si%.1f_FDR%d_NH%d_ND%d_MOD%d_EPS%.1f_SE%.1f_SHT%.1f' % (lbdscale, sigma, FDR, NUMHYP, NUMDRAWS,mod_choice, eps , sensitivity,shift)
            all_filenames = [filename for filename in os.listdir('./dat') if filename.startswith(filename_pre)]

            if all_filenames == []:
                print("No file found!")
                print(filename_pre)
                sys.exit()

            # Get different pis
            pos_PM_start = [all_filenames[i].index('PM') for i in range(len(all_filenames))]
            pos_PM_end = [all_filenames[i].index('_NR') for i in range(len(all_filenames))]
            PM_vec = [float(all_filenames[i][pos_PM_start[i] + 2:pos_PM_end[i]]) for i in range(len(all_filenames))]

            order = np.argsort(PM_vec)
            PM_list = sorted(set(np.array(PM_vec)[order]))

            # Initialize result matrices
            TDR_av.append(np.zeros([1, len(PM_list)]))
            TDR_std.append(np.zeros([1, len(PM_list)]))
            FDR_av.append(np.zeros([1, len(PM_list)]))
            FDR_std.append(np.zeros([1, len(PM_list)]))
            TDR_vec = np.zeros(len(PM_list))
            FDR_vec = np.zeros(len(PM_list))
            TDR_vec_std = np.zeros(len(PM_list))
            FDR_vec_std = np.zeros(len(PM_list))

            # Merge everything with the same NA and NH
            for k, PM in enumerate(PM_list):
                indices = np.where(np.array(PM_vec) == PM)[0]
                result_mat = []
                # Load resultmats and append
                for j, idx in enumerate(indices):
                    result_mat_cache = np.loadtxt('./dat/%s' % all_filenames[idx])
                    result_mat_cache = result_mat_cache[6*NUMHYP:6*NUMHYP+2,0:200]
                    if (j == 0):
                        result_mat = result_mat_cache
                    else:
                        result_mat = np.c_[result_mat, result_mat_cache]

                numrun = len(result_mat[0])
                # Get first vector for TDR
                TDR_vec[k] = np.average(result_mat[0])
                TDR_vec_std[k] = np.true_divide(np.std(result_mat[0]),np.sqrt(numrun))
                # FDR
                FDR_vec[k] = np.average(result_mat[1])
                FDR_vec_std[k] = np.true_divide(np.std(result_mat[1]), np.sqrt(numrun))
            TDR_av[ind] = [TDR_vec[k] for k in range(len(PM_list))]
            TDR_std[ind] = [TDR_vec_std[k] for k in range(len(PM_list))]
            FDR_av[ind] = [FDR_vec[k] for k in range(len(PM_list))]
            FDR_std[ind] = [FDR_vec_std[k] for k in range(len(PM_list))]

            ind = ind + 1



        # -------- PLOT ---------------
        xs = PM_list
        x_label = '$\pi_1$'

        # Create legend
        legends_list = np.array([r'$\theta_i=1.80$',r'$\theta_i=1.90$',r'$\theta_i=2.00$'])

        ##### FDR vs pi #####

        filename = 'FDRvsPI_NH%d_ND%d_MOD%d_EPS%.1f_FDR%d' %  (NUMHYP, NUMDRAWS, mod_choice,eps,FDR)
        plot_errors_mat(xs, FDR_av, FDR_std, legends_list, plot_dirname, filename, x_label, 'FDR')

        ##### TDR vs pi ####
        filename = 'PowervsPI_NH%d_ND%d_MOD%d_EPS%.1f_FDR%d' %  (NUMHYP, NUMDRAWS, mod_choice, eps,FDR)
        plot_errors_mat(xs, TDR_av, TDR_std, legends_list, plot_dirname, filename, x_label, 'Power')   


def just_plot_results(plot_style, whichrun, FDRrange, pirange, hyprange, mu_gap,lbd_scale, sigma, NUMHYP, num_runs, mod_choice, eps, epsrange, sensitivity,shift,shiftrange,lbdrange, NUMDRAWS = 1):

    plot_dirname = './plots'
    numrun = 100000

    #%%%%%%%%%%%%%%%%%%%%  PLOTS vs. Hyp (time)  %%%%%%%%%%%%%%%%%%%%%%

    if plot_style == 0:

        numFDR = len(FDRrange)
        
        pi = pirange[3]

        # ----------- LOAD DATA --------
        FDR_mat = [None]*len(FDRrange)
        wealth_mat = [None]*len(FDRrange)
        TDR_mat = [None]*len(FDRrange)
        alpha_mat = [None]*len(FDRrange)

        for FDR_j, FDR in enumerate(FDRrange):


            # filename_pre = 'Lbd%.2f_Si%.1f_FDR%d_NH%d_ND%d_MOD%d_EPS%.1f_SE%.1f_SHT%.1f_PM%.2f_NR%d' % (lbd_scale, sigma, FDR, NUMHYP, NUMDRAWS, mod_choice,eps, sensitivity,shift, pi, num_runs)
            filename_pre = 'federated%d' % (50)
            all_filenames = [filename for filename in os.listdir('./dat') if filename.startswith(filename_pre)]

            if all_filenames == []:
                print("No files found!")
                print(filename_pre)
                sys.exit()

            # Load results
            result_mat = np.loadtxt('./dat/%s' % all_filenames[0])
            FDR_vec = result_mat[0:NUMHYP, whichrun]
            rej_vec = result_mat[1*NUMHYP:2*NUMHYP, whichrun]
            falrej_vec = result_mat[2*NUMHYP:3*NUMHYP, whichrun]
            wealth_vec = result_mat[3*NUMHYP:4*NUMHYP, whichrun]
            alpha_vec = result_mat[5*NUMHYP:6*NUMHYP, whichrun]

            # Get true Hypo vector
            Hypo = get_hyp(pi, NUMHYP)
            Hypo = Hypo.astype(int)

            # Save to matrix
            FDR_mat[FDR_j] = FDR_vec
            wealth_mat[FDR_j] = wealth_vec
            # TDR_mat[FDR_j] = TDR_vec
            alpha_mat[FDR_j] = alpha_vec
        
        # -------- PLOT ---------------
        # Set x axis
        if len(hyprange) == 1:
            xs = range(NUMHYP)
            hyplen = NUMHYP
        else:
            # Cut the matrices
            xs = hyprange
            hyplen = len(hyprange)
            FDR_mat = np.array(FDR_mat)[:,0:len(hyprange)]
            wealth_mat = np.array(wealth_mat)[:,0:len(hyprange)]
            alpha_mat = np.array(alpha_mat)[:,0:len(hyprange)]
        
        legends_list = np.array(proc_list).take([f - 1 for f in FDRrange])
        #[0:numFDR]


        leg_col = 1

        #### Wealth vs HYP ####
        filename = 'WealthvsHP_Lbd%.2f_Si%.1f_NH%d_ND%d_PM%.2f_HR%d_R%d_MOD%d_EPS%.1f_SE%.1f_SHT%.1f' %  (lbd_scale, sigma, NUMHYP, NUMDRAWS, pi, hyplen, whichrun, mod_choice, eps, sensitivity,shift)
        plot_curves_mat(xs, wealth_mat, legends_list, plot_dirname, filename,  'Hypothesis index', 'Wealth($J$)', 0, leg_col = leg_col)

        #### alpha vs. HYP ####
        filename = 'alphavsHP_Lbd%.2f_Si%.1f_NH%d_ND%d_PM%.2f_HR%d_R%d_MOD%d_EPS%.1f_SE%.1f_SHT%.1f' %  (lbd_scale, sigma, NUMHYP, NUMDRAWS, pi, hyplen, whichrun, mod_choice, eps, sensitivity,shift)
        plot_curves_mat(xs, alpha_mat, legends_list, plot_dirname, filename,  'Hypothesis index', '$alpha(J)$', 0, leg_col = leg_col)

        #%%%%%%%%%%%%%%%%%%%  PLOTS vs. pi1 %%%%%%%%%%%%%%%%%%%%%%%%%%
        
    elif plot_style == 1:

        TDR_av = []
        TDR_std = []
        FDR_av = []
        FDR_std = []
        ind = 0
        
        for index, FDR in enumerate(FDRrange):

            # FIXME: modify here
            filename_pre = 'federated%d' % (100)
            all_filenames = [filename for filename in os.listdir('./dat') if filename.startswith(filename_pre)]

            if all_filenames == []:
                print("No file found!")
                print(filename_pre)
                sys.exit()

            # Get different pis
            # pos_PM_start = [all_filenames[i].index('PM') for i in range(len(all_filenames))]
            # pos_PM_end = [all_filenames[i].index('_NR') for i in range(len(all_filenames))]
            PM_vec = [0.1]

            order = np.argsort(PM_vec)
            PM_list = sorted(set(np.array(PM_vec)[order]))

            # Initialize result matrices
            TDR_av.append(np.zeros([1, len(PM_list)]))
            TDR_std.append(np.zeros([1, len(PM_list)]))
            FDR_av.append(np.zeros([1, len(PM_list)]))
            FDR_std.append(np.zeros([1, len(PM_list)]))
            TDR_vec = np.zeros(len(PM_list))
            FDR_vec = np.zeros(len(PM_list))
            TDR_vec_std = np.zeros(len(PM_list))
            FDR_vec_std = np.zeros(len(PM_list))

            # Merge everything with the same NA and NH
            for k, PM in enumerate(PM_list):
                indices = np.where(np.array(PM_vec) == PM)[0]
                result_mat = []
                # Load resultmats and append
                for j, idx in enumerate(indices):
                    # FIXME: modify here
                    result_mat_cache = np.loadtxt('./dat/%s' % all_filenames[idx])
                    result_mat_cache = result_mat_cache[6*NUMHYP:6*NUMHYP+2,0:200]
                    if (j == 0):
                        result_mat = result_mat_cache
                    else:
                        result_mat = np.c_[result_mat, result_mat_cache]

                numrun = len(result_mat[0])
                # Get first vector for TDR
                TDR_vec[k] = np.average(result_mat[0])
                TDR_vec_std[k] = np.true_divide(np.std(result_mat[0]),np.sqrt(numrun))
                # FDR
                FDR_vec[k] = np.average(result_mat[1])
                FDR_vec_std[k] = np.true_divide(np.std(result_mat[1]), np.sqrt(numrun))
            TDR_av[ind] = [TDR_vec[k] for k in range(len(PM_list))]
            TDR_std[ind] = [TDR_vec_std[k] for k in range(len(PM_list))]
            FDR_av[ind] = [FDR_vec[k] for k in range(len(PM_list))]
            FDR_std[ind] = [FDR_vec_std[k] for k in range(len(PM_list))]

            ind = ind + 1



        # -------- PLOT ---------------
        xs = PM_list
        x_label = '$\pi_1$'

        # Create legend
        legends_list = np.array(proc_list).take([f - 1 for f in FDRrange])

        ##### FDR vs pi #####

        filename = 'FDRvsPI_Lbd%.2f_Si%.1f_NH%d_ND%d_MOD%d_EPS%.1f_SE%.1f_SHT%.1f' %  (lbd_scale, sigma, NUMHYP, NUMDRAWS, mod_choice, eps, sensitivity,shift)
        plot_errors_mat(xs, FDR_av, FDR_std, legends_list, plot_dirname, filename, x_label, 'FDR')

        ##### TDR vs pi ####
        filename = 'PowervsPI_Lbd%.2f_Si%.1f_NH%d_ND%d_MOD%d_EPS%.1f_SE%.1f_SHT%.1f' %  (lbd_scale, sigma, NUMHYP, NUMDRAWS, mod_choice, eps, sensitivity,shift)
        plot_errors_mat(xs, TDR_av, TDR_std, legends_list, plot_dirname, filename, x_label, 'Power')

    elif plot_style == 2:
        plot_dirname = './plots'
        numrun = 100000
        FDR=FDRrange[0]

        TDR_av = []
        TDR_std = []
        FDR_av = []
        FDR_std = []
        ind = 0
        
        for index, epsilon in enumerate(epsrange):


            filename_pre = 'Lbd%.2f_Si%.1f_FDR%d_NH%d_ND%d_MOD%d_EPS%.1f_SE%.1f_SHT%.1f' % (lbd_scale, sigma, FDR, NUMHYP, NUMDRAWS,mod_choice, epsilon , sensitivity,shift)
            all_filenames = [filename for filename in os.listdir('./dat') if filename.startswith(filename_pre)]

            if all_filenames == []:
                print("No file found!")
                print(filename_pre)
                sys.exit()

            # Get different pis
            pos_PM_start = [all_filenames[i].index('PM') for i in range(len(all_filenames))]
            pos_PM_end = [all_filenames[i].index('_NR') for i in range(len(all_filenames))]
            PM_vec = [float(all_filenames[i][pos_PM_start[i] + 2:pos_PM_end[i]]) for i in range(len(all_filenames))]

            order = np.argsort(PM_vec)
            PM_list = sorted(set(np.array(PM_vec)[order]))

            # Initialize result matrices
            TDR_av.append(np.zeros([1, len(PM_list)]))
            TDR_std.append(np.zeros([1, len(PM_list)]))
            FDR_av.append(np.zeros([1, len(PM_list)]))
            FDR_std.append(np.zeros([1, len(PM_list)]))
            TDR_vec = np.zeros(len(PM_list))
            FDR_vec = np.zeros(len(PM_list))
            TDR_vec_std = np.zeros(len(PM_list))
            FDR_vec_std = np.zeros(len(PM_list))

            # Merge everything with the same NA and NH
            for k, PM in enumerate(PM_list):
                indices = np.where(np.array(PM_vec) == PM)[0]
                result_mat = []
                # Load resultmats and append
                for j, idx in enumerate(indices):
                    result_mat_cache = np.loadtxt('./dat/%s' % all_filenames[idx])
                    result_mat_cache = result_mat_cache[6*NUMHYP:6*NUMHYP+2,0:200]
                    if (j == 0):
                        result_mat = result_mat_cache
                    else:
                        result_mat = np.c_[result_mat, result_mat_cache]

                numrun = len(result_mat[0])
                # Get first vector for TDR
                TDR_vec[k] = np.average(result_mat[0])
                TDR_vec_std[k] = np.true_divide(np.std(result_mat[0]),np.sqrt(numrun))
                # FDR
                FDR_vec[k] = np.average(result_mat[1])
                FDR_vec_std[k] = np.true_divide(np.std(result_mat[1]), np.sqrt(numrun))
            TDR_av[ind] = [TDR_vec[k] for k in range(len(PM_list))]
            TDR_std[ind] = [TDR_vec_std[k] for k in range(len(PM_list))]
            FDR_av[ind] = [FDR_vec[k] for k in range(len(PM_list))]
            FDR_std[ind] = [FDR_vec_std[k] for k in range(len(PM_list))]

            ind = ind + 1



        # -------- PLOT ---------------
        xs = PM_list
        x_label = '$\pi_1$'

        # Create legend
        legends_list = np.array(['EPS = 3', 'EPS = 5', 'EPS = 10'])

        ##### FDR vs pi #####

        filename = 'FDRvsPI_NH%d_ND%d_MOD%d_SHT%.1f_FDR%d' %  (NUMHYP, NUMDRAWS, mod_choice,shift,FDR)
        plot_errors_mat(xs, FDR_av, FDR_std, legends_list, plot_dirname, filename, x_label, 'FDR')

        ##### TDR vs pi ####
        filename = 'PowervsPI_NH%d_ND%d_MOD%d_SHT%.1f_FDR%d' %  (NUMHYP, NUMDRAWS, mod_choice, shift,FDR)
        plot_errors_mat(xs, TDR_av, TDR_std, legends_list, plot_dirname, filename, x_label, 'Power')

    elif plot_style == 3:
        plot_dirname = './plots'
        numrun = 100000
        FDR=FDRrange[0]

        TDR_av = []
        TDR_std = []
        FDR_av = []
        FDR_std = []
        ind = 0
        
        for index, s in enumerate(shiftrange):


            filename_pre = 'Lbd%.2f_Si%.1f_FDR%d_NH%d_ND%d_MOD%d_EPS%.1f_SE%.1f_SHT%.1f' % (lbd_scale, sigma, FDR, NUMHYP, NUMDRAWS, mod_choice, eps , sensitivity, s)
            all_filenames = [filename for filename in os.listdir('./dat') if filename.startswith(filename_pre)]

            if all_filenames == []:
                print("No file found!")
                print(filename_pre)
                sys.exit()

            # Get different pis
            pos_PM_start = [all_filenames[i].index('PM') for i in range(len(all_filenames))]
            pos_PM_end = [all_filenames[i].index('_NR') for i in range(len(all_filenames))]
            PM_vec = [float(all_filenames[i][pos_PM_start[i] + 2:pos_PM_end[i]]) for i in range(len(all_filenames))]

            order = np.argsort(PM_vec)
            PM_list = sorted(set(np.array(PM_vec)[order]))

            # Initialize result matrices
            TDR_av.append(np.zeros([1, len(PM_list)]))
            TDR_std.append(np.zeros([1, len(PM_list)]))
            FDR_av.append(np.zeros([1, len(PM_list)]))
            FDR_std.append(np.zeros([1, len(PM_list)]))
            TDR_vec = np.zeros(len(PM_list))
            FDR_vec = np.zeros(len(PM_list))
            TDR_vec_std = np.zeros(len(PM_list))
            FDR_vec_std = np.zeros(len(PM_list))

            # Merge everything with the same NA and NH
            for k, PM in enumerate(PM_list):
                indices = np.where(np.array(PM_vec) == PM)[0]
                result_mat = []
                # Load resultmats and append
                for j, idx in enumerate(indices):
                    result_mat_cache = np.loadtxt('./dat/%s' % all_filenames[idx])
                    result_mat_cache = result_mat_cache[6*NUMHYP:6*NUMHYP+2,0:200]
                    if (j == 0):
                        result_mat = result_mat_cache
                    else:
                        result_mat = np.c_[result_mat, result_mat_cache]

                numrun = len(result_mat[0])
                # Get first vector for TDR
                TDR_vec[k] = np.average(result_mat[0])
                TDR_vec_std[k] = np.true_divide(np.std(result_mat[0]),np.sqrt(numrun))
                # FDR
                FDR_vec[k] = np.average(result_mat[1])
                FDR_vec_std[k] = np.true_divide(np.std(result_mat[1]), np.sqrt(numrun))
            TDR_av[ind] = [TDR_vec[k] for k in range(len(PM_list))]
            TDR_std[ind] = [TDR_vec_std[k] for k in range(len(PM_list))]
            FDR_av[ind] = [FDR_vec[k] for k in range(len(PM_list))]
            FDR_std[ind] = [FDR_vec_std[k] for k in range(len(PM_list))]

            ind = ind + 1



        # -------- PLOT ---------------
        xs = PM_list
        x_label = '$\pi_1$'

        # Create legend
        legends_list = np.array(['s=0.5','s=1','s=1.5','s=2'])

        ##### FDR vs pi #####

        filename = 'FDRvsPI_NH%d_ND%d_MOD%d_EPS%.1f_FDR%d' %  (NUMHYP, NUMDRAWS, mod_choice,eps,FDR)
        plot_errors_mat(xs, FDR_av, FDR_std, legends_list, plot_dirname, filename, x_label, 'FDR')

        ##### TDR vs pi ####
        filename = 'PowervsPI_NH%d_ND%d_MOD%d_EPS%.1f_FDR%d' %  (NUMHYP, NUMDRAWS, mod_choice, eps,FDR)
        plot_errors_mat(xs, TDR_av, TDR_std, legends_list, plot_dirname, filename, x_label, 'Power')
        
    elif plot_style == 4:
        plot_dirname = './plots'
        numrun = 100000
        FDR=FDRrange[0]

        TDR_av = []
        TDR_std = []
        FDR_av = []
        FDR_std = []
        ind = 0
        
        for index, lbdscale in enumerate(lbdrange):


            filename_pre = 'Lbd%.2f_Si%.1f_FDR%d_NH%d_ND%d_MOD%d_EPS%.1f_SE%.1f_SHT%.1f' % (lbdscale, sigma, FDR, NUMHYP, NUMDRAWS,mod_choice, eps , sensitivity,shift)
            all_filenames = [filename for filename in os.listdir('./dat') if filename.startswith(filename_pre)]

            if all_filenames == []:
                print("No file found!")
                print(filename_pre)
                sys.exit()

            # Get different pis
            pos_PM_start = [all_filenames[i].index('PM') for i in range(len(all_filenames))]
            pos_PM_end = [all_filenames[i].index('_NR') for i in range(len(all_filenames))]
            PM_vec = [float(all_filenames[i][pos_PM_start[i] + 2:pos_PM_end[i]]) for i in range(len(all_filenames))]

            order = np.argsort(PM_vec)
            PM_list = sorted(set(np.array(PM_vec)[order]))

            # Initialize result matrices
            TDR_av.append(np.zeros([1, len(PM_list)]))
            TDR_std.append(np.zeros([1, len(PM_list)]))
            FDR_av.append(np.zeros([1, len(PM_list)]))
            FDR_std.append(np.zeros([1, len(PM_list)]))
            TDR_vec = np.zeros(len(PM_list))
            FDR_vec = np.zeros(len(PM_list))
            TDR_vec_std = np.zeros(len(PM_list))
            FDR_vec_std = np.zeros(len(PM_list))

            # Merge everything with the same NA and NH
            for k, PM in enumerate(PM_list):
                indices = np.where(np.array(PM_vec) == PM)[0]
                result_mat = []
                # Load resultmats and append
                for j, idx in enumerate(indices):
                    result_mat_cache = np.loadtxt('./dat/%s' % all_filenames[idx])
                    result_mat_cache = result_mat_cache[6*NUMHYP:6*NUMHYP+2,0:200]
                    if (j == 0):
                        result_mat = result_mat_cache
                    else:
                        result_mat = np.c_[result_mat, result_mat_cache]

                numrun = len(result_mat[0])
                # Get first vector for TDR
                TDR_vec[k] = np.average(result_mat[0])
                TDR_vec_std[k] = np.true_divide(np.std(result_mat[0]),np.sqrt(numrun))
                # FDR
                FDR_vec[k] = np.average(result_mat[1])
                FDR_vec_std[k] = np.true_divide(np.std(result_mat[1]), np.sqrt(numrun))
            TDR_av[ind] = [TDR_vec[k] for k in range(len(PM_list))]
            TDR_std[ind] = [TDR_vec_std[k] for k in range(len(PM_list))]
            FDR_av[ind] = [FDR_vec[k] for k in range(len(PM_list))]
            FDR_std[ind] = [FDR_vec_std[k] for k in range(len(PM_list))]

            ind = ind + 1



        # -------- PLOT ---------------
        xs = PM_list
        x_label = '$\pi_1$'

        # Create legend
        legends_list = np.array([r'$\theta_i=1.80$',r'$\theta_i=1.90$',r'$\theta_i=2.00$'])

        ##### FDR vs pi #####

        filename = 'FDRvsPI_NH%d_ND%d_MOD%d_EPS%.1f_FDR%d' %  (NUMHYP, NUMDRAWS, mod_choice,eps,FDR)
        plot_errors_mat(xs, FDR_av, FDR_std, legends_list, plot_dirname, filename, x_label, 'FDR')

        ##### TDR vs pi ####
        filename = 'PowervsPI_NH%d_ND%d_MOD%d_EPS%.1f_FDR%d' %  (NUMHYP, NUMDRAWS, mod_choice, eps,FDR)
        plot_errors_mat(xs, TDR_av, TDR_std, legends_list, plot_dirname, filename, x_label, 'Power')