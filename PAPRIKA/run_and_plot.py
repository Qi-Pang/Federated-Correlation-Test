import logging, argparse
import numpy as np
from exp_FDR_batch_new import*
from plot_batch_results import*
from toimport import *

def main():

    if not os.path.exists('./dat'):
        os.makedirs('./dat')

    #########%%%%%%  SET PARAMETERS FOR RUNNING EXPERIMENT %%%%%%%##########

    FDRrange = str2list(args.FDRrange)
    pirange = str2list(args.pirange, 'float')
    epsrange = str2list(args.epsrange,'float')
    shiftrange = str2list(args.shiftrange,'float')
    lbdrange = str2list(args.lbdrange,'float')
    mu_gap = args.mu_gap
    hyprange = [0]

    ########%%%%%%%%%%%%%%%%% RUN EXPERIMENT %%%%%%%%########################
    if args.justplot == 0:
        for pi in pirange:
            # Run single FDR
            if args.plot_style < 2:
                for FDR in FDRrange:
                    # Prevent from running if data already exists
                    filename_pre = 'Lbd%.2f_Si1.0_FDR%d_NH%d_ND%d_MOD%d_EPS%.1f_SE%.1f_SHT%.1f_PM%.2f_NR%d_dropout%d' % (args.lbd_scale, FDR, args.num_hyp, 1, args.mod_choice, args.eps, args.sensitivity, args.shift, pi, args.num_runs, args.dropout)
                    all_filenames = [filename for filename in os.listdir('./dat') if filename.startswith(filename_pre)]
                    # all_filenames = []

                    # Run experiment if data doesn't exist yet
                    if all_filenames == []:
                        print("Running experiment for FDR procedure %s and pi %.2f" % (proc_list[FDR - 1], pi))
                        run_single(args.num_runs, args.num_hyp, 1, mu_gap, args.lbd_scale, pi, args.alpha0, args.mod_choice, FDR, args.eps, args.sensitivity,args.shift,sigma = 1, verbose = False, dropout=args.dropout)
                    else:
                            print("Experiments for FDR procedure %s and pi %.2f are already run" % (proc_list[FDR-1], pi))
            if args.plot_style == 2:
                FDR = FDRrange[0]
                for eps in epsrange:
                    # Prevent from running if data already exists
                    filename_pre = 'Lbd%.2f_Si1.0_FDR%d_NH%d_ND%d_MOD%d_EPS%.1f_SE%.1f_SHT%.1f_PM%.2f_NR%d' % (args.lbd_scale, FDR, args.num_hyp, 1, args.mod_choice, eps, args.sensitivity, args.shift, pi, args.num_runs)
                    all_filenames = [filename for filename in os.listdir('./dat') if filename.startswith(filename_pre)]
                    # all_filenames = []

                    # Run experiment if data doesn't exist yet
                    if all_filenames == []:
                        print("Running experiment for FDR procedure %s with epsilon %.1f and pi %.2f" % (proc_list[FDR - 1], eps, pi))
                        run_single(args.num_runs, args.num_hyp, 1, mu_gap, args.lbd_scale, pi, args.alpha0, args.mod_choice, FDR, eps, args.sensitivity,args.shift,sigma = 1, verbose = False)
                    else:
                        print("Experiments for FDR procedure %s with epsilon %.1f and pi %.2f are already run" % (proc_list[FDR-1], eps, pi))
            if args.plot_style == 3:
                FDR = FDRrange[0]
                for shift in shiftrange:
                    # Prevent from running if data already exists
                    filename_pre = 'Lbd%.2f_Si1.0_FDR%d_NH%d_ND%d_MOD%d_EPS%.1f_SE%.1f_SHT%.1f_PM%.2f_NR%d' % (args.lbd_scale, FDR, args.num_hyp, 1, args.mod_choice, args.eps, args.sensitivity, shift, pi, args.num_runs)
                    all_filenames = [filename for filename in os.listdir('./dat') if filename.startswith(filename_pre)]
                    # all_filenames = []

                    # Run experiment if data doesn't exist yet
                    if all_filenames == []:
                        print("Running experiment for FDR procedure %s with shift %.1f and pi %.2f" % (proc_list[FDR - 1], shift, pi))
                        run_single(args.num_runs, args.num_hyp, 1, mu_gap, args.lbd_scale, pi, args.alpha0, args.mod_choice, FDR, args.eps, args.sensitivity,shift,sigma = 1, verbose = False)
                    else:
                        print("Experiments for FDR procedure %s with shift %.1f and pi %.2f are already run" % (proc_list[FDR-1], shift, pi))
            if args.plot_style == 4:
                FDR = FDRrange[0]
                for lbd_scale in lbdrange:
                    # Prevent from running if data already exists
                    filename_pre = 'Lbd%.2f_Si1.0_FDR%d_NH%d_ND%d_MOD%d_EPS%.1f_SE%.1f_SHT%.1f_PM%.2f_NR%d' % (lbd_scale, FDR, args.num_hyp, 1, args.mod_choice, args.eps, args.sensitivity, args.shift, pi, args.num_runs)
                    all_filenames = [filename for filename in os.listdir('./dat') if filename.startswith(filename_pre)]
                    # all_filenames = []

                    # Run experiment if data doesn't exist yet
                    if all_filenames == []:
                        print("Running experiment for FDR procedure %s with lbdscale %.2f and pi %.2f" % (proc_list[FDR - 1], lbd_scale, pi))
                        run_single(args.num_runs, args.num_hyp, 1, mu_gap, lbd_scale, pi, args.alpha0, args.mod_choice, FDR, args.eps, args.sensitivity,args.shift,sigma = 1, verbose = False)
                    else:
                        print("Experiments for FDR procedure %s with lbdscale %.2f and pi %.2f are already run" % (proc_list[FDR-1], lbd_scale, pi))
            break
    

        # Plot different measures over hypotheses for different FDR
        print("Now plotting ... ")
        plot_results(args.plot_style, 0, FDRrange, pirange, hyprange, mu_gap, args.lbd_scale, 1, args.num_hyp, args.num_runs, args.mod_choice, args.eps, epsrange, args.sensitivity,args.shift,shiftrange,lbdrange, dropout=args.dropout)
    
    else:
        print("Now plotting ... ")
        just_plot_results(args.plot_style, 0, FDRrange, pirange, hyprange, mu_gap, args.lbd_scale, 1, args.num_hyp, args.num_runs, args.mod_choice, args.eps, epsrange, args.sensitivity,args.shift,shiftrange,lbdrange)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--FDRrange', type=str, default = '1') # choice of algorithms and parameters 1 for SAFFRON, 2 for LORD, 3 for Alpha-investing, 4 for SAFFRON AI, 5 for PAPRIKA AI, 6 for PAPRIKA
    parser.add_argument('--num-runs', type=int, default = 100) # number of independent trials
    parser.add_argument('--num-hyp', type=int, default = 100) # number of hypotheses
    parser.add_argument('--plot-style', type = int, default = 1) # 0 for plots vs hyp, 1 for plots vs pi1, 2 for varying espilon, 3 for varying shift magnitude, 4 for varying signal in the truncated exponential distribution example
    parser.add_argument('--alpha0', type=float, default = 0.05) # test level
    parser.add_argument('--mu-gap', type=float, default = 0) # mu_c for gaussian tests
    parser.add_argument('--lbd-scale', type=float, default = 0) # used for alternative signal in truncated exponential example
    parser.add_argument('--mod-choice', type=int, default = 5) # 1 for gaussian tests, 2 for beta alternatives, 3 for truncated exponential example, 4 for bernoulli example (*set mu-gap=0 when not doing gaussian tests*) 
    parser.add_argument('--pirange', type=str, default = '0.01,0.02,0.03,0.04,0.05') # range of pi1
    parser.add_argument('--eps', type=float, default = 5) #privacy parameter epsilon
    parser.add_argument('--epsrange', type=str, default = '3,5,10') #used for varying epsilon for private algorithms
    parser.add_argument('--sensitivity', type=float, default = np.sqrt(np.log(1000)/1000)) #multiplicative sensitivity use np.sqrt(np.log(1000)/1000) 
    parser.add_argument('--shift', type=float, default = 1) #shift magnitude c
    parser.add_argument('--shiftrange', type=str, default = '0.5,1,1.5,2') #used for varying shift magnitude for private algorithms
    parser.add_argument('--lbdrange', type=str, default = '1.80,1.90,2.00') #used for varying alternative signal in truncated exponential example
    parser.add_argument('--justplot', type=int, default = 0) #plot only
    parser.add_argument('--dropout', type=float, default = 0)
    args = parser.parse_args()
    logging.info(args)
    main()
