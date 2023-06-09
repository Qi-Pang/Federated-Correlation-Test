import numpy as np
from numpy import sqrt, log, exp, mean, cumsum, sum, zeros, ones, argsort, argmin, argmax, array, maximum, concatenate
from numpy.random import randn, rand
np.set_printoptions(precision = 4)

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['font.size'] = 24
mpl.rcParams['axes.labelsize'] = 36
mpl.rcParams['xtick.labelsize']= 28
mpl.rcParams['ytick.labelsize']= 28

import matplotlib.pyplot as plt
plt.switch_backend('agg')
pgf_with_rc_fonts = {"pgf.texsystem": "pdflatex"}
matplotlib.rcParams.update(pgf_with_rc_fonts)


## Plotting settings

# for normal plots
# plot_style = [':',  '--', '-.', '-'] # for varying aggressiveness
plot_style = ['-', '-']
plot_col = ['gold', 'darkorange', 'firebrick', 'maroon','indianred','chocolate'] # SAFFRON colors
# plot_col = ['mediumslateblue', 'mediumorchid', 'darkviolet', 'indigo'] # LORD colors
# plot_col = ['firebrick', 'darkviolet', 'green'] # SAFFRON, LORD, Alpha-investing
plot_mark = [ 'x', 'o', '^', 'v', 'D', 'x', '+']
plots_ind = 1

def saveplot(direc, filename, lgd, ext = 'pdf',  close = True, verbose = True):
    filename = "%s.%s" % (filename, ext)
    if not os.path.exists(direc):
        os.makedirs(direc)
    savepath = os.path.join(direc, filename)
    plt.savefig(savepath, bbox_extra_artists=(lgd,), bbox_inches='tight')
    if verbose:
        print("Saving figure to %s" % savepath)
    if close:
        plt.close()


def plot_errors_mat(xs, matrix_av, matrix_err, labels, dirname, filename, xlabel, ylabel):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    no_lines = len(matrix_av)
    for i in range(no_lines):
            ys = np.array(matrix_av[i])
            zs = np.array(matrix_err[i])
            print('FDR: ', ys)
            ax.errorbar(xs, ys, yerr = zs, color = plot_col[i % len(plot_col)], marker = plot_mark[i % len(plot_mark)], linestyle = plot_style[i % len(plot_style)], lw= 3, markersize =10, label=labels[i])
    lgd = ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, handletextpad=0.3,
                       ncol=min(no_lines,2), mode="expand", borderaxespad=0., prop={'size': 17})
    ax.set_xlabel(xlabel, labelpad=10)
    ax.set_ylabel(ylabel, labelpad=10)
    ax.set_xlim((min(xs), max(xs)))
    if ylabel == 'Power':
        ax.set_ylim((0, 1))
    else:
        ax.set_ylim((0, 0.2))  #orgin 0.05
    ax.grid(True)
    saveplot(dirname, filename, lgd)

# Plot single curves without error (e.g. because it's not an average)
def plot_curves_mat(xs, matrix_av, labels, dirname, filename, xlabel, ylabel, bool_markers, leg_col = 2):

        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    no_lines = len(matrix_av)
    for i in range(no_lines):
            ys = np.array(matrix_av[i])
            if bool_markers == 1:
                ax.plot(xs, ys, color = plot_col[i % len(plot_col)], marker = plot_mark[i % len(plot_mark)], linestyle = plot_style[i % len(plot_style)], lw=3, markersize =5, label=labels[i])
            else:
                ax.plot(xs, ys, color = plot_col[i % len(plot_col)], linestyle = plot_style[i % len(plot_style)], lw=3,  label=labels[i])
    lgd = ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, handletextpad=0.3,
                       ncol=min(no_lines,leg_col), mode="expand", borderaxespad=0.)
    ax.set_xlabel(xlabel, labelpad=10)
    ax.set_ylabel(ylabel, labelpad=10)
    ax.set_xlim((min(xs), max(xs)))
    ax.grid(True)
    saveplot(dirname, filename, lgd)
    

def plotsingle_shaded_mat(xs, matrix, dirname, filename, xlabel, ylabel):

    no_lines = len(matrix)

    # Compute max of all rows and min
    max_vec = np.array(matrix).max(axis=0)
    min_vec = np.array(matrix).min(axis=0)
    mean_vec = np.array(matrix).mean(axis=0)

    # Compute mean
    for i in range(no_lines):
        ys = np.array(matrix[i])
        plt.plot(xs, ys, color = plot_col[plots_ind],  lw=3)
    plt.fill_between(xs, max_vec, min_vec, facecolor=plot_col[plots_ind], alpha=0.2)
    plt.plot(xs, mean_vec, 'r--')
    plt.xlabel(xlabel, labelpad=10)
    plt.ylabel(ylabel, labelpad=10)
    saveplot(dirname, filename, [])
    
