import random
import matplotlib.pyplot as plt
import numpy as np
import model
import rates

import pandas as pd
pd.set_option('display.float_format', '{:.2f}'.format)


from stats import *
from rw import *

import bootstrap_test as bst

    

def read_distance_csv_no_header(filepath):
    """
    Reads a CSV file without a header, assuming columns are:
    distance, mean, SD (in that order).

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame with columns ['distance', 'Mean', 'SD'].
    """
    df = pd.read_csv(filepath)
    return df.astype(float).set_index('Distance').sort_index()


def expected_mean_sd(Mean0, Var0, rate_bifurcation, rate_annihilation, bin_size):
    # Apply row labels if provided
    assert len(rate_bifurcation) == len(rate_annihilation)
    r = {'Distance':np.arange(0, len(rate_bifurcation) + 1) * bin_size,
         'Mean':[Mean0],
         'Var':[Var0]}

    for i in range(len(rate_bifurcation)):
        Ez0 = r['Mean'][-1]
        Varz0 = r['Var'][-1]
        Ez1 = model.mean_Z(bin_size, Ez0, rate_bifurcation[i], rate_annihilation[i])
        Varz1 = model.var_Z(bin_size, Ez0, Varz0, rate_bifurcation[i], rate_annihilation[i])
        r['Mean'].append(Ez1)
        r['Var'].append(Varz1)

    r = pd.DataFrame(r)
    r['SD'] = np.sqrt(r['Var'])
    r = r.drop('Var', axis=1).set_index('Distance')
    return r


def rowwise_mean_sd(data, bin_size, index_names=None):
    """
    Calculates the mean and standard deviation for each row of a 2D array or DataFrame.

    Parameters:
        data (np.ndarray or pd.DataFrame): Input 2D numerical data.
        index_names (list of str, optional): Row names for the result. If None, uses default or DataFrame indices.

    Returns:
        pd.DataFrame: DataFrame with 'Mean' and 'Standard Deviation' for each row.
    """
    # Convert to DataFrame if needed
    if isinstance(data, np.ndarray):
        df = pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise TypeError("Input must be a NumPy array or Pandas DataFrame.")

    # Apply row labels if provided
    if index_names is not None:
        if len(index_names) != df.shape[0]:
            raise ValueError("Length of index_names must match number of rows.")
        df.index = index_names

        
    result = pd.DataFrame({
        'Distance':np.arange(df.shape[0]) * bin_size,
        'Mean': df.mean(axis=1),
        'SD': df.std(axis=1)
    }).set_index('Distance')

    return result


def plot_histogram(data, bins=10, title='', xlabel='number of branch points', ylabel='frequency',
                   color='black', edgecolor=None, exp_mean=None, exp_std=None, xlim=None, ylim=None):
    """
    Plots a histogram from a NumPy array using matplotlib and shows:
    - Mean ± std of the data
    - Optional second error bar (exp_mean ± exp_std) above it

    Parameters:
        data (np.ndarray): Input numerical data array.
        bins (int or sequence): Number of bins or bin edges.
        title (str): Title of the histogram.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        color (str): Fill color of the bars and error bars.
        edgecolor (str or None): Color of the bar edges.
        exp_mean (float or None): Optional second mean for comparison.
        exp_std (float or None): Optional second standard deviation.
    """
    mean = np.mean(data)
    std = np.std(data)

    plt.figure(figsize=(3.5, 4.8/6.4*4))
    if xlim is None:
        xlim = [0, round(max(data) / 5) * 5]
    counts, bins_edges, _ = plt.hist(data, bins=np.linspace(xlim[0], xlim[1], bins), color=color, edgecolor=edgecolor)

    
    # Positioning error bars
    y_max = max(counts)
    y1 = y_max + y_max * 0.05       # position for main mean±SD
    y2 = y1 + y_max * 0.15          # position for exp_mean±exp_std

    # First error bar: sample mean ± std
    plt.errorbar(mean, y1, xerr=std, fmt='o', color=color, capsize=0,
                 label=f'Sim')

    # Second error bar: expected/reference mean ± std
    if exp_mean is not None and exp_std is not None:
        plt.errorbar(exp_mean, y2, xerr=exp_std, fmt='o', color='lightgray', mfc='white', capsize=0,
                     label=f'Exp')

    # Formatting
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xlim)
    plt.ylim(ylim)
    #plt.ylim([0,50]) #(top=y2 + y_max * 0.15)  # ensure both error bars are visible
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    #plt.show()

    
def plot_two_curves_with_errorbars(x1, y1, err1, x2, y2, err2, 
                                    label1='Sim', label2='Exp', 
                                    color1='black', color2='lightgray', 
                                    xlabel='distance from soma($\mu$m)', ylabel='intersections', title='',
                                    significant_bins1=[], significant_bins2=[], marker_height1=0.1, marker_height2=10, xlim=[-5, 725], dx=50, ylim=[0, 25], figsize=(7, 4.8/6.4*4)):
    
    return plot_three_curves_with_errorbars(x1, y1, err1, x2, y2, err2, y3=None, err3=None,
                                    label1=label1, label2=label2, 
                                    color1=color1, color2=color2, 
                                    xlabel=xlabel, ylabel=ylabel, title=title,
                                    significant_bins1=significant_bins1, significant_bins2=significant_bins2, marker_height1=marker_height1, marker_height2=marker_height2, xlim=xlim, dx=dx, ylim=ylim, figsize=figsize)
                                     
    
def plot_three_curves_with_errorbars(x1, y1, err1, x2, y2, err2, x3=None, y3=None, err3=None,
                                    label1='Sim', label2='Exp', label3='Test',
                                    color1='black', color2='lightgray', color3='red',
                                    xlabel='distance from soma($\mu$m)', ylabel='intersections', title='',
                                    significant_bins1=[], significant_bins2=[], marker_height1=0.1, marker_height2=10, xlim=[-5, 725], dx=50, ylim=[0, 25], figsize=(7, 4.8/6.4*4)):
    """
    Plots two curves with error bars and asterisk markers for statistically significant bins.

    Parameters:
        x (array-like): Shared x-axis values.
        y1, y2 (array-like): Mean values for each curve.
        err1, err2 (array-like): Corresponding error (std or SEM) for each curve.
        label1, label2 (str): Labels for the curves.
        color1, color2 (str): Colors for the curves.
        xlabel, ylabel, title (str): Axis and plot labels.
        significant_bins (list of bool or indices): Marks "*" on x positions where True or present.
        marker_height (float): Additional vertical space above the highest error bar at a bin.
    """
    plt.figure(figsize=figsize)

    pad = 700 * 0.0075 #(x[-1] - x[0]) * 0.01

    # Plot error bars
    if y3 is not None and err3 is not None:
        plt.errorbar(x3 - pad, y3, yerr=err3, label=label3, fmt='-o', color=color3, capsize=0)
        plt.errorbar(x2, y2, yerr=err2, label=label2, fmt='-o', color=color2, capsize=0)
        plt.errorbar(x1 + pad, y1, yerr=err1, label=label1, fmt='-o', color=color1, capsize=0)
    else:
        plt.errorbar(x2 + pad, y2, yerr=err2, label=label2, fmt='-o', color=color2, capsize=0, mfc='white')
        plt.errorbar(x1 - pad, y1, yerr=err1, label=label1, fmt='-o', color=color1, capsize=0)
        
    # Handle significance markers
    for ii, xi in enumerate(x1 if len(x1) < len(x2) else x2):
            # Calculate the higher point between the two ± error
            top1 = y1[xi] + err1[xi]
            top2 = y2[xi] + err2[xi]
            ymax = max(top1, top2)
            if xi in significant_bins1:
                plt.text((x1[ii]+x2[ii])*0.5, ymax + marker_height1, "*", ha='center', va='bottom', fontsize=14, color='black')
            if xi in significant_bins2:
                plt.text((x1[ii]+x2[ii])*0.5, ymax + marker_height2, "#", ha='center', va='bottom', fontsize=14, color='black')

                
    # Format plot
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xlim([xlim[0]-dx/2,xlim[1]+dx/2])
    plt.xticks(np.arange(xlim[0], xlim[1]+dx, dx))
    plt.ylim(ylim)
    plt.yticks(np.arange(ylim[0], ylim[1]+5, 5))
    #plt.show()
    



def compare_dataframes(df1, df2, one_sample=False):
    p_values = {'Distance':list(), 'Mean':list(), 'SD':list()}
    print(df1.index, df2.index)
    #assert (df1.index == df2.index).all()
    index = df1.index if len(df1.index) < len(df2.index) else df2.index
    for d in index:
        p_values['Distance'].append(d)
        sample1 = df1.loc[d, df1.columns[df1.columns.str.startswith('Count')]]
        if one_sample:
            r = bst.bootstrap_test_vs_theor(sample1, df2.loc[d, 'Mean'], df2.loc[d, 'SD'] ** 2)
            p_values['Mean'].append(r['mean']['p_value'])
            p_values['SD'].append(r['variance']['p_value'])
        else:
            sample2 = df2.loc[d, df2.columns[df2.columns.str.startswith('Count')]]
            p_values['Mean'].append(bst.bootstrap_pvalue_mean_diff(sample1, sample2)[0])
            p_values['SD'].append(bst.bootstrap_pvalue_var_ratio(sample1, sample2)[0])
    return pd.DataFrame(p_values).set_index('Distance')


def total_bifurcation_mean_std(dx, rate_bifurcation, rate_annihilation, mean_z, var_z):
    kappa = rate_bifurcation - rate_annihilation

    bif_mean_theor = 0
    for i in range(len(rate_bifurcation)):
        bif_mean_theor += model.mean_B(dx, mean_z[i], rate_bifurcation[i], rate_annihilation[i])

    
    bif_var_theor = 0
    for i in range(len(rate_bifurcation)):
        bif_var_theor += model.var_B(dx, mean_z[i], var_z[i], rate_bifurcation[i], rate_annihilation[i])

    for i in range(1, len(rate_bifurcation)):           
        for j in range(i + 1, len(rate_bifurcation) + 1):
            term1 = rate_bifurcation[i - 1] * (np.exp(kappa[i - 1] * dx) - 1) / kappa[i - 1] * np.exp(-kappa[i - 1] * dx) if i >= 1 else 1
            term2 = rate_bifurcation[j - 1] * (np.exp(kappa[j - 1] * dx) - 1) / kappa[j - 1] * np.exp(np.sum(kappa[np.arange(i, j - 1)]) * dx)
            bif_var_theor += 2 * term1 * term2 * var_z[i]

    bif_std_theor = np.sqrt(bif_var_theor)

    return bif_mean_theor, bif_std_theor

def plot_custom_boxplots(data, box_colors, labels):
    """
    Plots boxplots with no background, custom colors for borders/whiskers/outliers,
    x-axis labels, and a legend.

    Parameters:
    - data: list of arrays or lists, each representing a dataset.
    - box_colors: list of colors (str or tuple), one for each boxplot.
    - labels: list of str, x-axis labels and legend labels.
    - outlier_kwargs: dict of keyword arguments for outlier styling (optional).
    """

    if not (len(data) == len(box_colors) == len(labels)):
        raise ValueError("data, box_colors, and labels must have the same length.")

    # Default outlier style if not provided
    outlier_kwargs = {'marker': 'o', 'markersize': 6, 'alpha': 0.7}

    fig, ax = plt.subplots()

    # Remove background
    ax.set_facecolor('none')
    fig.patch.set_facecolor('none')

    # Create boxplot
    box = ax.boxplot(data, patch_artist=True)

    for i, color in enumerate(box_colors):
        # Box
        box['boxes'][i].set(color=color, linewidth=2, facecolor='none')

        # Whiskers
        box['whiskers'][2*i].set(color=color, linewidth=2)
        box['whiskers'][2*i+1].set(color=color, linewidth=2)

        # Caps
        box['caps'][2*i].set(color=color, linewidth=2)
        box['caps'][2*i+1].set(color=color, linewidth=2)

        # Median
        box['medians'][i].set(color=color, linewidth=2)

        # Fliers (outliers) — match box color
        box['fliers'][i].set(marker=outlier_kwargs.get('marker', 'o'),
                             markersize=outlier_kwargs.get('markersize', 6),
                             alpha=outlier_kwargs.get('alpha', 0.7),
                             markerfacecolor=color,
                             markeredgecolor=color,
                             linestyle='none')

    # Set x-axis labels
    ax.set_xticks(np.arange(1, len(labels)+1))
    ax.set_xticklabels(labels)

    # Create legend
##    legend_handles = [Patch(edgecolor=color, facecolor='none', label=label, linewidth=2) 
##                      for color, label in zip(box_colors, labels)]
##    ax.legend(handles=legend_handles, loc='upper right', frameon=False)

    # Optional: clean up spines
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()


# Example usage:
if __name__ == "__main__":

    

    import sys
    cell_type = sys.argv[sys.argv.index('--cell_type') + 1]    # get cell type from command line

    # number of bifurcations
    data_bif = {
        'PYR':{'Samples':pd.read_csv('pyr_bifurcations.txt')['0'].to_numpy()},
        'SL':{'Samples':pd.read_csv('sl_bifurcations.txt')['0'].to_numpy()},
        'neocortex':{'Samples':pd.read_csv('neocortex_bifurcations.txt')['0'].to_numpy()},
        'tufted':{'Samples':pd.read_csv('tufted_bifurcations.txt')['0'].to_numpy()},
        'mitral':{'Samples':pd.read_csv('mitral_bifurcations.txt')['0'].to_numpy()},
        }[cell_type]

    # re-calculate
    data_bif['Mean'] = np.mean(data_bif['Samples'])
    data_bif['SD'] = np.std(data_bif['Samples'])
##    
##    # get number of neurons from Bathellier et al
##    n_exp_neuron = {
##        'PYR':12,
##        'SL':3,
##        'hippocampus':222,
##        'neocortex':500,
##        }[cell_type]

    hist_xlim, hist_ylim, sholl_xlim, sholl_dx, sholl_ylim, figsize = {
        'SL':([0, 80], [0, 200], [0, 700], 50, [0, 20], (7, 4.8/6.4*4)),
        'PYR':([0, 80], [0, 200], [0, 700], 50, [0, 20], (7, 4.8/6.4*4)),
        'neocortex':([0, 160], [0, 200], [0, 1200], 100, [0, 20], (7, 4.8/6.4*4)),
        'mitral':([0, 160], [0, 200], [0, 1200], 100, [0, 35], (7, 4.8/6.4*4)),
        'tufted':([0, 160], [0, 200], [0, 1200], 100, [0, 35], (7, 4.8/6.4*4))
         }[cell_type]
    
    try:
        n_trials = int(sys.argv[sys.argv.index('--n_trials') + 1])
    except ValueError:
        n_trials = 200

    try:
        bif_factor = float(sys.argv[sys.argv.index('--n-bif-factor')+1])
    except ValueError:
        bif_factor = 1


    if '--n-bif-mean' in sys.argv:
        n_bif = (data_bif['Mean'] * bif_factor, None)
        suffix = '_bif_mean'
    elif '--n-bif-var' in sys.argv:
        n_bif = (None, (data_bif['SD'] * bif_factor) ** 2)
        suffix = '_bif_var'
    elif '--n-bif-both' in sys.argv:
        n_bif = (data_bif['Mean'] * bif_factor, (data_bif['SD'] * bif_factor) ** 2)
        suffix = '_bif_mean_var'
        hist_ylim = {'mitral':[0, 200], 'tufted':[0, 200], 'SL':[0, 100], 'PYR':[0, 100], 'neocortex':[0, 100]}[cell_type]
    else:
        n_bif = (None, None)
        suffix = ''

        
    try:
        step_size = float(sys.argv[sys.argv.index('--step_size')+1])
    except ValueError:
        step_size = 0.1

    try:
        prob_fugal = float(sys.argv[sys.argv.index('--prob_fugal')+1])
    except ValueError:
        prob_fugal = 1       

    try:
        alpha = float(sys.argv[sys.argv.index('--alpha')+1])
    except ValueError:
        alpha = 0.01
        
    
    filepath = {'PYR':'pyr_apical_sholl_plot.txt', 'SL':'sl_apical_sholl_plot.txt', 'neocortex':'neocortex_apical_sholl_plot.txt',
                'mitral':'mitral_sholl_plot.txt', 'tufted':'tufted_sholl_plot.txt'}[cell_type] # file containing sholl plots
    
    # load the sholl plots
    exp_data = read_distance_csv_no_header(filepath)
    exp_data_cpy = exp_data.copy()
    
    data_columns = exp_data.columns[exp_data.columns.str.startswith('Count')]
    exp_data['Mean'] = exp_data[data_columns].mean(axis=1)
    exp_data['SD'] = exp_data[data_columns].std(axis=1)
    exp_data.drop(data_columns, axis=1, inplace=True)
    exp_data.drop(exp_data[exp_data.Mean == 0].index, inplace=True)

    exp_data_cpy = exp_data_cpy.loc[exp_data.index, :] # drop 0s

    print(n_bif)
    # assess equal size of intervals
    dx = np.unique(np.diff(exp_data.index))
    assert dx.size == 1
    dx = dx[0]

    # estimate bifurcation and annihilation rates
    rate_bifurcation, rate_annihilation = rates.solve_qp(step_size, dx, exp_data.Mean.to_numpy(), exp_data.SD.to_numpy(), n_bif=n_bif, kappa_Penalty_Var=1)

    print(rate_bifurcation)
    print(rate_annihilation)
    
    # initial number of dendrites
    init_num = exp_data.loc[0, ['Mean', 'SD']].tolist()

    # run multiple trials
    xp = np.arange(0, rate_bifurcation.size) * dx
    all_walks, num_bifurcations = run_multiple_trials(np.array([xp, rate_bifurcation]).T, np.array([xp, rate_annihilation]).T,
                                                      prob_fugal=prob_fugal, init_num=init_num, bin_size_interp=dx, n_trials=n_trials, step_size=step_size, base_seed=450)
    # get the number of bifurcations
    num_bifurcations = [ tmp[-1, 1] for tmp in num_bifurcations ]

    # calculate the sholl plots for simulation data
    tmp = np.array([walk[:, 1].reshape(-1) for walk in all_walks]).T
    sim_data_cpy = pd.DataFrame(tmp)
    sim_data_cpy.columns = [ 'Count%d' % i for i in sim_data_cpy.columns ]
    sim_data_cpy['Distance'] = all_walks[0][:, 0]
    sim_data_cpy.set_index('Distance', inplace=True)

    sim_data = rowwise_mean_sd(tmp, dx)
    sim_data.drop(sim_data[sim_data.Mean == 0].index, inplace=True)

    sim_data_cpy = sim_data_cpy.loc[sim_data.index, :] # drop 0s

    # calculate theoretical sholl plots
    theor_data = expected_mean_sd(exp_data.loc[0, 'Mean'], exp_data.loc[0, 'SD'] ** 2, rate_bifurcation, rate_annihilation, dx)

    print(sim_data)
    print(exp_data)

    # perform tests between sholl plots
    #if sim_data.size == exp_data.size:
    r = compare_dataframes(sim_data_cpy, exp_data_cpy)
    print('\nsim vs exp')
    print(r, '\n\n')
    significant_bins1 = r[(r.SD < alpha / r.size) & (exp_data.SD > 0)].index
    significant_bins2 = r[r.Mean < alpha  / r.size].index
    
    r = compare_dataframes(sim_data_cpy, theor_data, one_sample=True)
    
    print(r)
    r.Mean = r.Mean < alpha / r.size
    r.SD = r.SD < alpha / r.size
    print('\nsim vs theor')
    print(r, '\n\n')

    r = compare_dataframes(exp_data_cpy, theor_data, one_sample=True)
    print(r)
    r.Mean = r.Mean < alpha / r.size
    r.SD = r.SD < alpha / r.size
    print('\nexp vs theor')
    print(r, '\n\n')
    
    print('Exp\n', exp_data, '\n')
    print('Sim\n', sim_data, '\n')
    print('Th.\n', theor_data, '\n')
    
    # compare mean and variances

        
    bif_mean_exp = data_bif['Mean']
    bif_std_exp = data_bif['SD']
    bif_mean_sim = np.mean(num_bifurcations)
    bif_std_sim = np.std(num_bifurcations)

    bif_mean_theor, bif_std_theor = total_bifurcation_mean_std(dx, rate_bifurcation, rate_annihilation, theor_data['Mean'].to_numpy(), np.power(theor_data['SD'].to_numpy(), 2))

    
    print(f'Exp:\t{bif_mean_exp:.1f}+/-{bif_std_exp:.1f}\nSim:\t{bif_mean_sim:.1f}+/-{bif_std_sim:.1f}\nTheor.:\t{bif_mean_theor:.1f}+/-{bif_std_theor:.1f}')
    r = bst.bootstrap_test_vs_theor(data_bif['Samples'], bif_mean_theor, bif_std_theor ** 2)
    print('exp vs th', r)
                          
    r = bst.bootstrap_test_vs_theor(num_bifurcations, bif_mean_theor, bif_std_theor ** 2)
    print('sim vs th', r)

##    r = two_samples_tests(bif_mean_sim, bif_std_sim ** 2, n_trials, bif_mean_exp, bif_std_exp ** 2, n_exp_neuron)
    r = [ bst.bootstrap_pvalue_mean_diff(num_bifurcations, data_bif['Samples'])[0], bst.bootstrap_pvalue_var_ratio(num_bifurcations, data_bif['Samples'])[0] ]
    print('sim vs exp', r)
    
        
    # Plotting the result
    plt.rcParams['font.size'] = 10
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'
    
    plot_histogram(num_bifurcations, exp_mean=bif_mean_exp, exp_std=bif_std_exp, xlim=hist_xlim, ylim=hist_ylim)
    #plt.ylim([0, n_trials])
    plt.savefig(f'hist_{cell_type}{suffix}.png', dpi=300)
    plt.show()
    
    plot_two_curves_with_errorbars(sim_data.index, sim_data.Mean, sim_data.SD, exp_data.index, exp_data.Mean, exp_data.SD, significant_bins1=significant_bins1, significant_bins2=significant_bins2, xlim=sholl_xlim, ylim=sholl_ylim, dx=sholl_dx, figsize=figsize)
    plt.savefig(f'sholl_plots_{cell_type}{suffix}.png', dpi=300)
    plt.show()
