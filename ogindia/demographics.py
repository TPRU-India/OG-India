'''
------------------------------------------------------------------------
Functions for generating demographic objects necessary for the OG-India
model
------------------------------------------------------------------------
'''
# Import packages
import os
import numpy as np
import scipy.optimize as opt
import scipy.interpolate as si
import pandas as pd
from ogindia import utils
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

'''
------------------------------------------------------------------------
Define functions
------------------------------------------------------------------------
'''


def get_fert(totpers, min_yr, max_yr, graph=False):
    '''
    This function generates a vector of fertility rates by model period
    age that corresponds to the fertility rate data by age in years
    (Source: Office of the Registrar General & Census Commissioner: See
    Statement [Table] 19 of
    http://www.censusindia.gov.in/vital_statistics/SRS_Report_2016/
    7.Chap_3-Fertility_Indicators-2016.pdf)

    Args:
        totpers (int): total number of agent life periods (E+S), >= 3
        min_yr (int): age in years at which agents are born, >= 0
        max_yr (int): age in years at which agents die with certainty,
            >= 4
        graph (bool): =True if want graphical output

    Variable in function:
        fert_data (NumPy array): births per 1000 females divided by 2000
            to include men

    Returns:
        fert_rates (Numpy array): fertility rates for each model period
            of life

    '''
    # Get current population data (2011) for weighting
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    # Get current population data (2011) for weighting
    pop_file = utils.read_file(cur_path,
                               "data/demographic/india_pop_data.csv")
    pop_data = pd.read_table(pop_file, sep=',')
    pop_data_samp = pop_data[(pop_data['Age'] >= min_yr - 1) &
                             (pop_data['Age'] <= max_yr - 1)]
    age_year_all = pop_data_samp['Age'] + 1
    curr_pop = np.array(pop_data_samp['2011'], dtype='f')
    curr_pop_pct = curr_pop / curr_pop.sum()
    # Get fertility rate by age-bin data
    fert_data = np.array([0.0, 1.0, 3.0, 10.7, 135.4, 166.0, 91.7, 32.7,
                          11.3, 4.1, 1.0, 0.0]) / 2000
    age_midp = np.array([9, 12, 15, 17, 22, 27, 32, 37, 42, 47, 52, 57])
    # Generate interpolation functions for fertility rates
    fert_func = si.interp1d(age_midp, fert_data, kind='cubic')
    # Calculate average fertility rate in each age bin using trapezoid
    # method with a large number of points in each bin.
    binsize = (max_yr - min_yr + 1) / totpers
    num_sub_bins = float(10000)
    len_subbins = (np.float64(100 * num_sub_bins)) / totpers
    age_sub = (np.linspace(np.float64(binsize) / num_sub_bins,
               np.float64(max_yr), int(num_sub_bins*max_yr)) -
               0.5 * np.float64(binsize) / num_sub_bins)
    curr_pop_sub = np.repeat(np.float64(curr_pop_pct) /
                             num_sub_bins, num_sub_bins)
    fert_rates_sub = np.zeros(curr_pop_sub.shape)
    pred_ind = (age_sub > age_midp[0]) * (age_sub < age_midp[-1])
    age_pred = age_sub[pred_ind]
    fert_rates_sub[pred_ind] = np.float64(fert_func(age_pred))
    fert_rates = np.zeros(totpers)
    end_sub_bin = 0
    for i in range(totpers):
        beg_sub_bin = int(end_sub_bin)
        end_sub_bin = int(np.rint((i + 1) * len_subbins))
        fert_rates[i] = ((curr_pop_sub[beg_sub_bin:end_sub_bin] *
                          fert_rates_sub[beg_sub_bin:end_sub_bin]).sum()
                         / curr_pop_sub[beg_sub_bin:end_sub_bin].sum())

    if graph:
        '''
        ----------------------------------------------------------------
        age_fine_pred  = (300,) vector, equally spaced support of ages
                         between the minimum and maximum interpolating
                         ages
        fert_fine_pred = (300,) vector, interpolated fertility rates
                         based on age_fine_pred
        age_fine       = (300+some,) vector of ages including leading
                         and trailing zeros
        fert_fine      = (300+some,) vector of fertility rates including
                         leading and trailing zeros
        age_mid_new    = (totpers,) vector, midpoint age of each model
                         period age bin
        output_fldr    = string, folder in current path to save files
        output_dir     = string, total path of OUTPUT folder
        output_path    = string, path of file name of figure to be saved
        ----------------------------------------------------------------
        '''
        # Generate finer age vector and fertility rate vector for
        # graphing cubic spline interpolating function
        age_fine_pred = np.linspace(age_midp[0], age_midp[-1], 300)
        fert_fine_pred = fert_func(age_fine_pred)
        age_fine = np.hstack((min_yr, age_fine_pred, max_yr))
        fert_fine = np.hstack((0, fert_fine_pred, 0))
        age_mid_new = (np.linspace(np.float(max_yr) / totpers, max_yr,
                                   totpers) - (0.5 * np.float(max_yr) /
                                               totpers))

        fig, ax = plt.subplots()
        plt.scatter(age_midp[3:-2], fert_data[3:-2], s=100, c='blue',
                    marker='o', label='Data')
        plt.scatter(np.append(age_midp[:3], age_midp[-2:]),
                    np.append(fert_data[:3], fert_data[-2:]), s=100,
                    c='green', marker='o', label='Non-Data for fitting')
        plt.scatter(age_mid_new, fert_rates, s=40, c='red', marker='d',
                    label='Model period (interpolated)')
        plt.plot(age_fine, fert_fine, label='Cubic spline')
        # for the minor ticks, use no labels; default NullFormatter
        minorLocator = MultipleLocator(1)
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        # plt.title('Fitted fertility rate function by age ($f_{s}$)',
        #     fontsize=20)
        plt.xlabel(r'Age $s$')
        plt.ylabel(r'Fertility rate $f_{s}$')
        plt.xlim((min_yr - 1, max_yr + 1))
        plt.ylim((-0.15 * (fert_fine_pred.max()),
                  1.15 * (fert_fine_pred.max())))
        plt.legend(loc='upper right')
        plt.text(-13, -0.035,
                 "Source:  Census of India, 2016, Chapter 3, " +
                 "Estimates of Fertility Indicators, Statement 20",
                 fontsize=9)
        plt.tight_layout(rect=(0, 0.03, 1, 1))
        # Create directory if OUTPUT directory does not already exist
        output_fldr = "OUTPUT/Demographics"
        output_dir = os.path.join(cur_path, output_fldr)
        if not os.access(output_dir, os.F_OK):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, "fert_rates")
        plt.savefig(output_path)

    return fert_rates


# def get_mort_bins(totpers, min_yr, max_yr, graph=False):
#     '''
#     This function generates a vector of mortality rates by model period
#     age.
#     (Source: data.in.gov
#     estagespdeathsex2006-2011.csv)

#     Args:
#         totpers (int): total number of agent life periods (E+S), >= 3
#         min_yr (int): age in years at which agents are born, >= 0
#         max_yr (int): age in years at which agents die with certainty,
#             >= 4
#         graph (bool): =True if want graphical output

#     Returns:
#         fert_rates (Numpy array): fertility rates for each model period
#             of life

#     '''
#     # Get current population data (2011) for weighting
#     cur_path = os.path.split(os.path.abspath(__file__))[0]
#     pop_file = utils.read_file(
#         cur_path, 'data/demographic/india_pop_data.csv')
#     pop_data = pd.read_csv(pop_file, sep=',', thousands=',')
#     pop_data_samp = pop_data[(pop_data['Age'] >= min_yr - 1) &
#                              (pop_data['Age'] <= max_yr - 1)]
#     curr_pop = np.array(pop_data_samp['2011'], dtype='f')
#     curr_pop_pct = curr_pop / curr_pop.sum()
#     # Get fertility rate by age-bin data
#     # Currently just pulling values for one education level
#     # need to get weighted avg across all education levels
#     infmort_rate = 48.2 / 1000  # not 100% sure if per 1000 individuals
#     mort_data = (np.array([2.9, 1, 0.7, 1.3, 1.6, 1.8, 2.3, 2.7, 4, 5.5,
#                            8.3, 12.2, 20.1, 33.2, 49.9, 73.6, 104.8,
#                            167.6]) / 1000)
#     # Mid points of age bins
#     age_midp = np.array([2.5, 6.5, 12, 17, 22, 27, 32, 37, 42, 47, 52,
#                          57, 62, 67, 72, 77, 82, 100])
#     # Generate interpolation functions for mortality rates
#     mort_func = si.interp1d(age_midp, mort_data, kind='cubic')
#     # Calculate average mortality rate in each age bin using trapezoid
#     # method with a large number of points in each bin.
#     binsize = (max_yr - min_yr + 1) / totpers
#     num_sub_bins = float(10000)
#     len_subbins = (np.float64(100 * num_sub_bins)) / totpers
#     age_sub = (np.linspace(np.float64(binsize) / num_sub_bins,
#                            np.float64(max_yr),
#                            int(num_sub_bins * max_yr)) - 0.5 *
#                np.float64(binsize) / num_sub_bins)
#     curr_pop_sub = np.repeat(np.float64(curr_pop_pct) / num_sub_bins,
#                              num_sub_bins)
#     mort_rates_sub = np.zeros(curr_pop_sub.shape)
#     pred_ind = (age_sub > age_midp[0]) * (age_sub < age_midp[-1])
#     age_pred = age_sub[pred_ind]
#     mort_rates_sub[pred_ind] = np.float64(mort_func(age_pred))
#     mort_rates = np.zeros(totpers)
#     end_sub_bin = 0
#     for i in range(totpers):
#         beg_sub_bin = int(end_sub_bin)
#         end_sub_bin = int(np.rint((i + 1) * len_subbins))
#         mort_rates[i] = ((
#             curr_pop_sub[beg_sub_bin:end_sub_bin] *
#             mort_rates_sub[beg_sub_bin:end_sub_bin]).sum() /
#             curr_pop_sub[beg_sub_bin:end_sub_bin].sum())
#     mort_rates[-1] = 1  # Mortality rate in last period is set to 1

#     if graph:
#         '''
#         ----------------------------------------------------------------
#         age_fine_pred  = (300,) vector, equally spaced support of ages
#                          between the minimum and maximum interpolating
#                          ages
#         mort_fine_pred = (300,) vector, interpolated mortality rates
#                          based on age_fine_pred
#         age_fine       = (300+some,) vector of ages including leading
#                          and trailing zeros
#         mort_fine      = (300+some,) vector of mortality rates including
#                          leading and trailing zeros
#         age_mid_new    = (totpers,) vector, midpoint age of each model
#                          period age bin
#         output_fldr    = string, folder in current path to save files
#         output_dir     = string, total path of OUTPUT folder
#         output_path    = string, path of file name of figure to be saved
#         ----------------------------------------------------------------
#         '''
#         # Generate finer age vector and mortality rate vector for
#         # graphing cubic spline interpolating function
#         age_fine_pred = np.linspace(age_midp[0], age_midp[-1], 300)
#         mort_fine_pred = mort_func(age_fine_pred)
#         age_fine = np.hstack((min_yr, age_fine_pred, max_yr))
#         mort_fine = np.hstack((0, mort_fine_pred, 0))
#         age_mid_new = (np.linspace(np.float(max_yr) / totpers, max_yr,
#                                    totpers) - (0.5 * np.float(max_yr) /
#                                                totpers))

#         fig, ax = plt.subplots()
#         plt.scatter(age_midp, mort_data, s=70, c='blue', marker='o',
#                     label='Data')
#         plt.scatter(age_mid_new, mort_rates, s=40, c='red', marker='d',
#                     label='Model period (integrated)')
#         plt.plot(age_fine, mort_fine, label='Cubic spline')
#         # for the minor ticks, use no labels; default NullFormatter
#         minorLocator = MultipleLocator(1)
#         ax.xaxis.set_minor_locator(minorLocator)
#         plt.grid(b=True, which='major', color='0.65', linestyle='-')
#         # plt.title('Fitted fertility rate function by age ($f_{s}$)',
#         #     fontsize=20)
#         plt.xlabel(r'Age $s$')
#         plt.ylabel(r'Mortality rate $\rho_{s}$')
#         plt.xlim((min_yr - 1, max_yr + 1))
#         plt.ylim((-0.15 * (mort_fine_pred.max()),
#                   1.15 * (mort_fine_pred.max())))
#         plt.legend(loc='upper right')
#         plt.text(-5, -0.018, 'Source: data.gov.in', fontsize=9)
#         plt.tight_layout(rect=(0, 0.03, 1, 1))
#         # Create directory if OUTPUT directory does not already exist
#         output_dir = os.path.join(cur_path, 'OUTPUT', 'Demographics')
#         if os.access(output_dir, os.F_OK) is False:
#             os.makedirs(output_dir)
#         output_path = os.path.join(output_dir, 'mort_rates')
#         plt.savefig(output_path)

#     return mort_rates, infmort_rate


def get_mort(totpers, min_yr, max_yr, graph=False):
    '''
    This function generates a vector of mortality rates by model period
    age.
    (Source: Census of India, 2011)

    Args:
        totpers (int): total number of agent life periods (E+S), >= 3
        min_yr (int): age in years at which agents are born, >= 0
        max_yr (int): age in years at which agents die with certainty,
            >= 4
        graph (bool): =True if want graphical output

    Returns:
        mort_rates (Numpy array) mortality rates that correspond to each
            period of life
        infmort_rate (scalar): infant mortality rate from 2015 U.S. CIA
            World Factbook

    '''
    # Get mortality rate by age data
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    # Get current population data (2011) for weighting
    pop_file = utils.read_file(cur_path,
                               'data/demographic/india_pop_data.csv')
    pop_data = pd.read_table(pop_file, sep=',')
    pop_data_samp = pop_data[(pop_data['Age'] >= min_yr - 1) &
                             (pop_data['Age'] <= max_yr - 1)]
    age_year_all = pop_data_samp['Age'] + 1
    curr_pop = np.array(pop_data_samp['2011'], dtype='f')
    curr_pop_pct = curr_pop / curr_pop.sum()
    # Get mortality rate by age data
    infmort_rate = 0.0482
    # Get fertility rate by age-bin data
    mort_data = (np.array([2.9, 1.0, 0.7, 1.3, 1.6, 1.8, 2.3, 2.7, 4.0,
                           5.5, 8.3, 12.2, 20.1, 33.2, 49.9, 73.6, 104.8,
                           167.6]) / 1000)
    age_midp = np.array([2.5, 7, 12, 17, 22, 27, 32,  37, 42, 47, 52,
                         57, 62, 67, 72, 77, 82, 100])
    # Generate interpolation functions for fertility rates
    mort_func = si.interp1d(age_midp, mort_data, kind='cubic')
    # Calculate average fertility rate in each age bin using trapezoid
    # method with a large number of points in each bin.
    binsize = (max_yr - min_yr + 1) / totpers
    num_sub_bins = float(10000)
    len_subbins = (np.float64(100 * num_sub_bins)) / totpers
    age_sub = (np.linspace(np.float64(binsize) / num_sub_bins,
               np.float64(max_yr), int(num_sub_bins * max_yr)) -
               0.5 * np.float64(binsize) / num_sub_bins)
    curr_pop_sub = np.repeat(np.float64(curr_pop_pct) /
                             num_sub_bins, num_sub_bins)
    mort_rates_sub = np.zeros(curr_pop_sub.shape)
    pred_ind = (age_sub > age_midp[0]) * (age_sub < age_midp[-1])
    age_pred = age_sub[pred_ind]
    mort_rates_sub[pred_ind] = np.float64(mort_func(age_pred))
    mort_rates = np.zeros(totpers)
    end_sub_bin = 0
    for i in range(totpers):
        beg_sub_bin = int(end_sub_bin)
        end_sub_bin = int(np.rint((i + 1) * len_subbins))
        mort_rates[i] = ((curr_pop_sub[beg_sub_bin:end_sub_bin] *
                          mort_rates_sub[beg_sub_bin:end_sub_bin]).sum()
                         / curr_pop_sub[beg_sub_bin:end_sub_bin].sum())
    mort_rates[-1] = 1  # Mortality rate in last period is set to 1

    if graph:
        '''
        ----------------------------------------------------------------
        age_mid_new = (totpers,) vector, midpoint age of each model
                      period age bin
        output_fldr = string, folder in current path to save files
        output_dir  = string, total path of OUTPUT folder
        output_path = string, path of file name of figure to be saved
        ----------------------------------------------------------------
        '''
        age_mid_new = (np.linspace(np.float(max_yr) / totpers, max_yr,
                                   totpers) - (0.5 * np.float(max_yr) /
                                               totpers))
        fig, ax = plt.subplots()
        plt.scatter(np.hstack([0, age_midp]),
                    np.hstack([infmort_rate, mort_data]),
                    s=100, c='blue', marker='o', label='Data')
        plt.scatter(np.hstack([0, age_mid_new]),
                    np.hstack([infmort_rate, mort_rates]),
                    s=40, c='red', marker='d',
                    label='Model period (interpolated)')
        plt.axvline(x=max_yr, color='black', linestyle='-',
                    linewidth=1)
        # for the minor ticks, use no labels; default NullFormatter
        minorLocator = MultipleLocator(1)
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        # plt.title('Fitted mortality rate function by age ($rho_{s}$)',
        #     fontsize=20)
        plt.xlabel(r'Age $s$')
        plt.ylabel(r'Mortality rate $\rho_{s}$')
        plt.xlim((min_yr - 2, age_year_all.max() + 2))
        plt.ylim((-0.05, 1.05))
        plt.legend(loc='upper left')
        plt.text(-13, -0.30, "Source: Ministry of Health and Family " +
                 "Welfare, Department of Health and Family Welfare",
                 fontsize=9)
        plt.tight_layout(rect=(0, 0.03, 1, 1))
        # Create directory if OUTPUT directory does not already exist
        output_fldr = "OUTPUT/Demographics"
        output_dir = os.path.join(cur_path, output_fldr)
        if os.access(output_dir, os.F_OK) is False:
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, "mort_rates")
        plt.savefig(output_path)

    return mort_rates, infmort_rate


def pop_rebin(curr_pop_dist, totpers_new):
    '''
    For cases in which totpers (E+S) is less than the number of periods
    in the population distribution data, this function calculates a new
    population distribution vector with totpers (E+S) elements.

    Args:
        curr_pop_dist (Numpy array): population distribution over N
            periods
    totpers_new (int): number of periods to which we are
        transforming the population distribution, >= 4

    Returns:
        curr_pop_new (Numpy array): new population distribution over
            totpers (E+S) periods that approximates curr_pop_dist

    '''
    # Number of periods in original data
    totpers_orig = len(curr_pop_dist)
    if int(totpers_new) == totpers_orig:
        curr_pop_new = curr_pop_dist
    elif int(totpers_new) < totpers_orig:
        num_sub_bins = float(10000)
        curr_pop_sub = np.repeat(np.float64(curr_pop_dist) /
                                 num_sub_bins, num_sub_bins)
        len_subbins = ((np.float64(totpers_orig*num_sub_bins)) /
                       totpers_new)
        curr_pop_new = np.zeros(totpers_new, dtype=np.float64)
        end_sub_bin = 0
        for i in range(totpers_new):
            beg_sub_bin = int(end_sub_bin)
            end_sub_bin = int(np.rint((i + 1) * len_subbins))
            curr_pop_new[i] = \
                curr_pop_sub[beg_sub_bin:end_sub_bin].sum()
        # Return curr_pop_new to single precision float (float32)
        # datatype
        curr_pop_new = np.float32(curr_pop_new)

    return curr_pop_new


def get_imm_resid(totpers, min_yr, max_yr, graph=True):
    '''
    Calculate immigration rates by age as a residual given population
    levels in different periods, then output average calculated
    immigration rate. We have to replace the first mortality rate in
    this function in order to adjust the first implied immigration rate
    (Source: India Census, 2001 and 2011)

    Args:
        totpers (int): total number of agent life periods (E+S), >= 3
        min_yr (int): age in years at which agents are born, >= 0
        max_yr (int): age in years at which agents die with certainty,
            >= 4
        graph (bool): =True if want graphical output

    Returns:
        imm_rates (Numpy array):immigration rates that correspond to
            each period of life, length E+S

    '''
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    pop_file = utils.read_file(cur_path,
                               "data/demographic/india_pop_data.csv")
    pop_data = pd.read_csv(pop_file, sep=',', thousands=',')
    pop_data_samp = pop_data[(pop_data['Age'] >= min_yr - 1) &
                             (pop_data['Age'] <= max_yr - 1)]
    age_year_all = pop_data_samp['Age'] + 1
    pop_2001, pop_2011 = (
        np.array(pop_data_samp['2001'], dtype='f'),
        np.array(pop_data_samp['2011'], dtype='f'))
    pop_2001_EpS = pop_rebin(pop_2001, totpers)
    pop_2011_EpS = pop_rebin(pop_2011, totpers)
    # Create three years of estimated immigration rates for youngest age
    # individuals
    imm_mat = np.zeros((2, totpers))
    fert_rates = get_fert(totpers, min_yr, max_yr, False)
    mort_rates, infmort_rate = get_mort(totpers, min_yr, max_yr, False)
    newbornvec = np.dot(fert_rates, pop_2001_EpS).T
    # imm_mat[:, 0] = ((pop_2011_EpS[0] - (1 - infmort_rate) * newbornvec)
    #                  / pop_2001_EpS[0])
    imm_mat[:, 0] = 0
    # Estimate immigration rates for all other-aged
    # individuals
    mort_rate10 = np.zeros_like(mort_rates[:-10])  # 10-year mort rate
    for i in range(10):
        mort_rate10 = mort_rates[i:-10 + i] + mort_rate10
    mort_rate10[mort_rate10 > 1.0] = 1.0
    imm_mat[:, 10:] = ((pop_2011_EpS[10:] - (1 - mort_rate10) *
                       pop_2001_EpS[:-10]) / pop_2001_EpS[10:])
    # Final estimated immigration rates are the averages over years
    imm_rates = imm_mat.mean(axis=0)
    neg_rates = imm_rates < 0
    # For India, data were 10 years apart, so make annual rate
    imm_rates = ((1 + np.absolute(imm_rates)) ** (1 / 10)) - 1
    imm_rates[neg_rates] = -1 * imm_rates[neg_rates]
    age_per = np.linspace(1, totpers, totpers)

    if graph:
        '''
        ----------------------------------------------------------------
        output_fldr = string, path of the OUTPUT folder from cur_path
        output_dir  = string, total path of OUTPUT folder
        output_path = string, path of file name of figure to be saved
        ----------------------------------------------------------------
        '''
        fig, ax = plt.subplots()
        plt.scatter(age_per, imm_rates, s=40, c='red', marker='d')
        plt.plot(age_per, imm_rates)
        # for the minor ticks, use no labels; default NullFormatter
        minorLocator = MultipleLocator(1)
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        # plt.title('Fitted immigration rates by age ($i_{s}$), residual',
        #     fontsize=20)
        plt.xlabel(r'Age $s$ (model periods)')
        plt.ylabel(r'Imm. rate $i_{s}$')
        plt.xlim((0, totpers + 1))
        # Create directory if OUTPUT directory does not already exist
        output_fldr = "OUTPUT/Demographics"
        output_dir = os.path.join(cur_path, output_fldr)
        if os.access(output_dir, os.F_OK) is False:
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, "imm_rates_orig")
        plt.savefig(output_path)

    return imm_rates


def immsolve(imm_rates, *args):
    '''
    This function generates a vector of errors representing the
    difference in two consecutive periods stationary population
    distributions. This vector of differences is the zero-function
    objective used to solve for the immigration rates vector, similar to
    the original immigration rates vector from get_imm_resid(), that
    sets the steady-state population distribution by age equal to the
    population distribution in period int(1.5*S)

    Args:
        imm_rates (Numpy array):immigration rates that correspond to
            each period of life, length E+S
        args (tuple): (fert_rates, mort_rates, infmort_rate, omega_cur,
            g_n_SS)

    Returns:
        omega_errs (Numpy array): difference between omega_new and
            omega_cur_pct, length E+S

    '''
    fert_rates, mort_rates, infmort_rate, omega_cur_lev, g_n_SS = args
    omega_cur_pct = omega_cur_lev / omega_cur_lev.sum()
    totpers = len(fert_rates)
    OMEGA = np.zeros((totpers, totpers))
    OMEGA[0, :] = ((1 - infmort_rate) * fert_rates +
                   np.hstack((imm_rates[0], np.zeros(totpers-1))))
    OMEGA[1:, :-1] += np.diag(1 - mort_rates[:-1])
    OMEGA[1:, 1:] += np.diag(imm_rates[1:])
    omega_new = np.dot(OMEGA, omega_cur_pct) / (1 + g_n_SS)
    omega_errs = omega_new - omega_cur_pct

    return omega_errs


def get_pop_objs(E, S, T, min_yr, max_yr, curr_year, GraphDiag=True):
    '''
    This function produces the demographics objects to be used in the
    OG-India model package.

    Args:
        E (int): number of model periods in which agent is not
            economically active, >= 1
        S (int): number of model periods in which agent is economically
            active, >= 3
        T (int): number of periods to be simulated in TPI, > 2*S
        min_yr (int): age in years at which agents are born, >= 0
        max_yr (int): age in years at which agents die with certainty,
            >= 4
        curr_year (int): current year for which analysis will begin,
            >= 2016
        GraphDiag (bool): =True if want graphical output and printed
                diagnostics

    Returns:
        omega_path_S (Numpy array), time path of the population
            distribution from the current state to the steady-state,
            size T+S x S
        g_n_SS (scalar): steady-state population growth rate
        omega_SS (Numpy array): normalized steady-state population
            distribution, length S
        surv_rates (Numpy array): survival rates that correspond to
            each model period of life, lenght S
        mort_rates (Numpy array): mortality rates that correspond to
            each model period of life, length S
        g_n_path (Numpy array): population growth rates over the time
            path, length T + S

    '''
    # age_per = np.linspace(min_yr, max_yr, E+S)
    fert_rates = get_fert(E + S, min_yr, max_yr, graph=False)
    mort_rates, infmort_rate = get_mort(E + S, min_yr, max_yr,
                                        graph=False)
    mort_rates_S = mort_rates[-S:]
    imm_rates_orig = get_imm_resid(E + S, min_yr, max_yr, graph=False)
    OMEGA_orig = np.zeros((E + S, E + S))
    OMEGA_orig[0, :] = ((1 - infmort_rate) * fert_rates +
                        np.hstack((imm_rates_orig[0], np.zeros(E+S-1))))
    OMEGA_orig[1:, :-1] += np.diag(1 - mort_rates[:-1])
    OMEGA_orig[1:, 1:] += np.diag(imm_rates_orig[1:])

    # Solve for steady-state population growth rate and steady-state
    # population distribution by age using eigenvalue and eigenvector
    # decomposition
    eigvalues, eigvectors = np.linalg.eig(OMEGA_orig)
    g_n_SS = (eigvalues[np.isreal(eigvalues)].real).max() - 1
    eigvec_raw =\
        eigvectors[:,
                   (eigvalues[np.isreal(eigvalues)].real).argmax()].real
    omega_SS_orig = eigvec_raw / eigvec_raw.sum()

    # Generate time path of the nonstationary population distribution
    omega_path_lev = np.zeros((E + S, T + S))
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    pop_file = utils.read_file(cur_path,
                               "data/demographic/india_pop_data.csv")
    pop_data = pd.read_csv(pop_file, sep=',', thousands=',')
    pop_data_samp = pop_data[(pop_data['Age'] >= min_yr - 1) &
                             (pop_data['Age'] <= max_yr - 1)]
    pop_2011 = np.array(pop_data_samp['2011'], dtype='f')
    # Generate the current population distribution given that E+S might
    # be less than max_yr-min_yr+1
    age_per_EpS = np.arange(1, E + S + 1)
    pop_2011_EpS = pop_rebin(pop_2011, E + S)
    pop_2011_pct = pop_2011_EpS / pop_2011_EpS.sum()
    # Age most recent population data to the current year of analysis
    pop_curr = pop_2011_EpS.copy()
    data_year = 2011
    pop_next = np.dot(OMEGA_orig, pop_curr)
    g_n_curr = ((pop_next[-S:].sum() - pop_curr[-S:].sum()) /
                pop_curr[-S:].sum())  # g_n in 2011
    pop_past = pop_curr  # assume 2010-2011 pop
    # Age the data to the current year
    for per in range(curr_year - data_year):
        pop_next = np.dot(OMEGA_orig, pop_curr)
        g_n_curr = ((pop_next[-S:].sum() - pop_curr[-S:].sum()) /
                    pop_curr[-S:].sum())
        pop_past = pop_curr
        pop_curr = pop_next

    # Generate time path of the population distribution
    omega_path_lev[:, 0] = pop_curr.copy()
    for per in range(1, T + S):
        pop_next = np.dot(OMEGA_orig, pop_curr)
        omega_path_lev[:, per] = pop_next.copy()
        pop_curr = pop_next.copy()

    # Force the population distribution after 1.5*S periods to be the
    # steady-state distribution by adjusting immigration rates, holding
    # constant mortality, fertility, and SS growth rates
    imm_tol = 1e-14
    fixper = int(1.5 * S)
    omega_SSfx = (omega_path_lev[:, fixper] /
                  omega_path_lev[:, fixper].sum())
    imm_objs = (fert_rates, mort_rates, infmort_rate,
                omega_path_lev[:, fixper], g_n_SS)
    imm_fulloutput = opt.fsolve(immsolve, imm_rates_orig,
                                args=(imm_objs), full_output=True,
                                xtol=imm_tol)
    imm_rates_adj = imm_fulloutput[0]
    imm_diagdict = imm_fulloutput[1]
    omega_path_S = (omega_path_lev[-S:, :] /
                    np.tile(omega_path_lev[-S:, :].sum(axis=0), (S, 1)))
    omega_path_S[:, fixper:] = \
        np.tile(omega_path_S[:, fixper].reshape((S, 1)),
                (1, T + S - fixper))
    g_n_path = np.zeros(T + S)
    g_n_path[0] = g_n_curr.copy()
    g_n_path[1:] = ((omega_path_lev[-S:, 1:].sum(axis=0) -
                    omega_path_lev[-S:, :-1].sum(axis=0)) /
                    omega_path_lev[-S:, :-1].sum(axis=0))
    g_n_path[fixper + 1:] = g_n_SS
    omega_S_preTP = (pop_past.copy()[-S:]) / (pop_past.copy()[-S:].sum())
    imm_rates_mat = np.hstack((
        np.tile(np.reshape(imm_rates_orig[E:], (S, 1)), (1, fixper)),
        np.tile(np.reshape(imm_rates_adj[E:], (S, 1)), (1, T + S - fixper))))

    if GraphDiag:
        # Check whether original SS population distribution is close to
        # the period-T population distribution
        omegaSSmaxdif = np.absolute(omega_SS_orig -
                                    (omega_path_lev[:, T] /
                                     omega_path_lev[:, T].sum())).max()
        if omegaSSmaxdif > 0.0003:
            print("POP. WARNING: Max. abs. dist. between original SS " +
                  "pop. dist'n and period-T pop. dist'n is greater than" +
                  " 0.0003. It is " + str(omegaSSmaxdif) + ".")
        else:
            print("POP. SUCCESS: orig. SS pop. dist is very close to " +
                  "period-T pop. dist'n. The maximum absolute " +
                  "difference is " + str(omegaSSmaxdif) + ".")

        # Plot the adjusted steady-state population distribution versus
        # the original population distribution. The difference should be
        # small
        omegaSSvTmaxdiff = np.absolute(omega_SS_orig - omega_SSfx).max()
        if omegaSSvTmaxdiff > 0.0003:
            print("POP. WARNING: The maximimum absolute difference " +
                  "between any two corresponding points in the original"
                  + " and adjusted steady-state population " +
                  "distributions is" + str(omegaSSvTmaxdiff) + ", " +
                  "which is greater than 0.0003.")
        else:
            print("POP. SUCCESS: The maximum absolute difference " +
                  "between any two corresponding points in the original"
                  + " and adjusted steady-state population " +
                  "distributions is " + str(omegaSSvTmaxdiff))
        fig, ax = plt.subplots()
        plt.plot(age_per_EpS, omega_SS_orig, label="Original Dist'n")
        plt.plot(age_per_EpS, omega_SSfx, label="Fixed Dist'n")
        # for the minor ticks, use no labels; default NullFormatter
        minorLocator = MultipleLocator(1)
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        plt.title(
            'Original steady-state population distribution vs. fixed',
            fontsize=20)
        plt.xlabel(r'Age $s$')
        plt.ylabel(r"Pop. dist'n $\omega_{s}$")
        plt.xlim((0, E + S + 1))
        plt.legend(loc='upper right')
        # Create directory if OUTPUT directory does not already exist
        '''
        ----------------------------------------------------------------
        output_fldr = string, path of the OUTPUT folder from cur_path
        output_dir  = string, total path of OUTPUT folder
        output_path = string, path of file name of figure to be saved
        ----------------------------------------------------------------
        '''
        cur_path = os.path.split(os.path.abspath(__file__))[0]
        output_fldr = "OUTPUT/Demographics"
        output_dir = os.path.join(cur_path, output_fldr)
        if os.access(output_dir, os.F_OK) is False:
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, "OrigVsFixSSpop")
        plt.savefig(output_path)

        # Print whether or not the adjusted immigration rates solved the
        # zero condition
        immtol_solved = \
            np.absolute(imm_diagdict['fvec'].max()) < imm_tol
        if immtol_solved:
            print("POP. SUCCESS: Adjusted immigration rates solved " +
                  "with maximum absolute error of " +
                  str(np.absolute(imm_diagdict['fvec'].max())) +
                  ", which is less than the tolerance of " +
                  str(imm_tol))
        else:
            print("POP. WARNING: Adjusted immigration rates did not " +
                  "solve. Maximum absolute error of " +
                  str(np.absolute(imm_diagdict['fvec'].max())) +
                  " is greater than the tolerance of " + str(imm_tol))

        # Test whether the steady-state growth rates implied by the
        # adjusted OMEGA matrix equals the steady-state growth rate of
        # the original OMEGA matrix
        OMEGA2 = np.zeros((E + S, E + S))
        OMEGA2[0, :] = ((1 - infmort_rate) * fert_rates +
                        np.hstack((imm_rates_adj[0], np.zeros(E+S-1))))
        OMEGA2[1:, :-1] += np.diag(1 - mort_rates[:-1])
        OMEGA2[1:, 1:] += np.diag(imm_rates_adj[1:])
        eigvalues2, eigvectors2 = np.linalg.eig(OMEGA2)
        g_n_SS_adj = (eigvalues[np.isreal(eigvalues2)].real).max() - 1
        if np.max(np.absolute(g_n_SS_adj - g_n_SS)) > 10 ** (-8):
            print("FAILURE: The steady-state population growth rate" +
                  " from adjusted OMEGA is different (diff is " +
                  str(g_n_SS_adj - g_n_SS) + ") than the steady-" +
                  "state population growth rate from the original" +
                  " OMEGA.")
        elif np.max(np.absolute(g_n_SS_adj - g_n_SS)) <= 10 ** (-8):
            print("SUCCESS: The steady-state population growth rate" +
                  " from adjusted OMEGA is close to (diff is " +
                  str(g_n_SS_adj - g_n_SS) + ") the steady-" +
                  "state population growth rate from the original" +
                  " OMEGA.")

        # Do another test of the adjusted immigration rates. Create the
        # new OMEGA matrix implied by the new immigration rates. Plug in
        # the adjusted steady-state population distribution. Hit is with
        # the new OMEGA transition matrix and it should return the new
        # steady-state population distribution
        omega_new = np.dot(OMEGA2, omega_SSfx)
        omega_errs = np.absolute(omega_new - omega_SSfx)
        print("The maximum absolute difference between the adjusted " +
              "steady-state population distribution and the " +
              "distribution generated by hitting the adjusted OMEGA " +
              "transition matrix is " + str(omega_errs.max()))

        # Plot the original immigration rates versus the adjusted
        # immigration rates
        immratesmaxdiff = \
            np.absolute(imm_rates_orig - imm_rates_adj).max()
        print("The maximum absolute distance between any two points " +
              "of the original immigration rates and adjusted " +
              "immigration rates is " + str(immratesmaxdiff))
        fig, ax = plt.subplots()
        plt.plot(age_per_EpS, imm_rates_orig, label="Original Imm. Rates")
        plt.plot(age_per_EpS, imm_rates_adj, label="Adj. Imm. Rates")
        # for the minor ticks, use no labels; default NullFormatter
        minorLocator = MultipleLocator(1)
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        plt.title(
            'Original immigration rates vs. adjusted',
            fontsize=20)
        plt.xlabel(r'Age $s$')
        plt.ylabel(r"Imm. rates $i_{s}$")
        plt.xlim((0, E + S + 1))
        plt.legend(loc='upper center')
        # Create directory if OUTPUT directory does not already exist
        output_path = os.path.join(output_dir, "OrigVsAdjImm")
        plt.savefig(output_path)

        # Plot population distributions for data_year, curr_year,
        # curr_year+20, omega_SSfx, and omega_SS_orig
        fig, ax = plt.subplots()
        plt.plot(age_per_EpS, pop_2011_pct, label="2011 pop.")
        plt.plot(age_per_EpS, (omega_path_lev[:, 0] /
                               omega_path_lev[:, 0].sum()),
                 label=str(curr_year) + " pop.")
        plt.plot(age_per_EpS, (omega_path_lev[:, int(0.5 * S)] /
                               omega_path_lev[:, int(0.5 * S)].sum()),
                 label="T=" + str(int(0.5 * S)) + " pop.")
        plt.plot(age_per_EpS, (omega_path_lev[:, int(S)] /
                               omega_path_lev[:, int(S)].sum()),
                 label="T=" + str(int(S)) + " pop.")
        plt.plot(age_per_EpS, omega_SSfx, label="Adj. SS pop.")
        # for the minor ticks, use no labels; default NullFormatter
        minorLocator = MultipleLocator(1)
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        plt.title(
            'Population distribution at points in time path',
            fontsize=20)
        plt.xlabel(r'Age $s$')
        plt.ylabel(r"Pop. dist'n $\omega_{s}$")
        plt.xlim((0, E+S+1))
        plt.legend(loc='lower left')
        # Create directory if OUTPUT directory does not already exist
        output_path = os.path.join(output_dir, "PopDistPath")
        plt.savefig(output_path)

    # return omega_path_S, g_n_SS, omega_SSfx, survival rates,
    # mort_rates_S, and g_n_path
    return (omega_path_S.T, g_n_SS, omega_SSfx[-S:] /
            omega_SSfx[-S:].sum(), 1-mort_rates_S, mort_rates_S,
            g_n_path, imm_rates_mat.T, omega_S_preTP)
