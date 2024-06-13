import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
from scipy.optimize import curve_fit
from scipy.stats import t


FIGURE_PATH = 'figures'
DATA_PATH = 'data'

CPMG_ALBUMEN = 'CPMGalbume1B.dat'
CPMG_YOLK = 'CPMGtuorlo1B.dat'

IR_ALBUMEN = 'IRalbumeH.dat'
IR_YOLK = 'IRtuorloH.dat'


def sns_graphic_options():
    sns.set_theme(style="ticks")
    sns.set_context("paper", font_scale=2.0)
    plt.rcParams["font.family"] = "serif"  
    plt.rcParams["axes.grid"] = True  
    plt.rcParams["grid.linestyle"] = ":" 
    plt.rcParams['axes.spines.right'] = False 
    plt.rcParams['axes.spines.top'] = False

    return None

def read_table(PATH):
    """
    Returns a pd.dataframe obtained from .dat file
    """
    data = os.path.join(DATA_PATH, PATH)
    table = pd.read_csv(data) 
    table = table.dropna()
    table = table.map(lambda x: x.strip() if isinstance(x, str) else x)
    table.columns = table.columns.str.strip()  # Remove leading/trailing spaces

    return table


def cut_dataframe(table, column, threshold):
    """
    Filters all the dataframe element with value greater then a given threshold
    """
    table = table.loc[table[column] <= threshold].copy()
    return table


def T2_estimate_weighted_avg(table):
    """
    Returns an estimate of T2 computed as the weighted average
    of all T2 obtained from UPEN.
    """
    table["Sig_Np"] = pd.to_numeric(table["Sig_Np"], errors='coerce')
    table["T"] = pd.to_numeric(table["T"], errors='coerce')
    table["SigT"] = pd.to_numeric(table["SigT"], errors='coerce')
    table["Sig"] = pd.to_numeric(table["Sig"], errors='coerce')

    sumDistrT2 = table["Sig_Np"].sum(skipna=True)
    weightsT2 = table["Sig_Np"] / sumDistrT2
    T2 = (table["T"] * weightsT2).sum(skipna=True)

    return T2


def T2_plots(table, title1, figure_name1, estimated_T2, title2,  figure_name2, title3, figure_name3):
    """
    Creates three figures related to T2:
        - figure1 = UPEN output after the inversion
        - figure2 = Signal behaviour vs time
        - figure3 = Fitted function and expected one comparison

    Args:
        -table: pd.dataframe containing lab data
        -title*: title for figure number *
        -figure_name*: name with which figure number *  needs to be saved
        -estimated_T2: T2 estimated via T2_estimate_weighted_avg()
    Returns:
        None
    """

    # Converting columns to numeric types
    table["Sig_Np"] = pd.to_numeric(table["Sig_Np"], errors='coerce')
    table["T"] = pd.to_numeric(table["T"], errors='coerce')
    table["SigT"] = pd.to_numeric(table["SigT"], errors='coerce')
    table["Sig"] = pd.to_numeric(table["Sig"], errors='coerce')
    
    # Customize
    sns_graphic_options()

    # Figure 1: Plot with log-scaled x-axis
    plt.figure(figsize=(10, 6))
    plt.plot(table["T"], table["Sig_Np"])
    plt.xscale('log')
    plt.xlabel("T2 (ms)")
    plt.ylabel("Signal density (a.u.)")
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) 

    plt.title(title1)
    plt.grid(True)

    filename1 = figure_name1
    filepath1 = os.path.join(FIGURE_PATH, filename1)
    plt.savefig(filepath1)
    #plt.show()


    table["Sig"] = table["Sig"].apply(lambda x: abs(x) if x <= 0 else x)
    table["SigT"] = table["SigT"].apply(lambda x: abs(x) if x <= 0 else x)

    table = cut_dataframe(table, 'SigT', threshold=5000)

    fit_parameters = stats.linregress(table["SigT"], np.log(table["Sig"]))
    intercept = fit_parameters[1] 
    slope = fit_parameters[0]
    intercept_stderr = fit_parameters.intercept_stderr
    slope_stderr = fit_parameters.stderr
    r_squared = (fit_parameters.rvalue)**2

    x_space = np.linspace(np.finfo(float).eps, max(table["SigT"]), 1000)
    y_pred = np.exp(intercept + slope * x_space)

    # Calculate confidence interval
    # Calculate standard error of prediction (SEP) for each x_space value
    # This accounts for the variability in the predictions along the regression line
    se_pred = slope_stderr * np.sqrt(1 + 1/len(table) + (x_space - table["SigT"].mean())**2 / np.sum((table["SigT"] - table["SigT"].mean())**2))

    # 95% confidence interval using the t-distribution
    t_value = stats.t.ppf(0.975, len(table) - 2)  # 2 parameters estimated (intercept and slope)
    conf_interval = t_value * se_pred

    # Plot confidence interval
    lower_bound = np.exp((intercept + conf_interval) + (slope - conf_interval)* x_space)
    upper_bound = np.exp((intercept - conf_interval) + (slope + conf_interval) * x_space)

    plt.figure(figsize=(10, 6))
    sns.regplot(data=table, x="SigT", y="Sig", fit_reg=False, label="Data points")
    plt.plot(x_space, y_pred, color='r', linestyle='-', linewidth=2, label="Fit curve")
    plt.fill_between(x_space, lower_bound, upper_bound, color='r', alpha=0.2, label='95% Confidence Interval')
    plt.xlabel("TE (ms)")
    plt.ylabel("Signal (a.u.)")
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) 

    plt.title(title2)
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    
    # Save Figure 2
    filename2 = figure_name2
    filepath2 = os.path.join(FIGURE_PATH, filename2)
    plt.savefig(filepath2)

    plt.figure(figsize=(10, 6))
    expected_curve = np.exp(intercept - (1 / estimated_T2) * x_space)
    plt.plot(x_space, y_pred, color='r', linestyle='-', linewidth=2, label="Fit Curve")
    plt.plot(x_space, expected_curve, color='g', linestyle='dashed', linewidth=2, label="Expected curve")
    plt.plot(x_space, y_pred - expected_curve, color='b', linestyle='-', linewidth=2, label="Fit - Expected curve")
    plt.xlabel("TE (ms)")
    plt.ylabel("Signal (a.u.)")
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) 

    plt.title(title3)
    #plt.yscale('log')
    plt.legend()
    plt.grid(True)
    
    # Save Figure 2
    filename3 = figure_name3
    filepath3 = os.path.join(FIGURE_PATH, filename3)
    plt.savefig(filepath3)
    #plt.show()

    return (intercept, slope), (intercept_stderr, slope_stderr), r_squared


def T1_estimate_weighted_avg(table):
    sumDistrT1 = table["Sig_Np"].sum(skipna=True)
    weightsT1 = table["Sig_Np"] / sumDistrT1
    T1 = (table["T"] * weightsT1).sum(skipna=True)

    return T1


def fnc(x, b1, b2):
    return b1 * (1 - 2 * np.exp(b2 * x))


def compute_r_squared(custom_function, ydata):
    # Residuals (difference between model and data)
    residuals = ydata - custom_function
    # Sum of squares of residuals
    ss_res = np.sum(residuals**2)
    # Total sum of squares (proportional to variance of data)
    ss_tot = np.sum((ydata - np.mean(ydata))**2)

    # R-squared
    r_squared = 1 - (ss_res / ss_tot)

    return r_squared


def T1_plots(table, title1, figure_name1, estimated_T1, title2, figure_name2, title3, figure_name3):

    # Converting columns to numeric types
    table["Sig_Np"] = pd.to_numeric(table["Sig_Np"], errors='coerce')
    table["T"] = pd.to_numeric(table["T"], errors='coerce')
    table["SigT"] = pd.to_numeric(table["SigT"], errors='coerce')
    table["Sig"] = pd.to_numeric(table["Sig"], errors='coerce')

    # Customize
    sns_graphic_options()

    plt.figure(figsize=(10, 6))
    plt.plot(table["T"], table["Sig_Np"])
    plt.xscale('log')
    plt.xlabel("T1 (ms)")
    plt.ylabel("Signal density (a.u.)")
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) 

    plt.title(title1)
    plt.grid(True)
    plt.savefig(os.path.join(FIGURE_PATH, figure_name1))

    time = table["SigT"].values[:64]
    signal = -1 * table["Sig"].values[:64]
    signal = signal - np.min(signal) / 2
    plt.figure(figsize=(10,6))
    sns.regplot(x = time, y = signal, fit_reg=False, label="Data points")
    #plt.scatter(time, signal)
    
    B0 = [1.5e06, -1/1000]
    popt, pcov = curve_fit(fnc, time, signal, p0=B0)
    perr = np.sqrt(np.diag(pcov))

    plt.plot(time, fnc(time, *popt), color='r', label='Fit curve')
    plt.xlabel("IT (ms)")
    plt.ylabel("Signal (a.u.)")
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) 
    plt.title(title2)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(FIGURE_PATH, figure_name2))


    plt.figure(figsize=(10,6))   
    fit_curve = fnc(time, *popt) 
    expected_curve = fnc(time, popt[0],-1 / estimated_T1) 
    plt.plot(time, fit_curve, color='r', label='Fit curve', linewidth=3)
    plt.plot(time, expected_curve, color='g', label='Expected curve', linestyle='dashed', linewidth=3, alpha = 0.8)
    plt.plot(time, fit_curve - expected_curve, color='b', label='Fit - Expected curve', linewidth=3, alpha = 0.8)
    plt.xlabel("IT (ms)")
    plt.ylabel("Signal (a.u.)")
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) 
    plt.title(title3)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(FIGURE_PATH, figure_name3))

    r_squared = compute_r_squared(fit_curve, signal)

    return popt, perr, r_squared


def inversion_recovery_plots(table_yolk, table_albumen, figure_name):
    # Converting columns to numeric types
    table_yolk["Sig_Np"] = pd.to_numeric(table_yolk["Sig_Np"], errors='coerce')
    table_yolk["T"] = pd.to_numeric(table_yolk["T"], errors='coerce')
    table_yolk["SigT"] = pd.to_numeric(table_yolk["SigT"], errors='coerce')
    table_yolk["Sig"] = pd.to_numeric(table_yolk["Sig"], errors='coerce')

    table_albumen["Sig_Np"] = pd.to_numeric(table_albumen["Sig_Np"], errors='coerce')
    table_albumen["T"] = pd.to_numeric(table_albumen["T"], errors='coerce')
    table_albumen["SigT"] = pd.to_numeric(table_albumen["SigT"], errors='coerce')
    table_albumen["Sig"] = pd.to_numeric(table_albumen["Sig"], errors='coerce')

    # Customize
    sns_graphic_options()

    time_yolk = table_yolk["SigT"].values[:64]
    signal_yolk = -1 * table_yolk["Sig"].values[:64]
    signal_yolk = signal_yolk - np.min(signal_yolk) / 2
    plt.figure(figsize=(10,6))
    sns.regplot(x = time_yolk, y = signal_yolk, fit_reg=False, label="Yolk data points")

    time_albumen = table_albumen["SigT"].values[:64]
    signal_albumen = -1 * table_albumen["Sig"].values[:64]
    signal_albumen = signal_albumen - np.min(signal_albumen) / 2
    sns.regplot(x = time_albumen, y = signal_albumen, fit_reg=False, label="Albumen data points")

    plt.xlabel("IT (ms)")
    plt.ylabel("Signal (a.u.)")
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) 
    plt.title("Inversion Recovery data for yolk and albumen")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(FIGURE_PATH, figure_name))


def compute_T_with_uncertainty(fit_param, std):
    T = - 1 / fit_param
    
    fract_uncertainty = std / fit_param
    T_std = T * np.abs(fract_uncertainty)

    return T, T_std

if __name__ == '__main__':
    table_CPMG_albumen = read_table(CPMG_ALBUMEN)
    table_CPMG_yolk = read_table(CPMG_YOLK)
    table_IR_albumen = read_table(IR_ALBUMEN)
    table_IR_yolk = read_table(IR_YOLK)

    
    T2_albumen = T2_estimate_weighted_avg(table_CPMG_albumen)
    fit_outcome_T2_albumen = T2_plots(table_CPMG_albumen, "Albumen Signal density vs T2", "ALBUMEN_T2_Upen_Inverted", T2_albumen,
              "Albumen Signal vs Echo Time", "ALBUMEN_T2_Upen_output&fit", "Albumen T2 signal Fitted & Expected curve", "ALBUMEN_T2_fitted&expected")

    T2_yolk = T2_estimate_weighted_avg(table_CPMG_yolk)
    fit_outcome_T2_yolk = T2_plots(table_CPMG_yolk, "Yolk Signal density vs T2", "YOLK_T2_Upen_Inverted", T2_yolk,
             "Yolk Signal vs Echo Time", "YOLK_T2_Upen_output&fit", "Yolk T2 signal Fitted & Expected curve",
             "YOLK_T2_fitted&expected")

    T1_albumen = T1_estimate_weighted_avg(table_IR_albumen)
    fit_outcome_T1_albumen = T1_plots(table_IR_albumen, "Albumen Signal density vs T1", "ALBUMEN_T1_Upen_Inverted", T1_albumen,
             "Albumen Signal vs Inversion Time", "ALBUMEN_T1_Upen_output&fit", "Albumen T1 signal Fitted & Expected curve", "ALBUMEN_T1_fitted&expected")
    
    T1_yolk = T1_estimate_weighted_avg(table_IR_yolk)
    fit_outcome_T1_yolk = T1_plots(table_IR_yolk, "Yolk Signal density vs T1", "YOLK_T1_Upen_Inverted", T1_yolk, 
             "Yolk Signal vs Inversion Time", "YOLK_T1_Upen_output&fit", "Yolk T1 signal Fitted & Expected curve",
             "YOLK_T1_fitted&expected")

    inversion_recovery_plots(table_IR_yolk, table_IR_albumen, "IR_scatter_plots")


    dict_fit_T2_yolk = {
        'Intercept (M_0)': f"{fit_outcome_T2_yolk[0][0]:.2f} ± {fit_outcome_T2_yolk[1][0]:.2f}",
        'Slope (-1/T2)': f"{fit_outcome_T2_yolk[0][1]:.5f} ± {fit_outcome_T2_yolk[1][1]:.5f}",
        'R^2': fit_outcome_T2_yolk[2]
        }

    dict_fit_T2_albumen = {
        'Intercept (M_0)': f"{fit_outcome_T2_albumen[0][0]:.2f} ± {fit_outcome_T2_albumen[1][0]:.2f}",
        'Slope (-1/T2)': f"{fit_outcome_T2_albumen[0][1]:.5f} ± {fit_outcome_T2_albumen[1][1]:.5f}",
        'R^2': fit_outcome_T2_albumen[2]
    }

    dict_fit_T1_yolk = {
        'Intercept (M_0)': f"{fit_outcome_T1_yolk[0][0]:.2f} ± {fit_outcome_T1_yolk[1][0]:.2f}",
        'Slope (-1/T2)': f"{fit_outcome_T1_yolk[0][1]:.5f} ± {fit_outcome_T1_yolk[1][1]:.5f}",
        'R^2': fit_outcome_T1_yolk[2]
    }    

    dict_fit_T1_albumen = {
        'Intercept (M_0)': f"{fit_outcome_T1_albumen[0][0]:.2f} ± {fit_outcome_T1_albumen[1][0]:.2f}",
        'Slope (-1/T2)': f"{fit_outcome_T1_albumen[0][1]:.5f} ± {fit_outcome_T1_albumen[1][1]:.5f}",
        'R^2': fit_outcome_T1_albumen[2]
    }    

    dict_T1_yolk = {
        'T1 weighted average': T1_yolk,
        'T1 from fit': f"{compute_T_with_uncertainty(fit_outcome_T1_yolk[0][1], fit_outcome_T1_yolk[1][1])[0]:.5f} ± {compute_T_with_uncertainty(fit_outcome_T1_yolk[0][1], fit_outcome_T1_yolk[1][1])[1]:.5f}"
    }

    dict_T1_albumen = {
        'T1 weighted average': T1_albumen,
        'T1 from fit': f"{compute_T_with_uncertainty(fit_outcome_T1_albumen[0][1], fit_outcome_T1_albumen[1][1])[0]:.5f} ± {compute_T_with_uncertainty(fit_outcome_T1_albumen[0][1], fit_outcome_T1_albumen[1][1])[1]:.5f}"
    }


    dict_T2_yolk = {
        'T2 weighted average': T2_yolk,
        'T2 from fit': f"{compute_T_with_uncertainty(fit_outcome_T2_yolk[0][1], fit_outcome_T2_yolk[1][1])[0]:.5f} ± {compute_T_with_uncertainty(fit_outcome_T2_yolk[0][1], fit_outcome_T2_yolk[1][1])[1]:.5f}"
    }

    dict_T2_albumen = {
        'T2 weighted average': T2_albumen,
        'T2 from fit': f"{compute_T_with_uncertainty(fit_outcome_T2_albumen[0][1], fit_outcome_T2_albumen[1][1])[0]:.5f} ± {compute_T_with_uncertainty(fit_outcome_T2_albumen[0][1], fit_outcome_T2_albumen[1][1])[1]:.5f}"
    }


    # Create a lists of dictionaries
    results_fit_T2 = [dict_fit_T2_albumen, dict_fit_T2_yolk]
    results_fit_T1 = [dict_fit_T1_albumen, dict_fit_T1_yolk]
    results_T1 = [dict_T1_albumen, dict_T1_yolk]
    results_T2 = [dict_T2_albumen, dict_T2_yolk]

    # Create pandas dataframes containing results     
    results_fit_T2_df = pd.DataFrame(results_fit_T2, index=['Albumen', 'Yolk'])
    results_fit_T1_df = pd.DataFrame(results_fit_T1, index=['Albumen', 'Yolk'])
    results_T1_df = pd.DataFrame(results_T1, index=['Albumen', 'Yolk'])
    results_T2_df = pd.DataFrame(results_T2, index=['Albumen', 'Yolk'])
    
    
    print(results_fit_T2_df.to_latex())
    print(results_fit_T1_df.to_latex())
    print(results_T1_df.to_latex())
    print(results_T2_df.to_latex())
    
