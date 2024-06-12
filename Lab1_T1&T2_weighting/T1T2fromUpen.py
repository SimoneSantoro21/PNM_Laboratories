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
    sumDistrT2 = table["Sig_Np"].sum(skipna=True)
    weightsT2 = table["Sig_Np"] / sumDistrT2
    T2 = (table["T"] * weightsT2).sum(skipna=True)

    return T2


def T2_plots(table, figure_name1, figure_name2):
    """
    Creates two figures related to T2:
        - figure1 = UPEN output after the inversion
        - figure2 = Signal behaviour vs time

    Args:
        -table: pd.dataframe containing lab data
        -figure_name1: name with which the first figure needs to be saved
        -figure_name2: name with which the second figure needs to be saved
    Returns:
        None
    """

    # Converting columns to numeric types
    table["Sig_Np"] = pd.to_numeric(table["Sig_Np"], errors='coerce')
    table["T"] = pd.to_numeric(table["T"], errors='coerce')
    table["SigT"] = pd.to_numeric(table["SigT"], errors='coerce')
    table["Sig"] = pd.to_numeric(table["Sig"], errors='coerce')
    
    # Customize
    sns.set_theme(style="ticks")
    sns.set_context("paper", font_scale=1.5)
    plt.rcParams["font.family"] = "serif"  
    plt.rcParams["axes.grid"] = True  
    plt.rcParams["grid.linestyle"] = ":" 
    plt.rcParams['axes.spines.right'] = False 
    plt.rcParams['axes.spines.top'] = False

    # Figure 1: Plot with log-scaled x-axis
    plt.figure(figsize=(10, 6))
    plt.plot(table["T"], table["Sig_Np"])
    plt.xscale('log')
    plt.xlabel("T")
    plt.ylabel("Sig_Np")
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) 

    plt.title("Plot with Log-Scaled X-Axis")
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
    stderr = fit_parameters.stderr

    x_space = np.linspace(np.finfo(float).eps, max(table["SigT"]), 1000)
    y_pred = np.exp(intercept + slope * x_space)

    # Calculate confidence interval
    # Calculate standard error of prediction (SEP) for each x_space value
    # This accounts for the variability in the predictions along the regression line
    se_pred = stderr * np.sqrt(1 + 1/len(table) + (x_space - table["SigT"].mean())**2 / np.sum((table["SigT"] - table["SigT"].mean())**2))

    # 95% confidence interval using the t-distribution
    t_value = stats.t.ppf(0.975, len(table) - 2)  # 2 parameters estimated (intercept and slope)
    conf_interval = t_value * se_pred

    # Plot confidence interval
    lower_bound = np.exp((intercept + conf_interval) + (slope - conf_interval)* x_space)
    upper_bound = np.exp((intercept - conf_interval) + (slope + conf_interval) * x_space)

    plt.figure(figsize=(10, 6))
    sns.regplot(data=table, x="SigT", y="Sig", fit_reg=False, label="Data points")
    plt.plot(x_space, y_pred, color='r', linestyle='-', linewidth=2, label=f'y = e^({intercept:.5f} + {slope:.5f}x)')
    plt.fill_between(x_space, lower_bound, upper_bound, color='r', alpha=0.2, label='95% Confidence Interval')
    plt.xlabel("SigT")
    plt.ylabel("Sig")
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) 

    plt.title("Scatter Plot with Log-Transformed Y-Axis and Linear Regression")
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    
    # Save Figure 2
    filename2 = figure_name2
    filepath2 = os.path.join(FIGURE_PATH, filename2)
    plt.savefig(filepath2)
    #plt.show()

    return None


def T1_estimate_weighted_avg(table):
    sumDistrT1 = table["Sig_Np"].sum(skipna=True)
    weightsT1 = table["Sig_Np"] / sumDistrT1
    T1 = (table["T"] * weightsT1).sum(skipna=True)

    return T1


def fnc(x, b1, b2, b3):
    return b1 * (1 - (1 + b3) * np.exp(b2 * x))


def T1_plots(table, figure_name1, figure_name2):

    # Converting columns to numeric types
    table["Sig_Np"] = pd.to_numeric(table["Sig_Np"], errors='coerce')
    table["T"] = pd.to_numeric(table["T"], errors='coerce')
    table["SigT"] = pd.to_numeric(table["SigT"], errors='coerce')
    table["Sig"] = pd.to_numeric(table["Sig"], errors='coerce')

    # Customize
    sns.set_theme(style="ticks")
    sns.set_context("paper", font_scale=1.5)
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.linestyle"] = ":" 
    plt.rcParams['axes.spines.right'] = False 
    plt.rcParams['axes.spines.top'] = False

    plt.figure(figsize=(10, 6))
    plt.plot(table["T"], table["Sig_Np"])
    plt.xscale('log')
    plt.xlabel("T")
    plt.ylabel("Sig_Np")
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) 

    plt.title("T vs Sig_Np Plot with Log-Scaled X-Axis")
    plt.grid(True)
    plt.savefig(os.path.join(FIGURE_PATH, figure_name1))

    time = table["SigT"].values[:64]
    signal = -1 * table["Sig"].values[:64]
    signal = signal - np.min(signal) / 2
    plt.figure(figsize=(10,6))
    sns.regplot(x = time, y = signal, fit_reg=False, label="Data points")
    #plt.scatter(time, signal)
    
    B0 = [1.5e06, -1/1000, 0.8]
    popt, _ = curve_fit(fnc, time, signal, p0=B0)
    plt.plot(time, fnc(time, *popt), color='r', label='Fitted curve')
    plt.xlabel("SigT")
    plt.ylabel("Sig")
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) 
    plt.title("Non-linear Model Fit")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(FIGURE_PATH, figure_name2))


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
    sns.set_theme(style="ticks")
    sns.set_context("paper", font_scale=1.5)
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.linestyle"] = ":" 
    plt.rcParams['axes.spines.right'] = False 
    plt.rcParams['axes.spines.top'] = False

    time_yolk = table_yolk["SigT"].values[:64]
    signal_yolk = -1 * table_yolk["Sig"].values[:64]
    signal_yolk = signal_yolk - np.min(signal_yolk) / 2
    plt.figure(figsize=(10,6))
    sns.regplot(x = time_yolk, y = signal_yolk, fit_reg=False, label="Yolk data points")

    time_albumen = table_albumen["SigT"].values[:64]
    signal_albumen = -1 * table_albumen["Sig"].values[:64]
    signal_albumen = signal_albumen - np.min(signal_albumen) / 2
    sns.regplot(x = time_albumen, y = signal_albumen, fit_reg=False, label="Albumen data points")

    plt.xlabel("SigT")
    plt.ylabel("Sig")
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) 
    plt.title("Non-linear Model Fit")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(FIGURE_PATH, figure_name))

if __name__ == '__main__':
    table_CPMG_albumen = read_table(CPMG_ALBUMEN)
    table_CPMG_yolk = read_table(CPMG_YOLK)
    table_IR_albumen = read_table(IR_ALBUMEN)
    table_IR_yolk = read_table(IR_YOLK)

    
    T2_plots(table_CPMG_albumen, "ALBUMEN_T2_Upen_Inverted", "ALBUMEN_T2_Upen_output&fit")
    T2 = T2_estimate_weighted_avg(table_CPMG_albumen)
    print(T2)
    T2_plots(table_CPMG_yolk, "YOLK_T2_Upen_Inverted", "YOLK_T2_Upen_output&fit")
    T2_estimate_weighted_avg(table_CPMG_yolk)

    T1_plots(table_IR_albumen, "ALBUMEN_T1_Upen_Inverted", "ALBUMEN_T1_Upen_output&fit")
    T1_estimate_weighted_avg(table_IR_albumen)
    T1_plots(table_IR_yolk, "YOLK_T1_Upen_Inverted", "YOLK_T1_Upen_output&fit")
    T1_estimate_weighted_avg(table_IR_yolk)

    inversion_recovery_plots(table_IR_yolk, table_IR_albumen, "IR_scatter_plots")
