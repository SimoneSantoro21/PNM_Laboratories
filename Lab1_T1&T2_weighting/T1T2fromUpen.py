import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats


FIGURE_PATH = 'figures'
DATA_PATH = 'data'

CPMG_ALBUMEN = 'CPMGalbume1B.dat'
CPMG_YOLK = 'CPMGtuorlo1B.dat'

IR_ALBUMEN = 'IRalbumeH.dat'
IR_YOLK = 'IRtuorloH.dat'

def read_table(PATH):
    data = os.path.join(DATA_PATH, PATH)
    return pd.read_csv(data)


def T_2_estimation(T, figure_name1, figure_name2):
    """Converts and visualizes data from a .dat table (T) using Seaborn,
       saving the final figure to the FIGURE_PATH" """
    
    # Summing and weighting
    T = T.dropna(subset=["SigT", "Sig"])
    sumDistrT2 = T["Sig_Np"].sum(skipna=True)
    weightsT2 = T["Sig_Np"] / sumDistrT2
    T2 = (T["T"] * weightsT2).sum(skipna=True)
    print(f'T2: {T2}')
    
    # Figure 1: Plot with log-scaled x-axis
    plt.figure()
    plt.plot(T["T"], T["Sig_Np"])
    plt.xscale('log')
    plt.xlabel("T")
    plt.ylabel("Sig_Np")
    plt.title("Plot with Log-Scaled X-Axis")
    plt.grid(True)

    filename1 = figure_name1
    filepath1 = os.path.join(FIGURE_PATH, filename1)
    plt.savefig(filepath1)
    plt.show()


    T["Sig"] = T["Sig"].apply(lambda x: abs(x) if x <= 0 else x)
    
    slope, intercept, _, _, _ = stats.linregress(T["SigT"], np.log(T["Sig"]))
    x_space = np.linspace(np.finfo(float).eps, max(T["SigT"]), 1000)
    y_pred = np.exp(intercept + slope * x_space)

    plt.figure(figsize=(10, 6))
    sns.regplot(data=T, x="SigT", y="Sig", fit_reg=False, scatter_kws={'alpha':0.6})  # Disable default reg line
    plt.plot(x_space, y_pred, color='r', linestyle='-', linewidth=2, label=f'y = e^({intercept:.2f} + {slope:.2f}x)')
    plt.xlabel("SigT")
    plt.ylabel("Sig")
    plt.title("Scatter Plot with Log-Transformed Y-Axis and Linear Regression")
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    
    # Save Figure 2
    filename2 = figure_name2
    filepath2 = os.path.join(FIGURE_PATH, filename2)
    plt.savefig(filepath2)
    plt.show()


if __name__ == '__main__':
    table = read_table(CPMG_ALBUMEN)
    table = table.map(lambda x: x.strip() if isinstance(x, str) else x)
    table.columns = table.columns.str.strip()  # Remove leading/trailing spaces

    # Converting columns to numeric types
    table["Sig_Np"] = pd.to_numeric(table["Sig_Np"], errors='coerce')
    table["T"] = pd.to_numeric(table["T"], errors='coerce')
    table["SigT"] = pd.to_numeric(table["SigT"], errors='coerce')
    table["Sig"] = pd.to_numeric(table["Sig"], errors='coerce')
    
    T_2_estimation(table, "Upen_Inverted", "Upen_output&fit")
