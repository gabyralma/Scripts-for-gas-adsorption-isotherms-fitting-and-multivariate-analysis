import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.ticker as ticker
from pathlib import Path

# ---------------------- plotting style ----------------------
matplotlib.rcParams.update({
    "font.family": "Arial",
    "axes.labelsize": 18,
    "font.size": 18,
    "legend.fontsize": 16,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "text.usetex": False,     
})

# ---------------------- isotherm models ----------------------
def langmuir(P, Qm, b):
    """Langmuir: single-site adsorption."""
    return (Qm * b * P) / (1.0 + b * P)

def sips(P, Qm, b, n):
    """Sips (Langmuir–Freundlich)."""
    return (Qm * b * P ** n) / (1.0 + b * P ** n)

# ---------------------- fitting + plot helper ----------------------
def fit_and_plot(data, label, color, marker, ax,
                 show_residuals=False):
    """Fit Langmuir & Sips to one T-series and plot."""
    P, V = data[:, 0], data[:, 1]

    # ---------- Langmuir ----------
    p0_L  = [V.max(), 0.1]                     # [Qm, b]
    bndsL = ([0, 1e-6], [np.inf, np.inf])
    params_L, _ = curve_fit(
        langmuir, P, V, p0=p0_L, bounds=bndsL,
        method="trf", max_nfev=5000
    )

    # ---------- Sips ----------
    p0_S  = [V.max(), 0.1, 1.0]                # [Qm, b, n]
    bndsS = ([0, 1e-6, 0.1], [np.inf, np.inf, 5.0])
    params_S, _ = curve_fit(
        sips, P, V, p0=p0_S, bounds=bndsS,
        method="trf", max_nfev=100000000
    )

    # Predictions
    V_L = langmuir(P, *params_L)
    V_S = sips(P, *params_S)

    # Plot experimental points
    ax.scatter(P, V, color=color, marker=marker, s=70, label=label)

    # Plot model curves on the same P grid
    P_fit = np.linspace(P.min(), P.max(), 300)
    ax.plot(P_fit, langmuir(P_fit, *params_L),
            ls="--", color="blue",  lw=1.8, label=None)
    ax.plot(P_fit, sips(P_fit, *params_S),
            ls="--", color="red",   lw=1.8, label=None)

    # ---------------- metrics summary ----------------
    metrics = {
        "Langmuir": {
            "Qm": params_L[0], "b": params_L[1],
            "R2": r2_score(V, V_L),
            "RMSE": np.sqrt(mean_squared_error(V, V_L))
        },
        "Sips": {
            "Qm": params_S[0], "b": params_S[1], "n": params_S[2],
            "R2": r2_score(V, V_S),
            "RMSE": np.sqrt(mean_squared_error(V, V_S))
        }
    }
    print(f"\n=== {label} ===")
    for model, vals in metrics.items():
        params_str = ", ".join(f"{k}={v:.4g}" for k, v in vals.items()
                               if k not in ("R2", "RMSE"))
        print(f"{model:8s}: {params_str:<30s}"
              f"  R²={vals['R2']:.4f}  RMSE={vals['RMSE']:.4f}")

    # ------------- optional residual plot -------------
    if show_residuals:
        plt.figure()
        plt.scatter(P, V - V_S, c=color)
        plt.axhline(0, ls="--")
        plt.xlabel("Pressure (bar)")
        plt.ylabel("Residual (mmol g⁻¹)")
        plt.title(f"Residuals – Sips fit ({label})")
        plt.tight_layout()

# ---------------------- main script ----------------------
def main():
    for fname in ("data.dat", "data2.dat", "data3.dat"):
        if not Path(fname).is_file():
            raise FileNotFoundError(f"Missing {fname!r}")

    data_298K = np.loadtxt("data.dat")
    data_323K = np.loadtxt("data2.dat")
    data_348K = np.loadtxt("data3.dat")

    fig, ax = plt.subplots(figsize=(6, 8))
    ax.plot([], [], ls="--", color="blue",  label="Langmuir fit")
    ax.plot([], [], ls="--", color="red",   label="Sips fit")

    # Fit & plot each temperature
    fit_and_plot(data_298K, "298 K", "black", "o", ax)
    fit_and_plot(data_323K, "323 K", "green", "D", ax)
    fit_and_plot(data_348K, "348 K", "brown", "s", ax)

    ax.set_xlabel("Pressure (bar)")
    ax.set_ylabel("Quantity adsorbed (mmol g⁻¹)")
    ax.set_xlim(0, 35)
    ax.set_ylim(0, 12)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    ax.tick_params(axis="both", which="major", direction="out",
                   length=6, width=2)
    ax.tick_params(axis="both", which="minor", direction="out",
                   length=4, width=1)

    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
