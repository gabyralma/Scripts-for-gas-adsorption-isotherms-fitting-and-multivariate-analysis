import os, math
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.cross_decomposition import PLSRegression

# --------- Matplotlib config: Arial + mathtext ----------
matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "font.size": 20,
    "axes.titlesize": 24,
    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "figure.titlesize": 24,
})
CMAP = "RdBu_r"  # red–blue divergent colormap

OUTDIR = "./outputs"
os.makedirs(OUTDIR, exist_ok=True)

def savefig(fig, name):
    path = os.path.join(OUTDIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=1000)
    plt.close(fig)
    print("Saved:", path)

def savecsv(df, name):
    path = os.path.join(OUTDIR, name)
    df.to_csv(path, index=False)
    print("Saved:", path)

# --------- Embedded dataset (12 samples, expanded responses) ----------
raw = [
    # Sample, A_BET, S_NLDFT, Vultra, Vsuper, Vmicro, Vmeso, Vtotal,
    # U_0.05bar_273K, U_0.2bar_273K, U_0.2bar_298K, U_0.2bar_323K,
    # U_1bar_273K, U_1bar_298K, U_1bar_323K, U_30bar_298K, U_30bar_323K
    ["Xerogel 4",  789,1081,0.19988168,0.129005098,0.328886777,0.146574074,0.457891875,
                   0.307552,0.990332,0.462046934,0.227214241,2.61372,1.2185,0.728262885,6.559496447,4.77702504],
    ["Xerogel 7",  798,1074,0.19970707,0.134920547,0.334627617,0.10667671, 0.469548164,
                   0.379914,0.911595,0.474178616,0.233903173,2.32328,1.3017,0.744839765,6.900092754,5.367491371],
    ["Xerogel 9",  738,1008,0.190958199,0.122047069,0.313005268,0.089240206,0.435052337,
                   0.291607,0.967967,0.453742802,0.215488501,2.55054,1.1979,0.660237249,6.418155933,4.577576977],
    ["Xerogel 11", 846,1145,0.192349461,0.147938135,0.340287597,0.318920993,0.488225732,
                   0.271653,0.958752,0.546895879,0.235857044,2.60023,1.4207,0.801330267,6.735614491,5.046580063],
    ["Aerogel 7",  1153,1323,0.118315382,0.238471717,0.3567871,  2.179677009,0.595258817,
                   0.20931, 0.5533,  0.364463033,0.176734919,1.65336,1.109618812,0.84,     9.988846944,6.95],
    ["Xero-Si",    629, 797,0.103786347,0.16352365, 0.267309998,0.208658053,11.00850455,
                   0.194651,0.523159,0.270495405,0.138456701,1.46925,0.821111991,0.51,    6.733612387,4.8],
    ["Aero-Si",    1497,1576,0.160943531,0.214618645,0.375562176,1.214199219,13.31272161,
                   0.176651,0.49986, 0.259301998,0.123070446,1.6239,1.278939915,0.5,      9.11197686, 5.79],
    ["Xero-SiAl4", 1378,1506,0.149528303,0.3726565,  0.522184803,0.655789249,6.553394684,
                   0.189274,0.549932,0.24212628, 0.118897711,1.73783,0.843544117,0.49,    7.722543526,5.44],
    ["Aero-SiAl4", 1928,1906,0.184833381,0.356098807,0.540932188,0.970773907,7.631080703,
                   0.170561,0.493961,0.266858655,0.122252667,1.61227,0.944075051,0.48,    8.766662016,5.89],
    ["Xero-SiZr",  1482,1239,0.101652367,0.327195409,0.428847777,0.599822979,2.67622545,
                   0.279795,0.686587,0.472921651,0.214646977,1.9592,1.528957493,0.87,     13.1884672, 8.32],
    ["Aero-SiZr",  1623,1612,0.147251777,0.259891082,0.407142859,1.052587366,8.230281667,
                   0.1943161,0.549846,0.471167678,0.229239825,1.78629,1.570965123,0.86,   11.41885452,7.7],
]

cols = ["Sample","A_BET","S_NLDFT","Vultra","Vsuper","Vmicro","Vmeso","Vtotal",
        "Upt_5e-2bar_273K","Upt_2e-1bar_273K","Upt_2e-1bar_298K","Upt_2e-1bar_323K",
        "Upt_1bar_273K","Upt_1bar_298K","Upt_1bar_323K","Upt_30bar_298K","Upt_30bar_323K"]
df = pd.DataFrame(raw, columns=cols)

# --------- Variable sets ----------
X_cols = ["A_BET","S_NLDFT","Vultra","Vsuper","Vmicro","Vmeso","Vtotal"]
Y_cols = [
    "Upt_5e-2bar_273K","Upt_2e-1bar_273K","Upt_2e-1bar_298K","Upt_2e-1bar_323K",
    "Upt_1bar_273K","Upt_1bar_298K","Upt_1bar_323K","Upt_30bar_298K","Upt_30bar_323K"
]

# --------- Pretty (mathtext) labels ----------
pretty = {
    "A_BET":          r'$A_{\mathrm{BET}}$',
    "S_NLDFT":        r'$S_{\mathrm{NLDFT}}$',
    "Vultra":         r'$V_{\mathrm{ultra}}$',
    "Vsuper":         r'$V_{\mathrm{super}}$',
    "Vmicro":         r'$V_{\mathrm{micro}}$',
    "Vmeso":          r'$V_{\mathrm{meso}}$',
    "Vtotal":         r'$V_{\mathrm{total}}$',
    "Upt_5e-2bar_273K": r'CO$_2$ uptake (0.05 bar, 273 K)',
    "Upt_2e-1bar_273K":  r'CO$_2$ uptake (0.2 bar, 273 K)',
    "Upt_2e-1bar_298K":  r'CO$_2$ uptake (0.2 bar, 298 K)',
    "Upt_2e-1bar_323K":  r'CO$_2$ uptake (0.2 bar, 323 K)',
    "Upt_1bar_273K":    r'CO$_2$ uptake (1 bar, 273 K)',
    "Upt_1bar_298K":    r'CO$_2$ uptake (1 bar, 298 K)',
    "Upt_1bar_323K":    r'CO$_2$ uptake (1 bar, 323 K)',
    "Upt_30bar_298K":   r'CO$_2$ uptake (30 bar, 298 K)',
    "Upt_30bar_323K":   r'CO$_2$ uptake (30 bar, 323 K)',
}

def plabel(name):  # pretty label with fallback
    return pretty.get(name, name)

# --------- 1) Correlation heatmap (X + Y) ----------
all_cols = X_cols + Y_cols
R = np.zeros((len(all_cols), len(all_cols)))
for i,a in enumerate(all_cols):
    for j,b in enumerate(all_cols):
        if i <= j:
            r,_ = pearsonr(df[a], df[b])
            R[i,j] = r
            R[j,i] = r

fig, ax = plt.subplots(figsize=(18,12))
im = ax.imshow(R, vmin=-1, vmax=1, cmap=CMAP)
ax.set_xticks(range(len(all_cols))); ax.set_yticks(range(len(all_cols)))
ax.set_xticklabels([plabel(c) for c in all_cols], rotation=45, ha='right')
ax.set_yticklabels([plabel(c) for c in all_cols])
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_title("Correlation matrix (predictors & responses)")
savefig(fig, "corr_all_redblue.png")
savecsv(pd.DataFrame(R, index=all_cols, columns=all_cols).round(3), "corr_matrix.csv")

# --------- 2) PCA on X ----------
X = df[X_cols].values
Xs = StandardScaler().fit_transform(X)
pca = PCA(n_components=2, random_state=0)
scores = pca.fit_transform(Xs)
loadings = pca.components_.T
expl = pca.explained_variance_ratio_

fig, ax = plt.subplots(figsize=(8,6))
ax.scatter(scores[:,0], scores[:,1])
for (x, y, label) in zip(scores[:,0], scores[:,1], df["Sample"].values):
    ax.text(x, y, label, fontsize=10)
scale = (np.max(np.abs(scores)) or 1.0)
for i, name in enumerate(X_cols):
    ax.arrow(0, 0, loadings[i,0]*scale, loadings[i,1]*scale,
             head_width=0.03*scale, length_includes_head=True)
    ax.text(loadings[i,0]*scale*1.08, loadings[i,1]*scale*1.08, plabel(name))
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
ax.set_title(f"PCA biplot (X only). Var: PC1={expl[0]:.2f}, PC2={expl[1]:.2f}")
savefig(fig, "pca_biplot.png")
savecsv(pd.DataFrame({"PC":["PC1","PC2"], "ExplainedVariance": expl[:2]}),
        "PCA_explained_variance.csv")

# --------- 3) PLS2 (X -> multiple Y) with CV, VIP, coefficients ----------
Y = df[Y_cols].values
Ys = StandardScaler().fit_transform(Y)

def mean_cv_r2(nc):
    pls = PLSRegression(n_components=nc)
    yhat = cross_val_predict(pls, Xs, Ys, cv=KFold(n_splits=5, shuffle=True, random_state=0))
    r2s = []
    for i in range(Ys.shape[1]):
        yt, yp = Ys[:,i], yhat[:,i]
        ss_res = np.sum((yt-yp)**2); ss_tot = np.sum((yt-np.mean(yt))**2)
        r2s.append(1 - ss_res/ss_tot)
    return float(np.mean(r2s))

max_comp = min(Xs.shape[1], Xs.shape[0]-1)
cv_rows, best_c, best_score = [], 1, -1e9
for c in range(1, max_comp+1):
    m = mean_cv_r2(c)
    cv_rows.append([c, m])
    if m > best_score:
        best_score, best_c = m, c

cv_df = pd.DataFrame(cv_rows, columns=["n_components","mean_CV_R2"])
savecsv(cv_df, "PLS_CV_results.csv")

fig, ax = plt.subplots(figsize=(6,4))
ax.plot(cv_df["n_components"], cv_df["mean_CV_R2"], marker="o")
ax.set_xlabel("PLS components"); ax.set_ylabel("Mean CV R$^2$ (across responses)")
ax.set_title("PLS component selection")
savefig(fig, "pls_cv_curve.png")

# Fit final PLS
pls = PLSRegression(n_components=best_c)
pls.fit(Xs, Ys)

# VIP (aggregated across Y)
T = pls.x_scores_; W = pls.x_weights_; Q = pls.y_loadings_
p, h = W.shape
SSY = np.sum((T @ Q.T)**2, axis=0)
total_SSY = np.sum(SSY)
vip = np.zeros((p,))
for j in range(p):
    wj = W[j, :]
    vip[j] = math.sqrt(p * np.sum(SSY * (wj**2) / np.sum(W**2, axis=0)) / total_SSY)
vip_df = pd.DataFrame({"feature": X_cols, "VIP": vip}).sort_values("VIP", ascending=False)
savecsv(vip_df, "PLS_VIP_scores.csv")

# Standardized coefficients per response
coef = pls.coef_  # (p x q), standardized
coef_df = pd.DataFrame(coef, index=X_cols, columns=Y_cols)
savecsv(coef_df.reset_index().rename(columns={"index":"feature"}),
        "PLS_coefficients_standardized.csv")

for j, yname in enumerate(Y_cols):
    fig, ax = plt.subplots(figsize=(9,4))
    ax.bar(range(len(X_cols)), coef[:,j])
    ax.set_xticks(range(len(X_cols)))
    ax.set_xticklabels([plabel(x) for x in X_cols], rotation=45, ha="right")
    ax.set_title(f"PLS standardized coefficients → {plabel(yname)}  (n_comp={best_c})")
    savefig(fig, f"pls_coef_{yname}.png")

# Predicted vs True (original units) with pretty labels
yhat_std = cross_val_predict(PLSRegression(n_components=best_c), Xs, Ys,
                             cv=KFold(n_splits=5, shuffle=True, random_state=0))
Y_scaler = StandardScaler().fit(Y)
yhat = Y_scaler.inverse_transform(yhat_std)

rows = []
for i, yname in enumerate(Y_cols):
    yt, yp = Y[:,i], yhat[:,i]
    ss_res = np.sum((yt-yp)**2); ss_tot = np.sum((yt-np.mean(yt))**2)
    r2 = 1 - ss_res/ss_tot
    rows.append([yname, r2])

    fig, ax = plt.subplots(figsize=(5,4))
    ax.scatter(yt, yp)
    mn = float(min(yt.min(), yp.min())); mx = float(max(yt.max(), yp.max()))
    ax.plot([mn, mx], [mn, mx])
    ax.set_xlabel("True"); ax.set_ylabel("Predicted (CV)")
    ax.set_title(f"{plabel(yname)} — CV R$^2$ = {r2:.2f}  (PLS comps = {best_c})")
    savefig(fig, f"pred_vs_true_{yname}.png")

savecsv(pd.DataFrame(rows, columns=["response","CV_R2"]), "PLS_CV_R2_per_response.csv")

print("\nDone. See the 'outputs6' folder for figures and CSV summaries.")
