import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.stats.diagnostic import het_white, acorr_breusch_godfrey
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, coherence, correlate
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from statsmodels.stats.stattools import jarque_bera
import imageio
import matplotlib
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# INDEX LABELS — full names for the nine global stock market indices
# Used only in diagnostic output; does not affect any computation.
# ==============================================================================
INDEX_NAMES = {
    "NYA":   "NYSE Composite (NYA)",
    "N100":  "Euronext 100 (N100)",
    "RUT":   "Russell 2000 (RUT)",
    "GDAXI": "DAX 40 (GDAXI)",
    "N225":  "Nikkei 225 (N225)",
    "NDX":   "NASDAQ-100 (NDX)",
    "FCHI":  "CAC 40 (FCHI)",
    "SSMI":  "SMI Swiss (SSMI)",
    "TWII":  "TAIEX Taiwan (TWII)",
}

def get_pair_full_label(label):
    parts = label.split("-")
    if len(parts) == 2:
        return f"{INDEX_NAMES.get(parts[0], parts[0])} / {INDEX_NAMES.get(parts[1], parts[1])}"
    return label

def run_adf_test(series, series_name="Series"):
    """
    Augmented Dickey-Fuller test for unit roots.
    Reference: Dickey & Fuller (1979). JASA, 74, 427-431.
    """
    result = adfuller(series, autolag='AIC')
    adf_stat, p_value, used_lag, nobs, crit_vals, _ = result
    print(f"\n  ADF — {series_name}")
    print(f"    Statistic: {adf_stat:.4f}  p: {p_value:.4f}  lags: {used_lag}  N: {nobs}")
    for k, v in crit_vals.items():
        print(f"    CV {k}: {v:.4f}")
    return {"stat": adf_stat, "pvalue": p_value, "lags": used_lag, "nobs": nobs}


def run_phillips_perron_proxy(series, series_name="Series"):
    """
    Phillips-Perron test proxy (ADF with zero lags).
    Reference: Phillips & Perron (1988). Biometrika, 75(2), 335-346.
    """
    result = adfuller(series, maxlag=0, autolag=None, regression='c')
    stat, p_value = result[0], result[1]
    nobs = result[3]
    print(f"\n  PP-proxy — {series_name}")
    print(f"    Statistic: {stat:.4f}  p: {p_value:.4f}  N: {nobs}")
    return {"stat": stat, "pvalue": p_value, "nobs": nobs}


def run_full_unit_root_suite(series_dict):
    """
    ADF and PP tests for all nine indices.
    Note: with trending financial series, low p-values are expected and do not
    imply substantive economic significance per se.
    """
    print("\n" + "=" * 70)
    print("UNIT ROOT TESTS — ADF and Phillips-Perron proxy")
    print("Note: low p-values in trending series do not imply substantive significance.")
    print("=" * 70)
    rows = []
    for name, series in series_dict.items():
        adf = run_adf_test(np.log(series), series_name=name)
        pp  = run_phillips_perron_proxy(np.log(series), series_name=name)
        rows.append({"Index": name,
                     "ADF Stat": round(adf["stat"], 4), "ADF p": round(adf["pvalue"], 4),
                     "PP Stat":  round(pp["stat"],  4), "PP p":  round(pp["pvalue"],  4),
                     "N": adf["nobs"]})
    df = pd.DataFrame(rows)
    print("\nSummary:\n", df.to_string(index=False))
    return df


def run_engle_granger_cointegration(PX, PY, label="Pair"):
    """
    Engle-Granger two-step cointegration test on log prices.
    Reference: Engle & Granger (1987). Econometrica, 55(2), 251-276.
    """
    min_len = min(len(PX), len(PY))
    stat, pval, cvs = coint(np.log(PX[:min_len]), np.log(PY[:min_len]))
    print(f"\n  Engle-Granger — {label}")
    print(f"    EG stat: {stat:.4f}  p: {pval:.4f}  CV(1%,5%,10%): {cvs.round(4)}")
    return {"stat": stat, "pvalue": pval, "crit_vals": cvs}


def print_ols_diagnostics(models_dict):
    """
    For each fitted OLS model (from regression_with_at_proxy), prints:
      - Full coefficient table: coef, SE, t, p, 95% CI
      - N, df_resid, R², Adj.R², F, Prob(F)
      - Breusch-Godfrey (4 lags) serial correlation test
      - White heteroskedasticity test
      - Jarque-Bera normality test on residuals

    References:
      Breusch & Godfrey (1978). Journal of Econometrics, 7(3), 333-349.
      White (1980). Econometrica, 48(4), 817-838.
      Jarque & Bera (1987). International Statistical Review, 55, 163-172.

    IMPORTANT CAVEAT printed with each model: the AT proxy is a deterministic
    sigmoid — a monotonically increasing function of time. Regressions of
    declining spectral frequencies on this proxy mechanically yield negative
    coefficients and non-trivial R². The R² reflects temporal co-movement
    between two trends, NOT causal identification of AT. With trending time
    series, low p-values are expected and do not indicate substantive
    significance. Results are descriptive only.
    """
    print("\n" + "=" * 70)
    print("OLS DIAGNOSTIC TABLES (added for reviewer)")
    print("CAVEAT: AT proxy is deterministic. R² measures temporal co-movement,")
    print("        not causal identification. Low p-values expected in trending series.")
    print("=" * 70)

    rows = []
    for label, model in models_dict.items():
        full_lbl = get_pair_full_label(label)
        print(f"\n{'─'*70}")
        print(f"Pair: {full_lbl}  [{label}]")
        print(f"{'─'*70}")
        print(f"  N={int(model.nobs)}  df_resid={int(model.df_resid)}"
              f"  R²={model.rsquared:.4f}  Adj.R²={model.rsquared_adj:.4f}"
              f"  F={model.fvalue:.4f}  Prob(F)={model.f_pvalue:.4f}")

        # Coefficient table with 95% CI
        ci_raw = model.conf_int(alpha=0.05)
        ci = pd.DataFrame(ci_raw) if not hasattr(ci_raw, 'iloc') else ci_raw
        print(f"\n  {'Variable':<18} {'Coef':>10} {'SE':>10} {'t':>10} "
              f"{'p':>8} {'CI_low':>10} {'CI_high':>10}")
        print(f"  {'─'*78}")
        for idx_p, pname in enumerate(model.model.exog_names):
            print(f"  {pname:<18} {model.params[idx_p]:>10.4f} "
                  f"{model.bse[idx_p]:>10.4f} "
                  f"{model.tvalues[idx_p]:>10.4f} "
                  f"{model.pvalues[idx_p]:>8.4f} "
                  f"{ci.iloc[idx_p, 0]:>10.4f} "
                  f"{ci.iloc[idx_p, 1]:>10.4f}")

        # Breusch-Godfrey
        bg_stat, bg_p, _, _ = acorr_breusch_godfrey(model, nlags=4)

        # White
        try:
            w_stat, w_p, _, _ = het_white(model.resid, model.model.exog)
        except Exception:
            w_stat, w_p = np.nan, np.nan

        # Jarque-Bera
        jb_stat, jb_p, skew, kurt = jarque_bera(model.resid)

        print(f"\n  Breusch-Godfrey (4 lags): stat={bg_stat:.4f}  p={bg_p:.4f}")
        print(f"  White heterosked.       : stat={w_stat:.4f}  p={w_p:.4f}")
        print(f"  Jarque-Bera             : stat={jb_stat:.4f}  p={jb_p:.4f}"
              f"  skew={skew:.3f}  kurt={kurt:.3f}")
        print(f"\n  ⚠ R²={model.rsquared:.2%} reflects temporal co-movement, NOT causal AT effect.")

        rows.append({
            "label": label, "N": int(model.nobs), "df": int(model.df_resid),
            "R2": model.rsquared, "AdjR2": model.rsquared_adj,
            "F": model.fvalue, "ProbF": model.f_pvalue,
            "const": model.params[0], "const_SE": model.bse[0],
            "const_t": model.tvalues[0], "const_p": model.pvalues[0],
            "AT_coef": model.params[1], "AT_SE": model.bse[1],
            "AT_t": model.tvalues[1], "AT_p": model.pvalues[1],
            "AT_CI_low": ci.iloc[1, 0], "AT_CI_high": ci.iloc[1, 1],
            "BG_stat": bg_stat, "BG_p": bg_p,
            "White_stat": w_stat, "White_p": w_p,
            "JB_stat": jb_stat, "JB_p": jb_p,
            "skew": skew, "kurt": kurt,
        })

    df_diag = pd.DataFrame(rows)
    return df_diag


def export_ols_latex(df_diag, filepath="ols_table.tex"):
    """Exports the OLS diagnostic table to LaTeX."""
    cols = ["label", "N", "R2", "AdjR2", "AT_coef", "AT_SE", "AT_t", "AT_p",
            "AT_CI_low", "AT_CI_high", "BG_p", "White_p", "JB_p"]
    rename = {
        "label": "Pair", "N": "N", "R2": r"$R^2$", "AdjR2": r"Adj.$R^2$",
        "AT_coef": "Coef.", "AT_SE": "SE", "AT_t": r"$t$", "AT_p": r"$p$",
        "AT_CI_low": r"CI$_{low}$", "AT_CI_high": r"CI$_{high}$",
        "BG_p": "BG $p$", "White_p": "White $p$", "JB_p": "JB $p$"
    }
    latex = df_diag[cols].rename(columns=rename).to_latex(
        index=False, float_format="%.4f",
        caption=("OLS: freq\\_dom2 $\\sim$ AT diffusion proxy (sigmoid, $t_0=2010$, $k=0.25$). "
                 "Standard OLS SE. BG=Breusch-Godfrey (4 lags); White=heteroskedasticity test; "
                 "JB=Jarque-Bera. "
                 "Note: low $p$-values expected in trending series; $R^2$ reflects "
                 "temporal co-movement, not causal identification."),
        label="tab:ols_diagnostics")
    with open(filepath, "w") as f:
        f.write(latex)
    print(f"\nLaTeX table saved to {filepath}")

def advanced_oscillator_analysis(results_list):
    freq_keys = ['freq_dom1', 'freq_dom2', 'freq_dom3']
    mag_keys = ['mag_dom1', 'mag_dom2', 'mag_dom3']
    n = len(results_list)
    labels = [res['label'] for res in results_list]

    def compute_metrics(series1, series2):
        pearson = pearsonr(series1, series2)[0]
        s1 = np.asarray(series1, dtype=float)
        s2 = np.asarray(series2, dtype=float)
        mask = ~np.isnan(s1) & ~np.isnan(s2)
        s1 = s1[mask]; s2 = s2[mask]
        ds1 = np.diff(s1); ds2 = np.diff(s2)
        if len(ds1) < 8:
            coherence_avg = np.nan
        else:
            nperseg = max(8, min(128, len(ds1) // 3))
            f, Cxy = coherence(ds1, ds2, fs=1.0, nperseg=nperseg, detrend='constant')
            coherence_avg = np.nanmean(Cxy)
        corr = correlate(series1 - np.mean(series1), series2 - np.mean(series2), mode='full')
        lags = np.arange(-len(series1) + 1, len(series1))
        optimal_lag = lags[np.argmax(corr)]
        max_corr = np.max(corr) / (np.std(series1) * np.std(series2) * len(series1))
        mi = mutual_info_regression(np.array(series1).reshape(-1, 1), np.array(series2))[0]
        return pearson, coherence_avg, max_corr, mi, optimal_lag

    for fk in freq_keys:
        print(f"\nAnalysis for {fk.replace('_', ' ').title()}")
        all_series = [res[fk] for res in results_list]
        pearson_mat = np.eye(n); coherence_mat = np.eye(n)
        crosscorr_mat = np.eye(n); mi_mat = np.eye(n); lag_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    p, c, cc, mi, lag = compute_metrics(all_series[i], all_series[j])
                    pearson_mat[i, j] = p; coherence_mat[i, j] = c
                    crosscorr_mat[i, j] = cc; mi_mat[i, j] = mi; lag_mat[i, j] = lag
        metrics = {
            'Pearson': (pearson_mat, -1, 1),
            'Coherence': (coherence_mat, 0, 1),
            'Cross Corr': (crosscorr_mat, -1, 1),
            'Mutual Info': (mi_mat, 0, np.max(mi_mat))
        }
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Coupling Analysis for {fk.replace("_", " ").title()}', y=1.02)
        for idx, (title, (data, vmin, vmax)) in enumerate(metrics.items()):
            ax = axs[idx // 2, idx % 2]
            cax = ax.matshow(data, cmap='viridis' if title == 'Mutual Info' else 'coolwarm',
                             vmin=vmin, vmax=vmax)
            plt.colorbar(cax, ax=ax)
            ax.set_title(f"{title} (Average: {np.nanmean(data[np.eye(n) == 0]):.2f})")
            ax.set_xticks(range(n)); ax.set_yticks(range(n))
            ax.set_xticklabels(labels, rotation=90); ax.set_yticklabels(labels)
        plt.tight_layout(); plt.show()

    for mk in mag_keys:
        print(f"\nAnalysis for {mk.replace('_', ' ').title()}")
        all_series = [res[mk] for res in results_list]
        corr_mat = np.eye(n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    corr_mat[i, j] = pearsonr(all_series[i], all_series[j])[0]
        plt.figure(figsize=(8, 6))
        plt.matshow(corr_mat, cmap='coolwarm', vmin=-1, vmax=1, fignum=0)
        plt.colorbar()
        plt.title(f'Pearson Correlation for {mk.replace("_", " ").title()}\n'
                  f'Average: {np.nanmean(corr_mat[np.eye(n) == 0]):.2f}')
        plt.xticks(range(n), labels, rotation=90); plt.yticks(range(n), labels)
        plt.show()


def pairwise_correlation_analysis(results_list):
    freq_keys = ['freq_dom1', 'freq_dom2', 'freq_dom3']
    mag_keys  = ['mag_dom1',  'mag_dom2',  'mag_dom3']

    def process_and_plot(key, data_label):
        data_all = []; labels = []
        for res in results_list:
            if len(res[key]) > 0:
                data_all.append(res[key]); labels.append(res['label'])
        if len(data_all) < 2:
            print(f"Not enough series for {key}."); return
        min_length = min(len(arr) for arr in data_all)
        data_all = np.array([arr[:min_length] for arr in data_all])
        corr_matrix = np.corrcoef(data_all)
        n = corr_matrix.shape[0]
        avg_corr = (np.sum(corr_matrix) - np.trace(corr_matrix)) / (n * (n - 1))
        print(f"Average correlation for {data_label} ({key}): {avg_corr:.2f}")
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, xticklabels=labels, yticklabels=labels, cmap='coolwarm')
        plt.title(f'Correlation Matrix: {data_label} ({key})\nAverage: {avg_corr:.2f}')
        plt.tight_layout(); plt.show()

    for key in freq_keys:
        process_and_plot(key, "Dominant Frequency")
    for key in mag_keys:
        process_and_plot(key, "Magnitude")

    def weighted_average_correlation(results_list):
        weighted_series = []; labels = []
        for res in results_list:
            if all(len(res[k]) > 0 for k in
                   ['freq_dom1','freq_dom2','freq_dom3','mag_dom1','mag_dom2','mag_dom3']):
                min_len = min(len(res[k]) for k in
                              ['freq_dom1','freq_dom2','freq_dom3','mag_dom1','mag_dom2','mag_dom3'])
                f1=np.array(res['freq_dom1'][:min_len]); f2=np.array(res['freq_dom2'][:min_len])
                f3=np.array(res['freq_dom3'][:min_len]); m1=np.array(res['mag_dom1'][:min_len])
                m2=np.array(res['mag_dom2'][:min_len]); m3=np.array(res['mag_dom3'][:min_len])
                weighted_freq = (f1*m1 + f2*m2 + f3*m3) / (m1 + m2 + m3)
                weighted_series.append(weighted_freq); labels.append(res['label'])
        common_length = min(len(s) for s in weighted_series)
        weighted_series = np.array([s[:common_length] for s in weighted_series])
        print(f"Shape of weighted series: {weighted_series.shape}")
        corr_matrix = np.corrcoef(weighted_series)
        n = corr_matrix.shape[0]
        mean_corr = (np.sum(corr_matrix) - np.trace(corr_matrix)) / (n * (n - 1))
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, xticklabels=labels, yticklabels=labels, cmap='coolwarm')
        plt.title(f'Correlation Matrix of Weighted Averages\nAverage: {mean_corr:.2f}')
        plt.tight_layout(); plt.show()

    weighted_average_correlation(results_list)


def regression_with_at_proxy(results, t0=2010, k=0.25):
    """
    t0=2010, k=0.25 are the original defaults.
    """
    metrics_list = []; models = {}
    for res in results:
        label = res["label"]
        times = np.array(res["times"])
        freq1 = np.array(res["freq_dom1"])
        AT_proxy = 1 / (1 + np.exp(-k * (times - t0)))
        X = sm.add_constant(AT_proxy)
        model = sm.OLS(freq1, X).fit()   # standard OLS, no cov_type change
        models[label] = model
        metrics = {}
        metrics["label"] = label
        metrics["R_squared"] = model.rsquared
        metrics["Adj_R_squared"] = model.rsquared_adj
        metrics["F_statistic"] = model.fvalue
        metrics["Prob_F"] = model.f_pvalue
        metrics["const_coef"] = model.params[0]
        metrics["const_pvalue"] = model.pvalues[0]
        metrics["AT_proxy_coef"] = model.params[1]
        metrics["AT_proxy_pvalue"] = model.pvalues[1]
        jb_stat, jb_pvalue, skew, kurtosis = jarque_bera(model.resid)
        metrics["JB_stat"] = jb_stat; metrics["JB_pvalue"] = jb_pvalue
        metrics["Skew"] = skew; metrics["Kurtosis"] = kurtosis
        metrics_list.append(metrics)
        print(f"Regression results for series {label}:")
        print(model.summary())
        print("\n" + "-" * 80 + "\n")
    df_metrics = pd.DataFrame(metrics_list)
    return df_metrics, models


def plot_regression_comparison(df_metrics):
    sns.set(style="whitegrid")
    for col, title, palette, threshold in [
        ("R_squared",       "R-squared per Series",                 "Blues_d",   None),
        ("AT_proxy_coef",   "AT_proxy Coefficient per Series",      "Greens_d",  None),
        ("AT_proxy_pvalue", "AT_proxy P-value per Series",          "Reds_d",    0.05),
        ("JB_stat",         "Jarque-Bera Statistic per Series",     "Purples_d", None),
        ("Kurtosis",        "Kurtosis per Series",                  "Oranges_d", None),
        ("JB_pvalue",       "Jarque-Bera P-value per Series",       "Greys_d",   0.05),
    ]:
        plt.figure(figsize=(10, 5))
        sns.barplot(x="label", y=col, data=df_metrics, palette=palette)
        plt.title(title); plt.xlabel("Series"); plt.ylabel(col); plt.xticks(rotation=45)
        mean_val = df_metrics[col].mean()
        plt.axhline(mean_val, color="black", linestyle="--", label=f"Average: {mean_val:.3f}")
        if threshold is not None:
            plt.axhline(threshold, color="blue", linestyle=":", label=f"Threshold: {threshold}")
        plt.legend(); plt.tight_layout(); plt.show()


def aggregate_fft_results(series_list, step):
    results_list = []
    for PX, PY, label in series_list:
        result = analyze_fft_evolution(PX, PY, step, False)
        result['label'] = label
        results_list.append(result)

    for dom_num, key in enumerate(['freq_dom1', 'freq_dom2', 'freq_dom3'], 1):
        plt.figure(figsize=(14, 8))
        for res in results_list:
            plt.plot(res['times'], res[key], marker='o', label=f"{res['label']} - {dom_num}st Freq")
        plt.xlabel('Year'); plt.ylabel('Frequency')
        plt.title(f'Comparison of {dom_num}st Dominant Frequency')
        plt.legend(); plt.tight_layout(); plt.show()

    for dom_num, key in enumerate(['mag_dom1', 'mag_dom2', 'mag_dom3'], 1):
        plt.figure(figsize=(14, 8))
        for res in results_list:
            plt.plot(res['times'], res[key], marker='o', label=f"{res['label']} - Magnitude {dom_num}st")
        plt.xlabel('Year'); plt.ylabel('Magnitude')
        plt.title(f'Comparison of Magnitudes - {dom_num}st Dominant Frequency')
        plt.legend(); plt.tight_layout(); plt.show()

    for dom_num, key in enumerate(['rel_change_freq1','rel_change_freq2','rel_change_freq3'], 1):
        markers = ['o', 'x', 's']
        plt.figure(figsize=(14, 8))
        for res in results_list:
            plt.plot(res['times'], res[key], marker=markers[dom_num-1],
                     label=f"{res['label']} - Rel. Δ {dom_num}st Freq")
        plt.xlabel('Year'); plt.ylabel('Relative Change')
        plt.title(f'Instantaneous Relative Change - {dom_num}st Frequency')
        plt.legend(); plt.tight_layout(); plt.show()

    for dom_num, key in enumerate(['cum_rel_change_freq1','cum_rel_change_freq2','cum_rel_change_freq3'], 1):
        markers = ['o', 'x', 's']
        plt.figure(figsize=(14, 8))
        for res in results_list:
            plt.plot(res['times'], res[key], marker=markers[dom_num-1],
                     label=f"{res['label']} - Cumul. Δ {dom_num}st Freq")
        plt.xlabel('Year'); plt.ylabel('Cumulative Relative Change')
        plt.title(f'Cumulative Change - {dom_num}st Frequency')
        plt.legend(); plt.tight_layout(); plt.show()

    for dom_num, key in enumerate(['rel_change_mag1','rel_change_mag2','rel_change_mag3'], 1):
        markers = ['o', 'x', 's']
        plt.figure(figsize=(14, 8))
        for res in results_list:
            plt.plot(res['times'], res[key], marker=markers[dom_num-1],
                     label=f"{res['label']} - Rel. Δ Mag {dom_num}st")
        plt.xlabel('Year'); plt.ylabel('Relative Change')
        plt.title(f'Instantaneous Relative Change - {dom_num}st Magnitude')
        plt.legend(); plt.tight_layout(); plt.show()

    for dom_num, key in enumerate(['cum_rel_change_mag1','cum_rel_change_mag2','cum_rel_change_mag3'], 1):
        markers = ['o', 'x', 's']
        plt.figure(figsize=(14, 8))
        for res in results_list:
            plt.plot(res['times'], res[key], marker=markers[dom_num-1],
                     label=f"{res['label']} - Cumul. Δ Mag {dom_num}st")
        plt.xlabel('Year'); plt.ylabel('Cumulative Relative Change')
        plt.title(f'Cumulative Change - {dom_num}st Magnitude')
        plt.legend(); plt.tight_layout(); plt.show()

    pairwise_correlation_analysis(results_list)
    advanced_oscillator_analysis(results_list)
    return results_list


def analyze_fft_evolution(PX, PY, step, plot=False):
    a, b = calculate_cointegration_params(PX, PY)
    num_points = (len(PX) - 365) // step
    years_array = np.linspace(2000, 2024, num_points)
    times = []; freq_dom1=[]; freq_dom2=[]; freq_dom3=[]
    mag_dom1=[]; mag_dom2=[]; mag_dom3=[]
    for i, year in zip(range(0, len(PX) - 365, step), years_array):
        fft, freq, peaks, _, _, _ = calculate_fft_and_spreads(PX, PY, i, a, b)
        times.append(year)
        freq_dom1.append(freq[peaks[0]]); freq_dom2.append(freq[peaks[1]]); freq_dom3.append(freq[peaks[2]])
        mag_dom1.append(abs(fft[peaks[0]])); mag_dom2.append(abs(fft[peaks[1]])); mag_dom3.append(abs(fft[peaks[2]]))
    times=np.array(times); freq_dom1=np.array(freq_dom1); freq_dom2=np.array(freq_dom2); freq_dom3=np.array(freq_dom3)
    mag_dom1=np.array(mag_dom1); mag_dom2=np.array(mag_dom2); mag_dom3=np.array(mag_dom3)

    rel_change_freq1=np.full_like(freq_dom1,np.nan,dtype=float)
    rel_change_freq2=np.full_like(freq_dom2,np.nan,dtype=float)
    rel_change_freq3=np.full_like(freq_dom3,np.nan,dtype=float)
    rel_change_mag1=np.full_like(mag_dom1,np.nan,dtype=float)
    rel_change_mag2=np.full_like(mag_dom2,np.nan,dtype=float)
    rel_change_mag3=np.full_like(mag_dom3,np.nan,dtype=float)
    for j in range(1, len(times)):
        rel_change_freq1[j]=(freq_dom1[j]-freq_dom1[j-1])/freq_dom1[j-1]
        rel_change_freq2[j]=(freq_dom2[j]-freq_dom2[j-1])/freq_dom2[j-1]
        rel_change_freq3[j]=(freq_dom3[j]-freq_dom3[j-1])/freq_dom3[j-1]
        rel_change_mag1[j]=(mag_dom1[j]-mag_dom1[j-1])/mag_dom1[j-1]
        rel_change_mag2[j]=(mag_dom2[j]-mag_dom2[j-1])/mag_dom2[j-1]
        rel_change_mag3[j]=(mag_dom3[j]-mag_dom3[j-1])/mag_dom3[j-1]
    cum_rel_change_freq1=np.nancumsum(np.nan_to_num(rel_change_freq1))
    cum_rel_change_freq2=np.nancumsum(np.nan_to_num(rel_change_freq2))
    cum_rel_change_freq3=np.nancumsum(np.nan_to_num(rel_change_freq3))
    cum_rel_change_mag1=np.nancumsum(np.nan_to_num(rel_change_mag1))
    cum_rel_change_mag2=np.nancumsum(np.nan_to_num(rel_change_mag2))
    cum_rel_change_mag3=np.nancumsum(np.nan_to_num(rel_change_mag3))
    return {
        'times': times, 'freq_dom1': freq_dom1, 'freq_dom2': freq_dom2, 'freq_dom3': freq_dom3,
        'mag_dom1': mag_dom1, 'mag_dom2': mag_dom2, 'mag_dom3': mag_dom3,
        'rel_change_freq1': rel_change_freq1, 'rel_change_freq2': rel_change_freq2, 'rel_change_freq3': rel_change_freq3,
        'rel_change_mag1': rel_change_mag1, 'rel_change_mag2': rel_change_mag2, 'rel_change_mag3': rel_change_mag3,
        'cum_rel_change_freq1': cum_rel_change_freq1, 'cum_rel_change_freq2': cum_rel_change_freq2, 'cum_rel_change_freq3': cum_rel_change_freq3,
        'cum_rel_change_mag1': cum_rel_change_mag1, 'cum_rel_change_mag2': cum_rel_change_mag2, 'cum_rel_change_mag3': cum_rel_change_mag3
    }


def load_stock_data(file_name):
    path = f"C:/q/dash/sample/data/stocks/{file_name}.csv"
    try:
        df = pd.read_csv(path)
        print(f"CSV file {file_name} loaded successfully.")
        if 'Close' in df.columns:
            return df['Close']
        else:
            print(f"Error: 'Close' column not found in {file_name}."); return None
    except FileNotFoundError:
        print(f"Error: File {path} not found."); return None
    except Exception as e:
        print(f"Error: {e}"); return None


def calculate_frequency_differences(PX, PY, step):
    a, b = calculate_cointegration_params(PX, PY)
    last_fft, last_freq, last_peaks, _, _, p_value = calculate_fft_and_spreads(PX, PY, 0, a, b)
    diffs1=[]; diffs2=[]; diffs3=[]; dm1=[]; dm2=[]; dm3=[]; p_values=[]; f=[]
    for i in range(0, len(PX) - 365, step):
        fft, freq, peaks, _, _, p_value = calculate_fft_and_spreads(PX, PY, i, a, b)
        diffs1.append((freq[peaks[0]]-last_freq[last_peaks[0]])/last_freq[last_peaks[0]])
        diffs2.append((freq[peaks[1]]-last_freq[last_peaks[1]])/last_freq[last_peaks[1]])
        diffs3.append((freq[peaks[2]]-last_freq[last_peaks[2]])/last_freq[last_peaks[2]])
        dm1.append(float((fft[peaks[0]]-last_fft[last_peaks[0]])/last_fft[last_peaks[0]]))
        dm2.append(float((fft[peaks[1]]-last_fft[last_peaks[1]])/last_fft[last_peaks[1]]))
        dm3.append(float((fft[peaks[2]]-last_fft[last_peaks[2]])/last_fft[last_peaks[2]]))
        p_values.append(p_value); f.append(freq[peaks[0]])
        last_freq, last_fft, last_peaks = freq, fft, peaks
    return diffs1, diffs2, diffs3, dm1, dm2, dm3, p_values, f


def calculate_cointegration_pvalue(PX, PY, n):
    min_length = min(len(PX), len(PY))
    PX = PX[:min_length - n]; PY = PY[:min_length - n]
    _, p_value, _ = coint(PX, PY)
    return p_value


def sigmoid(t, L, k, t0):
    return L / (1 + np.exp(-k * (t - t0)))


def calculate_moving_average(data, window_percentage):
    window_size = int(len(data) * window_percentage)
    moving_averages = []
    for i in range(len(data)):
        if i < window_size:
            current_window = data[:i + 1]
        else:
            current_window = data[i - window_size + 1:i + 1]
        moving_averages.append(np.mean(current_window))
    return moving_averages


def calculate_cointegration_params(PX, PY):
    min_length = min(len(PX), len(PY))
    PX = PX[:min_length - 1]; PY = PY[:min_length - 1]
    PX = np.log(PX); PY = np.log(PY)
    combined_df = pd.DataFrame({'PX': PX, 'PY': PY})
    X = sm.add_constant(combined_df['PX'])
    res = sm.OLS(combined_df['PY'], X).fit()
    return res.params[0], res.params[1]


def calculate_fft_and_spreads(PX, PY, n, alpha, beta):
    if PX is None or PY is None: return None
    PX = PX[:365 + n]; PY = PY[:365 + n]
    _, p_value, _ = coint(PX, PY)
    PX = np.log(PX); PY = np.log(PY)
    combined_df = pd.DataFrame({'PX': PX, 'PY': PY})
    X = sm.add_constant(combined_df['PX'])
    res = sm.OLS(combined_df['PY'], X).fit()
    alpha, beta = res.params
    spreads = combined_df['PY'] - combined_df['PX'] * beta - alpha
    moving_averages = calculate_moving_average(spreads, 0.05)
    fft = np.fft.fft(moving_averages)
    freq = np.fft.fftfreq(len(moving_averages), d=1)
    positive_magnitudes = abs(fft[freq >= 0])
    peaks, _ = find_peaks(positive_magnitudes, height=0.1)
    peaks = sorted(peaks, key=lambda x: positive_magnitudes[x], reverse=True)
    return fft, freq, peaks, spreads, moving_averages, p_value


def generate_fft_gif(PX, PY, step):
    matplotlib.use('Agg'); frames = []
    a, b = calculate_cointegration_params(PX, PY)
    j = 0; years_array = np.linspace(2001, 2024, (len(PX) - 365))
    for i in range(0, len(PX) - 365, step):
        fft, freq, peaks, _, _, _ = calculate_fft_and_spreads(PX, PY, i, a, b)
        plt.figure()
        plt.plot(abs(freq), abs(fft), label='FFT')
        plt.scatter(freq[peaks[0]], abs(fft[peaks[0]]), color='red', label='1st Dominant Frequency')
        plt.scatter(freq[peaks[1]], abs(fft[peaks[1]]), color='green', label='2nd Dominant Frequency')
        plt.scatter(freq[peaks[2]], abs(fft[peaks[2]]), color='purple', label='3rd Dominant Frequency')
        plt.xlim(left=0, right=0.015); plt.ylim(bottom=0, top=650)
        plt.title(f'{years_array[j]}'); plt.legend()
        plt.gca().figure.canvas.draw()
        frame = np.array(plt.gca().figure.canvas.renderer.buffer_rgba())
        frames.append(frame); plt.close(); j += 30
    imageio.mimsave('C:/q/dash/sample/fft.gif', frames, duration=6.75, loop=0)
    plt.show()


def generate_spreads_gif(PX, PY, step):
    matplotlib.use('Agg'); frames = []
    a, b = calculate_cointegration_params(PX, PY)
    for i in range(0, len(PX) - 365, step):
        _, _, _, spreads, moving_averages, _ = calculate_fft_and_spreads(PX, PY, i, a, b)
        plt.figure()
        plt.plot(np.arange(len(spreads)), spreads, label='Spreads')
        plt.plot(np.arange(len(moving_averages)), moving_averages, label='Moving average')
        plt.xlabel('Years'); plt.ylabel('Price'); plt.legend(); plt.title(f'+{i}')
        plt.gca().figure.canvas.draw()
        frame = np.array(plt.gca().figure.canvas.renderer.buffer_rgba())
        frames.append(frame); plt.close()
    imageio.mimsave('C:/q/dash/sample/animation.gif', frames, duration=6.75, loop=0)
    plt.show()


def plot_difference_analysis(diffs, axs, y0, yt, subplot_idx, title):
    X = np.linspace(y0, yt, len(diffs)).reshape(-1, 1)
    y = np.array(diffs); acum_errors = np.cumsum(y)
    colors = np.where(acum_errors >= 0, 'green', 'red')
    axs[subplot_idx].scatter(X, acum_errors, c=colors, alpha=0.6, edgecolors='w',
                             label='Accumulated errors (Red if <0 | Green otherwise)')
    axs[subplot_idx].plot(X, y, label='Errors')
    axs[subplot_idx].axhline(y=0, color='g', linestyle='--')
    axs[subplot_idx].axvline(x=2008, color='g', linestyle='--')
    axs[subplot_idx].set_xlabel('Year'); axs[subplot_idx].set_ylabel('Error rate')
    axs[subplot_idx].set_title(title)
    model = LinearRegression(); model.fit(X, acum_errors)
    X_flat = X.flatten()
    coefficients_poly = np.polyfit(X_flat, acum_errors, 4)
    Y_fit_poly = np.polyval(coefficients_poly, X_flat)
    axs[subplot_idx].plot(X, Y_fit_poly, 'r-', label='Polynomial regression')
    axs[subplot_idx].legend()
    years = np.linspace(2000, 2024, len(Y_fit_poly))
    res = sigmoid(years, 100, 0.1, 2011)
    correlation, p_value = pearsonr(res, Y_fit_poly)
    print(f'Correlation coefficient: {correlation}, p-value: {p_value}')


def plot_all_differences(PX, PY, step, y0, yt):
    diffs1, diffs2, diffs3, dm1, dm2, dm3, _, _ = calculate_frequency_differences(PX, PY, step)
    matplotlib.use('TkAgg')
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    axs[0].plot(np.linspace(y0, yt, len(diffs1)), diffs1, label='Error')
    axs[0].set_title('1st dominant frequency errors'); axs[0].legend()
    axs[1].plot(np.linspace(y0, yt, len(diffs2)), diffs2, label='Error')
    axs[1].set_title('2nd dominant frequency errors'); axs[1].legend()
    axs[2].plot(np.linspace(y0, yt, len(diffs3)), diffs3, label='Error')
    axs[2].set_title('3rd dominant frequency errors'); axs[2].legend()
    plt.tight_layout(); plt.show()
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    plot_difference_analysis(diffs1, axs, y0, yt, 0, '1st dominant frequency errors')
    plot_difference_analysis(diffs2, axs, y0, yt, 1, '2nd dominant frequency errors')
    plot_difference_analysis(diffs3, axs, y0, yt, 2, '3rd dominant frequency errors')
    plt.tight_layout(); plt.show()
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    axs[0].plot(np.linspace(y0, yt, len(dm1)), dm1, label='Error')
    axs[0].set_title('1st dominant frequency magnitude errors'); axs[0].legend()
    axs[1].plot(np.linspace(y0, yt, len(dm2)), dm2, label='Error')
    axs[1].set_title('2nd dominant frequency magnitude errors'); axs[1].legend()
    axs[2].plot(np.linspace(y0, yt, len(dm3)), dm3, label='Error')
    axs[2].set_title('3rd dominant frequency magnitude errors'); axs[2].legend()
    plt.tight_layout(); plt.show()
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    plot_difference_analysis(dm1, axs, y0, yt, 0, '1st dominant frequency magnitude errors')
    plot_difference_analysis(dm2, axs, y0, yt, 1, '2nd dominant frequency magnitude errors')
    plot_difference_analysis(dm3, axs, y0, yt, 2, '3rd dominant frequency magnitude errors')
    plt.tight_layout(); plt.show()


# ==============================================================================
# MAIN PIPELINE 
# ==============================================================================

def main():
    # Step 1: Load stock data 
    NYA   = np.array(load_stock_data("NYA2000").ffill())
    N100  = np.array(load_stock_data("N1002000").ffill())
    RUT   = np.array(load_stock_data("RUT2000").ffill())
    GDAXI = np.array(load_stock_data("GDAXI2000").ffill())
    N225  = np.array(load_stock_data("N2252000").ffill())
    NDX   = np.array(load_stock_data("NDX2000").ffill())
    FCHI  = np.array(load_stock_data("FCHI2000").ffill())
    SSMI  = np.array(load_stock_data("SSMI2000").ffill())
    TWII  = np.array(load_stock_data("TWII2000").ffill())

    # Step 2: Prepare series list 
    series = [
        (NYA, N100,  "NYA-N100"),
        (N100, RUT,  "N100-RUT"),
        (RUT, GDAXI, "RUT-GDAXI"),
        (NYA, N225,  "NYA-N225"),
        (NDX, N225,  "NDX-N225"),
        (FCHI, SSMI, "FCHI-SSMI"),
        (SSMI, N225, "SSMI-N225"),
        (NYA, TWII,  "NYA-TWII"),
        (NDX, TWII,  "NDX-TWII"),
    ]

    # Step 3: Trim to global minimum length  
    min_length = min(len(PX) for PX, _, _ in series)
    min_length = min(min_length, min(len(PY) for _, PY, _ in series))
    series = [(PX[:min_length], PY[:min_length], label) for PX, PY, label in series]

    # Step 4: Aggregate FFT results 
    r = aggregate_fft_results(series, 90)

    # Step 5: Regression with AT proxy  
    df, models = regression_with_at_proxy(r)

    # Step 6: Plot regression comparison 
    plot_regression_comparison(df)

    # Step 7: ADF and PP unit root tests for all nine indices
    series_dict = {
        "NYA": NYA, "N100": N100, "RUT": RUT, "GDAXI": GDAXI,
        "N225": N225, "NDX": NDX, "FCHI": FCHI, "SSMI": SSMI, "TWII": TWII
    }
    run_full_unit_root_suite(series_dict)

    # Step 8: Engle-Granger cointegration for each pair
    print("\n" + "=" * 70)
    print("ENGLE-GRANGER COINTEGRATION TESTS")
    print("=" * 70)
    for PX, PY, lbl in series:
        run_engle_granger_cointegration(PX, PY, label=lbl)

    # Step 9: Full OLS diagnostic table (SE, t, CI, BG, White, JB)
    df_diag = print_ols_diagnostics(models)
    export_ols_latex(df_diag, filepath="ols_table.tex")

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
