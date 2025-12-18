
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, coherence, correlate
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from statsmodels.stats.stattools import jarque_bera
import imageio
import matplotlib
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression

# --- Analysis Functions ---

def advanced_oscillator_analysis(results_list):
    """
    Performs advanced correlation analysis for dominant frequencies including:
    - Pearson correlation
    - Average spectral coherence
    - Maximum cross-correlation with optimal lag
    - Mutual information

    Parameters:
      - results_list: List of dictionaries with data for each spread, containing:
            'label': identifying label,
            'freq_dom1', 'freq_dom2', 'freq_dom3': time series of frequencies,
            'mag_dom1', 'mag_dom2', 'mag_dom3': time series of magnitudes.
    """
    freq_keys = ['freq_dom1', 'freq_dom2', 'freq_dom3']
    mag_keys = ['mag_dom1', 'mag_dom2', 'mag_dom3']
    n = len(results_list)
    labels = [res['label'] for res in results_list]

    def compute_metrics(series1, series2):
        """Computes multiple metrics for two time series"""
        # Pearson correlation
        pearson = pearsonr(series1, series2)[0]
        
        # Average spectral coherence

        s1 = np.asarray(series1, dtype=float)
        s2 = np.asarray(series2, dtype=float)

        mask = ~np.isnan(s1) & ~np.isnan(s2)
        s1 = s1[mask]
        s2 = s2[mask]

        ds1 = np.diff(s1)
        ds2 = np.diff(s2)

        if len(ds1) < 8:
            coherence_avg = np.nan
        else:
            
            nperseg = max(8, min(128, len(ds1) // 3))

            f, Cxy = coherence(
                ds1,
                ds2,
                fs=1.0,
                nperseg=nperseg,
                detrend='constant'
            )
            coherence_avg = np.nanmean(Cxy)
        
        # Cross-correlation with optimal lag
        corr = correlate(series1 - np.mean(series1), series2 - np.mean(series2), mode='full')
        lags = np.arange(-len(series1) + 1, len(series1))
        optimal_lag = lags[np.argmax(corr)]
        max_corr = np.max(corr) / (np.std(series1) * np.std(series2) * len(series1))
        
        # Mutual information (scikit-learn style)
        mi = mutual_info_regression(np.array(series1).reshape(-1, 1), np.array(series2))[0]
        
        return pearson, coherence_avg, max_corr, mi, optimal_lag

    # Analysis for frequencies with multiple metrics
    for fk in freq_keys:
        print(f"\nAnalysis for {fk.replace('_', ' ').title()}")
        all_series = [res[fk] for res in results_list]
        
        # Initialize matrices
        pearson_mat = np.eye(n)
        coherence_mat = np.eye(n)
        crosscorr_mat = np.eye(n)
        mi_mat = np.eye(n)
        lag_mat = np.zeros((n, n))
        
        # Fill matrices
        for i in range(n):
            for j in range(n):
                if i != j:
                    p, c, cc, mi, lag = compute_metrics(all_series[i], all_series[j])
                    pearson_mat[i, j] = p
                    coherence_mat[i, j] = c
                    crosscorr_mat[i, j] = cc
                    mi_mat[i, j] = mi
                    lag_mat[i, j] = lag

        # Calculate average metrics
        metrics = {
            'Pearson': (pearson_mat, -1, 1),
            'Coherence': (coherence_mat, 0, 1),
            'Cross Corr': (crosscorr_mat, -1, 1),
            'Mutual Info': (mi_mat, 0, np.max(mi_mat))
        }
        
        # Visualization
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Coupling Analysis for {fk.replace("_", " ").title()}', y=1.02)
        
        for idx, (title, (data, vmin, vmax)) in enumerate(metrics.items()):
            ax = axs[idx // 2, idx % 2]
            cax = ax.matshow(data, cmap='viridis' if title == 'Mutual Info' else 'coolwarm', 
                             vmin=vmin, vmax=vmax)
            plt.colorbar(cax, ax=ax)
            ax.set_title(f"{title} (Average: {np.nanmean(data[np.eye(n) == 0]):.2f})")
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(labels, rotation=90)
            ax.set_yticklabels(labels)
        
        plt.tight_layout()
        plt.show()

    # Analysis for magnitudes (classic Pearson)
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
        plt.xticks(range(n), labels, rotation=90)
        plt.yticks(range(n), labels)
        plt.show()

def pairwise_correlation_analysis(results_list):
    """
    Calculates and plots correlation matrices for the 3 dominant frequencies and the 3 magnitudes,
    also calculating the average correlation (ignoring the diagonal) for each group.

    Parameters:
      - results_list: List of dictionaries with data for each spread, which must contain:
            'label': identifying label,
            'freq_dom1', 'freq_dom2', 'freq_dom3': vectors of dominant frequencies,
            'mag_dom1', 'mag_dom2', 'mag_dom3': vectors of magnitudes.
    """
    # List of keys for frequencies and magnitudes
    freq_keys = ['freq_dom1', 'freq_dom2', 'freq_dom3']
    mag_keys = ['mag_dom1', 'mag_dom2', 'mag_dom3']

    def process_and_plot(key, data_label):
        data_all = []
        labels = []
        for res in results_list:
            if len(res[key]) > 0:
                data_all.append(res[key])
                labels.append(res['label'])
        if len(data_all) < 2:
            print(f"Not enough series for {key}.")
            return
        # Trim to the same length
        min_length = min(len(arr) for arr in data_all)
        data_all = np.array([arr[:min_length] for arr in data_all])
        
        # Calculate the correlation matrix
        corr_matrix = np.corrcoef(data_all)
        
        # Calculate the average correlation ignoring the diagonal
        n = corr_matrix.shape[0]
        avg_corr = (np.sum(corr_matrix) - np.trace(corr_matrix)) / (n * (n - 1))
        print(f"Average correlation for {data_label} ({key}): {avg_corr:.2f}")
        
        # Plot the heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, xticklabels=labels, yticklabels=labels, cmap='coolwarm')
        plt.title(f'Correlation Matrix: {data_label} ({key})\nAverage: {avg_corr:.2f}')
        plt.tight_layout()
        plt.show()

    # Process each dominant frequency
    for key in freq_keys:
        process_and_plot(key, "Dominant Frequency")
    
    # Process each magnitude
    for key in mag_keys:
        process_and_plot(key, "Magnitude")
    
    # Optional: joint analysis of the 3 frequencies
    def weighted_average_correlation(results_list):
        """
        For each series in 'results_list', calculates the weighted average of the three dominant frequencies,
        using their magnitudes as weights, and then calculates and plots the pairwise correlation matrix
        between these weighted average series.

        The weighted average is calculated as:
        
            weighted_freq = (f1*m1 + f2*m2 + f3*m3) / (m1 + m2 + m3)
        
        where f1, f2, f3 are the dominant frequencies (freq_dom1, freq_dom2, freq_dom3)
        and m1, m2, m3 are the corresponding magnitudes (mag_dom1, mag_dom2, mag_dom3).

        Parameters:
        - results_list: list of dictionaries with the keys:
                'label': series label,
                'freq_dom1', 'freq_dom2', 'freq_dom3': vectors of dominant frequencies,
                'mag_dom1', 'mag_dom2', 'mag_dom3': vectors of magnitudes.
        """
        weighted_series = []
        labels = []
        
        # Loop through each series and calculate the weighted average
        for res in results_list:
            # Check that the required data exists
            if (len(res['freq_dom1']) > 0 and len(res['freq_dom2']) > 0 and len(res['freq_dom3']) > 0 and
                len(res['mag_dom1']) > 0 and len(res['mag_dom2']) > 0 and len(res['mag_dom3']) > 0):
                
                # Find the minimum length to ensure consistency in this series
                min_len = min(len(res['freq_dom1']), len(res['freq_dom2']), len(res['freq_dom3']),
                              len(res['mag_dom1']), len(res['mag_dom2']), len(res['mag_dom3']))
                
                f1 = np.array(res['freq_dom1'][:min_len])
                f2 = np.array(res['freq_dom2'][:min_len])
                f3 = np.array(res['freq_dom3'][:min_len])
                m1 = np.array(res['mag_dom1'][:min_len])
                m2 = np.array(res['mag_dom2'][:min_len])
                m3 = np.array(res['mag_dom3'][:min_len])
                
                # Calculate the weighted average for each instant
                weighted_freq = (f1 * m1 + f2 * m2 + f3 * m3) / (m1 + m2 + m3)
                
                weighted_series.append(weighted_freq)
                labels.append(res['label'])
        
        # Ensure all series have the same length by trimming to the common minimum
        common_length = min(len(series) for series in weighted_series)
        weighted_series = np.array([series[:common_length] for series in weighted_series])
        
        print(f"Shape of weighted series: {weighted_series.shape}")
        
        # Calculate the correlation matrix
        corr_matrix = np.corrcoef(weighted_series)
        
        # Calculate the average correlation ignoring the diagonal
        n = corr_matrix.shape[0]
        mean_corr = (np.sum(corr_matrix) - np.trace(corr_matrix)) / (n * (n - 1))
        
        # Plot the correlation matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, xticklabels=labels, yticklabels=labels, cmap='coolwarm')
        plt.title(f'Correlation Matrix of Weighted Averages\nAverage: {mean_corr:.2f}')
        plt.tight_layout()
        plt.show()

    weighted_average_correlation(results_list)

def regression_with_at_proxy(results, t0=2010, k=0.25):
    """
    For each series in 'results', builds the sigmoid proxy variable,
    estimates an OLS regression model, and extracts key metrics for comparison.
    
    Parameters:
      - results: List of dictionaries with 'times', 'freq_dom1' and 'label'
      - t0: Central year of the transition (default=2012)
      - k: Sigmoid slope (default=0.5)
    
    Returns:
      - df_metrics: DataFrame with the extracted metrics for each series.
      - models: Dictionary with the fitted models (key: label)
    """
    metrics_list = []
    models = {}

    for res in results:
        label = res["label"]
        times = np.array(res["times"])
        freq1 = np.array(res["freq_dom2"])
        
        # Build the sigmoid proxy variable for AT
        AT_proxy = 1 / (1 + np.exp(-k * (times - t0)))
        
        # Prepare the data for regression (adding constant)
        X = sm.add_constant(AT_proxy)
        model = sm.OLS(freq1, X).fit()
        models[label] = model
        
        # Extract model metrics
        metrics = {}
        metrics["label"] = label
        metrics["R_squared"] = model.rsquared
        metrics["Adj_R_squared"] = model.rsquared_adj
        metrics["F_statistic"] = model.fvalue
        metrics["Prob_F"] = model.f_pvalue
        
        # Coefficients and p-values: [const, AT_proxy]
        metrics["const_coef"] = model.params[0]
        metrics["const_pvalue"] = model.pvalues[0]
        metrics["AT_proxy_coef"] = model.params[1]
        metrics["AT_proxy_pvalue"] = model.pvalues[1]
        
        # Diagnostic statistics using jarque_bera on residuals
        jb_stat, jb_pvalue, skew, kurtosis = jarque_bera(model.resid)
        metrics["JB_stat"] = jb_stat
        metrics["JB_pvalue"] = jb_pvalue
        metrics["Skew"] = skew
        metrics["Kurtosis"] = kurtosis

        metrics_list.append(metrics)
        
        # Print the model summary
        print(f"Regression results for series {label}:")
        print(model.summary())
        print("\n" + "-" * 80 + "\n")
        
    df_metrics = pd.DataFrame(metrics_list)
    return df_metrics, models

def plot_regression_comparison(df_metrics):
    """
    Generates comparative plots for the metrics extracted from the regressions.
    
    Plots are created for:
      - R-squared
      - AT_proxy coefficient
      - P-value of the AT_proxy coefficient (with significance line at 0.05 and average)
      - Jarque-Bera statistic
      - Kurtosis
      - Jarque-Bera p-value
     
    Additionally, a horizontal line showing the average of that metric across all series is drawn in each plot.
    """
    sns.set(style="whitegrid")

    # Plot 1: R-squared per series
    plt.figure(figsize=(10, 5))
    sns.barplot(x="label", y="R_squared", data=df_metrics, palette="Blues_d")
    plt.title("R-squared per Series")
    plt.xlabel("Series")
    plt.ylabel("R-squared")
    plt.xticks(rotation=45)
    mean_r2 = df_metrics["R_squared"].mean()
    plt.axhline(mean_r2, color="black", linestyle="--", label=f"Average: {mean_r2:.3f}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot 2: Coefficient of AT_proxy per series
    plt.figure(figsize=(10, 5))
    sns.barplot(x="label", y="AT_proxy_coef", data=df_metrics, palette="Greens_d")
    plt.title("AT_proxy Coefficient per Series")
    plt.xlabel("Series")
    plt.ylabel("Coefficient")
    plt.xticks(rotation=45)
    mean_coef = df_metrics["AT_proxy_coef"].mean()
    plt.axhline(mean_coef, color="black", linestyle="--", label=f"Average: {mean_coef:.3f}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot 3: P-value of AT_proxy coefficient
    plt.figure(figsize=(10, 5))
    sns.barplot(x="label", y="AT_proxy_pvalue", data=df_metrics, palette="Reds_d")
    plt.title("AT_proxy P-value per Series")
    plt.xlabel("Series")
    plt.ylabel("P-value")
    plt.axhline(0.05, color="black", linestyle="--", label="Significance (0.05)")
    mean_pvalue = df_metrics["AT_proxy_pvalue"].mean()
    plt.axhline(mean_pvalue, color="blue", linestyle="--", label=f"Average: {mean_pvalue:.3f}")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Plot 4: Jarque-Bera Statistic per series
    plt.figure(figsize=(10, 5))
    sns.barplot(x="label", y="JB_stat", data=df_metrics, palette="Purples_d")
    plt.title("Jarque-Bera Statistic per Series")
    plt.xlabel("Series")
    plt.ylabel("JB Statistic")
    plt.xticks(rotation=45)
    mean_jb = df_metrics["JB_stat"].mean()
    plt.axhline(mean_jb, color="black", linestyle="--", label=f"Average: {mean_jb:.2f}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot 5: Kurtosis per series
    plt.figure(figsize=(10, 5))
    sns.barplot(x="label", y="Kurtosis", data=df_metrics, palette="Oranges_d")
    plt.title("Kurtosis per Series")
    plt.xlabel("Series")
    plt.ylabel("Kurtosis")
    plt.xticks(rotation=45)
    mean_kurtosis = df_metrics["Kurtosis"].mean()
    plt.axhline(mean_kurtosis, color="black", linestyle="--", label=f"Average: {mean_kurtosis:.2f}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot 6: Jarque-Bera P-value per series
    plt.figure(figsize=(10, 5))
    sns.barplot(x="label", y="JB_pvalue", data=df_metrics, palette="Greys_d")
    plt.title("Jarque-Bera P-value per Series")
    plt.xlabel("Series")
    plt.ylabel("JB P-value")
    plt.xticks(rotation=45)
    mean_jb_pvalue = df_metrics["JB_pvalue"].mean()
    plt.axhline(mean_jb_pvalue, color="black", linestyle="--", label=f"Average: {mean_jb_pvalue:.3f}")
    plt.legend()
    plt.tight_layout()
    plt.show()

# --- FFT Aggregation and Visualization ---

def aggregate_fft_results(series_list, step):
    """
    Receives a list of series and aggregates the FFT analysis results.
    
    Parameters:
      - series_list: list of tuples (PX, PY, label), where:
            PX, PY: Asset data series.
            label: Identifying label for the series.
      - step: Number of points to advance in each window for FFT analysis.
    
    For each series, the function:
      - Calls analyze_fft_evolution to obtain results.
      - Aggregates these results into a list.
    
    Then generates comparative plots for frequencies, magnitudes, and changes.
      
    Returns a list with the result dictionaries for further analysis.
    """
    # List to store the results of each series
    results_list = []
    
    # Process each pair of series
    for PX, PY, label in series_list:
        result = analyze_fft_evolution(PX, PY, step, False)
        result['label'] = label
        results_list.append(result)
    
    # Comparative plots for dominant frequencies
    for dom_num, key in enumerate(['freq_dom1', 'freq_dom2', 'freq_dom3'], 1):
        plt.figure(figsize=(14, 8))
        for res in results_list:
            plt.plot(res['times'], res[key], marker='o', label=f"{res['label']} - {dom_num}st Freq")
        plt.xlabel('Year')
        plt.ylabel('Frequency')
        plt.title(f'Comparison of {dom_num}st Dominant Frequency')
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Comparative plots for dominant magnitudes
    for dom_num, key in enumerate(['mag_dom1', 'mag_dom2', 'mag_dom3'], 1):
        plt.figure(figsize=(14, 8))
        for res in results_list:
            plt.plot(res['times'], res[key], marker='o', label=f"{res['label']} - Magnitude {dom_num}st")
        plt.xlabel('Year')
        plt.ylabel('Magnitude')
        plt.title(f'Comparison of Magnitudes - {dom_num}st Dominant Frequency')
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Comparative plots for relative changes in frequencies
    for dom_num, key in enumerate(['rel_change_freq1', 'rel_change_freq2', 'rel_change_freq3'], 1):
        markers = ['o', 'x', 's']
        plt.figure(figsize=(14, 8))
        for res in results_list:
            plt.plot(res['times'], res[key], marker=markers[dom_num-1], label=f"{res['label']} - Rel. Δ {dom_num}st Freq")
        plt.xlabel('Year')
        plt.ylabel('Relative Change')
        plt.title(f'Instantaneous Relative Change - {dom_num}st Frequency')
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Comparative plots for cumulative changes in frequencies
    for dom_num, key in enumerate(['cum_rel_change_freq1', 'cum_rel_change_freq2', 'cum_rel_change_freq3'], 1):
        markers = ['o', 'x', 's']
        plt.figure(figsize=(14, 8))
        for res in results_list:
            plt.plot(res['times'], res[key], marker=markers[dom_num-1], label=f"{res['label']} - Cumul. Δ {dom_num}st Freq")
        plt.xlabel('Year')
        plt.ylabel('Cumulative Relative Change')
        plt.title(f'Cumulative Change - {dom_num}st Frequency')
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Comparative plots for relative changes in magnitudes
    for dom_num, key in enumerate(['rel_change_mag1', 'rel_change_mag2', 'rel_change_mag3'], 1):
        markers = ['o', 'x', 's']
        plt.figure(figsize=(14, 8))
        for res in results_list:
            plt.plot(res['times'], res[key], marker=markers[dom_num-1], label=f"{res['label']} - Rel. Δ Mag {dom_num}st")
        plt.xlabel('Year')
        plt.ylabel('Relative Change')
        plt.title(f'Instantaneous Relative Change - {dom_num}st Magnitude')
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Comparative plots for cumulative changes in magnitudes
    for dom_num, key in enumerate(['cum_rel_change_mag1', 'cum_rel_change_mag2', 'cum_rel_change_mag3'], 1):
        markers = ['o', 'x', 's']
        plt.figure(figsize=(14, 8))
        for res in results_list:
            plt.plot(res['times'], res[key], marker=markers[dom_num-1], label=f"{res['label']} - Cumul. Δ Mag {dom_num}st")
        plt.xlabel('Year')
        plt.ylabel('Cumulative Relative Change')
        plt.title(f'Cumulative Change - {dom_num}st Magnitude')
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Perform pairwise correlation and advanced analysis
    pairwise_correlation_analysis(results_list)
    advanced_oscillator_analysis(results_list)
    
    return results_list

# --- FFT Evolution Analysis ---

def analyze_fft_evolution(PX, PY, step, plot=False):
    """
    Analyzes the evolution of the three dominant frequencies and their magnitudes from sliding windows,
    also calculating relative changes and their cumulatives.
    """
    a, b = calculate_cointegration_params(PX, PY)
    
    num_points = (len(PX) - 365) // step
    years_array = np.linspace(2000, 2024, num_points)
    times = []
    freq_dom1, freq_dom2, freq_dom3 = [], [], []
    mag_dom1, mag_dom2, mag_dom3 = [], [], []
    
    for i, year in zip(range(0, len(PX) - 365, step), years_array):
        fft, freq, peaks, _, _, _ = calculate_fft_and_spreads(PX, PY, i, a, b)
        times.append(year)
        freq_dom1.append(freq[peaks[0]])
        freq_dom2.append(freq[peaks[1]])
        freq_dom3.append(freq[peaks[2]])
        mag_dom1.append(abs(fft[peaks[0]]))
        mag_dom2.append(abs(fft[peaks[1]]))
        mag_dom3.append(abs(fft[peaks[2]]))
    
    times = np.array(times)
    freq_dom1 = np.array(freq_dom1)
    freq_dom2 = np.array(freq_dom2)
    freq_dom3 = np.array(freq_dom3)
    mag_dom1 = np.array(mag_dom1)
    mag_dom2 = np.array(mag_dom2)
    mag_dom3 = np.array(mag_dom3)
    
    rel_change_freq1 = np.full_like(freq_dom1, np.nan, dtype=float)
    rel_change_freq2 = np.full_like(freq_dom2, np.nan, dtype=float)
    rel_change_freq3 = np.full_like(freq_dom3, np.nan, dtype=float)
    rel_change_mag1 = np.full_like(mag_dom1, np.nan, dtype=float)
    rel_change_mag2 = np.full_like(mag_dom2, np.nan, dtype=float)
    rel_change_mag3 = np.full_like(mag_dom3, np.nan, dtype=float)
    
    for j in range(1, len(times)):
        rel_change_freq1[j] = (freq_dom1[j] - freq_dom1[j - 1]) / freq_dom1[j - 1]
        rel_change_freq2[j] = (freq_dom2[j] - freq_dom2[j - 1]) / freq_dom2[j - 1]
        rel_change_freq3[j] = (freq_dom3[j] - freq_dom3[j - 1]) / freq_dom3[j - 1]
        rel_change_mag1[j] = (mag_dom1[j] - mag_dom1[j - 1]) / mag_dom1[j - 1]
        rel_change_mag2[j] = (mag_dom2[j] - mag_dom2[j - 1]) / mag_dom2[j - 1]
        rel_change_mag3[j] = (mag_dom3[j] - mag_dom3[j - 1]) / mag_dom3[j - 1]
    
    cum_rel_change_freq1 = np.nancumsum(np.nan_to_num(rel_change_freq1))
    cum_rel_change_freq2 = np.nancumsum(np.nan_to_num(rel_change_freq2))
    cum_rel_change_freq3 = np.nancumsum(np.nan_to_num(rel_change_freq3))
    cum_rel_change_mag1 = np.nancumsum(np.nan_to_num(rel_change_mag1))
    cum_rel_change_mag2 = np.nancumsum(np.nan_to_num(rel_change_mag2))
    cum_rel_change_mag3 = np.nancumsum(np.nan_to_num(rel_change_mag3))
    
    if plot:
        # Implement plotting code here if needed
        pass
    
    return {
        'times': times,
        'freq_dom1': freq_dom1,
        'freq_dom2': freq_dom2,
        'freq_dom3': freq_dom3,
        'mag_dom1': mag_dom1,
        'mag_dom2': mag_dom2,
        'mag_dom3': mag_dom3,
        'rel_change_freq1': rel_change_freq1,
        'rel_change_freq2': rel_change_freq2,
        'rel_change_freq3': rel_change_freq3,
        'rel_change_mag1': rel_change_mag1,
        'rel_change_mag2': rel_change_mag2,
        'rel_change_mag3': rel_change_mag3,
        'cum_rel_change_freq1': cum_rel_change_freq1,
        'cum_rel_change_freq2': cum_rel_change_freq2,
        'cum_rel_change_freq3': cum_rel_change_freq3,
        'cum_rel_change_mag1': cum_rel_change_mag1,
        'cum_rel_change_mag2': cum_rel_change_mag2,
        'cum_rel_change_mag3': cum_rel_change_mag3
    }

# --- Utility Functions ---

def load_stock_data(file_name):
    path = f"C:/q/dash/sample/data/stocks/{file_name}.csv"
    try:
        df = pd.read_csv(path)
        print(f"CSV file {file_name} loaded successfully.")
        if 'Close' in df.columns:
            return df['Close']
        else:
            print(f"Error: 'Close' column not found in {file_name}.")
            return None
    except FileNotFoundError:
        print(f"Error: File {path} not found.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def calculate_frequency_differences(PX, PY, step):
    a, b = calculate_cointegration_params(PX, PY)
    last_fft, last_freq, last_peaks, _, _, p_value = calculate_fft_and_spreads(PX, PY, 0, a, b)
    diffs1, diffs2, diffs3 = [], [], []
    dm1, dm2, dm3 = [], [], []
    p_values, f = [], []
    
    for i in range(0, len(PX) - 365, step):
        fft, freq, peaks, _, _, p_value = calculate_fft_and_spreads(PX, PY, i, a, b)
        diffs1.append((freq[peaks[0]] - last_freq[last_peaks[0]]) / last_freq[last_peaks[0]])
        diffs2.append((freq[peaks[1]] - last_freq[last_peaks[1]]) / last_freq[last_peaks[1]])
        diffs3.append((freq[peaks[2]] - last_freq[last_peaks[2]]) / last_freq[last_peaks[2]])
        dm1.append(float((fft[peaks[0]] - last_fft[last_peaks[0]]) / last_fft[last_peaks[0]]))
        dm2.append(float((fft[peaks[1]] - last_fft[last_peaks[1]]) / last_fft[last_peaks[1]]))
        dm3.append(float((fft[peaks[2]] - last_fft[last_peaks[2]]) / last_fft[last_peaks[2]]))
        p_values.append(p_value)
        f.append(freq[peaks[0]])
        last_freq, last_fft, last_peaks = freq, fft, peaks
    
    return diffs1, diffs2, diffs3, dm1, dm2, dm3, p_values, f

def calculate_cointegration_pvalue(PX, PY, n):
    min_length = min(len(PX), len(PY))
    PX = PX[:min_length - n]
    PY = PY[:min_length - n]
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
    PX = PX[:min_length - 1]
    PY = PY[:min_length - 1]
    PX = np.log(PX)
    PY = np.log(PY)
    combined_df = pd.DataFrame({'PX': PX, 'PY': PY})
    PX_clean = combined_df['PX']
    PY_clean = combined_df['PY']
    X = sm.add_constant(PX_clean)
    model = sm.OLS(PY_clean, X)
    res = model.fit()
    alpha, beta = res.params
    return alpha, beta

def calculate_fft_and_spreads(PX, PY, n, alpha, beta):
    if PX is None or PY is None:
        return None
    PX = PX[:365 + n]
    PY = PY[:365 + n]
    _, p_value, _ = coint(PX, PY)
    PX = np.log(PX)
    PY = np.log(PY)
    combined_df = pd.DataFrame({'PX': PX, 'PY': PY})
    PX_clean = combined_df['PX']
    PY_clean = combined_df['PY']
    X = sm.add_constant(PX_clean)
    model = sm.OLS(PY_clean, X)
    res = model.fit()
    alpha, beta = res.params
    spreads = PY_clean - PX_clean * beta - alpha
    window_percentage = 0.05
    moving_averages = calculate_moving_average(spreads, window_percentage)
    fft = np.fft.fft(moving_averages)
    freq = np.fft.fftfreq(len(moving_averages), d=1)
    positive_frequencies = freq[freq >= 0]
    positive_magnitudes = abs(fft[freq >= 0])
    peaks, _ = find_peaks(positive_magnitudes, height=0.1)
    peaks = sorted(peaks, key=lambda x: positive_magnitudes[x], reverse=True)
    return fft, freq, peaks, spreads, moving_averages, p_value

def generate_fft_gif(PX, PY, step):
    matplotlib.use('Agg')
    frames = []
    a, b = calculate_cointegration_params(PX, PY)
    j = 0
    years_array = np.linspace(2001, 2024, (len(PX) - 365))
    for i in range(0, len(PX) - 365, step):
        fft, freq, peaks, _, _, _ = calculate_fft_and_spreads(PX, PY, i, a, b)
        plt.figure()
        plt.plot(abs(freq), abs(fft), label='FFT')
        plt.scatter(freq[peaks[0]], abs(fft[peaks[0]]), color='red', label='1st Dominant Frequency')
        plt.scatter(freq[peaks[1]], abs(fft[peaks[1]]), color='green', label='2nd Dominant Frequency')
        plt.scatter(freq[peaks[2]], abs(fft[peaks[2]]), color='purple', label='3rd Dominant Frequency')
        plt.xlim(left=0, right=0.015)
        plt.ylim(bottom=0, top=650)
        plt.title(f'{years_array[j]}')
        plt.legend()
        plt.gca().figure.canvas.draw()
        frame = np.array(plt.gca().figure.canvas.renderer.buffer_rgba())
        frames.append(frame)
        plt.close()
        j += 30
    imageio.mimsave('C:/q/dash/sample/fft.gif', frames, duration=6.75, loop=0)
    plt.show()

def generate_spreads_gif(PX, PY, step):
    matplotlib.use('Agg')
    frames = []
    a, b = calculate_cointegration_params(PX, PY)
    for i in range(0, len(PX) - 365, step):
        _, _, _, spreads, moving_averages, _ = calculate_fft_and_spreads(PX, PY, i, a, b)
        plt.figure()
        plt.plot(np.arange(len(spreads)), spreads, label='Spreads')
        plt.plot(np.arange(len(moving_averages)), moving_averages, label='Moving average')
        plt.xlabel('Years')
        plt.ylabel('Price')
        plt.legend()
        plt.title(f'+{i}')
        plt.gca().figure.canvas.draw()
        frame = np.array(plt.gca().figure.canvas.renderer.buffer_rgba())
        frames.append(frame)
        plt.close()
    imageio.mimsave('C:/q/dash/sample/animation.gif', frames, duration=6.75, loop=0)
    plt.show()

def plot_difference_analysis(diffs, axs, y0, yt, subplot_idx, title):
    X = np.linspace(y0, yt, len(diffs)).reshape(-1, 1)
    y = np.array(diffs)
    acum_errors = np.cumsum(y)
    colors = np.where(acum_errors >= 0, 'green', 'red')
    axs[subplot_idx].scatter(X, acum_errors, c=colors, alpha=0.6, edgecolors='w', label='Accumulated errors (Red if <0 | Green otherwise)')
    axs[subplot_idx].plot(X, y, label='Errors')
    axs[subplot_idx].axhline(y=0, color='g', linestyle='--')
    axs[subplot_idx].axvline(x=2008, color='g', linestyle='--')
    axs[subplot_idx].set_xlabel('Year')
    axs[subplot_idx].set_ylabel('Error rate')
    axs[subplot_idx].set_title(title)
    model = LinearRegression()
    model.fit(X, acum_errors)
    # axs[subplot_idx].plot(X, model.predict(X), label='Linear regression')
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
    axs[0].set_title('1st dominant frequency errors')
    axs[0].legend()
    axs[1].plot(np.linspace(y0, yt, len(diffs2)), diffs2, label='Error')
    axs[1].set_title('2nd dominant frequency errors')
    axs[1].legend()
    axs[2].plot(np.linspace(y0, yt, len(diffs3)), diffs3, label='Error')
    axs[2].set_title('3rd dominant frequency errors')
    axs[2].legend()
    plt.tight_layout()
    plt.show()
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    plot_difference_analysis(diffs1, axs, y0, yt, 0, '1st dominant frequency errors')
    plot_difference_analysis(diffs2, axs, y0, yt, 1, '2nd dominant frequency errors')
    plot_difference_analysis(diffs3, axs, y0, yt, 2, '3rd dominant frequency errors')
    plt.tight_layout()
    plt.show()
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    axs[0].plot(np.linspace(y0, yt, len(dm1)), dm1, label='Error')
    axs[0].set_title('1st dominant frequency magnitude errors')
    axs[0].legend()
    axs[1].plot(np.linspace(y0, yt, len(dm2)), dm2, label='Error')
    axs[1].set_title('2nd dominant frequency magnitude errors')
    axs[1].legend()
    axs[2].plot(np.linspace(y0, yt, len(dm3)), dm3, label='Error')
    axs[2].set_title('3rd dominant frequency magnitude errors')
    axs[2].legend()
    plt.tight_layout()
    plt.show()
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    plot_difference_analysis(dm1, axs, y0, yt, 0, '1st dominant frequency magnitude errors')
    plot_difference_analysis(dm2, axs, y0, yt, 1, '2nd dominant frequency magnitude errors')
    plot_difference_analysis(dm3, axs, y0, yt, 2, '3rd dominant frequency magnitude errors')
    plt.tight_layout()
    plt.show()

# --- Main Pipeline ---

def main():
    # Step 1: Load stock data
    NYA = np.array(load_stock_data("NYA2000").ffill())
    N100 = np.array(load_stock_data("N1002000").ffill())
    RUT = np.array(load_stock_data("RUT2000").ffill())
    GDAXI = np.array(load_stock_data("GDAXI2000").ffill())
    N225 = np.array(load_stock_data("N2252000").ffill())
    NDX = np.array(load_stock_data("NDX2000").ffill())
    FCHI = np.array(load_stock_data("FCHI2000").ffill())
    SSMI = np.array(load_stock_data("SSMI2000").ffill())
    TWII = np.array(load_stock_data("TWII2000").ffill())
    
    # Step 2: Prepare series list
    series = [
        (NYA, N100, "NYA-N100"),
        (N100, RUT, "N100-RUT"),
        (RUT, GDAXI, "RUT-GDAXI"),
        (NYA, N225, "NYA-N225"),
        (NDX, N225, "NDX-N225"),
        (FCHI, SSMI, "FCHI-SSMI"),
        (SSMI, N225, "SSMI-N225"),
        (NYA, TWII, "NYA-TWII"),
        (NDX, TWII, "NDX-TWII")
    ]
    
    # Step 3: Determine minimum length and trim series
    min_length = min(len(PX) for PX, _, _ in series)
    min_length = min(min_length, min(len(PY) for _, PY, _ in series))
    series = [(PX[:min_length], PY[:min_length], label) for PX, PY, label in series]
    
    # Step 4: Aggregate FFT results and perform correlations/advanced analysis
    r = aggregate_fft_results(series, 90)
    
    # Step 5: Perform regression with AT proxy
    df, _ = regression_with_at_proxy(r)
    
    # Step 6: Plot regression comparison
    plot_regression_comparison(df)

if __name__ == "__main__":
    main()
