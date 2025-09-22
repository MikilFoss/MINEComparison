import os
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt


def _infer_num_runs(raw_results):
    if not isinstance(raw_results, list) or not raw_results:
        return 1
    lengths = [len(item.get('errors', [])) for item in raw_results if isinstance(item, dict)]
    lengths = [l for l in lengths if l > 0]
    return max(lengths) if lengths else 1


def plot_varying_T(data_T: Dict, results_dir: str, dim_label: str) -> str:
    Ts = data_T['Ts']
    errors_T = data_T['errors_mean']
    error_stds_T = data_T['errors_std']
    timestamp = data_T.get('timestamp', 'latest')
    m_fixed = data_T.get('m_fixed', None)
    true_mi = float(data_T.get('true_mi', 0.0))
    true_js = float(data_T.get('true_js', true_mi))
    sklearn_mi = float(data_T.get('sklearn_mi_sum_features', 0.0))
    num_runs = _infer_num_runs(data_T.get('raw_results', []))

    errors_T_array = np.array(errors_T)
    error_ses_T_array = np.array(error_stds_T) / np.sqrt(max(num_runs, 1))

    fig, ax = plt.subplots(1, 1, figsize=(8, 2.24))
    ax.plot(Ts, errors_T, 'bo-', label='Neural Network Method')
    if len(errors_T) == len(error_stds_T):
        ax.fill_between(
            Ts,
            errors_T_array - 3 * error_ses_T_array,
            errors_T_array + 3 * error_ses_T_array,
            alpha=0.3,
            color='blue',
            label='±3 Standard Error'
        )

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of iterations (T)')
    ax.set_ylabel(r'$|D_{KL}^{\mathrm{true}} - D_{KL}^{\mathrm{approx}}|$')

    # Baseline line: compare sklearn MI against true JS (I(X;Y)) if available
    baseline = abs(true_js - sklearn_mi)
    if baseline > 0:
        ax.axhline(y=baseline, color='red', linestyle='--', label='Sklearn MI baseline')

    title_suffix = f' (fixed m = {m_fixed})' if m_fixed is not None else ''
    ax.set_title(f'{dim_label}: Varying T{title_suffix}')
    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    ax.grid(True)

    plt.tight_layout()
    out_path = os.path.join(results_dir, f'kl_experiments_{dim_label.lower()}_varying_T_{timestamp}.pdf')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return out_path


def plot_varying_m(data_m: Dict, results_dir: str, dim_label: str) -> str:
    ms = data_m['ms']
    errors_m = data_m['errors_mean']
    error_stds_m = data_m['errors_std']
    timestamp = data_m.get('timestamp', 'latest')
    T_fixed = data_m.get('T_fixed', None)
    true_mi = float(data_m.get('true_mi', 0.0))
    true_js = float(data_m.get('true_js', true_mi))
    sklearn_mi = float(data_m.get('sklearn_mi_sum_features', 0.0))
    num_runs = _infer_num_runs(data_m.get('raw_results', []))

    errors_m_array = np.array(errors_m)
    error_ses_m_array = np.array(error_stds_m) / np.sqrt(max(num_runs, 1))

    fig, ax = plt.subplots(1, 1, figsize=(8, 2.24))
    ax.plot(ms, errors_m, 'ro-', label='Neural Network Method')
    if len(errors_m) == len(error_stds_m):
        ax.fill_between(
            ms,
            errors_m_array - 3 * error_ses_m_array,
            errors_m_array + 3 * error_ses_m_array,
            alpha=0.3,
            color='red',
            label='±3 Standard Error'
        )

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of neurons (m)')
    ax.set_ylabel(r'$|D_{KL}^{\mathrm{true}} - D_{KL}^{\mathrm{approx}}|$')

    baseline = abs(true_js - sklearn_mi)
    if baseline > 0:
        ax.axhline(y=baseline, color='blue', linestyle='--', label='Sklearn MI baseline')

    title_suffix = f' (fixed T = {T_fixed})' if T_fixed is not None else ''
    ax.set_title(f'{dim_label}: Varying m{title_suffix}')
    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    ax.grid(True)

    plt.tight_layout()
    out_path = os.path.join(results_dir, f'kl_experiments_{dim_label.lower()}_varying_m_{timestamp}.pdf')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return out_path


