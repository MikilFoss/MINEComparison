import os
import time
import pickle
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
from tqdm import tqdm
from sklearn.feature_selection import mutual_info_regression



@dataclass
class SweepConfig:
    vary_T_values: List[int]
    vary_m_values: List[int]
    fixed_m_for_T: int
    fixed_T_for_m: int
    runs_per_setting_T: int
    runs_per_setting_m: int
    its_eval: int


@dataclass
class ExperimentIO:
    raw_dir: str
    results_dir: str


def ensure_dirs(io: ExperimentIO) -> None:
    os.makedirs(io.raw_dir, exist_ok=True)
    os.makedirs(io.results_dir, exist_ok=True)


class KLProblem:
    def __init__(self, input_dim: int):
        self.input_dim = input_dim

    def init_network(self, m: int) -> Tuple[np.ndarray, np.ndarray]:
        weights_raw = np.random.randn(m, self.input_dim)
        weights = weights_raw / np.linalg.norm(weights_raw, axis=1)[:, np.newaxis]
        biases = np.random.uniform(-2, 2, m)
        return weights, biases

    @staticmethod
    def phi(weights: np.ndarray, biases: np.ndarray, x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, weights.dot(x) + biases)

    def sample_p(self) -> np.ndarray:
        while True:
            x = np.random.multivariate_normal(mean=np.zeros(self.input_dim), cov=np.eye(self.input_dim))
            if np.all(np.abs(x) <= 2):
                return x

    def sample_q(self) -> np.ndarray:
        return np.random.uniform(-2, 2, self.input_dim)

    def update_step(self, theta: np.ndarray, z: float, zeta: Tuple[np.ndarray, np.ndarray], alpha: float, r: float,
                    weights: np.ndarray, biases: np.ndarray) -> Tuple[np.ndarray, float]:
        x_p, x_q = zeta
        phi_p = self.phi(weights, biases, x_p)
        phi_q = self.phi(weights, biases, x_q)
        z = np.clip(z, 1e-5, 1e5)
        exp_term = np.exp(np.sum(phi_q * theta))

        grad_term = phi_p - (exp_term / z) * phi_q
        theta_new = theta + alpha * r * grad_term
        z_new = z + alpha * (exp_term - z)
        bound = 2e1 / np.sqrt(len(theta))
        theta_new = np.clip(theta_new, -bound, bound)
        return theta_new, z_new

    def approximate_kl_dv(self, theta: np.ndarray, its: int, weights: np.ndarray, biases: np.ndarray) -> float:
        first_term_samples = []
        f_q_samples = []
        for _ in range(its):
            x_p = self.sample_p()
            x_q = self.sample_q()
            first_term_samples.append(np.sum(theta * self.phi(weights, biases, x_p)))
            f_q_samples.append(np.sum(theta * self.phi(weights, biases, x_q)))
        first_term = float(np.mean(first_term_samples))
        f_q_array = np.asarray(f_q_samples, dtype=np.float64)
        f_q_max = float(np.max(f_q_array))
        log_second_term = f_q_max + float(np.log(np.mean(np.exp(f_q_array - f_q_max))))
        return float(first_term - log_second_term)

    def estimate_true_kl(self, num_samples: int = 200000) -> float:
        d = self.input_dim
        test_samples = np.random.multivariate_normal(mean=np.zeros(d), cov=np.eye(d), size=1000000)
        in_cube = np.all(np.abs(test_samples) <= 2, axis=1)
        prob_in_cube = np.mean(in_cube)
        log_gauss_const = -0.5 * d * np.log(2 * np.pi)
        log_q_uniform = -d * np.log(4.0)
        log_p0_vals = []
        for _ in range(num_samples):
            x = self.sample_p()
            log_p0 = log_gauss_const - 0.5 * float(np.sum(x ** 2))
            log_p0_vals.append(log_p0)
        e_log_p0 = float(np.mean(log_p0_vals))
        return float(e_log_p0 - log_q_uniform - np.log(prob_in_cube))

    def estimate_mutual_info_sklearn(self, num_samples: int = 30000) -> Tuple[float, np.ndarray]:
        samples_p = []
        samples_q = []
        for _ in range(num_samples):
            samples_p.append(self.sample_p())
            samples_q.append(self.sample_q())
        X = np.vstack([np.array(samples_p), np.array(samples_q)])
        y = np.hstack([np.zeros(len(samples_p)), np.ones(len(samples_q))])
        mi_scores = mutual_info_regression(X, y, random_state=42)
        total_mi = float(np.sum(mi_scores))
        return total_mi, mi_scores


def _sweep_vary_T(problem, config: SweepConfig, io: ExperimentIO,
                   run_exp_fn: Callable[[int, int, int, float], float], label_prefix: str,
                   true_mi: float,
                   sklearn_mi_sum_features: float = 0.0) -> Dict:
    Ts = list(config.vary_T_values)
    errors_T: List[float] = []
    error_stds_T: List[float] = []
    raw_results_T: List[Dict] = []
    m_fixed = config.fixed_m_for_T

    for T in Ts:
        T_errors: List[float] = []
        for run_idx in tqdm(range(config.runs_per_setting_T), desc=f'Running the T={T} setting'):
            err = run_exp_fn(m_fixed, T, config.its_eval, true_mi)
            if not np.isnan(err):
                T_errors.append(float(err))
        raw_results_T.append({
            'T': T,
            'm': m_fixed,
            'errors': T_errors.copy(),
            'mean_error': float(np.mean(T_errors)) if T_errors else float('nan'),
            'std_error': float(np.std(T_errors, ddof=1)) if len(T_errors) > 1 else float('nan')
        })
        if T_errors:
            errors_T.append(float(np.mean(T_errors)))
            error_stds_T.append(float(np.std(T_errors, ddof=1)))
        else:
            errors_T.append(float('nan'))
            error_stds_T.append(float('nan'))

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    results_T = {
        'Ts': Ts,
        'errors_mean': errors_T,
        'errors_std': error_stds_T,
        'm_fixed': m_fixed,
        'raw_results': raw_results_T,
        'timestamp': timestamp,
        'true_mi': float(true_mi),
        'sklearn_mi_sum_features': float(sklearn_mi_sum_features),
    }
    with open(os.path.join(io.raw_dir, f'{label_prefix}_varying_T_{timestamp}.pkl'), 'wb') as f:
        pickle.dump(results_T, f)
    return results_T


def _sweep_vary_m(problem, config: SweepConfig, io: ExperimentIO,
                   run_exp_fn: Callable[[int, int, int, float], float], label_prefix: str,
                   true_mi: float,
                   sklearn_mi_sum_features: float = 0.0) -> Dict:
    ms = list(config.vary_m_values)
    errors_m: List[float] = []
    error_stds_m: List[float] = []
    raw_results_m: List[Dict] = []
    T_fixed = config.fixed_T_for_m

    for m_val in ms:
        m_errors: List[float] = []
        for run_idx in tqdm(range(config.runs_per_setting_m), desc=f'Running the m={m_val} setting'):
            err = run_exp_fn(m_val, T_fixed, config.its_eval, true_mi)
            if not np.isnan(err):
                m_errors.append(float(err))
        raw_results_m.append({
            'T': T_fixed,
            'm': m_val,
            'errors': m_errors.copy(),
            'mean_error': float(np.mean(m_errors)) if m_errors else float('nan'),
            'std_error': float(np.std(m_errors, ddof=1)) if len(m_errors) > 1 else float('nan')
        })
        if m_errors:
            errors_m.append(float(np.mean(m_errors)))
            error_stds_m.append(float(np.std(m_errors, ddof=1)))
        else:
            errors_m.append(float('nan'))
            error_stds_m.append(float('nan'))

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    results_m = {
        'ms': ms,
        'errors_mean': errors_m,
        'errors_std': error_stds_m,
        'T_fixed': T_fixed,
        'raw_results': raw_results_m,
        'timestamp': timestamp,
        'true_mi': float(true_mi),
        'sklearn_mi_sum_features': float(sklearn_mi_sum_features),
    }
    with open(os.path.join(io.raw_dir, f'{label_prefix}_varying_m_{timestamp}.pkl'), 'wb') as f:
        pickle.dump(results_m, f)
    return results_m




def make_default_config_2d() -> SweepConfig:
    return SweepConfig(
        vary_T_values=[100, 500, 1_000, 5_000, 10_000],
        vary_m_values=[10, 50, 100, 500, 1_000],
        fixed_m_for_T=100,
        fixed_T_for_m=10_000,
        runs_per_setting_T=50,
        runs_per_setting_m=50,
        its_eval=5_000,
    )


def make_default_config_5d() -> SweepConfig:
    return SweepConfig(
        vary_T_values=[100, 500, 1_000, 5_000, 10_000, 50_000, 100_000],
        vary_m_values=[10, 50, 100, 500, 1_000, 5_000],
        fixed_m_for_T=500,
        fixed_T_for_m=500_000,
        runs_per_setting_T=50,
        runs_per_setting_m=50,
        its_eval=5_000,
    )


def run_experiment_2d(io: ExperimentIO, config: SweepConfig, rerun: bool) -> Tuple[Dict, Dict]:
    ensure_dirs(io)
    label_prefix = 'results_2d'
    problem = KLProblem(2)

    def run_exp(m: int, T: int, its_eval: int, true_mi: float) -> float:
        weights, biases = problem.init_network(m)
        r = 1 / m
        alpha = T ** (-2 / 3)
        theta = np.zeros(m)
        z = 1.0
        for _ in range(T):
            zeta = (problem.sample_p(), problem.sample_q())
            theta, z = problem.update_step(theta, z, zeta, alpha, r, weights, biases)
        kl_estimate = problem.approximate_kl_dv(theta, its_eval, weights, biases)
        return abs(float(true_mi) - float(kl_estimate))

    if rerun:
        true_kl = problem.estimate_true_kl()
        sklearn_mi_sum, _ = problem.estimate_mutual_info_sklearn(num_samples=30000)
        results_T = _sweep_vary_T(problem, config, io, run_exp, label_prefix, true_kl, sklearn_mi_sum)
        results_m = _sweep_vary_m(problem, config, io, run_exp, label_prefix, true_kl, sklearn_mi_sum)
    else:
        results_T, results_m = load_latest_cached(io.raw_dir, label_prefix)
    # augment with metadata needed by plotting utils
    inject_metadata(results_T, results_m)
    return results_T, results_m


def run_experiment_5d(io: ExperimentIO, config: SweepConfig, rerun: bool) -> Tuple[Dict, Dict]:
    ensure_dirs(io)
    label_prefix = 'results_5d'
    problem = KLProblem(5)

    def run_exp(m: int, T: int, its_eval: int, true_mi: float) -> float:
        weights, biases = problem.init_network(m)
        r = 1 / m
        alpha = T ** (-2 / 3)
        theta = np.zeros(m, dtype=float)
        z = 1.0
        for _ in range(T):
            zeta = (problem.sample_p(), problem.sample_q())
            theta, z = problem.update_step(theta, z, zeta, alpha, r, weights, biases)
        kl_estimate = problem.approximate_kl_dv(theta, its_eval, weights, biases)
        return abs(float(true_mi) - float(kl_estimate))

    if rerun:
        true_kl = problem.estimate_true_kl()
        sklearn_mi_sum, _ = problem.estimate_mutual_info_sklearn(num_samples=30000)
        results_T = _sweep_vary_T(problem, config, io, run_exp, label_prefix, true_kl, sklearn_mi_sum)
        results_m = _sweep_vary_m(problem, config, io, run_exp, label_prefix, true_kl, sklearn_mi_sum)
    else:
        results_T, results_m = load_latest_cached(io.raw_dir, label_prefix)
    inject_metadata(results_T, results_m)
    return results_T, results_m


def load_latest_cached(raw_dir: str, label_prefix: str) -> Tuple[Dict, Dict]:
    import glob
    t_files = glob.glob(os.path.join(raw_dir, f'{label_prefix}_varying_T_*.pkl'))
    m_files = glob.glob(os.path.join(raw_dir, f'{label_prefix}_varying_m_*.pkl'))
    if not t_files or not m_files:
        raise FileNotFoundError('No cached experiment files found. Set rerun=True to generate data.')
    latest_t = max(t_files, key=os.path.getmtime)
    latest_m = max(m_files, key=os.path.getmtime)
    with open(latest_t, 'rb') as f:
        results_T = pickle.load(f)
    with open(latest_m, 'rb') as f:
        results_m = pickle.load(f)
    return results_T, results_m


def inject_metadata(results_T: Dict, results_m: Dict) -> None:
    if 'm_fixed' not in results_T and 'ms' in results_m and len(results_m['ms']) > 0:
        results_T['m_fixed'] = results_m['ms'][0]
    if 'T_fixed' not in results_m and 'Ts' in results_T and len(results_T['Ts']) > 0:
        results_m['T_fixed'] = results_T['Ts'][0]
    if 'true_mi' not in results_T:
        results_T['true_mi'] = float(0.0)
    if 'true_mi' not in results_m:
        results_m['true_mi'] = float(0.0)
    if 'sklearn_mi_sum_features' not in results_T:
        results_T['sklearn_mi_sum_features'] = float(results_T.get('D_KL_true', 0.0))
    if 'sklearn_mi_sum_features' not in results_m:
        results_m['sklearn_mi_sum_features'] = float(results_m.get('D_KL_true', 0.0))


