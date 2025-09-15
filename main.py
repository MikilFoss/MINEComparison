import argparse
import os

from experiments import (
    ExperimentIO,
    make_default_config_2d,
    make_default_config_5d,
    run_experiment_2d,
    run_experiment_5d,
)
from visualization import plot_varying_T, plot_varying_m


def get_base_dirs() -> ExperimentIO:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(base_dir, 'raw_data')
    results_dir = os.path.join(base_dir, 'results')
    return ExperimentIO(raw_dir=raw_dir, results_dir=results_dir)


def main():
    parser = argparse.ArgumentParser(description='Unified 2D/5D experiments and visualization')
    parser.add_argument('--five_d', action='store_true', help='Use 5D problem (default: 2D)')
    parser.add_argument('--rerun', action='store_true', help='Rerun experiments; otherwise use cached latest')
    parser.add_argument('--runs_T', type=int, default=None, help='Override runs per T setting')
    parser.add_argument('--runs_m', type=int, default=None, help='Override runs per m setting')
    parser.add_argument('--its', type=int, default=None, help='Override evaluation iterations per setting')
    parser.add_argument('--fixed_m', type=int, default=None, help='Override fixed m for T sweep')
    parser.add_argument('--fixed_T', type=int, default=None, help='Override fixed T for m sweep')
    args = parser.parse_args()

    io = get_base_dirs()

    if args.five_d:
        config = make_default_config_5d()
        dim_label = '5D'
        if args.runs_T is not None:
            config.runs_per_setting_T = args.runs_T
        if args.runs_m is not None:
            config.runs_per_setting_m = args.runs_m
        if args.its is not None:
            config.its_eval = args.its
        if args.fixed_m is not None:
            config.fixed_m_for_T = args.fixed_m
        if args.fixed_T is not None:
            config.fixed_T_for_m = args.fixed_T
        results_T, results_m = run_experiment_5d(io, config, rerun=args.rerun)
    else:
        config = make_default_config_2d()
        dim_label = '2D'
        if args.runs_T is not None:
            config.runs_per_setting_T = args.runs_T
        if args.runs_m is not None:
            config.runs_per_setting_m = args.runs_m
        if args.its is not None:
            config.its_eval = args.its
        if args.fixed_m is not None:
            config.fixed_m_for_T = args.fixed_m
        if args.fixed_T is not None:
            config.fixed_T_for_m = args.fixed_T
        results_T, results_m = run_experiment_2d(io, config, rerun=args.rerun)

    # Visualization
    out_T = plot_varying_T(results_T, io.results_dir, dim_label)
    out_m = plot_varying_m(results_m, io.results_dir, dim_label)

    print(f'Regenerated T plot: {out_T}')
    print(f'Regenerated m plot: {out_m}')


if __name__ == '__main__':
    main()


