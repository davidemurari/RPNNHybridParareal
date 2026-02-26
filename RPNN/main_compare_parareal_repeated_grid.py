import argparse
import csv
import os
import random
import time as time_lib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from tqdm.auto import tqdm

from scripts.ode_solvers import solver
from scripts.parareal import parallel_solver
from scripts.repeated_experiments import run_experiment


def parse_factors(text):
    vals = []
    for item in text.split(","):
        item = item.strip()
        if item:
            vals.append(float(item))
    if len(vals) == 0:
        raise ValueError("No coarse factors provided.")
    if any(v <= 0 for v in vals):
        raise ValueError("All coarse factors must be positive.")
    return vals


def summarize(values):
    arr = np.asarray(values, dtype=float)
    return float(np.mean(arr)), float(np.std(arr))


def compute_solution_on_uniform_grid(y0, t_final, dt, vec_ref):
    n_steps = int(np.ceil(t_final / dt))
    time_uniform = np.linspace(0.0, t_final, n_steps + 1)
    sol, time = solver([[y0, t_final, time_uniform], vec_ref], final=False)
    return time, sol


def sample_solution(time_src, sol_src, time_target):
    d = sol_src.shape[1]
    sampled = np.zeros((len(time_target), d))
    for j in range(d):
        sampled[:, j] = np.interp(time_target, time_src, sol_src[:, j])
    return sampled


if __name__ == "__main__":
    system_names = ["SIR", "Lorenz", "Brusselator", "Arenstorf", "Duffing", "Rober", "BurgerQ", "Burger1W", "BurgerSW"]

    parser = argparse.ArgumentParser(
        description="Compare RPNN-Parareal vs classical Parareal on the same macro grid used by main_repeated_experiments."
    )
    parser.add_argument("--system", choices=system_names, required=True, help="System to compare.")
    parser.add_argument("--nodes", choices=["uniform", "lobatto"], default="uniform", help="Collocation nodes for RPNN coarse solver.")
    parser.add_argument("--ab_init", choices=["uniform", "centred"], default="uniform", help="AB initialization for RPNN coarse solver.")
    parser.add_argument("--a_min", type=float, default=1.0, help="Lower bound for |a_i| when using centred AB init.")
    parser.add_argument("--a_max", type=float, default=1.0, help="Upper bound for |a_i| when using centred AB init.")
    parser.add_argument(
        "--coarse_factors",
        type=str,
        default="2,5,10,20,50,100",
        help="Comma-separated factors for classical coarse internal step: dt_coarse = factor * dt_fine.",
    )
    parser.add_argument("--num_trials", type=int, default=10, help="Number of repeated trials.")
    parser.add_argument("--seed_base", type=int, default=1234, help="Base seed for reproducible paired trials.")
    parser.add_argument("--no_progress", action="store_true", help="Disable progress bars.")
    args = parser.parse_args()

    if args.a_min > args.a_max:
        parser.error("--a_min must be <= --a_max.")
    if args.num_trials < 1:
        parser.error("--num_trials must be >= 1.")
    factors = parse_factors(args.coarse_factors)

    cwd = os.getcwd()
    os.chdir(cwd + "/RPNN")
    print("Current working directory:", os.getcwd())

    ab_folder = "centred" if args.ab_init == "centred" else "uniform"
    plots_dir = os.path.join("savedPlots", ab_folder)
    reports_dir = os.path.join("savedReports", ab_folder)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    rpnn_time = []
    rpnn_speedup = []
    rpnn_iters = []
    rpnn_e0 = []
    rpnn_efinal = []

    classical_time = {f: [] for f in factors}
    classical_speedup = {f: [] for f in factors}
    classical_iters = {f: [] for f in factors}
    classical_e0 = {f: [] for f in factors}
    classical_efinal = {f: [] for f in factors}

    dt_fine_values = []
    slab_dt_values = []

    total_solves = args.num_trials * (1 + len(factors))
    with tqdm(total=total_solves, desc="Comparison progress", unit="solve", disable=args.no_progress) as pbar:
        for trial in range(args.num_trials):
            seed = args.seed_base + trial
            np.random.seed(seed)
            random.seed(seed)

            data, time, dts, vec_ref, number_processors = run_experiment(
                [args.system, args.nodes, args.ab_init, args.a_min, args.a_max],
                setup_only=True,
            )

            y0 = data["y0"]
            t_final = time[-1]
            dt_fine_values.append(vec_ref.dt_fine)
            slab_dt_values.append(float(np.mean(dts)))

            start_serial = time_lib.time()
            _ = solver(([y0, t_final], vec_ref))
            serial_time = time_lib.time() - start_serial

            time_fine, sol_fine = compute_solution_on_uniform_grid(y0, t_final, vec_ref.dt_fine, vec_ref)
            fine_on_macro = sample_solution(time_fine, sol_fine, time)

            rpnn_out = parallel_solver(
                time=time,
                data=data,
                dts=dts,
                vecRef=vec_ref,
                number_processors=number_processors,
                verbose=False,
                track_history=True,
                fine_reference=fine_on_macro,
                coarse_mode="rpnn",
            )
            _, _, total_rpnn, _, _, diag_rpnn = rpnn_out
            rpnn_time.append(total_rpnn)
            rpnn_speedup.append(serial_time / total_rpnn)
            rpnn_iters.append(len(diag_rpnn["error_history"]))
            rpnn_e0.append(diag_rpnn["error_history"][0])
            rpnn_efinal.append(diag_rpnn["error_history"][-1])
            pbar.set_postfix_str(f"trial {trial + 1}/{args.num_trials} RPNN")
            pbar.update(1)

            for factor in factors:
                classical_out = parallel_solver(
                    time=time,
                    data=data,
                    dts=dts,
                    vecRef=vec_ref,
                    number_processors=number_processors,
                    verbose=False,
                    track_history=True,
                    fine_reference=fine_on_macro,
                    coarse_mode="classical",
                    coarse_dt=factor * vec_ref.dt_fine,
                )
                _, _, total_classical, _, _, diag_classical = classical_out
                classical_time[factor].append(total_classical)
                classical_speedup[factor].append(serial_time / total_classical)
                classical_iters[factor].append(len(diag_classical["error_history"]))
                classical_e0[factor].append(diag_classical["error_history"][0])
                classical_efinal[factor].append(diag_classical["error_history"][-1])
                pbar.set_postfix_str(f"trial {trial + 1}/{args.num_trials} classical x{factor:g}")
                pbar.update(1)

    dt_fine = float(np.mean(np.asarray(dt_fine_values)))
    slab_dt = float(np.mean(np.asarray(slab_dt_values)))
    report_path = os.path.join(reports_dir, f"compare_parareal_same_grid_{args.system}_{args.nodes}.txt")
    csv_path = os.path.join(reports_dir, f"compare_parareal_same_grid_{args.system}_{args.nodes}.csv")
    speedup_plot_path = os.path.join(plots_dir, f"compare_same_grid_speedup_{args.system}_{args.nodes}.pdf")
    iter_plot_path = os.path.join(plots_dir, f"compare_same_grid_iterations_{args.system}_{args.nodes}.pdf")

    with open(report_path, "w") as f:
        f.write("============================================================\n")
        f.write("Classical vs RPNN Parareal comparison (same macro grid)\n")
        f.write("============================================================\n")
        f.write(f"system: {args.system}\n")
        f.write(f"nodes: {args.nodes}\n")
        f.write(f"ab_init: {args.ab_init}\n")
        f.write(f"a_min: {args.a_min}\n")
        f.write(f"a_max: {args.a_max}\n")
        f.write(f"num_trials: {args.num_trials}\n")
        f.write(f"dt_fine (avg): {dt_fine}\n")
        f.write(f"macro slab dt (avg): {slab_dt}\n")
        f.write(f"coarse_factors (classical internal dt): {factors}\n\n")

        m_t, s_t = summarize(rpnn_time)
        m_su, s_su = summarize(rpnn_speedup)
        m_it, s_it = summarize(rpnn_iters)
        m_e0, s_e0 = summarize(rpnn_e0)
        m_ef, s_ef = summarize(rpnn_efinal)
        f.write("RPNN Parareal (single setup on same macro grid)\n")
        f.write("total_time_mean, total_time_std, speedup_mean, speedup_std, iters_mean, iters_std, E0_mean, E0_std, Efinal_mean, Efinal_std\n")
        f.write(f"{m_t:.6e},{s_t:.6e},{m_su:.6e},{s_su:.6e},{m_it:.6e},{s_it:.6e},{m_e0:.6e},{s_e0:.6e},{m_ef:.6e},{s_ef:.6e}\n\n")

        f.write("Classical Parareal by internal coarse dt factor\n")
        f.write("factor, dt_coarse_internal, total_time_mean, total_time_std, speedup_mean, speedup_std, iters_mean, iters_std, E0_mean, E0_std, Efinal_mean, Efinal_std\n")
        for factor in factors:
            m_t, s_t = summarize(classical_time[factor])
            m_su, s_su = summarize(classical_speedup[factor])
            m_it, s_it = summarize(classical_iters[factor])
            m_e0, s_e0 = summarize(classical_e0[factor])
            m_ef, s_ef = summarize(classical_efinal[factor])
            f.write(
                f"{factor},{factor * dt_fine:.6e},{m_t:.6e},{s_t:.6e},{m_su:.6e},{s_su:.6e},"
                f"{m_it:.6e},{s_it:.6e},{m_e0:.6e},{s_e0:.6e},{m_ef:.6e},{s_ef:.6e}\n"
            )

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "system",
                "nodes",
                "ab_init",
                "a_min",
                "a_max",
                "num_trials",
                "dt_fine",
                "macro_slab_dt",
                "method",
                "factor",
                "dt_coarse_internal",
                "total_time_mean",
                "total_time_std",
                "speedup_mean",
                "speedup_std",
                "iters_mean",
                "iters_std",
                "E0_mean",
                "E0_std",
                "Efinal_mean",
                "Efinal_std",
            ]
        )

        m_t, s_t = summarize(rpnn_time)
        m_su, s_su = summarize(rpnn_speedup)
        m_it, s_it = summarize(rpnn_iters)
        m_e0, s_e0 = summarize(rpnn_e0)
        m_ef, s_ef = summarize(rpnn_efinal)
        writer.writerow(
            [
                args.system,
                args.nodes,
                args.ab_init,
                args.a_min,
                args.a_max,
                args.num_trials,
                dt_fine,
                slab_dt,
                "rpnn",
                "",
                "",
                m_t,
                s_t,
                m_su,
                s_su,
                m_it,
                s_it,
                m_e0,
                s_e0,
                m_ef,
                s_ef,
            ]
        )

        for factor in factors:
            m_t, s_t = summarize(classical_time[factor])
            m_su, s_su = summarize(classical_speedup[factor])
            m_it, s_it = summarize(classical_iters[factor])
            m_e0, s_e0 = summarize(classical_e0[factor])
            m_ef, s_ef = summarize(classical_efinal[factor])
            writer.writerow(
                [
                    args.system,
                    args.nodes,
                    args.ab_init,
                    args.a_min,
                    args.a_max,
                    args.num_trials,
                    dt_fine,
                    slab_dt,
                    "classical",
                    factor,
                    factor * dt_fine,
                    m_t,
                    s_t,
                    m_su,
                    s_su,
                    m_it,
                    s_it,
                    m_e0,
                    s_e0,
                    m_ef,
                    s_ef,
                ]
            )

    factor_arr = np.asarray(factors, dtype=float)
    classical_speedup_mean = np.asarray([summarize(classical_speedup[f])[0] for f in factors], dtype=float)
    classical_speedup_std = np.asarray([summarize(classical_speedup[f])[1] for f in factors], dtype=float)
    rpnn_speedup_mean = summarize(rpnn_speedup)[0]
    rpnn_speedup_std = summarize(rpnn_speedup)[1]

    plt.figure(figsize=(9, 5))
    plt.errorbar(factor_arr, classical_speedup_mean, yerr=classical_speedup_std, marker="o", capsize=3, label="Classical Parareal")
    plt.axhline(rpnn_speedup_mean, color="tab:orange", linestyle="--", label="RPNN Parareal")
    plt.fill_between(factor_arr, rpnn_speedup_mean - rpnn_speedup_std, rpnn_speedup_mean + rpnn_speedup_std, color="tab:orange", alpha=0.2)
    plt.xlabel("Classical internal coarse factor (dt_coarse / dt_fine)")
    plt.ylabel("Speedup")
    plt.title(f"Parareal Speedup Comparison (same macro grid, {args.system}, {args.nodes})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(speedup_plot_path, bbox_inches="tight")
    plt.close()

    classical_iters_mean = np.asarray([summarize(classical_iters[f])[0] for f in factors], dtype=float)
    classical_iters_std = np.asarray([summarize(classical_iters[f])[1] for f in factors], dtype=float)
    rpnn_iters_mean = summarize(rpnn_iters)[0]
    rpnn_iters_std = summarize(rpnn_iters)[1]

    plt.figure(figsize=(9, 5))
    plt.errorbar(factor_arr, classical_iters_mean, yerr=classical_iters_std, marker="o", capsize=3, label="Classical Parareal")
    plt.axhline(rpnn_iters_mean, color="tab:orange", linestyle="--", label="RPNN Parareal")
    plt.fill_between(factor_arr, rpnn_iters_mean - rpnn_iters_std, rpnn_iters_mean + rpnn_iters_std, color="tab:orange", alpha=0.2)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel("Classical internal coarse factor (dt_coarse / dt_fine)")
    plt.ylabel("Parareal iterations")
    plt.title(f"Parareal Iteration Count Comparison (same macro grid, {args.system}, {args.nodes})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(iter_plot_path, bbox_inches="tight")
    plt.close()

    print(f"Saved comparison report: {report_path}")
    print(f"Saved comparison csv: {csv_path}")
    print(f"Saved speedup plot: {speedup_plot_path}")
    print(f"Saved iterations plot: {iter_plot_path}")
