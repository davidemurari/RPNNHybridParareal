import argparse
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, LogFormatterMathtext

from scripts.ode_solvers import solver
from scripts.parareal import parallel_solver
from scripts.repeated_experiments import run_experiment


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


def max_rel_error(sol_a, sol_b):
    diff_norm = np.linalg.norm(sol_a - sol_b, axis=1)
    ref_norm = np.linalg.norm(sol_b, axis=1)
    return np.max(diff_norm) / max(np.max(ref_norm), 1e-14)


if __name__ == "__main__":
    system_names = ["SIR", "Lorenz", "Brusselator", "Arenstorf", "Rober", "BurgerQ", "Burger1W", "BurgerSW"]

    parser = argparse.ArgumentParser(description="Run diagnostics: fine-solver accuracy and Parareal convergence plot.")
    parser.add_argument("--system", choices=system_names, required=True, help="System to analyze.")
    parser.add_argument("--nodes", choices=["uniform", "lobatto"], default="uniform", help="Collocation nodes.")
    parser.add_argument("--ab_init", choices=["uniform", "centred"], default="uniform", help="AB initialization.")
    parser.add_argument("--a_min", type=float, default=1.0, help="Lower bound for |a_i| in centred init.")
    parser.add_argument("--a_max", type=float, default=1.0, help="Upper bound for |a_i| in centred init.")
    parser.add_argument("--refine_factor", type=int, default=4, help="Reference refinement factor over dt_fine.")
    args = parser.parse_args()

    if args.a_min > args.a_max:
        parser.error("--a_min must be <= --a_max.")
    if args.refine_factor < 2:
        parser.error("--refine_factor must be >= 2.")

    cwd = os.getcwd()
    os.chdir(cwd + "/RPNN")
    print("Current working directory:", os.getcwd())
    
    # Keep diagnostics plot style aligned with scripts/plotting.py
    matplotlib.rcParams["text.usetex"] = True
    matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
    matplotlib.rcParams["font.family"] = "ptm"
    matplotlib.rcParams["font.size"] = 45

    _, _, _, _, data = run_experiment(
        [args.system, args.nodes, args.ab_init, args.a_min, args.a_max],
        return_nets=True,
        verbose=False,
    )

    y0 = data["y0"]
    time = data["time"]
    dts = data["dts"]
    vec_ref = data["vecRef"]
    number_processors = data["number_processors"]
    t_final = time[-1]

    dt_fine = vec_ref.dt_fine
    dt_ref = dt_fine / args.refine_factor

    time_fine, sol_fine = compute_solution_on_uniform_grid(y0, t_final, dt_fine, vec_ref)
    time_ref, sol_ref = compute_solution_on_uniform_grid(y0, t_final, dt_ref, vec_ref)

    sol_fine_on_coarse = sample_solution(time_fine, sol_fine, time)
    sol_ref_on_coarse = sample_solution(time_ref, sol_ref, time)

    fine_abs_error = np.max(np.linalg.norm(sol_fine_on_coarse - sol_ref_on_coarse, axis=1))
    fine_rel_error = max_rel_error(sol_fine_on_coarse, sol_ref_on_coarse)

    parareal_out = parallel_solver(
        time=time,
        data=data,
        dts=dts,
        vecRef=vec_ref,
        number_processors=number_processors,
        verbose=False,
        track_history=True,
        fine_reference=sol_ref_on_coarse,
    )

    coarse_approx, networks, total_time, _, avg_coarse_step, diagnostics = parareal_out
    _ = coarse_approx, networks, total_time, avg_coarse_step
    error_history = diagnostics["error_history"]
    iterations = np.arange(len(error_history))

    ab_folder = "centred" if args.ab_init == "centred" else "uniform"
    plots_dir = os.path.join("savedPlots", ab_folder)
    reports_dir = os.path.join("savedReports", ab_folder)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    plot_path = os.path.join(plots_dir, f"diagnostics_convergence_{args.system}_{args.nodes}.pdf")
    report_path = os.path.join(reports_dir, f"diagnostics_{args.system}_{args.nodes}.txt")
    summary_csv_path = os.path.join(reports_dir, f"diagnostics_{args.system}_{args.nodes}.csv")
    history_csv_path = os.path.join(reports_dir, f"diagnostics_convergence_{args.system}_{args.nodes}.csv")

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.semilogy(iterations, error_history, marker="o")
    ax.set_xlabel("Parareal Iteration")
    ax.set_ylabel("Relative error")
    ax.set_title("Parareal convergence")
    ax.grid(True, which="both", alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # Force major ticks at every decade in the displayed error range.
    y_vals = np.asarray(error_history)
    y_vals = y_vals[y_vals > 0]
    if y_vals.size > 0:
        exp_min = int(np.floor(np.log10(np.min(y_vals))))
        exp_max = int(np.ceil(np.log10(np.max(y_vals))))
        y_ticks = [10.0 ** e for e in range(exp_min, exp_max + 1)]
        ax.set_yticks(y_ticks)
        ax.set_ylim(10.0 ** exp_min, 10.0 ** exp_max)
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    fig.tight_layout()
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()

    with open(report_path, "w") as f:
        f.write("============================================================\n")
        f.write("Diagnostics report\n")
        f.write("============================================================\n")
        f.write(f"system: {args.system}\n")
        f.write(f"nodes: {args.nodes}\n")
        f.write(f"ab_init: {args.ab_init}\n")
        f.write(f"a_min: {args.a_min}\n")
        f.write(f"a_max: {args.a_max}\n")
        f.write(f"dt_fine: {dt_fine}\n")
        f.write(f"dt_ref: {dt_ref}\n")
        f.write(f"fine_abs_error_on_coarse_grid: {fine_abs_error}\n")
        f.write(f"fine_rel_error_on_coarse_grid: {fine_rel_error}\n")
        if len(error_history) > 0:
            f.write(f"parareal_initial_guess_rel_error_E0: {error_history[0]}\n")
            f.write(f"parareal_final_rel_error: {error_history[-1]}\n")
            f.write(f"parareal_iterations_recorded: {len(error_history)}\n")

    with open(summary_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "system",
                "nodes",
                "ab_init",
                "a_min",
                "a_max",
                "dt_fine",
                "dt_ref",
                "fine_abs_error_on_coarse_grid",
                "fine_rel_error_on_coarse_grid",
                "parareal_initial_guess_rel_error_E0",
                "parareal_final_rel_error",
                "parareal_iterations_recorded",
            ]
        )
        writer.writerow(
            [
                args.system,
                args.nodes,
                args.ab_init,
                args.a_min,
                args.a_max,
                dt_fine,
                dt_ref,
                fine_abs_error,
                fine_rel_error,
                float(error_history[0]) if len(error_history) > 0 else np.nan,
                float(error_history[-1]) if len(error_history) > 0 else np.nan,
                len(error_history),
            ]
        )

    with open(history_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "relative_error"])
        for k, err in enumerate(error_history):
            writer.writerow([k, float(err)])

    print(f"Saved convergence plot: {plot_path}")
    print(f"Saved diagnostics report: {report_path}")
    print(f"Saved diagnostics summary csv: {summary_csv_path}")
    print(f"Saved diagnostics convergence csv: {history_csv_path}")
    print(f"Fine solver relative error on coarse grid: {fine_rel_error:.6e}")
    if len(error_history) > 0:
        print(f"Parareal initial guess relative error E0: {error_history[0]:.6e}")
