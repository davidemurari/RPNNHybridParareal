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


def parse_num_t_values(text):
    if text is None:
        return []
    text = text.strip()
    if text == "":
        return []
    vals = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        v = int(item)
        if v < 2:
            raise ValueError("All num_t values must be >= 2.")
        vals.append(v)
    if len(vals) == 0:
        return []
    # stable dedup
    seen = set()
    uniq = []
    for v in vals:
        if v not in seen:
            uniq.append(v)
            seen.add(v)
    return uniq


def summarize(values):
    arr = np.asarray(values, dtype=float)
    return float(np.mean(arr)), float(np.std(arr))


def sum_profile_times(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.sum(arr))


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


def init_setting_store(factors):
    return {
        "dt_fine_values": [],
        "slab_dt_values": [],
        "rpnn": {
            "time": [],
            "speedup": [],
            "iters": [],
            "e0": [],
            "efinal": [],
            "fine_stage": [],
            "coarse_stage": [],
            "corr_stage": [],
            "lsq_time": [],
            "lsq_fast_calls": [],
            "lsq_full_calls": [],
            "lsq_fallbacks": [],
            "residual_time": [],
            "jac_time": [],
            "flowmap_time": [],
        },
        "classical": {
            f: {
                "time": [],
                "speedup": [],
                "iters": [],
                "e0": [],
                "efinal": [],
                "fine_stage": [],
                "coarse_stage": [],
                "corr_stage": [],
            }
            for f in factors
        },
    }


def get_rpnn_summary(store):
    rp = store["rpnn"]
    return {
        "total_time_mean": summarize(rp["time"])[0],
        "total_time_std": summarize(rp["time"])[1],
        "speedup_mean": summarize(rp["speedup"])[0],
        "speedup_std": summarize(rp["speedup"])[1],
        "iters_mean": summarize(rp["iters"])[0],
        "iters_std": summarize(rp["iters"])[1],
        "e0_mean": summarize(rp["e0"])[0],
        "e0_std": summarize(rp["e0"])[1],
        "efinal_mean": summarize(rp["efinal"])[0],
        "efinal_std": summarize(rp["efinal"])[1],
        "fine_stage_mean": summarize(rp["fine_stage"])[0],
        "fine_stage_std": summarize(rp["fine_stage"])[1],
        "coarse_stage_mean": summarize(rp["coarse_stage"])[0],
        "coarse_stage_std": summarize(rp["coarse_stage"])[1],
        "correction_stage_mean": summarize(rp["corr_stage"])[0],
        "correction_stage_std": summarize(rp["corr_stage"])[1],
        "rpnn_lsq_time_mean": summarize(rp["lsq_time"])[0],
        "rpnn_lsq_time_std": summarize(rp["lsq_time"])[1],
        "rpnn_lsq_fast_calls_mean": summarize(rp["lsq_fast_calls"])[0],
        "rpnn_lsq_fast_calls_std": summarize(rp["lsq_fast_calls"])[1],
        "rpnn_lsq_full_calls_mean": summarize(rp["lsq_full_calls"])[0],
        "rpnn_lsq_full_calls_std": summarize(rp["lsq_full_calls"])[1],
        "rpnn_lsq_fallbacks_mean": summarize(rp["lsq_fallbacks"])[0],
        "rpnn_lsq_fallbacks_std": summarize(rp["lsq_fallbacks"])[1],
        "rpnn_residual_time_mean": summarize(rp["residual_time"])[0],
        "rpnn_residual_time_std": summarize(rp["residual_time"])[1],
        "rpnn_jac_time_mean": summarize(rp["jac_time"])[0],
        "rpnn_jac_time_std": summarize(rp["jac_time"])[1],
        "rpnn_flowmap_time_mean": summarize(rp["flowmap_time"])[0],
        "rpnn_flowmap_time_std": summarize(rp["flowmap_time"])[1],
    }


def get_classical_summary(store, factor):
    cl = store["classical"][factor]
    return {
        "total_time_mean": summarize(cl["time"])[0],
        "total_time_std": summarize(cl["time"])[1],
        "speedup_mean": summarize(cl["speedup"])[0],
        "speedup_std": summarize(cl["speedup"])[1],
        "iters_mean": summarize(cl["iters"])[0],
        "iters_std": summarize(cl["iters"])[1],
        "e0_mean": summarize(cl["e0"])[0],
        "e0_std": summarize(cl["e0"])[1],
        "efinal_mean": summarize(cl["efinal"])[0],
        "efinal_std": summarize(cl["efinal"])[1],
        "fine_stage_mean": summarize(cl["fine_stage"])[0],
        "fine_stage_std": summarize(cl["fine_stage"])[1],
        "coarse_stage_mean": summarize(cl["coarse_stage"])[0],
        "coarse_stage_std": summarize(cl["coarse_stage"])[1],
        "correction_stage_mean": summarize(cl["corr_stage"])[0],
        "correction_stage_std": summarize(cl["corr_stage"])[1],
    }


def write_csv_row(writer, args, num_t, dt_fine, slab_dt, method, factor, row):
    writer.writerow(
        [
            args.system,
            args.nodes,
            args.ab_init,
            args.a_min,
            args.a_max,
            args.num_trials,
            num_t,
            dt_fine,
            slab_dt,
            method,
            factor if factor is not None else "",
            (factor * dt_fine) if factor is not None else "",
            row["total_time_mean"],
            row["total_time_std"],
            row["speedup_mean"],
            row["speedup_std"],
            row["iters_mean"],
            row["iters_std"],
            row["e0_mean"],
            row["e0_std"],
            row["efinal_mean"],
            row["efinal_std"],
            row["fine_stage_mean"],
            row["fine_stage_std"],
            row["coarse_stage_mean"],
            row["coarse_stage_std"],
            row["correction_stage_mean"],
            row["correction_stage_std"],
            row.get("rpnn_lsq_time_mean", ""),
            row.get("rpnn_lsq_time_std", ""),
            row.get("rpnn_lsq_fast_calls_mean", ""),
            row.get("rpnn_lsq_fast_calls_std", ""),
            row.get("rpnn_lsq_full_calls_mean", ""),
            row.get("rpnn_lsq_full_calls_std", ""),
            row.get("rpnn_lsq_fallbacks_mean", ""),
            row.get("rpnn_lsq_fallbacks_std", ""),
            row.get("rpnn_residual_time_mean", ""),
            row.get("rpnn_residual_time_std", ""),
            row.get("rpnn_jac_time_mean", ""),
            row.get("rpnn_jac_time_std", ""),
            row.get("rpnn_flowmap_time_mean", ""),
            row.get("rpnn_flowmap_time_std", ""),
        ]
    )


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
    parser.add_argument(
        "--num_t_values",
        type=str,
        default="",
        help="Optional comma-separated macro-grid point counts for RPNN/classical comparison on each grid (e.g. 26,51,101). If empty, use default grid from run_experiment.",
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
    num_t_values = parse_num_t_values(args.num_t_values)
    num_t_sweep = num_t_values if len(num_t_values) > 0 else [None]

    cwd = os.getcwd()
    os.chdir(cwd + "/RPNN")
    print("Current working directory:", os.getcwd())

    ab_folder = "centred" if args.ab_init == "centred" else "uniform"
    plots_dir = os.path.join("savedPlots", ab_folder)
    reports_dir = os.path.join("savedReports", ab_folder)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    stores = {}

    total_solves = args.num_trials * len(num_t_sweep) * (1 + len(factors))
    with tqdm(total=total_solves, desc="Comparison progress", unit="solve", disable=args.no_progress) as pbar:
        for num_t_choice in num_t_sweep:
            for trial in range(args.num_trials):
                seed = args.seed_base + trial
                np.random.seed(seed)
                random.seed(seed)

                data, time, dts, vec_ref, number_processors = run_experiment(
                    [args.system, args.nodes, args.ab_init, args.a_min, args.a_max],
                    setup_only=True,
                    num_t_override=num_t_choice,
                )

                num_t_key = int(data["num_t"])
                if num_t_key not in stores:
                    stores[num_t_key] = init_setting_store(factors)
                store = stores[num_t_key]

                y0 = data["y0"]
                t_final = time[-1]
                store["dt_fine_values"].append(vec_ref.dt_fine)
                store["slab_dt_values"].append(float(np.mean(dts)))

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

                rp = store["rpnn"]
                rp["time"].append(total_rpnn)
                rp["speedup"].append(serial_time / total_rpnn)
                rp["iters"].append(len(diag_rpnn["error_history"]))
                rp["e0"].append(diag_rpnn["error_history"][0])
                rp["efinal"].append(diag_rpnn["error_history"][-1])
                timing_rpnn = diag_rpnn.get("timing_profile", {})
                rp["fine_stage"].append(sum_profile_times(timing_rpnn.get("fine_stage_times", [])))
                rp["coarse_stage"].append(sum_profile_times(timing_rpnn.get("coarse_update_times", [])))
                rp["corr_stage"].append(sum_profile_times(timing_rpnn.get("correction_times", [])))
                rp_prof = timing_rpnn.get("rpnn_profile_totals", {})
                rp["lsq_time"].append(float(rp_prof.get("lsq_time", 0.0)))
                rp["lsq_fast_calls"].append(float(rp_prof.get("lsq_fast_calls", 0)))
                rp["lsq_full_calls"].append(float(rp_prof.get("lsq_full_calls", 0)))
                rp["lsq_fallbacks"].append(float(rp_prof.get("lsq_fallbacks", 0)))
                rp["residual_time"].append(float(rp_prof.get("residual_time", 0.0)))
                rp["jac_time"].append(float(rp_prof.get("jac_time", 0.0)))
                rp["flowmap_time"].append(float(rp_prof.get("flowmap_total_time", 0.0)))
                pbar.set_postfix_str(
                    f"num_t {num_t_key} trial {trial + 1}/{args.num_trials} RPNN"
                )
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
                    cl = store["classical"][factor]
                    cl["time"].append(total_classical)
                    cl["speedup"].append(serial_time / total_classical)
                    cl["iters"].append(len(diag_classical["error_history"]))
                    cl["e0"].append(diag_classical["error_history"][0])
                    cl["efinal"].append(diag_classical["error_history"][-1])
                    timing_classical = diag_classical.get("timing_profile", {})
                    cl["fine_stage"].append(sum_profile_times(timing_classical.get("fine_stage_times", [])))
                    cl["coarse_stage"].append(sum_profile_times(timing_classical.get("coarse_update_times", [])))
                    cl["corr_stage"].append(sum_profile_times(timing_classical.get("correction_times", [])))
                    pbar.set_postfix_str(
                        f"num_t {num_t_key} trial {trial + 1}/{args.num_trials} classical x{factor:g}"
                    )
                    pbar.update(1)

    num_t_keys = sorted(stores.keys())

    multi_num_t = len(num_t_keys) > 1
    suffix = "_numt_sweep" if multi_num_t else ""

    report_path = os.path.join(reports_dir, f"compare_parareal_same_grid_{args.system}_{args.nodes}{suffix}.txt")
    csv_path = os.path.join(reports_dir, f"compare_parareal_same_grid_{args.system}_{args.nodes}{suffix}.csv")
    speedup_plot_path = os.path.join(plots_dir, f"compare_same_grid_speedup_{args.system}_{args.nodes}{suffix}.pdf")
    iter_plot_path = os.path.join(plots_dir, f"compare_same_grid_iterations_{args.system}_{args.nodes}{suffix}.pdf")
    min_time_plot_path = os.path.join(plots_dir, f"compare_same_grid_min_time_{args.system}_{args.nodes}{suffix}.pdf")

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
        if len(num_t_values) > 0:
            f.write(f"num_t_values (requested): {num_t_values}\n")
        f.write(f"coarse_factors (classical internal dt): {factors}\n\n")

        global_best_rpnn = None
        global_best_classical = None

        for num_t_key in num_t_keys:
            store = stores[num_t_key]
            dt_fine = float(np.mean(np.asarray(store["dt_fine_values"])))
            slab_dt = float(np.mean(np.asarray(store["slab_dt_values"])))

            rp_row = get_rpnn_summary(store)
            f.write("------------------------------------------------------------\n")
            f.write(f"num_t: {num_t_key}\n")
            f.write(f"dt_fine (avg): {dt_fine}\n")
            f.write(f"macro slab dt (avg): {slab_dt}\n\n")

            f.write("RPNN Parareal\n")
            f.write("total_time_mean, total_time_std, speedup_mean, speedup_std, iters_mean, iters_std, E0_mean, E0_std, Efinal_mean, Efinal_std, fine_stage_mean, fine_stage_std, coarse_stage_mean, coarse_stage_std, correction_stage_mean, correction_stage_std, lsq_time_mean, lsq_time_std, lsq_fast_calls_mean, lsq_fast_calls_std, lsq_full_calls_mean, lsq_full_calls_std, lsq_fallbacks_mean, lsq_fallbacks_std, residual_time_mean, residual_time_std, jac_time_mean, jac_time_std, flowmap_time_mean, flowmap_time_std\n")
            f.write(
                f"{rp_row['total_time_mean']:.6e},{rp_row['total_time_std']:.6e},{rp_row['speedup_mean']:.6e},{rp_row['speedup_std']:.6e},{rp_row['iters_mean']:.6e},{rp_row['iters_std']:.6e},{rp_row['e0_mean']:.6e},{rp_row['e0_std']:.6e},{rp_row['efinal_mean']:.6e},{rp_row['efinal_std']:.6e},{rp_row['fine_stage_mean']:.6e},{rp_row['fine_stage_std']:.6e},{rp_row['coarse_stage_mean']:.6e},{rp_row['coarse_stage_std']:.6e},{rp_row['correction_stage_mean']:.6e},{rp_row['correction_stage_std']:.6e},{rp_row['rpnn_lsq_time_mean']:.6e},{rp_row['rpnn_lsq_time_std']:.6e},{rp_row['rpnn_lsq_fast_calls_mean']:.6e},{rp_row['rpnn_lsq_fast_calls_std']:.6e},{rp_row['rpnn_lsq_full_calls_mean']:.6e},{rp_row['rpnn_lsq_full_calls_std']:.6e},{rp_row['rpnn_lsq_fallbacks_mean']:.6e},{rp_row['rpnn_lsq_fallbacks_std']:.6e},{rp_row['rpnn_residual_time_mean']:.6e},{rp_row['rpnn_residual_time_std']:.6e},{rp_row['rpnn_jac_time_mean']:.6e},{rp_row['rpnn_jac_time_std']:.6e},{rp_row['rpnn_flowmap_time_mean']:.6e},{rp_row['rpnn_flowmap_time_std']:.6e}\n\n"
            )

            f.write("Classical Parareal by internal coarse dt factor\n")
            f.write("factor, dt_coarse_internal, total_time_mean, total_time_std, speedup_mean, speedup_std, iters_mean, iters_std, E0_mean, E0_std, Efinal_mean, Efinal_std, fine_stage_mean, fine_stage_std, coarse_stage_mean, coarse_stage_std, correction_stage_mean, correction_stage_std\n")
            best_cl_time = None
            best_cl_factor = None
            for factor in factors:
                cl_row = get_classical_summary(store, factor)
                f.write(
                    f"{factor},{factor * dt_fine:.6e},{cl_row['total_time_mean']:.6e},{cl_row['total_time_std']:.6e},{cl_row['speedup_mean']:.6e},{cl_row['speedup_std']:.6e},{cl_row['iters_mean']:.6e},{cl_row['iters_std']:.6e},{cl_row['e0_mean']:.6e},{cl_row['e0_std']:.6e},{cl_row['efinal_mean']:.6e},{cl_row['efinal_std']:.6e},{cl_row['fine_stage_mean']:.6e},{cl_row['fine_stage_std']:.6e},{cl_row['coarse_stage_mean']:.6e},{cl_row['coarse_stage_std']:.6e},{cl_row['correction_stage_mean']:.6e},{cl_row['correction_stage_std']:.6e}\n"
                )
                if best_cl_time is None or cl_row["total_time_mean"] < best_cl_time:
                    best_cl_time = cl_row["total_time_mean"]
                    best_cl_factor = factor

            f.write(
                f"best_classical_for_num_t,{best_cl_factor},{best_cl_time:.6e}\n\n"
            )

            if global_best_rpnn is None or rp_row["total_time_mean"] < global_best_rpnn["time"]:
                global_best_rpnn = {
                    "time": rp_row["total_time_mean"],
                    "num_t": num_t_key,
                    "slab_dt": slab_dt,
                }
            if global_best_classical is None or best_cl_time < global_best_classical["time"]:
                global_best_classical = {
                    "time": best_cl_time,
                    "num_t": num_t_key,
                    "slab_dt": slab_dt,
                    "factor": best_cl_factor,
                }

        f.write("============================================================\n")
        f.write("Global best timings\n")
        f.write("============================================================\n")
        f.write(
            f"rpnn_best_time: {global_best_rpnn['time']:.6e} at num_t={global_best_rpnn['num_t']} (macro dt={global_best_rpnn['slab_dt']:.6e})\n"
        )
        f.write(
            f"classical_best_time: {global_best_classical['time']:.6e} at num_t={global_best_classical['num_t']} (macro dt={global_best_classical['slab_dt']:.6e}), factor={global_best_classical['factor']}\n"
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
                "num_t",
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
                "fine_stage_mean",
                "fine_stage_std",
                "coarse_stage_mean",
                "coarse_stage_std",
                "correction_stage_mean",
                "correction_stage_std",
                "rpnn_lsq_time_mean",
                "rpnn_lsq_time_std",
                "rpnn_lsq_fast_calls_mean",
                "rpnn_lsq_fast_calls_std",
                "rpnn_lsq_full_calls_mean",
                "rpnn_lsq_full_calls_std",
                "rpnn_lsq_fallbacks_mean",
                "rpnn_lsq_fallbacks_std",
                "rpnn_residual_time_mean",
                "rpnn_residual_time_std",
                "rpnn_jac_time_mean",
                "rpnn_jac_time_std",
                "rpnn_flowmap_time_mean",
                "rpnn_flowmap_time_std",
            ]
        )

        for num_t_key in num_t_keys:
            store = stores[num_t_key]
            dt_fine = float(np.mean(np.asarray(store["dt_fine_values"])))
            slab_dt = float(np.mean(np.asarray(store["slab_dt_values"])))

            rp_row = get_rpnn_summary(store)
            write_csv_row(
                writer=writer,
                args=args,
                num_t=num_t_key,
                dt_fine=dt_fine,
                slab_dt=slab_dt,
                method="rpnn",
                factor=None,
                row=rp_row,
            )

            for factor in factors:
                cl_row = get_classical_summary(store, factor)
                write_csv_row(
                    writer=writer,
                    args=args,
                    num_t=num_t_key,
                    dt_fine=dt_fine,
                    slab_dt=slab_dt,
                    method="classical",
                    factor=factor,
                    row=cl_row,
                )

    if not multi_num_t:
        # Preserve previous plot behavior for the single-grid case.
        only_key = num_t_keys[0]
        store = stores[only_key]
        factor_arr = np.asarray(factors, dtype=float)
        classical_speedup_mean = np.asarray([summarize(store["classical"][f]["speedup"])[0] for f in factors], dtype=float)
        classical_speedup_std = np.asarray([summarize(store["classical"][f]["speedup"])[1] for f in factors], dtype=float)
        rpnn_speedup_mean = summarize(store["rpnn"]["speedup"])[0]
        rpnn_speedup_std = summarize(store["rpnn"]["speedup"])[1]

        plt.figure(figsize=(9, 5))
        plt.errorbar(factor_arr, classical_speedup_mean, yerr=classical_speedup_std, marker="o", capsize=3, label="Classical Parareal")
        plt.axhline(rpnn_speedup_mean, color="tab:orange", linestyle="--", label="RPNN Parareal")
        plt.fill_between(
            factor_arr,
            rpnn_speedup_mean - rpnn_speedup_std,
            rpnn_speedup_mean + rpnn_speedup_std,
            color="tab:orange",
            alpha=0.2,
        )
        plt.xlabel("Classical internal coarse factor (dt_coarse / dt_fine)")
        plt.ylabel("Speedup")
        plt.title(f"Parareal Speedup Comparison (same macro grid, {args.system}, {args.nodes})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(speedup_plot_path, bbox_inches="tight")
        plt.close()

        classical_iters_mean = np.asarray([summarize(store["classical"][f]["iters"])[0] for f in factors], dtype=float)
        classical_iters_std = np.asarray([summarize(store["classical"][f]["iters"])[1] for f in factors], dtype=float)
        rpnn_iters_mean = summarize(store["rpnn"]["iters"])[0]
        rpnn_iters_std = summarize(store["rpnn"]["iters"])[1]

        plt.figure(figsize=(9, 5))
        plt.errorbar(factor_arr, classical_iters_mean, yerr=classical_iters_std, marker="o", capsize=3, label="Classical Parareal")
        plt.axhline(rpnn_iters_mean, color="tab:orange", linestyle="--", label="RPNN Parareal")
        plt.fill_between(
            factor_arr,
            rpnn_iters_mean - rpnn_iters_std,
            rpnn_iters_mean + rpnn_iters_std,
            color="tab:orange",
            alpha=0.2,
        )
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.xlabel("Classical internal coarse factor (dt_coarse / dt_fine)")
        plt.ylabel("Parareal iterations")
        plt.title(f"Parareal Iteration Count Comparison (same macro grid, {args.system}, {args.nodes})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(iter_plot_path, bbox_inches="tight")
        plt.close()
    else:
        # Multi-grid mode: plot minimum timing curves vs macro slab dt.
        x_slab_dt = []
        y_rpnn_time = []
        y_classical_best_time = []

        for num_t_key in num_t_keys:
            store = stores[num_t_key]
            slab_dt = float(np.mean(np.asarray(store["slab_dt_values"])))
            rpnn_time_mean = summarize(store["rpnn"]["time"])[0]
            cl_best = min(summarize(store["classical"][f]["time"])[0] for f in factors)
            x_slab_dt.append(slab_dt)
            y_rpnn_time.append(rpnn_time_mean)
            y_classical_best_time.append(cl_best)

        x_slab_dt = np.asarray(x_slab_dt, dtype=float)
        y_rpnn_time = np.asarray(y_rpnn_time, dtype=float)
        y_classical_best_time = np.asarray(y_classical_best_time, dtype=float)

        order = np.argsort(x_slab_dt)
        x_slab_dt = x_slab_dt[order]
        y_rpnn_time = y_rpnn_time[order]
        y_classical_best_time = y_classical_best_time[order]

        plt.figure(figsize=(9, 5))
        plt.plot(x_slab_dt, y_rpnn_time, marker="o", label="RPNN")
        plt.plot(x_slab_dt, y_classical_best_time, marker="s", label="Classical (best factor)")
        plt.xlabel("Macro slab dt")
        plt.ylabel("Total time")
        plt.title(f"Best Timing Curves ({args.system}, {args.nodes})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(min_time_plot_path, bbox_inches="tight")
        plt.close()

    print(f"Saved comparison report: {report_path}")
    print(f"Saved comparison csv: {csv_path}")
    if multi_num_t:
        print(f"Saved minimum-time plot: {min_time_plot_path}")
    else:
        print(f"Saved speedup plot: {speedup_plot_path}")
        print(f"Saved iterations plot: {iter_plot_path}")
