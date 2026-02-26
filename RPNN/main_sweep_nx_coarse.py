import argparse
import csv
import os
import random
import time as time_lib
from datetime import datetime

import numpy as np
from tqdm.auto import tqdm

from scripts.ode_solvers import solver
from scripts.parareal import parallel_solver
from scripts.repeated_experiments import run_experiment


def parse_int_list(text):
    vals = []
    for item in text.split(","):
        item = item.strip()
        if item:
            vals.append(int(item))
    if len(vals) == 0:
        raise ValueError("No integer values provided.")
    return vals


def parse_float_list(text):
    vals = []
    for item in text.split(","):
        item = item.strip()
        if item:
            vals.append(float(item))
    if len(vals) == 0:
        raise ValueError("No float values provided.")
    if any(v <= 0 for v in vals):
        raise ValueError("All coarse factors must be positive.")
    return vals


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


def build_coarse_grid(t_final, dt_fine, factor):
    h_target = factor * dt_fine
    if h_target <= 0:
        raise ValueError("Computed coarse step must be positive.")
    time = np.arange(0.0, t_final, h_target)
    if len(time) == 0 or time[0] != 0.0:
        time = np.insert(time, 0, 0.0)
    # Avoid creating a tiny last interval due floating-point roundoff.
    if (t_final - time[-1]) <= 1e-9 * max(1.0, t_final, h_target):
        time[-1] = t_final
    elif (t_final - time[-1]) < 0.5 * dt_fine:
        time[-1] = t_final
    elif time[-1] < t_final:
        time = np.append(time, t_final)
    elif time[-1] > t_final:
        time[-1] = t_final
    return time, np.diff(time)


def mean_std(arr):
    a = np.asarray(arr, dtype=float)
    return float(np.mean(a)), float(np.std(a))


if __name__ == "__main__":
    system_names = ["SIR", "Lorenz", "Brusselator", "Arenstorf", "Duffing", "Rober", "BurgerQ", "Burger1W", "BurgerSW"]

    parser = argparse.ArgumentParser(
        description="Sweep n_x (=L) and coarse factors on a grid, comparing RPNN and classical Parareal."
    )
    parser.add_argument("--system", choices=system_names, required=True, help="System to run.")
    parser.add_argument("--nodes", choices=["uniform", "lobatto"], default="uniform", help="Collocation nodes.")
    parser.add_argument("--ab_init", choices=["uniform", "centred"], default="uniform", help="AB initialization.")
    parser.add_argument("--a_min", type=float, default=1.0, help="Lower bound for |a_i| with centred init.")
    parser.add_argument("--a_max", type=float, default=1.0, help="Upper bound for |a_i| with centred init.")
    parser.add_argument("--n_x_values", type=str, default="3,5,7", help="Comma-separated n_x values. L is set equal to n_x.")
    parser.add_argument("--coarse_factors", type=str, default="2,5,10,50", help="Comma-separated coarse factors (dt_coarse = factor * dt_fine).")
    parser.add_argument("--num_trials", type=int, default=10, help="Repeated runs per grid point.")
    parser.add_argument("--seed_base", type=int, default=1234, help="Base seed for reproducible runs.")
    parser.add_argument("--no_progress", action="store_true", help="Disable progress bars.")
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb_entity", type=str, default="dadeslam", help="W&B entity (team/user).")
    parser.add_argument("--wandb_project", type=str, default="RPNN", help="W&B project name.")
    parser.add_argument("--wandb_group", type=str, default="", help="Optional W&B group override.")
    parser.add_argument("--wandb_name", type=str, default="", help="Optional W&B run name override.")
    args = parser.parse_args()

    if args.a_min > args.a_max:
        parser.error("--a_min must be <= --a_max.")
    if args.num_trials < 1:
        parser.error("--num_trials must be >= 1.")

    nx_values = parse_int_list(args.n_x_values)
    factors = parse_float_list(args.coarse_factors)

    if args.nodes == "lobatto":
        allowed_lobatto = {3, 4, 5}
        bad = [n for n in nx_values if n not in allowed_lobatto]
        if len(bad) > 0:
            parser.error(f"lobatto nodes currently support n_x in {sorted(list(allowed_lobatto))}. Invalid: {bad}")

    cwd = os.getcwd()
    os.chdir(cwd + "/RPNN")
    print("Current working directory:", os.getcwd())

    ab_folder = "centred" if args.ab_init == "centred" else "uniform"
    reports_dir = os.path.join("savedReports", ab_folder)
    os.makedirs(reports_dir, exist_ok=True)

    raw_csv_path = os.path.join(
        reports_dir, f"sweep_nx_coarse_{args.system}_{args.nodes}_{args.ab_init}.csv"
    )
    summary_csv_path = os.path.join(
        reports_dir, f"sweep_nx_coarse_{args.system}_{args.nodes}_{args.ab_init}_summary.csv"
    )

    wandb = None
    wandb_run = None
    if args.use_wandb:
        try:
            import wandb as _wandb
        except ImportError as exc:
            raise SystemExit("W&B logging requested but `wandb` is not installed in this environment.") from exc
        wandb = _wandb
        default_group = f"sweep_{args.system}_{args.nodes}_{args.ab_init}"
        default_name = f"{default_group}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb_run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            group=args.wandb_group if args.wandb_group else default_group,
            name=args.wandb_name if args.wandb_name else default_name,
            config=vars(args),
            tags=["sweep", "parareal", args.system, args.nodes, args.ab_init],
        )

    rows = []

    total_solves = args.num_trials * len(nx_values) * len(factors) * 2
    with tqdm(total=total_solves, desc="Sweep progress", unit="solve", disable=args.no_progress) as pbar:
        for trial in range(args.num_trials):
            seed = args.seed_base + trial
            np.random.seed(seed)
            random.seed(seed)

            for nx in nx_values:
                data, time, _, vec_ref, number_processors = run_experiment(
                    [args.system, args.nodes, args.ab_init, args.a_min, args.a_max],
                    setup_only=True,
                    n_x_override=nx,
                    L_override=nx,
                )

                y0 = data["y0"]
                t_final = time[-1]
                dt_fine = vec_ref.dt_fine

                start_serial = time_lib.time()
                _ = solver(([y0, t_final], vec_ref))
                serial_time = time_lib.time() - start_serial

                time_fine, sol_fine = compute_solution_on_uniform_grid(y0, t_final, dt_fine, vec_ref)

                for factor in factors:
                    time_factor, dts_factor = build_coarse_grid(t_final, dt_fine, factor)
                    data_factor = data.copy()
                    data_factor["time"] = time_factor
                    data_factor["dts"] = dts_factor
                    data_factor["num_t"] = len(time_factor)
                    fine_on_coarse = sample_solution(time_fine, sol_fine, time_factor)

                    try:
                        rpnn_out = parallel_solver(
                            time=time_factor,
                            data=data_factor,
                            dts=dts_factor,
                            vecRef=vec_ref,
                            number_processors=number_processors,
                            verbose=False,
                            track_history=True,
                            fine_reference=fine_on_coarse,
                            coarse_mode="rpnn",
                        )
                        _, _, total_rpnn, _, _, diag_rpnn = rpnn_out
                        err_hist_rpnn = diag_rpnn["error_history"]
                        rows.append(
                            {
                                "trial": trial,
                                "seed": seed,
                                "system": args.system,
                                "nodes": args.nodes,
                                "ab_init": args.ab_init,
                                "a_min": args.a_min,
                                "a_max": args.a_max,
                                "n_x": nx,
                                "L": nx,
                                "method": "rpnn",
                                "coarse_factor": factor,
                                "dt_fine": dt_fine,
                                "dt_coarse": factor * dt_fine,
                                "serial_time": serial_time,
                                "total_time": total_rpnn,
                                "speedup": serial_time / total_rpnn,
                                "iters": len(err_hist_rpnn),
                                "E0": float(err_hist_rpnn[0]),
                                "Efinal": float(err_hist_rpnn[-1]),
                                "status": "ok",
                                "error": "",
                            }
                        )
                    except Exception as exc:
                        print(
                            f"[WARN] trial={trial + 1}, n_x={nx}, factor={factor}, method=rpnn failed: {exc}"
                        )
                        rows.append(
                            {
                                "trial": trial,
                                "seed": seed,
                                "system": args.system,
                                "nodes": args.nodes,
                                "ab_init": args.ab_init,
                                "a_min": args.a_min,
                                "a_max": args.a_max,
                                "n_x": nx,
                                "L": nx,
                                "method": "rpnn",
                                "coarse_factor": factor,
                                "dt_fine": dt_fine,
                                "dt_coarse": factor * dt_fine,
                                "serial_time": serial_time,
                                "total_time": np.nan,
                                "speedup": np.nan,
                                "iters": np.nan,
                                "E0": np.nan,
                                "Efinal": np.nan,
                                "status": "failed",
                                "error": str(exc),
                            }
                        )
                    pbar.set_postfix_str(f"trial {trial + 1}/{args.num_trials} n_x={nx} RPNN x{factor:g}")
                    pbar.update(1)

                    try:
                        classical_out = parallel_solver(
                            time=time_factor,
                            data=data_factor,
                            dts=dts_factor,
                            vecRef=vec_ref,
                            number_processors=number_processors,
                            verbose=False,
                            track_history=True,
                            fine_reference=fine_on_coarse,
                            coarse_mode="classical",
                            coarse_dt=factor * dt_fine,
                        )
                        _, _, total_classical, _, _, diag_classical = classical_out
                        err_hist_classical = diag_classical["error_history"]
                        rows.append(
                            {
                                "trial": trial,
                                "seed": seed,
                                "system": args.system,
                                "nodes": args.nodes,
                                "ab_init": args.ab_init,
                                "a_min": args.a_min,
                                "a_max": args.a_max,
                                "n_x": nx,
                                "L": nx,
                                "method": "classical",
                                "coarse_factor": factor,
                                "dt_fine": dt_fine,
                                "dt_coarse": factor * dt_fine,
                                "serial_time": serial_time,
                                "total_time": total_classical,
                                "speedup": serial_time / total_classical,
                                "iters": len(err_hist_classical),
                                "E0": float(err_hist_classical[0]),
                                "Efinal": float(err_hist_classical[-1]),
                                "status": "ok",
                                "error": "",
                            }
                        )
                    except Exception as exc:
                        print(
                            f"[WARN] trial={trial + 1}, n_x={nx}, factor={factor}, method=classical failed: {exc}"
                        )
                        rows.append(
                            {
                                "trial": trial,
                                "seed": seed,
                                "system": args.system,
                                "nodes": args.nodes,
                                "ab_init": args.ab_init,
                                "a_min": args.a_min,
                                "a_max": args.a_max,
                                "n_x": nx,
                                "L": nx,
                                "method": "classical",
                                "coarse_factor": factor,
                                "dt_fine": dt_fine,
                                "dt_coarse": factor * dt_fine,
                                "serial_time": serial_time,
                                "total_time": np.nan,
                                "speedup": np.nan,
                                "iters": np.nan,
                                "E0": np.nan,
                                "Efinal": np.nan,
                                "status": "failed",
                                "error": str(exc),
                            }
                        )
                    pbar.set_postfix_str(f"trial {trial + 1}/{args.num_trials} n_x={nx} classical x{factor:g}")
                    pbar.update(1)

    fieldnames = [
        "trial",
        "seed",
        "system",
        "nodes",
        "ab_init",
        "a_min",
        "a_max",
        "n_x",
        "L",
        "method",
        "coarse_factor",
        "dt_fine",
        "dt_coarse",
        "serial_time",
        "total_time",
        "speedup",
        "iters",
        "E0",
        "Efinal",
        "status",
        "error",
    ]

    with open(raw_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    grouped = {}
    for r in rows:
        key = (r["method"], int(r["n_x"]), float(r["coarse_factor"]))
        grouped.setdefault(key, []).append(r)

    summary_rows = []
    for (method, nx, factor), rr in sorted(grouped.items(), key=lambda k: (k[0][0], k[0][1], k[0][2])):
        rr_ok = [x for x in rr if x.get("status") == "ok"]
        num_total = len(rr)
        num_ok = len(rr_ok)
        num_failed = num_total - num_ok

        serial_time = [x["serial_time"] for x in rr]
        dt_fine = [x["dt_fine"] for x in rr]
        dt_coarse = [x["dt_coarse"] for x in rr]
        m_st, s_st = mean_std(serial_time)
        m_df, _ = mean_std(dt_fine)
        m_dc, _ = mean_std(dt_coarse)

        if num_ok > 0:
            total_time = [x["total_time"] for x in rr_ok]
            speedup = [x["speedup"] for x in rr_ok]
            iters = [x["iters"] for x in rr_ok]
            e0 = [x["E0"] for x in rr_ok]
            ef = [x["Efinal"] for x in rr_ok]
            m_t, s_t = mean_std(total_time)
            m_s, s_s = mean_std(speedup)
            m_it, s_it = mean_std(iters)
            m_e0, s_e0 = mean_std(e0)
            m_ef, s_ef = mean_std(ef)
        else:
            m_t = np.nan
            s_t = np.nan
            m_s = np.nan
            s_s = np.nan
            m_it = np.nan
            s_it = np.nan
            m_e0 = np.nan
            s_e0 = np.nan
            m_ef = np.nan
            s_ef = np.nan

        summary_rows.append(
            {
                "system": args.system,
                "nodes": args.nodes,
                "ab_init": args.ab_init,
                "a_min": args.a_min,
                "a_max": args.a_max,
                "n_x": nx,
                "L": nx,
                "method": method,
                "coarse_factor": factor,
                "num_samples": num_ok,
                "num_samples_total": num_total,
                "num_samples_failed": num_failed,
                "dt_fine_mean": m_df,
                "dt_coarse_mean": m_dc,
                "serial_time_mean": m_st,
                "serial_time_std": s_st,
                "total_time_mean": m_t,
                "total_time_std": s_t,
                "speedup_mean": m_s,
                "speedup_std": s_s,
                "iters_mean": m_it,
                "iters_std": s_it,
                "E0_mean": m_e0,
                "E0_std": s_e0,
                "Efinal_mean": m_ef,
                "Efinal_std": s_ef,
            }
        )

    summary_fields = [
        "system",
        "nodes",
        "ab_init",
        "a_min",
        "a_max",
        "n_x",
        "L",
        "method",
        "coarse_factor",
        "num_samples",
        "num_samples_total",
        "num_samples_failed",
        "dt_fine_mean",
        "dt_coarse_mean",
        "serial_time_mean",
        "serial_time_std",
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
    with open(summary_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    if wandb_run is not None:
        raw_table = wandb.Table(
            columns=fieldnames,
            data=[[r[c] for c in fieldnames] for r in rows],
        )
        summary_table = wandb.Table(
            columns=summary_fields,
            data=[[r[c] for c in summary_fields] for r in summary_rows],
        )
        wandb_run.log(
            {
                "raw_results_table": raw_table,
                "summary_results_table": summary_table,
            }
        )
        artifact_name = f"sweep-{args.system}-{args.nodes}-{args.ab_init}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        artifact = wandb.Artifact(name=artifact_name, type="results")
        artifact.add_file(raw_csv_path)
        artifact.add_file(summary_csv_path)
        wandb_run.log_artifact(artifact)
        wandb_run.finish()
        print(f"Logged results to W&B project {args.wandb_entity}/{args.wandb_project}")

    print(f"Saved sweep raw csv: {raw_csv_path}")
    print(f"Saved sweep summary csv: {summary_csv_path}")
