import multiprocessing
import numpy as np
import time as time_lib

from scripts.ode_solvers import solver
from scripts.utils import flowMap

def classical_coarse_step(y0, dt, vecRef, coarse_dt):
    if coarse_dt is None or coarse_dt <= 0:
        raise ValueError("coarse_dt must be a positive float for classical coarse mode.")
    n_steps = max(2, int(np.ceil(dt / coarse_dt)) + 1)
    t_eval = np.linspace(0.0, dt, n_steps)
    return solver(([y0, dt, t_eval], vecRef))

def fine_integrator(ics,dts,vecRef,number_processors,pool=None):
    if number_processors==1:
        output = np.zeros_like(ics)
        args = [(args,vecRef) for args in zip(ics,dts)]
        for i in range(len(dts)):
            output[i] = solver(args[i])
        return output
    else:
        tasks = [(args,vecRef) for args in zip(ics,dts)]
        if pool is not None:
            output = pool.map(solver,tasks)
        else:
            with multiprocessing.Pool(processes=number_processors) as local_pool:
                output = local_pool.map(solver,tasks)
        return np.asarray(output)


def getCoarse(time,data,vecRef,previous=None,networks=None,coarse_mode="rpnn",coarse_dt=None,dts=None):
    if previous is None:
        previous = []
    if networks is None:
        networks = []
    
    LB = data["LB"]
    UB = data["UB"]
    L = data["L"]
    y0 = data["y0"]
    weight = data["weight"]
    bias = data["bias"]
    n_x = data["n_x"]
    n_t = data["n_t"]
    system = data["system"]
    lsq_skip_tol = data.get("lsq_skip_tol", 1e-10)
    if dts is None:
        dts = np.diff(time)
    
    coarse_approx = np.zeros((len(time),len(y0)))
    coarse_approx[0] = y0
    
    initial_proj = np.kron(y0,np.ones(L))
    
    if len(previous)==0:
        if coarse_mode == "classical":
            for i in range(len(time)-1):
                coarse_approx[i+1] = classical_coarse_step(coarse_approx[i], dts[i], vecRef, coarse_dt)
        else:
            # First coarse pass:
            # keep using the fixed initial projection until we hit the first
            # slab with sufficiently small training residual; only then switch
            # to slab-to-slab warm starts.
            warm_start_proj = initial_proj.copy()
            use_warm_start_gate = data.get("use_warm_start_gate", False)
            warm_start_enabled = not use_warm_start_gate
            warm_start_activation_tol = data.get("warm_start_activation_tol", 1e-2)
            for i in range(len(time)-1):
                init_proj_i = warm_start_proj if warm_start_enabled else initial_proj
                flow = flowMap(y0=coarse_approx[i],initial_proj=init_proj_i,weight=weight,bias=bias,dt=dts[i],n_t=n_t,n_x=n_x,L=L,LB=LB,UB=UB,system=system,act_name="Tanh",vec=vecRef,lsq_skip_tol=lsq_skip_tol)
                flow.approximate_flow_map()
                coarse_approx[i+1] = flow.sol[-1]
                networks.append(flow)
                if flow.computed_projection_matrices.shape[0] > 0:
                    slab_res = float(flow.training_err_vec[-1])
                    if warm_start_enabled:
                        warm_start_proj = flow.computed_projection_matrices[-1].copy()
                    elif slab_res <= warm_start_activation_tol:
                        warm_start_enabled = True
                        warm_start_proj = flow.computed_projection_matrices[-1].copy()
    else:
        if coarse_mode == "classical":
            for i in range(len(time)-1):
                coarse_approx[i+1] = classical_coarse_step(previous[i], dts[i], vecRef, coarse_dt)
        else:
            for i in range(len(time)-1):
                flow = flowMap(y0=previous[i],initial_proj=initial_proj,weight=weight,bias=bias,dt=dts[i],n_t=n_t,n_x=n_x,L=L,LB=LB,UB=UB,system=system,act_name="Tanh",vec=vecRef,lsq_skip_tol=lsq_skip_tol)
                if len(networks)>0:
                    flow.computed_projection_matrices = networks[i].computed_projection_matrices.copy()
                flow.approximate_flow_map()
                coarse_approx[i+1] = flow.sol[-1]
                networks[i] = flow
            
    return coarse_approx, networks
    
def getNextCoarse(time,y,i,data,vecRef,networks=None, freeze=False,coarse_mode="rpnn",coarse_dt=None,dts=None):
    if networks is None:
        networks = []
    if dts is None:
        dts = np.diff(time)
    
    LB = data["LB"]
    UB = data["UB"]
    L = data["L"]
    weight = data["weight"]
    bias = data["bias"]
    n_x = data["n_x"]
    n_t = data["n_t"]
    system = data["system"]
    y0 = data["y0"]
    lsq_skip_tol = data.get("lsq_skip_tol", 1e-10)
    
    initial_proj = np.kron(y0,np.ones(L)).reshape(1,-1)
    
    if coarse_mode == "classical":
        return classical_coarse_step(y, dts[i], vecRef, coarse_dt), networks

    # Fast path for selective retraining: reuse already-trained slab model and
    # only propagate with updated initial condition.
    if freeze and len(networks) > i:
        flow = networks[i]
        flow.y0_supp = y
        flow.sol[0] = y
        flow.training_err_vec[0] = 0.0
        for j in range(flow.n_t - 1):
            xi_j = flow.computed_projection_matrices[j]
            y_seg = (flow.h - flow.h0) @ flow.to_mat(xi_j, flow.L, flow.d) + flow.y0_supp.reshape(1, -1)
            flow.y0_supp = y_seg[-1]
            flow.sol[j + 1] = flow.y0_supp
        next_val = flow.analyticalApproximateSolution(dts[i])
        if np.all(np.isfinite(next_val)):
            return next_val, networks
    
    flow = flowMap(y0=y,initial_proj=initial_proj,weight=weight,bias=bias,dt=dts[i],n_t=n_t,n_x=n_x,L=L,LB=LB,UB=UB,system=system,act_name="Tanh",vec=vecRef,lsq_skip_tol=lsq_skip_tol)
    if len(networks)>0:
        flow.computed_projection_matrices = networks[i].computed_projection_matrices.copy()
        #flow.y0 = y
    #flow.approximate_flow_map()
    #networks[i] = flow
    #return flow.analyticalApproximateSolution(dts[i]), networks

    if not freeze:
        # Normal behaviour: retrain on the new initial condition
        flow.y0 = y
        flow.approximate_flow_map()
    else:
        # FROZEN coarse integrator:
        # Do NOT change projection matrices; just propagate with new y
        flow.y0_supp = y
        flow.sol[0] = y
        flow.training_err_vec[0] = 0.0

        # Rebuild the trajectory using the fixed projection matrices
        for j in range(flow.n_t - 1):
            xi_j = flow.computed_projection_matrices[j]
            y_seg = (flow.h - flow.h0) @ flow.to_mat(xi_j, flow.L, flow.d) + flow.y0_supp.reshape(1, -1)
            flow.y0_supp = y_seg[-1]
            flow.sol[j + 1] = flow.y0_supp

    networks[i] = flow
    return flow.analyticalApproximateSolution(dts[i]), networks

def parallel_solver(
    time,
    data,
    dts,
    vecRef,
    number_processors,
    verbose=False,
    rel_thresh=1e-1,
    track_history=False,
    fine_reference=None,
    coarse_mode="rpnn",
    coarse_dt=None,
    ):

    prev_Xi = None       # will hold projection matrices from previous iterate
    freeze = False       # global flag
    iterates_history = []
    error_history = []
    
    y0 = data["y0"]
    if coarse_mode not in ("rpnn", "classical"):
        raise ValueError("coarse_mode must be either 'rpnn' or 'classical'.")
    if coarse_mode == "classical" and coarse_dt is None:
        coarse_dt = 10.0 * vecRef.dt_fine
    
    if fine_reference is not None:
        fine_reference = np.asarray(fine_reference)
        expected_shape = (len(time), len(y0))
        if fine_reference.shape != expected_shape:
            raise ValueError(
                f"fine_reference must have shape {expected_shape}, got {fine_reference.shape}"
            )
    
    max_it = 20 #maximum number of parareal iterates
    tol = 1e-4
    use_active_prefix = data.get("use_active_prefix", True)
    computational_times_per_iterate = []
    it = 0
    is_converged = False
    networks = []
    
    number_processors = min(number_processors,multiprocessing.cpu_count())

    initial_full = time_lib.time()
    pool = multiprocessing.Pool(processes=number_processors) if number_processors > 1 else None
    
    try:
        while it<max_it and is_converged==False:

            norm_difference = []
        
            if it==0:
                initial_time = time_lib.time()
                coarse_approx, networks = getCoarse(previous=[],time=time,data=data,vecRef=vecRef,networks=networks,coarse_mode=coarse_mode,coarse_dt=coarse_dt,dts=dts)
                coarse_values_parareal = coarse_approx.copy()            
                cost = time_lib.time()-initial_time
                computational_times_per_iterate.append(cost)
                if verbose:
                    print("Average cost per one coarse step : ",cost/len(dts))            
            else:
                initial_time = time_lib.time()
            
                # Parareal exactness property: after k correction iterates, first k
                # slab endpoints are already exact. Skip this converged prefix.
                active_start = 0
                if use_active_prefix:
                    active_start = min(max(it - 1, 0), len(dts))

                if active_start < len(dts):
                    start_fine = time_lib.time()
                    fine_int = fine_integrator(
                        coarse_values_parareal[active_start:-1],
                        dts[active_start:],
                        vecRef,
                        number_processors,
                        pool=pool,
                    )
                    if verbose:
                        print("Time required for the fine solver : ",time_lib.time()-start_fine)
                    use_selective_retrain = data.get("use_selective_retrain", True)
                    selective_retrain_tol = data.get("selective_retrain_tol", 1e-4)
                    selective_state_max = data.get("selective_retrain_state_max_norm", 1e6)
                    for loc_i, i in enumerate(range(active_start, len(time)-1)):
                        previous = coarse_values_parareal[i+1].copy()
                        slab_freeze = freeze
                        if (
                            coarse_mode == "rpnn"
                            and use_selective_retrain
                            and len(networks) > i
                        ):
                            corr = np.linalg.norm(coarse_values_parareal[i+1] - coarse_approx[i+1], 2)
                            corr_ref = max(np.linalg.norm(coarse_values_parareal[i+1], 2), 1e-12)
                            rel_corr = corr / corr_ref
                            y_norm = np.linalg.norm(coarse_values_parareal[i], 2)
                            finite_ok = np.all(np.isfinite(coarse_values_parareal[i])) and np.isfinite(rel_corr)
                            slab_freeze = slab_freeze or (
                                finite_ok
                                and y_norm <= selective_state_max
                                and rel_corr <= selective_retrain_tol
                            )
                        next,networks = getNextCoarse(y=coarse_values_parareal[i],i=i,time=time,data=data,vecRef=vecRef,networks=networks, freeze=slab_freeze,coarse_mode=coarse_mode,coarse_dt=coarse_dt,dts=dts)
                        coarse_values_parareal[i+1] = fine_int[loc_i] + next - coarse_approx[i+1]
                        coarse_approx[i+1] = next.copy()
                        norm_difference.append(np.linalg.norm(coarse_values_parareal[i+1]-previous,2))
            
                if verbose:
                    print("Difference norms : ",norm_difference)   
                computational_times_per_iterate.append(time_lib.time()-initial_time)
                if verbose:
                    if len(norm_difference) > 0:
                        print("Maximum norm of difference :",np.round(np.max(norm_difference),10))
                    else:
                        print("No active slabs to update (prefix skipping).")
                is_converged = (len(norm_difference) == 0) or (np.max(norm_difference)<tol)
        
            if track_history:
                iterates_history.append(coarse_values_parareal.copy())
            if fine_reference is not None:
                diff_norm = np.linalg.norm(coarse_values_parareal - fine_reference, axis=1)
                ref_norm = np.linalg.norm(fine_reference, axis=1)
                rel_err = np.max(diff_norm) / max(np.max(ref_norm), 1e-14)
                error_history.append(rel_err)
        
        # print(f"Iterate {it}")
            if len(networks) == 0:
                pass
            elif prev_Xi is None:
                # first time we have networks from two consecutive it’s: store and go on
                prev_Xi = [net.computed_projection_matrices.copy() for net in networks]
            else:
                rel_changes = []
                for j, net in enumerate(networks):
                    Xi = net.computed_projection_matrices
                    Xi_prev = prev_Xi[j]

                    abs_change = np.linalg.norm(Xi - Xi_prev)
                    rel_change = abs_change / (np.linalg.norm(Xi_prev) + 1e-12)
                    rel_changes.append(rel_change)

                    # update stored matrices for next iteration
                    prev_Xi[j] = Xi.copy()

                max_rel_change = max(rel_changes)
                # print("max rel change over slabs:", max_rel_change)

                # set global freeze flag for *next* iteration
                '''if max_rel_change < rel_thresh:
                    freeze = True
                    print("-> freezing coarse integrator from next iterate on")'''
        
            it+=1
            if verbose:
                print(f"Iterate {it} completed")
                print(f"Time for iterate {it} is {computational_times_per_iterate[-1]}")
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    
    total_time = time_lib.time()-initial_full

    if track_history or fine_reference is not None:
        diagnostics = {
            "iterates_history": iterates_history,
            "error_history": error_history,
        }
        return coarse_approx,networks,total_time,number_processors,cost/len(dts),diagnostics
    
    return coarse_approx,networks,total_time,number_processors,cost/len(dts)
