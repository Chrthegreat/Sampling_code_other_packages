import numpy as np
import pandas as pd
import time
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix

from polytopewalk import FacialReduction
from polytopewalk.sparse import SparseDikinWalk, SparseVaidyaWalk, SparseJohnWalk, SparseCenter
from polytope_diagnostics import *
from polytope_generators import *

# Note, sparse algorithms expect Constrained Polytope Form using equalities.

def get_batch_size(walk_type, dim):
    w = walk_type.lower()
    if "dikin" in w:
        return 2000 + 100 * dim
    if "vaidya" in w:
        return 2000 + 200 * dim
    if "john" in w:
        return 2000 + 200 * dim
    return 5000

def get_radius(walk_type, dim):
    w = walk_type.lower()
    if "dikin" in w:
        return 1.0
    if "vaidya" in w:
        return 1.8
    if "john" in w:
        return 5.0
    return 1.0

def get_thin(walk_type, dim):
    return 10  

def run_until_target_ess(walk, name, start_point, A, b, k_dim, target_ess=1000, 
                          batch_iter=1000, thin=10, time_limit=None):
    """
    Runs sparse random walk until target_ess is reached OR time_limit is exceeded.
    """
    print(f"[{name:<11}]   Batch size: {batch_iter}, thin: {thin}")

    all_samples = None 
    total_time = 0.0
    
    current_init = start_point.copy()
    current_ess = 0.0
    batch_count = 0
    max_batches = 100 
    
    # Generate a base seed for this run
    base_seed = int(time.time())

    while current_ess < target_ess and batch_count < max_batches:
        # Check Time Limit
        if time_limit is not None and total_time > time_limit:
            print(f"\n[{name:<11}]   -> TIMEOUT ({total_time:.1f}s > {time_limit}s). Stopping.")
            return 0, total_time, None

        batch_count += 1
        current_burnin = 500 if batch_count == 1 else 0 
        
        start_t = time.time()
        
        try:
            new_batch = walk.generateCompleteWalk(
                batch_iter, 
                current_init, 
                A, b, k_dim,  
                burnin=current_burnin, 
                thin=thin, 
                seed=base_seed
            )
        except Exception as e:
            print(f"\n[{name:<11}]   -> CRASHED in C++ backend: {e}")
            return 0, total_time, None
        
        batch_time = time.time() - start_t
        total_time += batch_time
        
        if all_samples is None:
            all_samples = new_batch
        else:
            all_samples = np.vstack([all_samples, new_batch])

        current_init = all_samples[-1, :]
        
        try:
            current_ess = ess(all_samples)
        except Exception:
            current_ess = 0
            
        print(f"[{name:<11}]   Batch {batch_count}: Samples={len(all_samples):,}, ESS={current_ess:>5.0f}, Time={total_time:>6.2f}s", end='\r', flush=True)

    if time_limit is not None and total_time > time_limit:
         print(f"\n[{name:<11}]   -> TIMEOUT after last batch ({total_time:.1f}s).")
         return 0, total_time, None

    print(f"\n[{name:<11}]   -> DONE. Final ESS: {current_ess:.0f}")
    
    return current_ess, total_time, all_samples

def run_sparse_benchmark():

    dims = [3] 
    TARGET_ESS = 200
    TIME_LIMIT_SEC = 60 * 30  # 30 Minutes
    
    # Active Flags
    active_methods = {
        "dikin": True,
        "vaidya": True,
        "john": True
    }

    results = []

    for N in dims:
        dim_ambient = N * N
        print(f"\n{'='*60}")
        print(f"*** Running Birkhoff N={N} (Ambient Dim {dim_ambient}) ***", flush=True)
        print(f"*** Target ESS: {TARGET_ESS} | Time Limit: {TIME_LIMIT_SEC}s ***")

        #GENERATE & REDUCE BIRKHOFF
        print("1. Generating & Reducing Polytope...", end=" ", flush=True)
        A_eq, b_eq = generate_birkhoff_equalities(N)
        k_ambient = A_eq.shape[1] # type: ignore
        
        fr = FacialReduction()
        try:
            fr_res = fr.reduce(A_eq, b_eq, k_ambient, True)
            A_red = fr_res.sparse_A
            b_red = fr_res.sparse_b.astype(np.float64)
            k_red = int(A_red.shape[1])
            print(f"Done. (Intrinsic Dim: {k_red})")
        except Exception as e:
            print(f"\n[Error] Reduction failed: {e}")
            continue

        # CENTER
        print("2. Finding Center...", end=" ", flush=True)
        sc = SparseCenter()
        try:
            init = sc.getInitialPoint(A_red, b_red, k_red)
            init = init.astype(np.float64)
            print("Done.")
        except Exception as e:
            print(f"\n[Error] Centering failed: {e}")
            continue

        # RUN WALKS
        # DIKIN 
        e_dikin, t_dikin, psrf_dikin = 0, 0, 0
        if active_methods["dikin"]:
            r = get_radius("dikin", k_red)
            bs = get_batch_size("dikin", k_red)
            thin = get_thin("dikin", k_red)
            
            dikin_walk = SparseDikinWalk(r=r)
            
            e_dikin, t_dikin, samples = run_until_target_ess(
                dikin_walk, "Dikin", init, A_red, b_red, k_red,
                target_ess=TARGET_ESS, batch_iter=bs, thin=thin, time_limit=TIME_LIMIT_SEC
            )
            
            if e_dikin > 0 and samples is not None:
                psrf_dikin = univariate_psrf(samples)
            elif e_dikin == 0:
                 print("!!! Dikin timed out. Disabling.")
                 active_methods["dikin"] = False

        # VAIDYA 
        e_vaidya, t_vaidya, psrf_vaidya = 0, 0, 0
        if active_methods["vaidya"]:
            r = get_radius("vaidya", k_red)
            bs = get_batch_size("vaidya", k_red)
            thin = get_thin("vaidya", k_red)
            
            vaidya_walk = SparseVaidyaWalk(r=r)
            
            e_vaidya, t_vaidya, samples = run_until_target_ess(
                vaidya_walk, "Vaidya", init, A_red, b_red, k_red,
                target_ess=TARGET_ESS, batch_iter=bs, thin=thin, time_limit=TIME_LIMIT_SEC
            )

            if e_vaidya > 0 and samples is not None:
                psrf_vaidya = univariate_psrf(samples)
            elif e_vaidya == 0:
                 print("!!! Vaidya timed out. Disabling.")
                 active_methods["vaidya"] = False

        # JOHN 
        e_john, t_john, psrf_john = 0, 0, 0
        if active_methods["john"]:
            r = get_radius("john", k_red)
            bs = get_batch_size("john", k_red)
            thin = get_thin("john", k_red)
            
            john_walk = SparseJohnWalk(r=r)
            
            e_john, t_john, samples = run_until_target_ess(
                john_walk, "John", init, A_red, b_red, k_red,
                target_ess=TARGET_ESS, batch_iter=bs, thin=thin, time_limit=TIME_LIMIT_SEC
            )

            if e_john > 0 and samples is not None:
                psrf_john = univariate_psrf(samples)
            elif e_john == 0:
                 print("!!! John timed out. Disabling.")
                 active_methods["john"] = False

        # SAVE RESULTS ROW
        row = {
            "N": N,
            "Intrinsic_Dim": k_red,
            "Dikin_ESS": e_dikin, "Dikin_Time": t_dikin, "Dikin_PSRF": psrf_dikin,
            "Vaidya_ESS": e_vaidya, "Vaidya_Time": t_vaidya, "Vaidya_PSRF": psrf_vaidya,
            "John_ESS": e_john, "John_Time": t_john, "John_PSRF": psrf_john
        }
        results.append(row)
        
        # Intermediate Save
        pd.DataFrame(results).to_csv("sparse_benchmark_partial.csv", index=False)

    # Final Save
    df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print(df.to_string())
    df.to_csv("sparse_benchmark_final.csv", index=False)

if __name__ == "__main__":
    run_sparse_benchmark()