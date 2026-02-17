import time
import numpy as np
import pandas as pd
import arviz as az
import math

from polytope_generators import generate_rotated_hypercube_direct, generate_rotated_simplex_direct, generate_birkhoff_direct
from polytope_diagnostics import *
from polywalk_3D_plot import plot_3d_samples, get_and_print_matrices, inspect_raw_generator

from polytopewalk.dense import BallWalk, HitAndRun
from polytopewalk.dense import DikinWalk, VaidyaWalk, JohnWalk, DikinLSWalk
from polytopewalk.dense import DenseCenter
from polytopewalk import FacialReduction

# Main function for sampling
def run_until_target_ess(walk, name, start_point, A, b, target_ess=3000, 
                         batch_iter=10000, r=0.5, thin=10, time_limit=None):
    """
    Runs random walk until target_ess is reached OR time_limit is exceeded.
    Returns (0, total_time, None) if timed out.
    """
    
    print(f"[{name:<11}]   Batch size: {batch_iter} , r: {r} , thin: {thin}")

    all_samples = None 
    total_time = 0.0
    
    current_init = start_point.copy()
    current_ess = 0.0
    batch_count = 0
    max_batches = 2000  
    
    while current_ess < target_ess and batch_count < max_batches:
        # Check Time Limit BEFORE running the next expensive batch
        if time_limit is not None and total_time > time_limit:
            print(f"\n[{name:<11}]   -> TIMEOUT ({total_time:.1f}s > {time_limit}s). Stopping method.")
            return 0, total_time, None

        batch_count += 1
        current_burnin = 100 if batch_count == 1 else 0 
        
        start_t = time.time()
        
        # *****Run C++ backend******
        new_batch = walk.generateCompleteWalk(
            batch_iter, current_init, A, b, 
            burnin=current_burnin, thin=thin, seed=8888 + batch_count
        )
        
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
            
        print(f"[{name:<11}]   Batch {batch_count}: Samples={len(all_samples):,}, ESS={current_ess:>5.0f}, Time={total_time:>6.4f}s", end='\r', flush=True)

    # Final check after loop finishes (in case the last batch pushed it over)
    if time_limit is not None and total_time > time_limit:
         print(f"\n[{name:<11}]   -> TIMEOUT after last batch ({total_time:.1f}s). Flagging as failed.")
         return 0, total_time, None

    print(f"\n[{name:<11}]   -> DONE. Final ESS: {current_ess:.0f}")
    
    return current_ess, total_time, all_samples

# Different batch size for each walk. Choose your own if needs be.
def get_batch_size(walk_type, dim):

    w = walk_type.lower()

    if "ball" in w:
        return 10000 + 1000 * dim
    
    if "hit" in w:
        return 5000 + 2000 * dim

    if "dikin" in w:
        return 12000 + 4000 * dim
    
    if "vaidya" in w:
        return 12000 + 4500 * dim
        
    if "john" in w:
        return 25000 + 4500 * dim
    
    if "dikinls" in w:
        return 12000 + 800 * dim
        
    return 10000

# These values are used after trial and error 
def get_radius(walk_type, dim):

    w = walk_type.lower()

    if "ball" in w:
        #return 8.5
        return math.floor(math.sqrt(dim))

    if "hit" in w:
        return 1.0

    if "dikin" in w:
        return 1.5
    
    if "vaidya" in w:
        return 1.8
    
    if "john" in w:
        return 5.0

    if "dikinls" in w:
        return 1.5

    return 1.0

def get_thin(walk_type, dim):

    w = walk_type.lower()

    if "ball" in w:
        return 100

    if "hit" in w:
        return 100

    if any(x in w for x in ["dikin", "vaidya", "john", "dikinls"]):
        return 100

    return 100



# CONFIGURATION
dims = [15]
TARGET_ESS = 500
TIME_LIMIT_SEC = 60 * 60  # 60 Minutes

# Status flags. Choose true of false for the methods you want 
active_methods = {
    "ball": False,
    "hit": False,
    "dikin": True,
    "vaidya": True,
    "john": True,
    "dikinls": False
}

results = []

for dim in dims:
    print(f"\n{'='*40}")
    print(f"***Running for dimension {dim} and ESS {TARGET_ESS}***", flush=True)

    # Generate Polytope. Comment and uncomment the one you like. Only one at a time.
    #init, dense_A, dense_b, name = generate_rotated_hypercube_direct(dim, angle_deg=53)
    #init, dense_A, dense_b, name = generate_rotated_simplex_direct(dim, angle_deg=53)
    init, dense_A, dense_b, name = generate_birkhoff_direct(dim*dim)
    print(f"Finished dim={dim} (Ambient Dim: {dim*dim}, Intrinsic Dim: {(dim-1)**2})")

    dc = DenseCenter()
    init = dc.getInitialPoint(dense_A, dense_b)
    print(f"Generated {name}.")

    r_dikin  = get_radius("dikin", dim)
    r_vaidya = get_radius("vaidya", dim)
    r_john   = get_radius("john", dim)
    
    thin_dikin  = get_thin("dikin", dim)
    thin_vaidya = get_thin("vaidya", dim)
    thin_john   = get_thin("john", dim)
    
    # RUN WALKS
    # --- DIKIN WALK ---
    if active_methods["dikin"]:
        dikin = DikinWalk(r=r_dikin)
        bs_dikin = get_batch_size("dikin", dim)
        
        # Pass the time_limit here
        e_dikin, t_dikin, samples_dikin = run_until_target_ess(
            dikin, "Dikin Walk", init, dense_A, dense_b, 
            TARGET_ESS, bs_dikin, r_dikin, thin_dikin, 
            time_limit=TIME_LIMIT_SEC
        )
        
        # If ESS is 0, it means it timed out -> Kill it for future dims
        if e_dikin == 0:
            print("!!! Dikin Walk timed out. Disabling for future dimensions.")
            active_methods["dikin"] = False
            psrf_dikin = 0 
        else:
            psrf_dikin = univariate_psrf(samples_dikin)
    else:
        # Method is disabled, fill with zeros/skips
        print("Skipping Dikin Walk (previously timed out).")
        e_dikin, t_dikin, psrf_dikin = 0, 0, 0


    # --- VAIDYA WALK ---
    if active_methods["vaidya"]:
        vaidya = VaidyaWalk(r=r_vaidya)
        bs_vaidya = get_batch_size("vaidya", dim)
        
        e_vaidya, t_vaidya, samples_vaidya = run_until_target_ess(
            vaidya, "Vaidya Walk", init, dense_A, dense_b, 
            TARGET_ESS, bs_vaidya, r_vaidya, thin_vaidya, 
            time_limit=TIME_LIMIT_SEC
        )

        if e_vaidya == 0:
            print("!!! Vaidya Walk timed out. Disabling for future dimensions.")
            active_methods["vaidya"] = False
            psrf_vaidya = 0
        else:
            psrf_vaidya = univariate_psrf(samples_vaidya)
    else:
        print("Skipping Vaidya Walk (previously timed out).")
        e_vaidya, t_vaidya, psrf_vaidya = 0, 0, 0


    # --- JOHN WALK ---
    if active_methods["john"]:
        john = JohnWalk(r=r_john)
        bs_john = get_batch_size("john", dim)
        
        e_john, t_john, samples_john = run_until_target_ess(
            john, "John Walk", init, dense_A, dense_b, 
            TARGET_ESS, bs_john, r_john, thin_john, 
            time_limit=TIME_LIMIT_SEC
        )

        if e_john == 0:
            print("!!! John Walk timed out. Disabling for future dimensions.")
            active_methods["john"] = False
            psrf_john = 0
        else:
            psrf_john = univariate_psrf(samples_john)
    else:
        print("Skipping John Walk (previously timed out).")
        e_john, t_john, psrf_john = 0, 0, 0


    # SAVE RESULTS
    results.append({
        "dim": dim,
        "Dikin_ESS": e_dikin,
        "Dikin_PSFR": psrf_dikin,
        "Dikin_Time": t_dikin,
        
        "Vaidya_ESS": e_vaidya,
        "Vaidya_PSFR": psrf_vaidya,
        "Vaidya_Time": t_vaidya,
        
        "John_ESS": e_john,
        "John_PSFR": psrf_john,
        "John_Time": t_john
    })
    
    # Save intermediate results after every dimension so we don't lose data if script crashes
    pd.DataFrame(results).to_csv("benchmark_results_partial.csv", index=False)

# Final Save
df_results = pd.DataFrame(results)
output_file = "benchmark_results.txt"
with open(output_file, "w") as f:
    f.write(df_results.to_string())
print(f"\nResults saved to {output_file}")