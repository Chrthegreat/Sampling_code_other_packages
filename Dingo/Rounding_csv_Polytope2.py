# The following lines of code are used to run the rounding 
# on different threads so that when the code crashes at some 
# eigen calculation it wont stop running the rest of the methods.

import multiprocessing as mp
import traceback

def _worker(A, b, method_name, return_dict):
    try:
        A_r, b_r, T, T_shift = PolytopeSampler.round_polytope(
            A, b, method=method_name
        )
        return_dict["success"] = True
        return_dict["A_r"] = A_r
        return_dict["b_r"] = b_r
        return_dict["T"] = T
        return_dict["T_shift"] = T_shift
    except Exception:
        return_dict["success"] = False
        return_dict["error"] = traceback.format_exc()


def safe_round(A, b, method_name, timeout=600):
    manager = mp.Manager()
    return_dict = manager.dict()

    p = mp.Process(target=_worker, args=(A, b, method_name, return_dict))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        return None, None, None, None, False, "timeout_or_crash"

    if return_dict.get("success"):
        return (
            return_dict["A_r"],
            return_dict["b_r"],
            return_dict["T"],
            return_dict["T_shift"],
            True,
            None,
        )

    return None, None, None, None, False, return_dict.get("error")


##############################################################################################

import numpy as np
import pandas as pd
from dingo import PolytopeSampler
import sys
import os
import time

# ==============================================================================
# 
# WHAT THIS CODE DOES:
# 1. Scans the 'INPUT FOLDER' folder for simplified polytope files (matrices A and b).
# 2. Loops through a list of problem names (e.g., 'afiro', 'blend').
# 3. Applies Rounding Methods. 
# 4. Measures the time taken for each rounding operation.
# 5. Exports the new, rounded matrices to the 'OUTPUT FOLDER' folder.
# =============================================================================

#INPUT_DIR = "polyround_output"   # Where the simplified files are
INPUT_DIR = "netlib_no_normalize"
OUTPUT_DIR = "rounded_no_normal_output"    # Where the rounded files will go
LOG_FILE = "rounding_benchmarks.txt" # print results

# Only include the base names (no _A, _simple, or extension)
PROBLEMS = [
    "afiro",
    "blend",
    "beaconfd",  
    "scorpion",  
    "agg",      
    "etamacro",
    "degen2"
    #"birkhoff500"
    # "sierra",  # Skipped (Too large for simplification)
    # "degen3",  # Skipped
    # "25fv47"   # Skipped
]

# ROUNDING METHODS
METHODS = [
    ("min_ellipsoid", False),   
    ("john_position", True), 
    ("isotropic_position", False),    
    ("log_barrier", True),       
    ("vaidya_barrier", True),    
    ("volumetric_barrier", True) 
]

def round_and_export_netlib(problem_name, method_name):
    print(f"\n{'='*60}")
    print(f"Processing: {problem_name} | Method: {method_name}")
    
    # Construct File Paths (Looking for SIMPLIFIED files)
    # File pattern: {name}_A_simple.csv inside polyround_output/
    A_file = os.path.join(INPUT_DIR, f"{problem_name}_A_simple.csv")
    b_file = os.path.join(INPUT_DIR, f"{problem_name}_b_simple.csv")

    # Check existence
    if not os.path.exists(A_file) or not os.path.exists(b_file):
        print(f"Skipping: Could not find files in {INPUT_DIR}")
        print(f"Expected: {A_file}")
        return

    # Load the matrices
    print("  Reading simplified CSV files...")
    try:
        # Force sep=',' and contiguous arrays for C-engine speed
        A = pd.read_csv(A_file, header=None, sep=',').values.astype(np.float64, order='C')
        b = pd.read_csv(b_file, header=None, sep=',').values.flatten().astype(np.float64, order='C')
        
        A = np.ascontiguousarray(A)
        b = np.ascontiguousarray(b)
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    print(f"Dimension: {A.shape[1]}")
    print(f"Constraints: {A.shape[0]}")

    # 3. Rounding with Timer
    print(f"Running {method_name}...")
    
    start_time = time.time()
    success = False
    
    # Uncomment this try-except if you remove the multiprocessing. It will crash though, if Eigen fails.
    # try:
    #     A_r, b_r, T, T_shift = PolytopeSampler.round_polytope(
    #         A, b, method=method_name
    #     )
    #     success = True
    # except Exception as e:
    #     print(f"Rounding failed: {e}")
    A_r, b_r, T, T_shift, success, err = safe_round(A, b, method_name)

    if not success:
        print(f"Rounding failed for {problem_name} | {method_name}: {err}")
        return
        
    # Stop Timer
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    if success:
        print(f"Success!")
        print(f"Time taken: {elapsed_time:.4f} seconds")

        # Diagnostics
        cond = None
        try:
            svals = np.linalg.svd(T, compute_uv=False)
            cond = svals.max() / svals.min()
            print(f"Condition number of T: {cond:.3e}")
        except:
            pass

        with open(LOG_FILE, "a") as f:
            # Format: Polytope | Method | Time(s) | Condition Number
            line = f"{problem_name.ljust(15)} | {method_name.ljust(20)} | {elapsed_time:.6f} | {cond if cond is not None else 'NA'}\n"
            f.write(line)

        # We save to the OUTPUT_DIR
        out_name_base = f"{problem_name}_{method_name}"
        output_A = os.path.join(OUTPUT_DIR, f"{out_name_base}_A.csv")
        output_b = os.path.join(OUTPUT_DIR, f"{out_name_base}_b.csv")
        
        print(f"  Saving to {OUTPUT_DIR}/...")
        np.savetxt(output_A, A_r, delimiter=",")
        np.savetxt(output_b, b_r, delimiter=",")

if __name__ == "__main__":
    # Ensure output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    print(f"Starting batch rounding for {len(PROBLEMS)} problems...")
    print(f"Input Directory: {INPUT_DIR}")
    
    # Outer Loop: Iterate through the list of problems
    for problem in PROBLEMS:
        
        # Inner Loop: Iterate through the list of methods
        for method_name, is_enabled in METHODS:
            if is_enabled:
                round_and_export_netlib(problem, method_name)
                
    print(f"\n{'='*60}")
    print("Batch processing complete.")