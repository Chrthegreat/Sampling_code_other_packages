import numpy as np
import pandas as pd
from dingo import PolytopeSampler
import sys
import os
import time

# ----------------------------------------------------------------------------------
# This code is used to round a csv polytope. The code reads some predetermined folder 
# for csv files and outputs the rounded polytopes in some other folder.
# ----------------------------------------------------------------------------------

INPUT_DIR = "polyround_output"   # Where your simplified files are
OUTPUT_DIR = "rounded_output"    # Where the rounded files will go

# The polytope names found in the input dir. The files must look like afiro_A.csv, afiro_b.csv for example.
PROBLEMS = [
    "afiro",
    "blend",
    "beaconfd",  
    "scorpion",  
    "agg",      
    "etamacro",
    "degen2"
    # "sierra",  # Skipped (One hour with no result)
    # "degen3",  # Skipped
    # "25fv47"   # Skipped
]

# Choose your rounding method by setting true of false
METHODS = [
    ("min_ellipsoid", False),   
    ("john_position", False), 
    ("max_ellipsoid", False),     # Not recognised
    ("log_barrier", False),       # Not recognised
    ("vaidya_barrier", False),    # Not recognised
    ("volumetric_barrier", False) # Not recognised
]

def round_and_export_netlib(problem_name, method_name):
    print(f"\n{'='*60}")
    print(f"Processing: {problem_name} | Method: {method_name}")
    
    # Construct File Paths (Looking for SIMPLIFIED files returned by Polyround)
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

    # Rounding with Timer
    print(f"Running {method_name}...")
    
    start_time = time.time()
    success = False
    
    try:
        # Calls dingo's rounding function
        A_r, b_r, T, T_shift = PolytopeSampler.round_polytope(
            A, b, method=method_name
        )
        success = True
    except Exception as e:
        print(f"Rounding failed: {e}")
        
    # Stop Timer
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    if success:
        print(f"Success!")
        print(f"Time taken: {elapsed_time:.4f} seconds")

        # Diagnostics 
        try:
            svals = np.linalg.svd(T, compute_uv=False)
            cond = svals.max() / svals.min()
            print(f"Condition number of T: {cond:.3e}")
        except:
            pass

        # Export. We save to the OUTPUT_DIR. Each file will have the method in its name
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