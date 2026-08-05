"""
A multiprocessing scheduler for the EEG preprocessing pipeline.
Distributes independent subject trials across multiple CPU cores.

Has dependency on main.py script.

Run one subject's multiple trials at a time using multi-processing.

Written based on the standard multiprocessing syntax from Python's documentation.

Author: ddh

Date: June 22, 2026
"""
import os
# Configure the environment variables.
def _set_thread_limits() -> None:
    """
    Limit BLAS/OMP (Basic Linear Algebra Subprograms/Open Multi-Processing)
    threads to avoid oversubscription in child processes during multiprocessing.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1") # "OMP" = OpenMP/Open MultiProcessing.
    os.environ.setdefault("MKL_NUM_THREADS", "1") # "MKL" = Math Kernel Library.
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1") # "OpenBLAS" = open source version of BLAS.
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1") # "NUMEXPR" = Numerical Expressions. For Numpy.

# Prevent oversubscription and deadlock caused by scientific packages.
_set_thread_limits() # Called before importing the scientific libraries.

import argparse
import time
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed, Future
from typing import List, Tuple, Dict

from main import process_single_trial


def parse_args() -> argparse.Namespace:
    """
    Parse arguments for the multiprocessing scheduler.
    """
    parser = argparse.ArgumentParser(description="Multiprocessing Scheduler for EEG Pipeline")

    parser.add_argument(
        "--subject_id", 
        type=int, 
        required=True, 
        help="The specific Subject ID being processed (e.g., 12)."
    )
    parser.add_argument(
        "--trial_dirs", 
        type=str, 
        nargs="+", 
        required=True, 
        help="List of directory paths for this subject's trials. Separated by space."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The path to the output directory"
    )
    parser.add_argument(
        "--n_jobs", 
        type=int, 
        default=4, 
        help="Number of parallel worker processes to run. Default to 4."
    )
    
    return parser.parse_args()


def main() -> None:
    """
    Main logic to initialize the process pool and map tasks.
    """
    # Initilization.
    args = parse_args()

    subject_id: int = args.subject_id
    trial_dirs: List[str] = args.trial_dirs 
    ouptut_dir: str = args.output_dir
    max_workers: int = args.n_jobs

    logging.basicConfig(level=logging.WARNING) # Prevent the printing conflict.
    
    print(f"Starting Multiprocessing Pipeline for Subject {subject_id}...")
    print(f"Total trials to process: {len(trial_dirs)}")
    print(f"CPU Workers assigned: {max_workers}\n")

    start_time: float = time.time()

    # Extract session info from directory names.
    sessions: List[int] = []
    for trial in trial_dirs:
        trial_path = Path(trial)

        for part in trial_path.parts:
            if part.startswith("ses-"):
                ses_str: str = part.split("-")[1]
        
        sessions.append(int(ses_str))
    
    # Record the successful and failed trials.
    successful_tasks: List[str] = []
    failed_tasks: List[Tuple[str, Exception]] = []

    # Commence the process pooling.
    with ProcessPoolExecutor(max_workers=max_workers) as executor: # Use context manager to instantialize.
                                                                   # Will ".shutdown()" automatically if crashed.
        # Mapping the future objects with trial folders.
        future_to_trial: Dict[Future, str] = {}

        for i, trial_dir in enumerate(trial_dirs):
            # Submit the task into the tasks pooling.
            future = executor.submit( # (task_function, arg1of_it, ...).
                process_single_trial, 
                trial_dir, 
                ouptut_dir,
                subject_id, 
                sessions[i]
                ) 
            future_to_trial[future] = trial_dir # Since future object only contains the "status" info, no "task_name".
                                                # Key can be anything, as lone as it is unchangeable/immutable/hashable...

        # Collect the finished tasks dynamically.
        for future in as_completed(future_to_trial):
            trial_path = future_to_trial[future]
                
            # Access the return value of currently process.
            try:
                future.result() # Return None if success (for this pipeline), concrete error message if crushed.
                successful_tasks.append(trial_path) 
                print(f"Trial at {trial_path} has been finished successfully")

            except Exception as exc:
                failed_tasks.append((trial_path, exc))
                print(f"Trial at {trial_path} generated an exception: {exc}")

    # Print the overall report of this multiprocessing run.
    total_time = time.time() - start_time
    print("\n" + "="*40)
    print("Multiprocessing Report")
    print("="*40)
    print(f"Total Time: {total_time:.3f}s")
    print(f"Success: {len(successful_tasks)}")
    print(f"Failed: {len(failed_tasks)}")

    if failed_tasks:
        print("\nFailed Trials Details:")

        for trial_dir, exc in failed_tasks:
            print(f"  - Trial {trial_dir}: {exc}")


if __name__ == "__main__":
    main()