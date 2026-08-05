"""
The main pipeline for EEG data preprocess.

Exclusively for .bdf (EEG format) files.

For single-processing only. Run "run_multiprocessing.py" for multiprocessing tasks.

Author: ddh

Date: June 21, 2026
"""

import argparse
import time
import logging
from typing import List, Optional, Dict, Tuple, Literal, Union
from pathlib import Path

import mne
import mne_icalabel
import numpy as np

from logger import PreprocessLogger
from all_channels_zero_management import FlatlineManager
from utils import detect_bad_channels, apply_ica_denoise, process_local_bad_channels, save_as_fif


# Global constants.
TARGET_SFREQ: int = 200 # Target sampling frequency, unit Hz.

FILTER_FREQ: Tuple[float, float] = (1.0, 47.0) # Bandpass filter frequencies in Hz.
NOTCH_FREQ: int = 50 # Notch filter frequency in Hz.
NOTCH_WIDTH: int = 4 # Notch filter width in Hz.
CUT_LENGTH: List[int] = [3, 60] # The lengths to be cut after filtering on each side, in seconds.

# ICA_COMPONENTS_RATIO: float = 1 # Ratio of PCA components to retain before ICA. Deprecated.
ICA_PROB_THRESHOLD: float = 0.8 # Confidence threshold (0 to 1) for ICLabel against artifacts.

BAD_CHANNELS_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    "constant_noise": (3.0, 0.4), # (multiplier, ratio) for bad channels detection logic.
    "abrupt_noise": (30.0, 0.01),
}

INTERPOLATION_THRESHOLD: int = 6 # Threshold for quitting interpolate the segment if too many bad channels. TODO

EEG_CHANNELS: List[str] = [ # The EEG channels for 10-20 system. Total 59.
    'Fpz', 'Fp1', 'Fp2',
    'AF3', 'AF4', 'AF7', 'AF8',
    'Fz', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8',
    'FCz', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'FT7', 'FT8',
    'Cz', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'T7', 'T8',
    'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'TP7', 'TP8',
    'Pz', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8',
    'POz', 'PO3', 'PO4', 'PO5', 'PO6', 'PO7', 'PO8',
    'Oz', 'O1', 'O2'
]

ICA_EXCLUDE_CHANNELS: List[str] = ['Fp1', 'Fp2', 'F7', 'F8', 'AF7', 'AF8']


def process_single_trial(
    data_directory: Union[str, Path], 
    output_directory: Union[str, Path],
    subject_id: int, 
    session: int, 
    verbose: bool = False
    ) -> None:
    """
    The core pipeline logic to process a single trial.
    Isolated into a function to support both single processing and multiprocessing.

    Args:
        data_directory (str); Path to the directory containing the .bdf file.
        ouptut_directory (str): Path to the output directory.
        subject_id (int): The ID of the subject.
        session (int): The session number of this trial.
        verbose (bool): If True, print all the tracing info when running.
                        Only recommended for single processing.
        ---
    """
    trial_path = Path(data_directory)
    output_directory = Path(output_directory)

    mne.set_log_level("INFO" if verbose else "ERROR") # Eliminate the mne log printing during filtering and ica.

    if verbose:
        print(f"\n--- Starting processing for subject {subject_id} at {trial_path} ---")

    # Load raw data and only pick the EEG channels.
    if verbose:
        print(f"Initialization finished. Begin loading data from {trial_path}")
    load_data_init_time = time.time()

    # raw_path: Path = trial_path / "data.bdf"
    # if not raw_path.exists():
    #     raise FileNotFoundError(f"Error: The data file cannot be found at {raw_path}")
    bdf_files: List[Path] = list(trial_path.rglob("*.bdf"))
    if not bdf_files:
        raise FileNotFoundError(
            f"Error: No .bdf file can be found in {trial_path} or its subdirectories"
        )
    raw_path: Path = bdf_files[0]

    raw: mne.io.BaseRaw = mne.io.read_raw_bdf(
        raw_path, 
        preload=True, # Prevent lazy loading. Load EEG data from hard drive to RAM.
                        # Mandatory for later "raw.resample()", which is DSP process in essence.
        verbose="ERROR" # Report until the program crashes.
        )

    raw.pick_channels(EEG_CHANNELS) # Pick out from "stand_1020". Prevent later crash.

    if verbose:
        print(f"Data loaded successfully, took {time.time()-load_data_init_time:.3f}s.\nBegin data filtering")

    # Downsampling the raw data to 200 Hz.
    raw.resample(TARGET_SFREQ)

    # Filter the data and then cut the edge (3s and 30s).
    raw.filter(l_freq=FILTER_FREQ[0], h_freq=FILTER_FREQ[1])
    raw.notch_filter(freqs=NOTCH_FREQ, notch_widths=NOTCH_WIDTH)

    raw.crop(tmin=CUT_LENGTH[0], tmax=raw.times[-1] - CUT_LENGTH[1]) # Should pass time in secs here.
    if verbose:
        print(f"Data filtered and cropped.\n")
    #=========================================================================================== 
    # Detect the flatline error.
    if verbose:
        print("Begin flatline error detection...")    
    detection_init_time = time.time()
    error_manager = FlatlineManager(raw, min_duration_sec=60.0)

    valid_raws: List[mne.io.BaseRaw] = error_manager.process() # The result will be printed after detection.

    if verbose:
        print(f"Flat line error detection finished. Took {time.time()-detection_init_time:.3f}s\n")
    #============================================================================================
    
    # Return to the common preprocessing pipeline.
    for idx, raw_seg in enumerate(valid_raws): # Loop through all the sliced eeg data segments, if there is any.
        # Prepare metadata logging for each loop.

        #----------------------------------CAN/SHOULD BE MODIFIED---------------------------------------------------------------------------
        if error_manager.has_error:
            # meta_dict = error_manager.segments_metadata[idx]
            # folder_name = f"seg{idx+1}_l{int(meta_dict['duration_sec'])}_s{int(meta_dict['start_sec'])}_e{int(meta_dict['end_sec'])}_preprocessed"
            folder_name = f"seg{idx+1:02d}"
            current_run_idx = idx + 1

            # segment_out_dir = trial_path / folder_name
            # json_save_dir = segment_out_dir
            segment_out_dir = output_directory / f"sub-{subject_id:02d}/ses-{session:02d}" / folder_name
            json_save_dir = segment_out_dir

        else:
            # folder_name = "preprocessed"
            current_run_idx = None

            # segment_out_dir = trial_path/ folder_name
            # json_save_dir = trial_path
            segment_out_dir = output_directory / f"sub-{subject_id:02d}/ses-{session:02d}"
            json_save_dir = segment_out_dir

        segment_out_dir.mkdir(parents=True, exist_ok=True)
        #-----------------------------------I CANNOT MAKE IT PERFECT NOW...------------------------------------------------------------------
    
        logger = PreprocessLogger(
            subject_id=subject_id,
            session=session,
            # trial_path=trial_path,
            run_idx=current_run_idx,
        )        
        
        if error_manager.has_error:
            logger.log_data["SegmentMetadata"] = error_manager.segments_metadata[idx]

        # Log the common prams.
        logger.log_sfreq(raw_seg.info['sfreq'])
        logger.log_filtering("bandpass", FILTER_FREQ)
        logger.log_filtering("notch", (NOTCH_FREQ, NOTCH_WIDTH)) 
        logger.log_begin_time(raw_seg.info['meas_date']) # Meta data won't change after slicing.     

        # 1st bad channels detection and interpolation.
        raw_seg.set_montage('standard_1020') # Set the montage before interpolation.

        if verbose:
            print(f"Begin the 1st and 2nd bad channels detection and interpolation")
        channel_detection_init_time = time.time()

        bads_1st = detect_bad_channels(raw_seg, thresholds=[
            BAD_CHANNELS_THRESHOLDS["constant_noise"], 
            BAD_CHANNELS_THRESHOLDS["abrupt_noise"]
            ])
        logger.log_bad_channels("1st", bads_1st)
        
        raw_seg.info['bads'] = [name for _, name, _ in bads_1st]
        if bads_1st:
            raw_seg.interpolate_bads(reset_bads=True)

        # 2nd bad channel detection and interpolation.
        raw_seg, bads_2nd = process_local_bad_channels(
            raw=raw_seg, 
            chunk_sec=60.0, 
            thresholds=[BAD_CHANNELS_THRESHOLDS['constant_noise'], BAD_CHANNELS_THRESHOLDS["abrupt_noise"]], 
            exclude_channels=ICA_EXCLUDE_CHANNELS
        )
        logger.log_bad_channels("2nd", bads_2nd)

        # Apply ICA to remove artifacts and log the rejected components.
        if verbose:
            print(f"1st and 2nd bad channels processed. Took {time.time()-channel_detection_init_time:.3f}s.")
            print("\nBegin ICA...")
        ica_init_time = time.time()

        raw_seg, full_list, exc_list, total_count, exc_count = apply_ica_denoise(raw_seg, prob_threshold=ICA_PROB_THRESHOLD)
        logger.log_ica_removal(full_list=full_list, exclude_list=exc_list, total_count=total_count, exclude_count=exc_count)

        # 3rd bad channels detection and interpolation.
        if verbose:
            print(f"ICA completed, took {time.time()-ica_init_time:.3f}s.\nBegin the 3rd bad channel processing")
        channel_detection_init_time = time.time()

        raw_seg, bads_3rd = process_local_bad_channels(
            raw=raw_seg, 
            chunk_sec=60.0, 
            thresholds=[BAD_CHANNELS_THRESHOLDS["abrupt_noise"]], # TODO tobe asked and insured.
            exclude_channels=[] # Empty list, check all channels
        )
        logger.log_bad_channels("3rd", bads_3rd)
        if verbose:
            print(f"The 3rd bad channels detection and interpolation finished, took {time.time()-channel_detection_init_time:.3f}")

        # Do the second re-referencing, using "average".
        raw_seg.set_eeg_reference('average') # Montange has already been set in apply_ica_denoise().
        logger.log_re_reference("average")
        if verbose:
            print("Data re-referenced, using 'average'")

        # Save the outputs to the expected places.
        logger.log_data_length(raw_seg)

        # output_filename = f"sub_{subject_id:02d}{'-seg'+str(current_run_idx) if current_run_idx else ''}_preprocessed"
        seg_info: str = f"_run-{current_run_idx:02d}" if current_run_idx is not None else ""
        output_filename = f"sub-{subject_id:02d}_ses-{session:02d}{seg_info}_task-dailylife_prep_eeg"

        save_as_fif(raw_seg, segment_out_dir, output_filename)
        # save_as_brainvision(raw_seg, segment_out_dir, output_filename)
        logger.save_log(output_dir=json_save_dir)
        
        if verbose:
            print(f"Segment {idx + 1} processing complete, saved to {segment_out_dir}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse the arguments for EEG data preprocess")

    parser.add_argument("--data_dir", type=str, required=True, help="Path to the directory that contains the .bdf file")
    parser.add_argument("--subject_id", type=int, choices=range(1, 16), required=True, help="Subject ID, indexing from 1 to 15")
    parser.add_argument("--verbose", action="store_true", help="If flagged, print the detailed processing logging")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory")

    return parser.parse_args()


def main() -> None:
    """
    Main entry point for single-process execution.
    """
    init_time: float = time.time() 
    args = parse_args()

    # Print the information from logging if verbose.
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s" # Keep the format the same as "print".
    )

    # Extract session info from directory name.
    data_path = Path(args.data_dir)
    for part in data_path.parts:
        if part.startswith("ses-"):
            ses_str: str = part.split("-")[1]
    session = int(ses_str)

    # Trigger the isolated pipeline logic.
    process_single_trial(
        args.data_dir, 
        args.output_dir, 
        args.subject_id, 
        session, 
        verbose=args.verbose
        )
    
    if args.verbose:
        print(f"\nAll tasks done. Total execution time: {time.time() - init_time:.2f}s")


if __name__ == "__main__":
    main()