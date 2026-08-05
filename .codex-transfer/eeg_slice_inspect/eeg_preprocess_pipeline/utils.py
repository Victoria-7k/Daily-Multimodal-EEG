"""
The helper functions utilized by the preprocess pipeline.

Author: ddh

Date: June 21, 2026
"""

import os
import logging
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import warnings

import mne 
import mne_icalabel
import numpy as np


def detect_bad_channels(
    raw: mne.io.BaseRaw, 
    thresholds: List[Tuple[float, float]],
    channel_names: Optional[List[str]] = None,
    ) -> List[Tuple[int, str, str]]:
    """
    The function that detects bad channels based on amplitude median thresholds.

    Args:
        raw (mne.io.BaseRaw): The object that corresponds to the data to be detected.
        thresholds (List[Tuple[float, float]]): A list containing (multiplier, ratio).
        channel_names (Optional[List[str]]): A list containing all the channel names to check.
                                             If None, used all EEG channels from the raw object.

    Returns:
        List[Tuple[int, str, str]]: A list containing (ch_idx, ch_name, brief_reason).
                                    "ch_idx" and "ch_name" might be duplicated here.
    ---
    """
    data: np.ndarray = raw.get_data(picks="eeg")
    total_samples: int = data.shape[1] 

    if channel_names is None:
        #     channel_names = [ # The EEG channels for 10-20 system. Total 59.
        #     'Fpz', 'Fp1', 'Fp2',
        #     'AF3', 'AF4', 'AF7', 'AF8',
        #     'Fz', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8',
        #     'FCz', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'FT7', 'FT8',
        #     'Cz', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'T7', 'T8',
        #     'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'TP7', 'TP8',
        #     'Pz', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8',
        #     'POz', 'PO3', 'PO4', 'PO5', 'PO6', 'PO7', 'PO8',
        #     'Oz', 'O1', 'O2'
        # ]
        # print(f"No channel_names input was detected. Using default xxx system to index")

        eeg_indices: np.ndarray = mne.pick_types(raw.info, eeg=True, meg=False) # Pick the indices for EEG channels only.
        channel_names = [raw.ch_names[idx] for idx in eeg_indices]
        # print("No channel_names input detected. Extract EEG channels from raw object") # TODO? WHat happended

    bad_channels = []

    for multiplier, ratio in thresholds:
        for ch_idx, ch_data in enumerate(data):
            median = np.median(np.abs(ch_data))# Make all samples positive. Only care about magnitude of artifacts
                                               # and they might be negative direction. 
                                               # Preprocessed samples/signal should swing equally across 0 axis.

            # Count the number of samples exceeding (median * ratio).
            high_values_count: int = np.sum(np.abs(ch_data) > (median * multiplier))
            high_values_ratio: float = high_values_count / total_samples

            if high_values_ratio > ratio:
                bad_channels.append(
                    (ch_idx, channel_names[ch_idx], f"multiplier_{multiplier}-ratio_{ratio}")
                    )

    return bad_channels


def apply_ica_denoise(
    raw: mne.io.BaseRaw,
    # component_ratio: float, # The default value? Deprecated.
    prob_threshold: float = 0.8
) -> Tuple[mne.io.BaseRaw, List[Tuple[int, str, float]], List[Tuple[int, str, float]], int, int]:
    """
    Apply (PCA and) Independent Component Analysis to the raw object that
    contains EEG data to remove artifacts. 
    Then collect and return all components' details, total count and the removed 
    artifact component details(including index, name, confidence) and total count.

    Args:
        raw (mne.io.BaseRaw): The raw object contains data to be processed..
        prob_threshold (float): Between 0 and 1.
                            The confidence threshold for non-brain's artifacts' rejection.
                        
    Returns:
        Tuple[mne.io.BaseRaw, List[Tuple[int, str, float]], List[Tuple[int, str, float]], int, int]: 
                - The cleaned data object.
                - The list containing (index, name, confidence) for all the components.
                - The list containing (index, name, confidence) for each rejected component.
                - The total count of all components.
                - The total count of the rejected components.                                       
    ---
    """
    # Standardize spatial location and reference.
    raw.set_montage('standard_1020') # Set 3-dimensional coordinates to the channel (names),
                                    # using standard 10-20 system.

    raw.set_eeg_reference('average') # Re-referencing, eliminate the "common-mode noises".

    # # Calculate the number of PCA components to retain. # This part of logic might be verified...
    # num_components: int = int(np.floor(len(raw.ch_names) * components_ratio)) # "raw.ch_names" is shortcut to "raw.info.ch_names".

    # Initialize the ICA object.
    ica = mne.preprocessing.ICA(
        # n_components=num_components, # So it can implement both PCA and ICA.
        n_components=None, # Mandatory here, can't be set to 59 manually.
                           # After re-referencing, the rank will lower to 58. So there will be one 0 eigenvalue. 
                           # Then there wil be problem to do PCA whitening (devide by eigenvalues).
        random_state=97, # For reproducibility. "97" is in mne's documentation.
        max_iter='auto',
        method='infomax', # Use the extended infomax algorithm.
        fit_params=dict(extended=True) # Specify the infomax details, use "extended infomax". Best practice.
    )

    # Fit the ICA model. # Question. If I am really going to write utilizable function or modules, if it is true that I should import all the dependencied inside this fucntion?
    with warnings.catch_warnings(): 
        warnings.filterwarnings( # Use warnings module to capture the obsessive warnings sent by mne (due to ICA).
            "ignore",
            category=RuntimeWarning,
            message=r".*has not been high-pass filtered.*", # ".*" is RegEx of Python. "."=any_symples, "*"=repeat_any_times.
                                                            # The dependent re package is already inside warning package.
            module=r"mne.*"
        )
        ica.fit(raw) # Time consuming! 
                     # "raw" will not be modified/written here.

    # Run ICLabel to classify the components.
    with warnings.catch_warnings(): 
        warnings.filterwarnings(
            "ignore",
            message=r"The provided raw instance is not filtered between 1 and 100 Hz.*",
            category=RuntimeWarning
        )
        component_dict = mne_icalabel.label_components(raw, ica, method='iclabel') # Pure Python dict.
                                                                                   # "brain" (or "eye blink", "other", etc.) may appear multiple times.
                                                                                   # Is this also time consuming? If so, how? TODO
                                                                                
    iclabel_labels: List[str] = component_dict['labels']
    iclabel_probs: List[float] = component_dict['y_pred_proba']

    # Identify and record the bad components.
    exclude_idxs: List[int] = []
    for idx, (label, prob) in enumerate(zip(iclabel_labels, iclabel_probs)):
        if label not in ["brain", "other"] and prob > prob_threshold:
            exclude_idxs.append(idx)

    # Copy raw data and reconstruct a cleaned data.
    raw_clean: mne.io.BaseRaw = ica.apply(raw.copy(), exclude=exclude_idxs)

    # Print results. No, "logging" results, compatible for multiprocessing.
    # print(f"{len(exclude_idxs)} artifact components were detected\n")
    logging.info(f"{len(exclude_idxs)} artifact components were detected\n")

    for idx in exclude_idxs:
        # print(f"Excluded: idx {idx:02d} | name {iclabel_labels[idx]:<10} | confidence {iclabel_probs[idx]:.4f}")
        logging.info(f"Excluded: idx {idx:02d} | name {iclabel_labels[idx]:<10} | confidence {iclabel_probs[idx]:.4f}")

    exclude_list = [
        (idx, iclabel_labels[idx], iclabel_probs[idx]) for idx in exclude_idxs
        ]
    full_list = [
        (idx, iclabel_labels[idx], iclabel_probs[idx]) for idx in range(len(iclabel_labels))
        ]
    return raw_clean, full_list, exclude_list, len(iclabel_labels), len(exclude_idxs)


def process_local_bad_channels(
    raw: mne.io.BaseRaw,
    chunk_sec: float,
    thresholds: List[Tuple[float, float]],
    exclude_channels: Optional[List[str]] = None
) -> Tuple[mne.io.BaseRaw, Dict[str, List]]:
    """
    Split the raw data into fixed-length chunks, detect and interpolate bad channels locally,
    and then seamlessly assemble them back.
    Will call "detect_bad_channels()".

    Args:
        raw (mne.io.BaseRaw): The raw object to process.
        chunk_sec (float): Duration of each chunk in seconds (e.g., 60.0).
        thresholds (List[Tuple[float, float]]): The detection thresholds.
        exclude_channels (List[str]): Channels to explicitly ignore during detection.

    Returns:
        Tuple[mne.io.BaseRaw, Dict[str, List]]: 
            - The reassembled, cleaned Raw object.
            - A dictionary containing log info per chunk (e.g., {"segment_1": [...]}).
    """
    exclude_channels = exclude_channels or []
    duration = raw.times[-1] # Get the time stamp of the final sample point, in seconds.
    
    processed_chunks = []
    log_dict: Dict[str, List] = {}

    # Define the boundaries for 1-minute chunks safely.
    starts: np.ndarray = np.arange(0, duration, chunk_sec) # "arange"="array range". 
                                                           # (start, stop[not included], step).
    
    for i, st in enumerate(starts):
        # Prevent the last tiny segment from crashing if less than 1 second.
        end = min(st + chunk_sec, duration)
        if (end - st) < 1.0:
            break # Exit the loop.
            
        # Safely crop a piece of continuous raw data.
        chunk = raw.copy().crop(tmin=st, tmax=end, include_tmax=False)
        
        # Use existing bad channel detector.
        detected_bads = detect_bad_channels(chunk, thresholds)
        
        # Filter out the explicitly excluded channels (like frontal lobes)
        valid_bads = [b for b in detected_bads if b[1] not in exclude_channels]
        
        # Apply interpolation if bad channels exist in this 1-minute chunk
        if valid_bads:
            chunk.info['bads'] = [b[1] for b in valid_bads]
            chunk.interpolate_bads(reset_bads=True)
            
        processed_chunks.append(chunk)
        log_dict[f"minute_{i+1:03d}"] = valid_bads

    # Reassemble all chunks back into one continuous raw object
    raw_assembled = mne.concatenate_raws(processed_chunks)
    
    return raw_assembled, log_dict


def save_as_fif(raw: mne.io.BaseRaw, out_dir: Union[str, Path], filename: str) -> None:
    """Save raw data to MNE's standard .fif format."""
    os.makedirs(out_dir, exist_ok=True)
    out_path = Path(out_dir) / f"{filename}.fif"
    raw.save(out_path, overwrite=True)
    # print(f"Data saved to {out_path}")
    logging.info(f"Data saved to {out_path}")


def save_as_brainvision(raw: mne.io.BaseRaw, out_dir: Union[str, Path], filename: str) -> None:
    """Save raw data to BIDS-compliant BrainVision format (.vhdr, .vmrk, .eeg)."""
    os.makedirs(out_dir, exist_ok=True)
    out_path = Path(out_dir) / f"{filename}.vhdr"

    # MNE export functionality requires the generic string representation of the file.
    mne.export.export_raw(str(out_path), raw, fmt='brainvision', overwrite=True)
    # print(f"Data saved to BrainVision format at {out_dir}")
    logging.info(f"Data saved to BrainVision format at {out_dir}")