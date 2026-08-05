"""
A script contains the total logics for handling one specific data error type,
for which the EEG signals will appear to be all zero across all the channels,
simultaneously and occasionally.

Author: ddh

Date: June 21, 2026
"""
import logging
from typing import Dict, List, Any, Tuple

import mne
import numpy as np

class FlatlineManager:
    """
    A manager to detect, slice, and track EEG data segments separated by 
    simultaneous "all-channel zero" artifacts.
    """
    def __init__(
        self,
        raw: mne.io.BaseRaw,
        min_duration_sec: float = 60.0,
        zero_tolerance: float = 1e-15
    ) -> None:
        """
        Args:
            raw (mne.io.BaseRaw): The raw EEG object to be inspected.
            min_duration_sec (float): Minimum duration in secs for a "valid segment" to be retained.
            zero_tolerance (float): The absolute voltage threshold to be considered as "zero".
        """
        self.raw: mne.io.BaseRaw = raw
        self.sfreq: float = raw.info['sfreq']
        self.min_samples: int = int(min_duration_sec * self.sfreq)
        self.zero_tolerance: float = zero_tolerance

        self.has_error: bool = False
        self.segments_metadata: List[Dict[str, Any]] = []

    def _find_valid_boundaries(self, is_zero_mask: np.ndarray) -> List[Tuple[int, int]]:
        """
        A optimized and vectorized method to find the start and end indices of all
        non-zero, valid contiguous segments.

        Args:
            is_zero_mask (np.ndarray): Boolean array where True means invalid data.

        Returns:
            List[Tuple[int, int]]: A list of tuples containing (start_idx, end_idx).
        ---
        """
        # Pad the mask with True at both ends to capture boundaries.
        padded_mask: np.ndarray = np.concatenate(([True], is_zero_mask, [True])) # Concatenate [True] at both before and after is_zero——mask.

        # Calculate differences.
        diffs: np.ndarray = np.diff(padded_mask.astype(int)) # "np.astype()" will convert True into 1, False into 0.
                                                             # For the difference movement, position[1]'=pos[1]-pos[0].
                                                             # Then shift the array/indices leftward/backward for 1 unit.

        starts: np.ndarray = np.where(diffs == -1)[0] # "-1" means True to False (bad data to good data), "1" means False to True.
                                                      # "np.where()" returns the indices numbers.
        ends: np.ndarray = np.where(diffs == 1)[0] -1 # "entrance of bad data" minus 1 to capture "the end of good data".

        valid_boundaries = [
            (int(st), int(en)) for st, en in zip(starts, ends)
        ]
        return valid_boundaries

    def process(self) -> List[mne.io.BaseRaw]:
        """
        Execute the detection and slicing logic.

        Return:
            List[mne.io.BaseRaw]: A list of valid, continuous MNE raw objects.
                                  Returns [self.raw] if no flatline error is detected.

        """
        self.segments_metadata.clear()
        
        # Pick out EEG channels only.
        eeg_picks: np.ndarray = mne.pick_types(self.raw.info, eeg=True, meg=False)
        data: np.ndarray = self.raw.get_data(picks=eeg_picks)

        # Check the flatline error.
        is_zero_mask: np.ndarray = np.all( # "np.all()" only return True when all conditioins are satisfied.
            np.abs(data) <= self.zero_tolerance, axis=0 # "axis=0" specifies that across all channels (but not lines).
            ) 

        if not np.any(is_zero_mask): # "np.any()" will return True as lone as there is at least one satisfied condition.
            self.has_error = False
            # print("No flat line error was detected, keep the original data")
            logging.info("No flat line error was detected, keep the original data")
            return [self.raw]          

        # Enter the slicing section if detected.
        self.has_error = True
        # print("The flatline error was detected. Begin slicing")
        logging.info("The flatline error was detected. Begin slicing")

        boundaries: List[Tuple[int, int]] = self._find_valid_boundaries(is_zero_mask)
        valid_raws: List[mne.io.BaseRaw] = [] 

        # Begin slicing and logging metadata.
        for idx, (start_idx, end_idx) in enumerate(boundaries):
            duration_samples: int = end_idx - start_idx + 1

            if duration_samples >= self.min_samples: # Filter out the segments that are shorter than one minute.
                start_sec: float = start_idx / self.sfreq
                end_sec: float = end_idx / self.sfreq 

                raw_segment = self.raw.copy().crop(tmin=start_sec, tmax=end_sec)
                valid_raws.append(raw_segment)

                self.segments_metadata.append({
                    "segment_idx": len(valid_raws), # Indexing from 1 instead of 0.
                    "start_sec": round(start_sec, 3),
                    "end_sec": round(end_sec, 3),
                    "duration_sec": round(end_sec - start_sec, 3),
                    "start_sample": start_idx,
                    "end_sample": end_idx
                })

        # print(f"Slicing finished\nOriginal trial was split into {len(valid_raws)} valid segments")
        logging.info(f"Slicing finished\nOriginal trial was split into {len(valid_raws)} valid segments")
        
        return valid_raws

    def get_metadata(self) -> Dict[str, Any]:
        """
        Retrieve the logging metadata for the slicing operation.
        """
        return {
            "FlatlineErrorOccurred": self.has_error,
            "ValidSegmentsCount": len(self.segments_metadata),
            "SegmentsDetails": self.segments_metadata
        }
