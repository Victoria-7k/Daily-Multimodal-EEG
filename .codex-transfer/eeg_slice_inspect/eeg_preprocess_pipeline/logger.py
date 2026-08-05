"""
Contains the PreprocessLogger class to be imported.

Specific to this preproces pipeline.

Author: ddh

Data: June 22, 2026
"""
import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Union, List, Dict, Tuple, Optional, Literal, Any

import mne # Because I used the type hint mne.io.BaseRaw... But import mne here shouldn't increase expenses.
 

class PreprocessLogger:
    """
    A unified logger for preprocess metadata.
    """
    def __init__(
        self, 
        subject_id: Union[int, str], 
        session: Union[int, str],
        # trial_path: Union[str, Path],
        run_idx: Optional[int] = None
        ) -> None:
        """
        Args:
            subject_id (Union[int, str]): the ID of the subject, from 1 to 15.
            trial_path (Union[str, Path]): The path to directory containing trial data (.bdf).
            run_idx (Optional[int]): The segment index if the trial was split. None if normal.
        """
        self.subject_id: int = int(subject_id)
        self.session: int = int(session)
        # self.trial_path = Path(trial_path)
        self.run_idx: int = run_idx

        # Extract and convert the time info from trial_path for logging.
        # months_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', "Jul", 'Aug', 'Sep', "Oct", 'Nov', "Dec"]
        # trial_time_str = self.trial_path.name[:14]
        # dt_obj: datetime = datetime.strptime(trial_time_str, "%Y%m%d%H%M%S")
        # self.trial_time: str = dt_obj.strftime("%Y-%b-%d_%H-%M-%S")

        self.log_data: Dict[str, Any] = {
            "Subject": int(subject_id),
            "Session": int(session),
            "BeginTime": None,
            # "InitialLength": None, # The initial length of this trial.
            "FinalLength": None, # The final length after cropping.

            "RunIndex": run_idx, # Record which segment it is, if sliced.

            "SamplingFrequency": None,
            "Filters": {}, # {type: "bandpass", range: (0, 47)}, {type: "notch", center: 50, width: 4}, in Hz.
            "1stBadChannels": None, # In full time length. All channels.
            "2ndBadChannels": None, # In one minu segments. Exclude come channels.
            "3rdBadChannels": None, # In one minu segments. All channels.

            "ICAComponents": {},
            "ICAComponentsRemoved": {},
            "ReReference": None, # A str demonstrating re-referencing method (e.g. "average").
        }
    def log_begin_time(self, begin_time: Union[str, datetime]) -> None:
        """
        Log the initial time of target trial.

        Args:
            begin_time (str): The time/date info for target trial (e.g. "2026-03-28 14:37:08+00:00").
        ---
        """
        self.log_data["BeginTime"] = str(begin_time)

    def log_sfreq(self, sfreq: float) -> None:
        self.log_data["SamplingFrequency"] = float(sfreq) # Use float is standard, rather than integer.

    def log_filtering(
        self, 
        filter_type: Literal["bandpass", "notch"], # Use "Literal" as type hints to limit the input.
        filter_prarms: Tuple[float, float]
    ) -> None: 
        """
        Log the filter info into the metadata.

        Args:
            filter_type Literal["bandpass", "notch"]: The specific type of filter applied.
            filter_parms: (Tuple[float, float]):
                - If "bandpass": A tuple of (high_pass_freq, low_pass_freq) in Hz.
                - If "notch": A tuple of (center_freq, bandwidth) in Hz.

        Raises:
            ValueError: If an unsupported "filter_type" is provided.
        """
        if filter_type == "bandpass":
            self.log_data["Filters"]["bandpass"] = {
                "type": "bandpass",
                "range": filter_prarms
            }
        elif filter_type == "notch":
            self.log_data["Filters"]["notch"] = {
                "type": "notch",
                "center": filter_prarms[0],
                "width": filter_prarms[1]
            }

    def log_bad_channels(
        self, 
        time: Literal["1st", "2nd", "3rd"],
        bad_channels: Union[List[Tuple[int, str, str]], Dict[str, List]]
        ) -> None:
        """
        Log the bad channels information after detection.
        Supports both full-length list logs and chunked dictionary logs.

        Args:
            time (Literal["1st", "2nd", "3rd"]): Specify which time/phase this bad channel 
                                                 detection is responsible for.
            bad_channels (List[Tuple[int, str, str]]): A list containing bad channels information.
                                                       Format: ((ch_idx, ch_name, brief_reason)).
                                                       Items might be duplicated.
        """
        self.log_data[f"{time}BadChannels"] = bad_channels

    def log_data_length(self, raw: mne.io.BaseRaw, ) -> None:
        """
        Receive a raw object, then calculate and log its pysical length.

        Args:  
            raw (mne.io.BaseRaw): The raw object who's length need to be calculated and logged.
        ---
        """
        total_secs: float = raw.n_times / raw.info['sfreq']

        # Calculated the time.
        minutes, seconds = divmod(total_secs, 60)
        hours, minutes = divmod(minutes, 60)

        output_str = f"{int(hours):02d}h {int(minutes):02d}m {seconds:06.3f}s, total {total_secs} secs"
        self.log_data["FinalLength"] = output_str

    def log_ica_removal(
        self,
        full_list: List[Tuple[int, str, float]],
        exclude_list: List[Tuple[int, str, float]],
        total_count: int,
        exclude_count: int,
        ) -> None:
        """
        Log the information of removed components after ICA.

        Args:
            full_list (List[Tuple[int, str, float]]): A list containing info for all components as a packed tuple.
                                                      Format (component_idx, identified_name, component_confidence).
            exclude_list (List[Tuple[int, str, float]]): A list containing info for rejected components as a packed tuple.
            total_count (int): The total number of all components.
            exclude_count (int): The total number of the removed components.
        ---
        """
        self.log_data["ICAComponents"]["total"] = int(total_count)
        self.log_data["ICAComponents"]["detail"] = [
            f'component idx:{a} name:"{b}" confidence:{c:.4f}' for a, b, c in full_list
            ]

        self.log_data["ICAComponentsRemoved"]["total"] = int(exclude_count)
        self.log_data["ICAComponentsRemoved"]["detail"] = [
            f'component idx:{a} name:"{b}" confidence:{c:.4f}' for a, b, c in exclude_list
            ]

    def log_re_reference(self, method: str) -> None:
        """
        Log the re-reference method adopted during the preprocessing.

        Args:
            method (str): A short str explaining the re-referencing method (e.g. "average").
        ---
        """
        self.log_data["ReReference"] = method

    def save_log(self, output_dir: Union[str, Path]) -> None:
        """
        Save the logged data as a JSON file. Default save to the trial_path,
        next to the .bdf data.
        """
        # if output_dir is None:
        #     output_dir = self.trial_path

        os.makedirs(output_dir, exist_ok=True)
        
        run_entity: str = f"_run-{self.run_idx:02d}" if self.run_idx is not None else ""
        # filename: str = f"{self.subject_id}_task-{self.trial_time}{run_entity}_preprocess.json"
        filename: str = f"sub-{self.subject_id:02d}_ses-{self.session:02d}{run_entity}_task-dailylife_prep.json"
        filepath: Path = Path(output_dir) / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.log_data, f, indent=4)
        # print(f"The log has been saved to {filepath}")
        logging.info(f"The log has been saved to {filepath}")
