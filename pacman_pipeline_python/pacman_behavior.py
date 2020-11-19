import datajoint as dj
import os, inspect, itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from churchland_pipeline_python import lab, acquisition, processing
from churchland_pipeline_python.utilities import datajointutils
from . import pacman_acquisition, pacman_processing
from sklearn import decomposition
from typing import List, Tuple

schema = dj.schema(dj.config.get('database.prefix') + 'churchland_analyses_pacman_behavior')

# =======
# LEVEL 0
# =======

@schema
class Force(dj.Computed):
    definition = """
    # Single trial force
    -> pacman_processing.BehaviorTrialAlignment
    -> pacman_processing.FilterParams
    ---
    force_raw:  longblob # raw (online), aligned force signal (V)
    force_filt: longblob # filtered, aligned, and calibrated force (N)
    """

    # batch process trials
    key_source = pacman_acquisition.Behavior.Condition \
        * pacman_processing.AlignmentParams \
        * pacman_processing.FilterParams \
        & (pacman_processing.BehaviorTrialAlignment & 'valid_alignment')

    def make(self, key):

        # trial source
        trial_source = (pacman_processing.BehaviorTrialAlignment & 'valid_alignment') \
            * pacman_processing.FilterParams & key

        # convert raw force signal to Newtons
        trial_rel = pacman_acquisition.Behavior.Trial & trial_source
        force_data = trial_rel.process_force(data_type='raw', apply_filter=False, keep_keys=True)

        # get filter kernel
        filter_key = (processing.Filter & (pacman_processing.FilterParams & key)).fetch1('KEY')
        filter_parts = datajointutils.get_parts(processing.Filter, context=inspect.currentframe())
        filter_rel = next(part for part in filter_parts if part & filter_key)

        # filter raw data
        fs = (acquisition.BehaviorRecording & key).fetch1('behavior_recording_sample_rate')
        [frc.update(force_filt_offline=filter_rel().filter(frc['force_raw_online'], fs)) for frc in force_data];

        # fetch alignment indices and cast as integers
        behavior_alignment = (trial_source).fetch('behavior_alignment', order_by='trial')
        behavior_alignment = list(map(lambda x: x.astype(int), behavior_alignment))

        # append key and align raw and filtered forces
        [frc.update(
            force_raw=frc['force_raw_online'][align_idx],
            force_filt=frc['force_filt_offline'][align_idx]
        )
        for frc, align_idx in zip(force_data, behavior_alignment)];

        # pop pre-aligned data
        for f_key in ['force_raw_online', 'force_filt_offline']:
            [frc.pop(f_key) for frc in force_data]

        # merge key with force data
        key = [dict(key, **frc) for frc in force_data]

        # insert aligned forces
        self.insert(key)