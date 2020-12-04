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
        [frc.update(force_filt_offline=filter_rel().filt(frc['force_raw_online'], fs)) for frc in force_data];

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


# =======
# LEVEL 1
# =======

@schema
class ForceMean(dj.Computed):
    definition = """
    # Trial-averaged forces
    -> pacman_processing.AlignmentParams
    -> pacman_processing.BehaviorBlock
    -> pacman_processing.BehaviorQualityParams
    -> pacman_processing.FilterParams
    ---
    force_raw_mean:  longblob # trial-averaged raw (online), aligned force signal (V)
    force_raw_sem:   longblob # raw mean force standard error
    force_filt_mean: longblob # trial-averaged filtered, aligned, and calibrated force (N)
    force_filt_sem:  longblob # filtered mean force standard error
    """

    # limit conditions with good trials
    key_source = pacman_processing.AlignmentParams \
        * pacman_processing.BehaviorBlock \
        * pacman_processing.BehaviorQualityParams \
        * pacman_processing.FilterParams \
        & Force \
        & (pacman_processing.GoodTrial & 'good_trial')

    def make(self, key):

        # fetch single-trial forces
        force_raw, force_filt = (Force & key & (pacman_processing.GoodTrial & 'good_trial')).fetch('force_raw', 'force_filt')
        force_raw = np.stack(force_raw)
        force_filt = np.stack(force_filt)

        # update key with mean and standard error
        key.update(
            force_raw_mean=force_raw.mean(axis=0),
            force_raw_sem=force_raw.std(axis=0, ddof=(1 if force_raw.shape[0] > 1 else 0))/np.sqrt(force_raw.shape[0]),
            force_filt_mean=force_filt.mean(axis=0),
            force_filt_sem=force_filt.std(axis=0, ddof=(1 if force_filt.shape[0] > 1 else 0))/np.sqrt(force_filt.shape[0]),
        )

        # insert forces
        self.insert1(key)