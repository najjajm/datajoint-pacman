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
    -> pacman_processing.TrialAlignment
    -> pacman_processing.FilterParams
    ---
    force_raw:  longblob # raw (online), aligned force signal (V)
    force_filt: longblob # filtered, aligned, and calibrated force (N)
    """

    key_source = (pacman_processing.TrialAlignment & 'valid_alignment') \
        * pacman_processing.FilterParams

    def make(self, key):

        # convert raw force signal to Newtons
        trial_rel = pacman_acquisition.Behavior.Trial & key
        force = trial_rel.process_force(data_type='raw', filter=False)

        # get filter kernel
        filter_key = (processing.Filter & (pacman_processing.FilterParams & key)).fetch1('KEY')
        filter_parts = datajointutils.get_parts(processing.Filter, context=inspect.currentframe())
        filter_rel = next(part for part in filter_parts if part & filter_key)

        # apply filter
        fs = (acquisition.BehaviorRecording & key).fetch1('behavior_recording_sample_rate')
        force_filt = filter_rel().filter(force.copy(), fs)

        # align force signal
        beh_align = (pacman_processing.TrialAlignment & key).fetch1('behavior_alignment')
        force_raw_align = force.copy()[beh_align]
        force_filt_align = force_filt[beh_align]

        key.update(
            force_raw=force_raw_align, 
            force_filt=force_filt_align
        )

        self.insert1(key)