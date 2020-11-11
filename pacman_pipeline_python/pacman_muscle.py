import datajoint as dj
import os, inspect, itertools
import pandas as pd
import numpy as np
import neo
import progressbar
import matplotlib.pyplot as plt
from churchland_pipeline_python import lab, acquisition, processing, equipment, reference
from churchland_pipeline_python.utilities import datasync, datajointutils
from . import pacman_acquisition, pacman_processing
from datetime import datetime
from sklearn import decomposition
from typing import List, Tuple

schema = dj.schema(dj.config.get('database.prefix') + 'churchland_analyses_pacman_muscle')

# =======
# LEVEL 0
# =======

@schema
class Emg(dj.Imported):
    definition = """
    # raw, trialized, and aligned EMG data
    -> acquisition.EmgChannelGroup.Channel
    -> pacman_processing.BehaviorBlock
    -> pacman_processing.TrialAlignment
    ---
    -> pacman_processing.GoodTrial
    emg_signal: longblob # EMG voltage signal
    """

    # process per channel group
    key_source = acquisition.EmgChannelGroup \
        * pacman_processing.BehaviorBlock \
        * (pacman_processing.TrialAlignment & 'valid_alignment') \
        & (pacman_acquisition.Behavior.Trial * pacman_processing.BehaviorBlock.SaveTag)

    def make(self, key):

        # fetch channel keys
        channel_keys = (acquisition.EmgChannelGroup.Channel & key).fetch('KEY')

        # read channel indices from keys
        channel_indices = [chan_key['ephys_channel_idx'] for chan_key in channel_keys]

        # fetch ephys alignment indices
        ephys_alignment = (pacman_processing.TrialAlignment & key).fetch1('ephys_alignment').astype(int)

        # fetch local ephys recording file path
        ephys_file_path = (acquisition.EphysRecording.File & key).projfilepath().fetch1('ephys_file_path')

        # ensure local path
        ephys_file_path = reference.EngramTier.ensurelocal(ephys_file_path)

        # read NSx file
        reader = neo.rawio.BlackrockRawIO(ephys_file_path)
        reader.parse_header()

        # read raw signals
        raw_signals = reader.get_analogsignal_chunk(
            block_index=0, 
            seg_index=0, 
            i_start=ephys_alignment[0], 
            i_stop=1+ephys_alignment[-1], 
            channel_indexes=channel_indices
        )

        # rescale raw signals to float
        emg_signals = reader.rescale_signal_raw_to_float(raw_signals, dtype='float64', channel_indexes=channel_indices)

        # fetch behavior quality params ID
        behavior_quality_params_id = (pacman_processing.BehaviorQualityParams & key).fetch1('behavior_quality_params_id')

        # fetch good trial indicator
        good_trial = (pacman_processing.GoodTrial & key).fetch1('good_trial')

        # update key with channel data
        keys = [dict(key, **chan_key, emg_signal=emg_signal, behavior_quality_params_id=behavior_quality_params_id, good_trial=good_trial) \
            for chan_key, emg_signal in zip(channel_keys, emg_signals)]

        # insert emg signal keys
        self.insert(keys)


@schema
class MotorUnitSpikeRaster(dj.Computed):
    definition = """
    # Aligned motor unit single-trial spike raster
    -> processing.MotorUnit
    -> pacman_processing.BehaviorBlock
    -> pacman_processing.TrialAlignment
    ---
    -> pacman_processing.GoodTrial
    motor_unit_spike_raster: longblob # motor unit trial-aligned spike raster (boolean array)
    """

    key_source = processing.MotorUnit \
        * pacman_processing.BehaviorBlock \
        * (pacman_processing.TrialAlignment & 'valid_alignment') \
        & (pacman_acquisition.Behavior.Trial * pacman_processing.BehaviorBlock.SaveTag)

    def make(self, key):

        # fetch ephys alignment indices for the current trial
        ephys_alignment = (pacman_processing.TrialAlignment & key).fetch1('ephys_alignment')

        # create spike bin edges centered around ephys alignment indices
        spike_bin_edges = np.append(ephys_alignment, ephys_alignment[-1]+1+np.arange(2)).astype(float)
        spike_bin_edges -= 0.5

        # fetch raw spike indices for the full recording
        motor_unit_spike_indices = (processing.MotorUnit & key).fetch1('motor_unit_spike_indices')

        # assign spike indices to bins
        spike_bins = np.digitize(motor_unit_spike_indices, spike_bin_edges) - 1

        # remove spike bins outside trial bounds
        spike_bins = spike_bins[(spike_bins >= 0) & (spike_bins < len(ephys_alignment))]

        # create trial spike raster
        spike_raster = np.zeros(len(ephys_alignment), dtype=bool)
        spike_raster[spike_bins] = 1

        key.update(
            motor_unit_spike_raster=spike_raster,
            behavior_quality_params_id=(pacman_processing.BehaviorQualityParams & key).fetch1('behavior_quality_params_id'),
            good_trial=(pacman_processing.GoodTrial & key).fetch1('good_trial')
        )

        # insert spike raster
        self.insert1(key)


# =======
# LEVEL 1
# =======

@schema
class MotorUnitRate(dj.Computed):
    definition = """
    # Aligned motor unit single-trial firing rate
    -> MotorUnitSpikeRaster
    -> pacman_processing.FilterParams
    ---
    -> pacman_processing.GoodTrial
    motor_unit_rate: longblob # motor unit trial-aligned firing rate (spikes/s)
    """

    # process per motor unit/condition
    key_source = processing.MotorUnit \
        * pacman_acquisition.Behavior.Condition \
        * pacman_processing.FilterParams \
        & MotorUnitSpikeRaster

    def make(self, key):

        # fetch behavior sample rate and time vector
        fs_beh = (acquisition.BehaviorRecording & key).fetch1('behavior_recording_sample_rate')
        t_beh = (pacman_acquisition.Behavior.Condition & key).fetch1('condition_time')
        n_samples = len(t_beh)

        # fetch spike rasters (ephys time base)
        spike_raster_keys = (MotorUnitSpikeRaster & key).fetch(as_dict=True)

        # resample time to ephys time base
        fs_ephys = (acquisition.EphysRecording & key).fetch1('ephys_recording_sample_rate')
        t_ephys = np.linspace(t_beh[0], t_beh[-1], 1+round(fs_ephys * np.ptp(t_beh)))

        # rebin spike raster to behavior time base 
        time_bin_edges = np.append(t_beh, t_beh[-1]+(1+np.arange(2))/fs_beh) - 1/(2*fs_beh)
        spike_bins = [np.digitize(t_ephys[spike_raster_key['motor_unit_spike_raster']], time_bin_edges) - 1 \
            for spike_raster_key in spike_raster_keys]
        spike_bins = [b[(b >= 0) & (b < len(t_beh))] for b in spike_bins]

        for spike_raster_key, spk_bins in zip(spike_raster_keys, spike_bins):
            spike_raster_key['motor_unit_spike_raster'] = np.zeros(n_samples, dtype=bool)
            spike_raster_key['motor_unit_spike_raster'][spk_bins] = 1

        # get filter kernel
        filter_key = (processing.Filter & (pacman_processing.FilterParams & key)).fetch1('KEY')
        filter_parts = datajointutils.getparts(processing.Filter, context=inspect.currentframe())
        filter_rel = next(part for part in filter_parts if part & filter_key)

        # filter rebinned spike raster
        motor_unit_rate_keys = spike_raster_keys.copy()
        [
            motor_unit_rate_key.update(
                filter_params_id = key['filter_params_id'],
                motor_unit_rate = fs_beh * filter_rel().filter(motor_unit_rate_key['motor_unit_spike_raster'], fs_beh),
                good_trial = (pacman_processing.GoodTrial & motor_unit_rate_key).fetch1('good_trial')
            )
            for motor_unit_rate_key in motor_unit_rate_keys
        ];

        # remove spike rasters
        [motor_unit_rate_key.pop('motor_unit_spike_raster') for motor_unit_rate_key in motor_unit_rate_keys];

        # insert motor unit rates
        self.insert(motor_unit_rate_keys, skip_duplicates=True)


# =======
# LEVEL 2
# =======

@schema
class MotorUnitPsth(dj.Computed):
    definition = """
    # Peri-stimulus time histogram
    -> processing.MotorUnit
    -> pacman_processing.BehaviorBlock
    -> pacman_processing.FilterParams
    ---
    motor_unit_psth: longblob # motor unit trial-averaged firing rate (spikes/s)
    """

    key_source = (processing.MotorUnit * pacman_processing.BehaviorBlock * pacman_processing.FilterParams) \
        & (pacman_acquisition.Behavior.Trial * pacman_processing.BehaviorBlock.SaveTag) \
        & MotorUnitRate

    def make(self, key):

        # fetch single-trial firing rates and average
        psth = (MotorUnitRate & key & 'good_trial').fetch('motor_unit_rate').mean(axis=0)

        # insert motor unit PSTH
        self.insert1(dict(**key, motor_unit_psth=psth))