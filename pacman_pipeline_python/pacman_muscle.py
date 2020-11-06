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

    key_source = acquisition.EmgChannelGroup.Channel \
        * pacman_processing.BehaviorBlock \
        * (pacman_processing.TrialAlignment & 'valid_alignment') \
        & (pacman_acquisition.Behavior.Trial * pacman_processing.BehaviorBlock.SaveTag)

    def make(self, key):

        # fetch local ephys recording file path
        ephys_file_path = (acquisition.EphysRecording.File & key).projfilepath().fetch1('ephys_file_path')

        # ensure local path
        ephys_file_path = (reference.EngramTier & {'engram_tier': 'locker'}).ensurelocal(ephys_file_path)

        # read NSx file
        reader = neo.rawio.BlackrockRawIO(ephys_file_path)
        reader.parse_header()

        # fetch channel ID and index
        chan_id, chan_idx = (acquisition.EphysRecording.Channel & key).fetch1('ephys_channel_id', 'ephys_channel_idx')

        # channel ID and gain
        id_idx, gain_idx = [
            idx for idx, name in enumerate(reader.header['signal_channels'].dtype.names) \
            if name in ['id','gain']
        ]
        chan_gain = next(chan[gain_idx] for chan in reader.header['signal_channels'] if chan[id_idx]==chan_id)

        # extract NSx channel data from memory map (within a nested dictionary)
        nsx_data = next(iter(reader.nsx_datas.values()))
        nsx_data = next(iter(nsx_data.values()))

        # fetch ephys alignment indices
        ephys_alignment = (pacman_processing.TrialAlignment & key).fetch1('ephys_alignment')

        # extract emg signal from NSx array and apply gain
        emg_signal = chan_gain * nsx_data[ephys_alignment, chan_idx]

        key.update(
            emg_signal=emg_signal,
            behavior_quality_params_id=(pacman_processing.BehaviorQualityParams & key).fetch1('behavior_quality_params_id'),
            good_trial=(pacman_processing.GoodTrial & key).fetch1('good_trial')
        )

        # insert emg signal
        self.insert1(key)


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

    def make(self, key):

        # fetch behavior sample rate and time vector
        fs_beh = (acquisition.BehaviorRecording & key).fetch1('behavior_recording_sample_rate')
        t_beh = (pacman_acquisition.Behavior.Condition & key).fetch1('condition_time')

        # fetch spike raster (ephys time base)
        spike_raster = (MotorUnitSpikeRaster & key).fetch1('motor_unit_spike_raster')

        if any(spike_raster):

            # resample time to ephys time base
            fs_ephys = (acquisition.EphysRecording & key).fetch1('ephys_recording_sample_rate')
            t_ephys = np.linspace(t_beh[0], t_beh[-1], 1+round(fs_ephys * np.ptp(t_beh)))

            # rebin spike raster to behavior time base 
            time_bin_edges = np.append(t_beh, t_beh[-1]+(1+np.arange(2))/fs_beh) - 1/(2*fs_beh)
            spike_bins = np.digitize(t_ephys[spike_raster], time_bin_edges) - 1
            spike_bins = spike_bins[(spike_bins >= 0) & (spike_bins < len(t_beh))]

            spike_raster = np.zeros(len(t_beh), dtype=bool)
            spike_raster[spike_bins] = 1

            # get filter kernel
            filter_key = (processing.Filter & (pacman_processing.FilterParams & key)).fetch1('KEY')
            filter_parts = datajointutils.getparts(processing.Filter, context=inspect.currentframe())
            filter_rel = next(part for part in filter_parts if part & filter_key)

            # filter rebinned spike raster
            rate = fs_beh * filter_rel().filter(spike_raster, fs_beh)

        else:
            rate = np.zeros(len(t_beh))

        key.update(
            motor_unit_rate=rate,
            behavior_quality_params_id=(pacman_processing.BehaviorQualityParams & key).fetch1('behavior_quality_params_id'),
            good_trial=(pacman_processing.GoodTrial & key).fetch1('good_trial')
        )

        # insert motor unit rate
        self.insert1(key)


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
        psth = (MotorUnitRate & key).fetch('motor_unit_rate').mean(axis=0)

        # insert motor unit PSTH
        self.insert1(dict(**key, motor_unit_psth=psth))