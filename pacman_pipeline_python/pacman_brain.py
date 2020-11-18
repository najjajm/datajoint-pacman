import datajoint as dj
import os, inspect, itertools
import pandas as pd
import numpy as np
import neo
import matplotlib.pyplot as plt
from churchland_pipeline_python import lab, acquisition, processing
from churchland_pipeline_python.utilities import datajointutils
from . import pacman_acquisition, pacman_processing
from sklearn import decomposition
from typing import List, Tuple

schema = dj.schema(dj.config.get('database.prefix') + 'churchland_analyses_pacman_brain')

# =======
# LEVEL 0
# =======

@schema
class NeuronSpikeRaster(dj.Computed):
    definition = """
    # Aligned neuron single-trial spike raster
    -> processing.Neuron
    -> pacman_processing.BehaviorBlock
    -> pacman_processing.TrialAlignment
    ---
    -> pacman_processing.GoodTrial
    neuron_spike_raster: longblob # neuron trial-aligned spike raster (boolean array)
    """

    key_source = processing.Neuron * pacman_processing.BehaviorBlock * (pacman_processing.TrialAlignment & 'valid_alignment') \
        & (pacman_acquisition.Behavior.Trial * pacman_processing.BehaviorBlock.SaveTag)

    def make(self, key):

        # fetch ephys alignment indices for the current trial
        ephys_alignment = (pacman_processing.TrialAlignment & key).fetch1('ephys_alignment')

        # create spike bin edges centered around ephys alignment indices
        spike_bin_edges = np.append(ephys_alignment, ephys_alignment[-1]+1+np.arange(2)).astype(float)
        spike_bin_edges -= 0.5

        # fetch raw spike indices for the full recording
        neuron_spike_indices = (processing.Neuron & key).fetch1('neuron_spike_indices')

        # assign spike indices to bins
        spike_bins = np.digitize(neuron_spike_indices, spike_bin_edges) - 1

        # remove spike bins outside trial bounds
        spike_bins = spike_bins[(spike_bins >= 0) & (spike_bins < len(ephys_alignment))]

        # create trial spike raster
        spike_raster = np.zeros(len(ephys_alignment), dtype=bool)
        spike_raster[spike_bins] = 1

        key.update(
            neuron_spike_raster=spike_raster,
            behavior_quality_params_id=(pacman_processing.BehaviorQualityParams & key).fetch1('behavior_quality_params_id'),
            good_trial=(pacman_processing.GoodTrial & key).fetch1('good_trial')
        )

        # insert spike raster
        self.insert1(key)


# =======
# LEVEL 1
# =======

@schema
class NeuronRate(dj.Computed):
    definition = """
    # Aligned neuron single-trial firing rate
    -> NeuronSpikeRaster
    -> pacman_processing.FilterParams
    ---
    -> pacman_processing.GoodTrial
    neuron_rate: longblob # neuron trial-aligned firing rate (spikes/s)
    """

    # process per neuron/condition
    key_source = processing.Neuron \
        * pacman_acquisition.Behavior.Condition \
        * pacman_processing.FilterParams \
        & NeuronSpikeRaster

    def make(self, key):

        # fetch behavior sample rate and time vector
        fs_beh = (acquisition.BehaviorRecording & key).fetch1('behavior_recording_sample_rate')
        t_beh = (pacman_acquisition.Behavior.Condition & key).fetch1('condition_time')
        n_samples = len(t_beh)

        # fetch spike rasters (ephys time base)
        spike_raster_keys = (NeuronSpikeRaster & key).fetch(as_dict=True)

        # resample time to ephys time base
        fs_ephys = (acquisition.EphysRecording & key).fetch1('ephys_recording_sample_rate')
        t_ephys = np.linspace(t_beh[0], t_beh[-1], 1+int(round(fs_ephys * np.ptp(t_beh))))

        # rebin spike raster to behavior time base 
        time_bin_edges = np.append(t_beh, t_beh[-1]+(1+np.arange(2))/fs_beh) - 1/(2*fs_beh)
        spike_bins = [np.digitize(t_ephys[spike_raster_key['neuron_spike_raster']], time_bin_edges) - 1 \
            for spike_raster_key in spike_raster_keys]
        spike_bins = [b[(b >= 0) & (b < len(t_beh))] for b in spike_bins]

        for spike_raster_key, spk_bins in zip(spike_raster_keys, spike_bins):
            spike_raster_key['neuron_spike_raster'] = np.zeros(n_samples, dtype=bool)
            spike_raster_key['neuron_spike_raster'][spk_bins] = 1

        # get filter kernel
        filter_key = (processing.Filter & (pacman_processing.FilterParams & key)).fetch1('KEY')
        filter_parts = datajointutils.get_parts(processing.Filter, context=inspect.currentframe())
        filter_rel = next(part for part in filter_parts if part & filter_key)

        # filter rebinned spike raster
        neuron_rate_keys = spike_raster_keys.copy()
        [
            neuron_rate_key.update(
                filter_params_id = key['filter_params_id'],
                neuron_rate = fs_beh * filter_rel().filter(neuron_rate_key['neuron_spike_raster'], fs_beh),
                good_trial = (pacman_processing.GoodTrial & neuron_rate_key).fetch1('good_trial')
            )
            for neuron_rate_key in neuron_rate_keys
        ];

        # remove spike rasters
        [neuron_rate_key.pop('neuron_spike_raster') for neuron_rate_key in neuron_rate_keys];

        # insert neuron rates
        self.insert(neuron_rate_keys, skip_duplicates=True)


# =======
# LEVEL 2
# =======

@schema
class NeuronPsth(dj.Computed):
    definition = """
    # Peri-stimulus time histogram
    -> processing.Neuron
    -> pacman_processing.BehaviorBlock
    -> pacman_processing.FilterParams
    ---
    neuron_psth:     longblob # neuron trial-averaged firing rate (spikes/s)
    neuron_psth_sem: longblob # neuron firing rate standard error (spikes/s)
    """

    key_source = (processing.Neuron * pacman_processing.BehaviorBlock * pacman_processing.FilterParams) \
        & (pacman_acquisition.Behavior.Trial * pacman_processing.BehaviorBlock.SaveTag) \
        & (NeuronRate & 'good_trial')

    def make(self, key):

        # fetch single-trial firing rates
        rates = np.stack((NeuronRate & key & 'good_trial').fetch('neuron_rate'))

        # update key with psth and standard error
        key.update(
            neuron_psth=rates.mean(axis=0),
            neuron_psth_sem=rates.std(axis=0, ddof=1)/np.sqrt(rates.shape[0])
        )

        # insert neuron PSTH
        self.insert1(key)