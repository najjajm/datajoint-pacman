import datajoint as dj
import os, inspect, itertools
import pandas as pd
import numpy as np
import neo
import matplotlib.pyplot as plt
import plotly.express as px
from churchland_pipeline_python import lab, acquisition, processing, reference
from churchland_pipeline_python.utilities import datajointutils
from . import pacman_acquisition, pacman_processing
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
    -> pacman_processing.EphysTrialAlignment
    ---
    emg_signal: longblob # EMG voltage signal
    """

    # process per channel group
    key_source = acquisition.EmgChannelGroup \
        * pacman_processing.EphysTrialAlignment \
        & (pacman_processing.BehaviorTrialAlignment & 'valid_alignment')

    def make(self, key):

        # fetch channel keys
        channel_keys = (acquisition.EmgChannelGroup.Channel & key).fetch('KEY')

        # read channel indices from keys
        channel_indices = [chan_key['ephys_channel_idx'] for chan_key in channel_keys]

        # fetch ephys alignment indices
        ephys_alignment = (pacman_processing.EphysTrialAlignment & key).fetch1('ephys_alignment').astype(int)

        # fetch local ephys recording file path
        ephys_file_path = (acquisition.EphysRecording.File & key).proj_file_path().fetch1('ephys_file_path')

        # ensure local path
        ephys_file_path = reference.EngramTier.ensure_local(ephys_file_path)

        # read NSx file
        reader = neo.rawio.BlackrockRawIO(ephys_file_path)
        reader.parse_header()

        # read raw emg signals and transpose to horizontal
        emg_signals = reader.get_analogsignal_chunk(
            block_index=0, 
            seg_index=0, 
            i_start=ephys_alignment[0], 
            i_stop=1+ephys_alignment[-1], 
            channel_indexes=channel_indices
        ).T

        # update key with channel data
        keys = [dict(key, **chan_key, emg_signal=emg_signal)
            for chan_key, emg_signal in zip(channel_keys, emg_signals)]

        # insert emg signal keys
        self.insert(keys)

    # =================
    def plot_trial(
        self,
        suppress_warning: bool=False,
    ) -> None:

        # ensure one trial selected
        trial_table = pacman_acquisition.Behavior.Trial

        if len(trial_table & self) > 1:

            trial_key = dj.U(*trial_table.primary_key) \
                .aggr(self, rnd='RAND()') \
                .fetch('KEY', order_by='rnd', limit=1)[0]

            if not suppress_warning:

                print(
                    'Randomly selected session {}, condition ID {}, trial {}' \
                        .format(trial_key['session_date'], trial_key['condition_id'], trial_key['trial'])
                )

            self = self & trial_key

        else:

            trial_key = (trial_table & self).fetch1('KEY')
        
        # make condition time vector
        fs = int((acquisition.EphysRecording & self).fetch1('ephys_recording_sample_rate'))
        t, _ = pacman_acquisition.ConditionParams.target_force_profile(trial_key['condition_id'], fs)

        # fetch emg attributes
        emg_attributes = self.fetch(as_dict=True)

        # append time vector
        [emg_attr.update(t=t) for emg_attr in emg_attributes];

        # filter EMG
        [emg_attr.update(
            emg_signal=processing.Filter.Butterworth().filt(emg_attr['emg_signal'], fs, order=2, low_cut=500)
            ) for emg_attr in emg_attributes];

        # flatten attributes and import to DataFrame
        df = pd.DataFrame.from_records(datajointutils.flatten_blobs(emg_attributes, ['t', 'emg_signal']))

        # plot with plotly express
        fig = px.line(df, x='t', y='emg_signal', facet_row='ephys_channel_idx', width=800, height=600)
        fig.show()
                



@schema
class MotorUnitSpikeRaster(dj.Computed):
    definition = """
    # Aligned motor unit single-trial spike raster
    -> processing.MotorUnit
    -> pacman_processing.EphysTrialAlignment
    ---
    motor_unit_spike_raster: longblob # motor unit trial-aligned spike raster (boolean array)
    """

    key_source = processing.MotorUnit \
        * pacman_processing.EphysTrialAlignment \
        & (pacman_processing.BehaviorTrialAlignment & 'valid_alignment')

    def make(self, key):

        # fetch ephys alignment indices for the current trial
        ephys_alignment = (pacman_processing.EphysTrialAlignment & key).fetch1('ephys_alignment')

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

        key.update(motor_unit_spike_raster=spike_raster)

        # insert spike raster
        self.insert1(key)

    
    def rebin(
        self, 
        fs: int=None, 
        as_raster: bool=False, 
        order_by: str=None
        ) -> (List[dict], np.ndarray):
        """Rebin spike rasters.

        Args:
            fs (int, optional): New sample rate. Defaults to behavior sample rate.
            as_raster (bool, optional): If True, returns output as raster. If False, returns spike indices. Defaults to False.
            order_by (str, optional): Attribute used to order the returned data. If None, returns the data without additional sorting. 

        Returns:
            keys (list): List of key dictionaries to identify each set of rebinned spikes with the original table entry
            spikes (np.ndarray): Array of rebinned spike indices or rasters, depending on as_raster value
        """

        # fetch behavior condition keys (retain only condition time vectors)
        condition_keys = (pacman_acquisition.Behavior.Condition & self).proj('condition_time').fetch(as_dict=True)

        # initialize list of spikes and keys
        keys = []
        spikes = []

        # loop unique conditions
        for cond_key in condition_keys:

            t_cond = cond_key['condition_time']

            # new time vector (behavior time base by default)
            if fs is None:
                fs_new = (acquisition.BehaviorRecording & cond_key).fetch1('behavior_recording_sample_rate')
                t_new = t_cond
            
            else:
                fs_new = fs
                t_new = np.linspace(t_cond[0], t_cond[-1], 1+int(round(np.ptp(t_cond)*fs_new)))

            # create time bins in new time base
            t_bins = np.concatenate((t_new[:-1,np.newaxis], t_new[1:,np.newaxis]), axis=1).mean(axis=1)
            t_bins = np.append(np.insert(t_bins, 0, t_new[0]-1/(2*fs_new)), np.Inf)

            # resample time vector to ephys time base
            fs_ephys = (acquisition.EphysRecording & cond_key).fetch1('ephys_recording_sample_rate')
            t_ephys = np.linspace(t_cond[0], t_cond[-1], 1+int(round(np.ptp(t_cond)*fs_ephys)))

            # rebin spike rasters to new time base
            raster_keys, spike_rasters = (self & cond_key).fetch('KEY', 'motor_unit_spike_raster')

            new_spikes = [np.digitize(t_ephys[raster], t_bins) - 1 for raster in spike_rasters]

            # convert spike indices to raster
            if as_raster:

                new_spikes = [[True if i in spk_idx else False for i in range(len(t_new))] for spk_idx in new_spikes]

            # append spike rasters and keys to list
            keys.extend(raster_keys)
            spikes.extend(new_spikes)

        # order the data
        if order_by is not None:

            # extract the ordering attribute from the keys
            order_attr = [key[order_by] for key in keys]

            # sort the spike data
            spike_data = [(key, spk) for _, key, spk in sorted(zip(order_attr, keys, spikes))]

            # unpack the keys and spike indices as 
            keys, spikes = map(list, zip(*spike_data))

        return keys, np.array(spikes)


# =======
# LEVEL 1
# =======

@schema
class EmgEnvelope(dj.Computed):
    definition = """
    # Rectified and filtered EMG data
    -> Emg
    -> pacman_processing.FilterParams
    ---
    emg_envelope: longblob # EMG envelope
    """

    def make(self, key):

        # fetch behavior and ephys sample rates
        fs_beh = int((acquisition.BehaviorRecording & key).fetch1('behavior_recording_sample_rate'))
        fs_ephys = int((acquisition.EphysRecording & key).fetch1('ephys_recording_sample_rate'))

        # fetch condition time (behavior time base)
        t_beh = (pacman_acquisition.Behavior.Condition & key).fetch1('condition_time')

        # make condition time in ephys time base
        t_ephys, _ = pacman_acquisition.ConditionParams.target_force_profile(key['condition_id'], fs_ephys)

        # fetch raw emg data
        emg_attributes = (Emg & key).fetch(as_dict=True)

        # highpass filter and rectify raw emg signals
        [emg_attr.update(emg_envelope=abs(processing.Filter.Butterworth().filt(emg_attr['emg_signal'], fs_ephys, order=2, low_cut=40)))
            for emg_attr in emg_attributes];

        # remove raw emg signal
        [emg_attr.pop('emg_signal') for emg_attr in emg_attributes];

        # get filter kernel
        filter_key = (processing.Filter & (pacman_processing.FilterParams & key)).fetch1('KEY')
        filter_parts = datajointutils.get_parts(processing.Filter, context=inspect.currentframe())
        filter_rel = next(part for part in filter_parts if part & filter_key)

        # smooth rectified emg signals to construct envelope and remove 
        [emg_attr.update(
            filter_params_id=key['filter_params_id'],
            emg_envelope=filter_rel().filt(emg_attr['emg_envelope'], fs_ephys)
        ) for emg_attr in emg_attributes];

        # resample emg to behavior time base
        [emg_attr.update(
            emg_envelope=np.interp(t_beh, t_ephys, emg_attr['emg_envelope'])
        ) for emg_attr in emg_attributes];

        # insert emg envelopes
        self.insert(emg_attributes)


@schema
class MotorUnitRate(dj.Computed):
    definition = """
    # Aligned motor unit single-trial firing rate
    -> MotorUnitSpikeRaster
    -> pacman_processing.FilterParams
    ---
    motor_unit_rate: longblob # motor unit trial-aligned firing rate (spikes/s)
    """

    # process per motor unit/condition
    key_source = processing.MotorUnit \
        * pacman_acquisition.Behavior.Condition \
        * pacman_processing.FilterParams \
        & MotorUnitSpikeRaster

    def make(self, key):

        # fetch behavior and ephys sample rates
        fs_beh = int((acquisition.BehaviorRecording & key).fetch1('behavior_recording_sample_rate'))
        fs_ephys = int((acquisition.EphysRecording & key).fetch1('ephys_recording_sample_rate'))

        # fetch condition time (behavior time base)
        t_beh = (pacman_acquisition.Behavior.Condition & key).fetch1('condition_time')
        n_samples = len(t_beh)

        # make condition time in ephys time base
        t_ephys, _ = pacman_acquisition.ConditionParams.target_force_profile(key['condition_id'], fs_ephys)

        # fetch spike rasters (ephys time base)
        spike_raster_keys = (MotorUnitSpikeRaster & key).fetch(as_dict=True)

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
        filter_parts = datajointutils.get_parts(processing.Filter, context=inspect.currentframe())
        filter_rel = next(part for part in filter_parts if part & filter_key)

        # filter rebinned spike raster
        motor_unit_rate_keys = spike_raster_keys.copy()
        [
            motor_unit_rate_key.update(
                filter_params_id = key['filter_params_id'],
                motor_unit_rate = fs_beh * filter_rel().filt(motor_unit_rate_key['motor_unit_spike_raster'], fs_beh)
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
class EmgEnvelopeMean(dj.Computed):
    definition = """
    # Trial-averaged rectified and filtered EMG data
    -> acquisition.EmgChannelGroup.Channel
    -> pacman_processing.AlignmentParams
    -> pacman_processing.BehaviorBlock
    -> pacman_processing.BehaviorQualityParams
    -> pacman_processing.FilterParams
    ---
    emg_envelope_mean: longblob # trial-averaged EMG envelope (au)
    emg_envelope_sem:  longblob # EMG envelope standard error across trials (au)
    """

    # limit conditions with good trials
    key_source = acquisition.EmgChannelGroup.Channel \
        * pacman_processing.AlignmentParams \
        * pacman_processing.BehaviorBlock \
        * pacman_processing.BehaviorQualityParams \
        * pacman_processing.FilterParams \
        & EmgEnvelope \
        & (pacman_processing.GoodTrial & 'good_trial')

    def make(self, key):

        # fetch single-trial emg envelopes
        emg_envelope = (EmgEnvelope & key & (pacman_processing.GoodTrial & 'good_trial')).fetch('emg_envelope')
        emg_envelope = np.stack(emg_envelope)

        # update key with mean and standard error
        key.update(
            emg_envelope_mean=emg_envelope.mean(axis=0),
            emg_envelope_sem=emg_envelope.std(axis=0, ddof=(1 if emg_envelope.shape[0] > 1 else 0))/np.sqrt(emg_envelope.shape[0])
        )

        # insert emg envelope means
        self.insert1(key)


@schema
class MotorUnitPsth(dj.Computed):
    definition = """
    # Peri-stimulus time histogram
    -> processing.MotorUnit
    -> pacman_processing.AlignmentParams
    -> pacman_processing.BehaviorBlock
    -> pacman_processing.BehaviorQualityParams
    -> pacman_processing.FilterParams
    ---
    motor_unit_psth:     longblob # motor unit trial-averaged firing rate (spikes/s)
    motor_unit_psth_sem: longblob # motor unit firing rate standard error (spikes/s)
    """

    # limit conditions with good trials
    key_source = processing.MotorUnit \
        * pacman_processing.AlignmentParams \
        * pacman_processing.BehaviorBlock \
        * pacman_processing.BehaviorQualityParams \
        * pacman_processing.FilterParams \
        & MotorUnitRate \
        & (pacman_processing.GoodTrial & 'good_trial')

    def make(self, key):

        # fetch single-trial firing rates
        rates = (MotorUnitRate & key & (pacman_processing.GoodTrial & 'good_trial')).fetch('motor_unit_rate')
        rates = np.stack(rates)

        # update key with psth and standard error
        key.update(
            motor_unit_psth=rates.mean(axis=0),
            motor_unit_psth_sem=rates.std(axis=0, ddof=(1 if rates.shape[0] > 1 else 0))/np.sqrt(rates.shape[0])
        )

        # insert motor unit PSTH
        self.insert1(key)