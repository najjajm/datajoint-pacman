import datajoint as dj
import os, inspect, itertools
import pandas as pd
import numpy as np
import neo
import progressbar
import matplotlib.pyplot as plt
from churchland_pipeline_python import lab, acquisition, processing, equipment, reference
from churchland_pipeline_python.utilities import datasync, datajointutils as dju
from . import pacman_acquisition
from datetime import datetime
from sklearn import decomposition

schema = dj.schema(dj.config.get('database.prefix') + 'churchland_analyses_pacman_processing')

# =======
# LEVEL 0
# =======

@schema
class AlignmentParams(dj.Manual):
    definition = """
    # Parameters for aligning trials
    -> pacman_acquisition.Behavior
    alignment_params_id:     tinyint unsigned # alignment params ID number
    ---
    -> pacman_acquisition.TaskState
    alignment_max_lag = 0.2: decimal(4,3)     # maximum absolute time lag for shifting each trial (s)
    """
    
    @classmethod
    def populate(self, 
        behavior_rel: pacman_acquisition.Behavior=pacman_acquisition.Behavior(), 
        task_state_rel: pacman_acquisition.TaskState=(pacman_acquisition.TaskState & {'task_state_name': 'InTarget'}), 
        max_lag: int=0.2) -> None:

        # check inputs
        assert isinstance(behavior_rel, pacman_acquisition.Behavior), 'Unrecognized behavior table'
        assert isinstance(task_state_rel, pacman_acquisition.TaskState), 'Unrecognized task state table'

        # construct "key source" from join of behavior and task state tables
        key_source = (behavior_rel * task_state_rel) - (self & {'alignment_max_lag': max_lag})

        behavior_source = behavior_rel & key_source.proj()
        task_state_source = task_state_rel & key_source.proj()

        # insert task state for every behavior
        for beh_key, task_state_key in itertools.product(behavior_source.fetch('KEY'), task_state_source.fetch('KEY')):

            # get filter params ID
            if not self & beh_key:
                new_param_id = 0
            else:
                all_param_id = (self & beh_key).fetch('alignment_params_id')
                new_param_id = next(i for i in range(2+max(all_param_id)) if i not in all_param_id)

            self.insert1(dict(**beh_key, alignment_params_id=new_param_id, **task_state_key, alignment_max_lag=max_lag))


@schema
class BehaviorBlock(dj.Manual):
    definition = """
    # Set of save tags and behavioral recording parameters for conducting analyses
    -> pacman_acquisition.Behavior
    behavior_block_id: tinyint unsigned # behavior block ID number
    ---
    -> pacman_acquisition.ArmPosture
    """
    
    class SaveTag(dj.Part):
        definition = """
        -> master
        -> pacman_acquisition.Behavior.SaveTag
        """

    @classmethod
    def insert_from_file(self,
        monkey: str
    ) -> None:
        """Inserts behavior block entries by reading data from a csv file."""

        # read table from metadata
        metadata_path = os.path.join(os.path.dirname(__file__), '..', 'metadata', '')
        behavior_block_df = pd.read_csv(metadata_path + monkey.lower() + '_behavior_block.csv')

        # convert dataframe to behavior block keys (remove secondary attributes)
        behavior_block_key = behavior_block_df\
            .drop(['arm_posture_id', 'save_tag'], axis=1)\
            .to_dict(orient='records')

        # prepend dataframe index to key
        behavior_block_key = [(idx, key) for idx, key in enumerate(behavior_block_key)]

        # filter keys by those in behavior table but not in behavior block table
        behavior_block_key = [(idx, key) for idx, key in behavior_block_key
            if (pacman_acquisition.Behavior & key) and not (self & key)]

        # insert entries
        for idx, key in behavior_block_key:

            # insert behavior block
            self.insert1(dict(**key, arm_posture_id=behavior_block_df.loc[idx, 'arm_posture_id']))

            # read save tags for block
            save_tags = eval('[' + behavior_block_df.loc[idx,'save_tag'] + ']')

            # insert behavior block save tags
            for tag in save_tags:
                
                self.SaveTag.insert1(dict(**key, save_tag=tag))


@schema
class EphysTrialStart(dj.Imported):
    definition = """
    # Synchronizes continuous acquisition ephys data with behavior trials
    -> pacman_acquisition.Behavior.Trial
    ---
    ephys_trial_start = null: int unsigned # sample index (ephys time base) corresponding to the trial start
    """

    key_source = pacman_acquisition.Behavior.Trial & (acquisition.Session & processing.SyncBlock)

    def make(self, key):

        session_key = (acquisition.Session & key).fetch1('KEY')

        # ephys sample rate
        fs_ephys = (acquisition.EphysRecording & session_key).fetch1('ephys_recording_sample_rate') 

        # all trial keys with simulation time
        trial_keys = (pacman_acquisition.Behavior.Trial & session_key).fetch('KEY','simulation_time',as_dict=True)

        # pop simulation time (Speedgoat clock) from trial key
        trial_time = [trial.pop('simulation_time',None) for trial in trial_keys]

        # sync block start index and encoded time stamp
        sync_block_start, sync_block_time = (processing.SyncBlock & session_key).fetch('sync_block_start', 'sync_block_time')

        # get trial start index in ephys time base
        ephys_trial_start_idx = datasync.ephystrialstart(fs_ephys, trial_time, sync_block_start, sync_block_time)

        # legacy adjustment
        if session_key['session_date'] <= datetime.strptime('2018-10-11','%Y-%m-%d').date():
            ephys_trial_start_idx += round(0.1 * fs_ephys)

        # append ephys trial start to key
        trial_keys = [dict(**trial, ephys_trial_start=i0) for trial,i0 in zip(trial_keys,ephys_trial_start_idx)]

        self.insert(trial_keys)


@schema
class FilterParams(dj.Manual):
    definition = """
    # Set of filter parameters for smoothing forces and spike trains
    -> pacman_acquisition.Behavior.Condition
    filter_params_id: tinyint unsigned # filter params ID number
    ---
    -> processing.Filter
    """

    @classmethod
    def populate(self,
        condition_rel: pacman_acquisition.Behavior.Condition=pacman_acquisition.Behavior.Condition(), 
        filter_attr: dict={'sd':25e-3,'width':4}):

        _, filter_parts = dju.joinparts(processing.Filter, filter_attr)
        filter_rel = next(x for x in filter_parts if x in dju.getchildren(processing.Filter))

        # check inputs
        assert isinstance(condition_rel, pacman_acquisition.Behavior.Condition), 'Unrecognized condition table'
        assert filter_rel in dju.getchildren(processing.Filter), 'Unrecognized filter table'

        # construct "key source" from join of condition and filter tables
        key_source = (condition_rel * filter_rel) - self

        cond_source = condition_rel & key_source.proj()
        filt_source = filter_rel & key_source.proj()

        # insert task state for every session
        for cond_key, filt_key in itertools.product(cond_source.fetch('KEY'), filt_source.fetch('KEY')):

            # get filter params ID
            if not self & cond_key:
                new_param_id = 0
            else:
                all_param_id = (self & cond_key).fetch('filter_params_id')
                new_param_id = next(i for i in range(2+max(all_param_id)) if i not in all_param_id)

            self.insert1(dict(**cond_key, filter_params_id=new_param_id, **filt_key))


# =======
# LEVEL 1
# =======

@schema
class TrialAlignment(dj.Computed):
    definition = """
    # Trial alignment indices for behavior and ephys data 
    -> EphysTrialStart
    -> AlignmentParams
    ---
    valid_alignment = 0:       bool     # whether the trial can be aligned
    behavior_alignment = null: longblob # trial alignment indices for behavioral data
    ephys_alignment = null:    longblob # trial alignment indices for ephys data
    """
    
    # restrict to trials with a defined start index
    key_source = ((EphysTrialStart & 'ephys_trial_start') & (pacman_acquisition.Behavior.Trial & 'successful_trial')) \
        * AlignmentParams

    def make(self, key):

        # trial table
        trial_rel = pacman_acquisition.Behavior.Trial & key

        # fetch all parameters from key source
        full_key = (self.key_source & key).fetch1()

        # set alignment index
        if pacman_acquisition.ConditionParams.Stim & trial_rel:

            # align to stimulationa
            stim = trial_rel.fetch1('stim')
            align_idx = next(i for i in range(len(stim)) if stim[i])

        else:
            # align to task state
            task_state = trial_rel.fetch1('task_state')
            align_idx = next(i for i in range(len(task_state)) if task_state[i] == full_key['task_state_id'])

        # behavioral sample rate
        fs_beh = (acquisition.BehaviorRecording & key).fetch1('behavior_recording_sample_rate')

        # fetch target force and time
        t, target_force = (pacman_acquisition.Behavior.Condition & trial_rel).fetch1('condition_time', 'condition_force')
        
        # behavior time indices and zero index
        t_idx_beh = (fs_beh * t).astype(int)
        zero_idx = np.argmax(t_idx_beh == 0)   

        # phase correct dynamic conditions
        if not pacman_acquisition.ConditionParams.Static & trial_rel:

            # generate lag range
            max_lag = float(full_key['alignment_max_lag'])
            max_lag_samp = int(round(fs_beh * max_lag))
            lags = range(-max_lag_samp, 1+max_lag_samp)

            # truncate time indices  ap
            precision = int(round(np.log10(fs_beh)))
            trunc_idx = np.nonzero((t>=round(t[0]+max_lag, precision)) & (t<=round(t[-1]-max_lag, precision)))[0]
            target_force = target_force[trunc_idx]
            align_idx_trunc = trunc_idx - zero_idx

            # process force signal
            force = trial_rel.processforce()

            # compute normalized mean squared error for each lag
            nmse = np.full(1+2*max_lag_samp, -np.inf)
            for idx, lag in enumerate(lags):
                if (align_idx + lag + align_idx_trunc[-1]) < len(force):
                    force_align = force[align_idx+lag+align_idx_trunc]
                    nmse[idx] = 1 - np.sqrt(np.mean((force_align-target_force)**2)/np.var(target_force))

            # shift alignment indices by optimal lag
            align_idx += lags[np.argmax(nmse)]

        # behavior alignment indices
        behavior_alignment = t_idx_beh + align_idx

        if behavior_alignment[-1] < len(trial_rel.fetch1('force_raw_online')):

            key.update(valid_alignment=1, behavior_alignment=behavior_alignment)

            # ephys alignment indices
            fs_ephys = (acquisition.EphysRecording & key).fetch1('ephys_recording_sample_rate')
            t_idx_ephys = (fs_ephys * np.linspace(t[0], t[-1], 1+round(fs_ephys * np.ptp(t)))).astype(int)
            ephys_alignment = t_idx_ephys + align_idx * round(fs_ephys/fs_beh)
            ephys_alignment += (EphysTrialStart & key).fetch1('ephys_trial_start')
            key.update(ephys_alignment=ephys_alignment)

        self.insert1(key)


# =======
# LEVEL 2
# =======

@schema
class Emg(dj.Imported):
    definition = """
    # raw, trialized, and aligned EMG data
    -> acquisition.EmgChannelGroup.Channel
    -> TrialAlignment
    ---
    emg_signal: longblob # EMG voltage signal
    """

    key_source = acquisition.EmgChannelGroup.Channel * (TrialAlignment & 'valid_alignment')

    def make(self, key):

        # fetch local ephys recording file path
        file_path, file_prefix, file_extension = (acquisition.EphysRecording * (acquisition.EphysRecording.File & key))\
            .fetch1('ephys_recording_path', 'ephys_file_prefix', 'ephys_file_extension')

        ephys_file_path = (reference.EngramTier & {'engram_tier': 'locker'})\
            .ensurelocal(file_path + file_prefix + '.' + file_extension)

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
        ephys_alignment = (TrialAlignment & key).fetch1('ephys_alignment')

        # extract emg signal from NSx array and apply gain
        emg_signal = chan_gain * nsx_data[ephys_alignment, chan_idx]

        # insert emg signal
        self.insert1(dict(**key, emg_signal=emg_signal))


@schema
class Force(dj.Computed):
    definition = """
    # Single trial force
    -> TrialAlignment
    -> FilterParams
    ---
    force_raw:  longblob # raw (online), aligned force signal (V)
    force_filt: longblob # filtered, aligned, and calibrated force (N)
    """

    key_source = (TrialAlignment & 'valid_alignment') * FilterParams

    def make(self, key):

        # convert raw force signal to Newtons
        trial_rel = pacman_acquisition.Behavior.Trial & key
        force = trial_rel.processforce(data_type='raw', filter=False)

        # get filter kernel
        filter_key = (processing.Filter & (FilterParams & key)).fetch1('KEY')
        filter_parts = dju.getparts(processing.Filter, context=inspect.currentframe())
        filter_rel = next(part for part in filter_parts if part & filter_key)

        # apply filter
        fs = (acquisition.BehaviorRecording & key).fetch1('behavior_recording_sample_rate')
        force_filt = filter_rel().filter(force.copy(), fs)

        # align force signal
        beh_align = (TrialAlignment & key).fetch1('behavior_alignment')
        force_raw_align = force.copy()[beh_align]
        force_filt_align = force_filt[beh_align]

        key.update(force_raw=force_raw_align, force_filt=force_filt_align)

        self.insert1(key)

    def plot(self):
        """Plot force trials."""

        condition_keys = (pacman_acquisition.ConditionParams & self).fetch('KEY')
        n_conditions = len(condition_keys)

        n_columns = np.ceil(np.sqrt(n_conditions)).astype(int)
        n_rows = np.ceil(n_conditions/n_columns).astype(int)

        max_force = (pacman_acquisition.ConditionParams.Force & self).fetch('force_max').max()

        _, axs = plt.subplots(n_rows, n_columns, figsize=(12,8))
        for idx, key in zip(np.ndindex((n_rows, n_columns)), condition_keys):

            t = next(iter((pacman_acquisition.Behavior.Condition & key).fetch('condition_time'))) #//TODO separate by session??
            trial_forces = (self & key).fetch('force_filt')

            axs[idx].plot(t, np.stack(trial_forces).T, 'k');
            axs[idx].set_ylim([0, 1.25*max_force])
            axs[idx].set_title('Cond. {} (n = {})'.format(key['condition_id'], trial_forces.shape[0]))
            axs[idx].set_xticks([])
            axs[idx].set_yticks([])


@schema
class MotorUnitSpikeRaster(dj.Computed):
    definition = """
    # Aligned motor unit single-trial spike raster
    -> processing.MotorUnit
    -> TrialAlignment
    ---
    motor_unit_spike_raster: longblob # motor unit trial-aligned spike raster (boolean array)
    """

    key_source = processing.MotorUnit * (TrialAlignment & 'valid_alignment')

    def make(self, key):

        # fetch ephys alignment indices for the current trial
        ephys_alignment = (TrialAlignment & key).fetch1('ephys_alignment')

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

        # insert spike raster
        self.insert1(dict(**key, motor_unit_spike_raster=spike_raster))


@schema
class NeuronSpikeRaster(dj.Computed):
    definition = """
    # Aligned neuron single-trial spike raster
    -> processing.Neuron
    -> TrialAlignment
    ---
    neuron_spike_raster: longblob # neuron trial-aligned spike raster (boolean array)
    """

    key_source = processing.Neuron * (TrialAlignment & 'valid_alignment')

    def make(self, key):

        # fetch ephys alignment indices for the current trial
        ephys_alignment = (TrialAlignment & key).fetch1('ephys_alignment')

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

        # insert spike raster
        self.insert1(dict(**key, neuron_spike_raster=spike_raster))


# =======
# LEVEL 3
# =======
    
@schema 
class GoodTrial(dj.Computed):
    definition = """
    # Trials that meet behavior quality thresholds
    -> Force
    -> AlignmentParams
    """

    # process keys per condition
    key_source = pacman_acquisition.Behavior.Condition & Force

    def make(self, key):

        # get force keys and single-trial filtered forces
        force_keys, trial_forces = (Force & key).fetch('KEY', 'force_filt')
        trial_forces = np.stack(trial_forces)

        # get target force profile
        target_force = (pacman_acquisition.Behavior.Condition & key).fetch1('condition_force')

        # compute maximum shift in samples
        fs = (acquisition.BehaviorRecording & key).fetch1('behavior_recording_sample_rate')
        max_lag_sec = (AlignmentParams & key).fetch1('alignment_max_lag')
        max_lag_samp = int(round(fs * max_lag_sec))

        # compute error tolerance for the condition (range of target force values within lag window)
        n_samples = len(target_force)
        err_tol = np.zeros(n_samples)

        for idx in range(n_samples):

            start_idx = max(0, idx-max_lag_samp)
            stop_idx = min(n_samples-1, idx+max_lag_samp)+1
            err_tol[idx] = target_force[start_idx:stop_idx].ptp()

        # bound error tolerance
        err_tol = np.maximum(2, 0.5*err_tol)

        # construct upper/lower bounds for trial forces
        mean_force = trial_forces.mean(axis=0)
        upper_bound = np.maximum(mean_force, target_force) + err_tol
        lower_bound = np.minimum(mean_force, target_force) - err_tol

        # identify good trials whose force values remain within bounds for at least 97% of the condition duration
        good_trials = np.mean((trial_forces < upper_bound) & (trial_forces > lower_bound), axis=1) > 0.97

        # remove any remaining extreme outliers (trials with momentary force values exceeding 3.5 x standard deviation)
        std_force = trial_forces.std(axis=0)
        good_trials = good_trials & np.all((trial_forces < (mean_force + 3.5*std_force)) & (trial_forces > (mean_force - 3.5*std_force)), axis=1)

        # filter force keys by good trials
        force_keys = [key for idx, key in enumerate(force_keys) if good_trials[idx]]

        # insert good trials
        self.insert(force_keys)


@schema
class MotorUnitRate(dj.Computed):
    definition = """
    # Aligned motor unit single-trial firing rate
    -> MotorUnitSpikeRaster
    -> FilterParams
    ---
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
            pre_pad_dur = (pacman_acquisition.ConditionParams.Target \
                & (pacman_acquisition.Behavior.Trial & key)).fetch1('target_pad_pre')

            fs_ephys = (acquisition.EphysRecording & key).fetch1('ephys_recording_sample_rate')
            t_ephys = (1 - round(fs_ephys * pre_pad_dur) + np.arange(round(fs_ephys/fs_beh) * len(t_beh)))/fs_ephys

            # rebin spike raster to behavior time base 
            time_bin_edges = np.append(t_beh, t_beh[-1]+(1+np.arange(2))/fs_beh) - 1/(2*fs_beh)
            spike_bins = np.digitize(t_ephys[spike_raster], time_bin_edges) - 1
            spike_bins = spike_bins[(spike_bins >= 0) & (spike_bins < len(t_beh))]

            spike_raster = np.zeros(len(t_beh), dtype=bool)
            spike_raster[spike_bins] = 1

            # get filter kernel
            filter_parts = dju.getparts(processing.Filter, context=inspect.currentframe())
            filter_rel = next(part for part in filter_parts if part & key)

            # filter rebinned spike raster
            rate = fs_beh * filter_rel().filter(spike_raster, fs_beh)

        else:
            rate = np.zeros(len(t_beh))

        # insert motor unit rate
        self.insert1(dict(**key, motor_unit_rate=rate))
    

@schema
class NeuronRate(dj.Computed):
    definition = """
    # Aligned neuron single-trial firing rate
    -> NeuronSpikeRaster
    -> FilterParams
    ---
    neuron_rate: longblob # neuron trial-aligned firing rate (spikes/s)
    """

    def make(self, key):

        # fetch behavior sample rate and time vector
        fs_beh = (acquisition.BehaviorRecording & key).fetch1('behavior_recording_sample_rate')
        t_beh = (pacman_acquisition.Behavior.Condition & key).fetch1('condition_time')

        # fetch spike raster (ephys time base)
        spike_raster = (NeuronSpikeRaster & key).fetch1('neuron_spike_raster')

        if any(spike_raster):

            # resample time to ephys time base
            pre_pad_dur = (pacman_acquisition.ConditionParams.Target \
                & (pacman_acquisition.Behavior.Trial & key)).fetch1('target_pad_pre')

            fs_ephys = (acquisition.EphysRecording & key).fetch1('ephys_recording_sample_rate')
            t_ephys = (1 - round(fs_ephys * pre_pad_dur) + np.arange(round(fs_ephys/fs_beh) * len(t_beh)))/fs_ephys

            # rebin spike raster to behavior time base 
            time_bin_edges = np.append(t_beh, t_beh[-1]+(1+np.arange(2))/fs_beh) - 1/(2*fs_beh)
            spike_bins = np.digitize(t_ephys[spike_raster], time_bin_edges) - 1
            spike_bins = spike_bins[(spike_bins >= 0) & (spike_bins < len(t_beh))]

            spike_raster = np.zeros(len(t_beh), dtype=bool)
            spike_raster[spike_bins] = 1

            # get filter kernel
            filter_parts = dju.getparts(processing.Filter, context=inspect.currentframe())
            filter_rel = next(part for part in filter_parts if part & key)

            # filter rebinned spike raster
            rate = fs_beh * filter_rel().filter(spike_raster, fs_beh)

        else:
            rate = np.zeros(len(t_beh))

        # insert neuron rate
        self.insert1(dict(**key, neuron_rate=rate))


# =======
# LEVEL 4
# =======

@schema
class MotorUnitPsth(dj.Computed):
    definition = """
    # Peri-stimulus time histogram
    -> processing.MotorUnit
    -> BehaviorBlock
    -> FilterParams
    ---
    motor_unit_psth: longblob # motor unit trial-averaged firing rate (spikes/s)
    """

    key_source = (processing.MotorUnit * BehaviorBlock * FilterParams) \
        & (TrialAlignment & 'valid_alignment')

    def make(self, key):

        # fetch behavior sample rate and time vector
        fs_beh = (acquisition.BehaviorRecording & key).fetch1('behavior_recording_sample_rate')
        t_beh = (pacman_acquisition.Behavior.Condition & key).fetch1('condition_time')

        # select good trials with valid alignments within the current behavior block
        trial_rel = pacman_acquisition.Behavior.Trial \
            & GoodTrial \
            & (TrialAlignment & 'valid_alignment') \
            & (BehaviorBlock.SaveTag & key)

        # fetch spike rasters (ephys time base)
        spike_rasters = (MotorUnitSpikeRaster & trial_rel).fetch('motor_unit_spike_raster')

        if np.stack(spike_rasters).any():

            # resample time to ephys time base
            pre_pad_dur = (pacman_acquisition.ConditionParams.Target \
                & (pacman_acquisition.Behavior.Trial & key)).fetch1('target_pad_pre')

            fs_ephys = (acquisition.EphysRecording & key).fetch1('ephys_recording_sample_rate')
            t_ephys = (1 - round(fs_ephys * pre_pad_dur) + np.arange(round(fs_ephys/fs_beh) * len(t_beh)))/fs_ephys

            # rebin spike rasters to behavior time base 
            time_bin_edges = np.append(t_beh, t_beh[-1]+(1+np.arange(2))/fs_beh) - 1/(2*fs_beh)
            spike_bins = [np.digitize(t_ephys[rast], time_bin_edges) - 1 for rast in spike_rasters]
            spike_bins = [b[(b >= 0) & (b < len(t_beh))] for b in spike_bins]

            spike_rasters = [np.zeros(len(t_beh), dtype=bool) for i in range(len(spike_rasters))]
            for rast, b in zip(spike_rasters, spike_bins):
                rast[b] = 1

            # get filter kernel
            filter_parts = dju.getparts(processing.Filter, context=inspect.currentframe())
            filter_rel = next(part for part in filter_parts if part & key)

            # filter rebinned spike rasters
            rate = [fs_beh * filter_rel().filter(rast, fs_beh) for rast in spike_rasters]

            # trial average
            psth = np.stack(rate).mean(axis=0)

        else:
            psth = np.zeros(len(t_beh))

        # insert motor unit PSTH
        self.insert1(dict(**key, motor_unit_psth=psth))


@schema
class NeuronPsth(dj.Computed):
    definition = """
    # Peri-stimulus time histogram
    -> processing.Neuron
    -> BehaviorBlock
    -> FilterParams
    ---
    neuron_psth: longblob # neuron trial-averaged firing rate (spikes/s)
    """

    key_source = (processing.Neuron * BehaviorBlock * FilterParams) \
        & (TrialAlignment & 'valid_alignment')

    def make(self, key):

        # fetch behavior sample rate and time vector
        fs_beh = (acquisition.BehaviorRecording & key).fetch1('behavior_recording_sample_rate')
        t_beh = (pacman_acquisition.Behavior.Condition & key).fetch1('condition_time')

        # select good trials with valid alignments within the current behavior block
        trial_rel = pacman_acquisition.Behavior.Trial \
            & GoodTrial \
            & (TrialAlignment & 'valid_alignment') \
            & (BehaviorBlock.SaveTag & key)

        # fetch spike rasters (ephys time base)
        spike_rasters = (NeuronSpikeRaster & trial_rel).fetch('neuron_spike_raster')

        if np.stack(spike_rasters).any():

            # resample time to ephys time base
            pre_pad_dur = (pacman_acquisition.ConditionParams.Target \
                & (pacman_acquisition.Behavior.Trial & key)).fetch1('target_pad_pre')

            fs_ephys = (acquisition.EphysRecording & key).fetch1('ephys_recording_sample_rate')
            t_ephys = (1 - round(fs_ephys * pre_pad_dur) + np.arange(round(fs_ephys/fs_beh) * len(t_beh)))/fs_ephys

            # rebin spike rasters to behavior time base 
            time_bin_edges = np.append(t_beh, t_beh[-1]+(1+np.arange(2))/fs_beh) - 1/(2*fs_beh)
            spike_bins = [np.digitize(t_ephys[rast], time_bin_edges) - 1 for rast in spike_rasters]
            spike_bins = [b[(b >= 0) & (b < len(t_beh))] for b in spike_bins]

            spike_rasters = [np.zeros(len(t_beh), dtype=bool) for i in range(len(spike_rasters))]
            for rast, b in zip(spike_rasters, spike_bins):
                rast[b] = 1

            # get filter kernel
            filter_parts = dju.getparts(processing.Filter, context=inspect.currentframe())
            filter_rel = next(part for part in filter_parts if part & key)

            # filter rebinned spike rasters
            rate = [fs_beh * filter_rel().filter(rast, fs_beh) for rast in spike_rasters]

            # trial average
            psth = np.stack(rate).mean(axis=0)

        else:
            psth = np.zeros(len(t_beh))

        # insert neuron PSTH
        self.insert1(dict(**key, neuron_psth=psth))