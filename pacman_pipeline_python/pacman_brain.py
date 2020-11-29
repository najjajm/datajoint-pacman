import datajoint as dj
import os, inspect, itertools
import pandas as pd
import numpy as np
import neo
import matplotlib.pyplot as plt
import matplotlib.colors as mplcol
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
    -> pacman_processing.EphysTrialAlignment
    ---
    neuron_spike_raster: longblob # neuron trial-aligned spike raster (boolean array)
    """

    key_source = processing.Neuron \
        * pacman_processing.EphysTrialAlignment \
        & (pacman_processing.BehaviorTrialAlignment & 'valid_alignment')

    def make(self, key):

        # fetch ephys alignment indices for the current trial
        ephys_alignment = (pacman_processing.EphysTrialAlignment & key).fetch1('ephys_alignment')

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

        key.update(neuron_spike_raster=spike_raster)

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
    neuron_rate: longblob # neuron trial-aligned firing rate (spikes/s)
    """

    # process per neuron/condition
    key_source = processing.Neuron \
        * pacman_acquisition.Behavior.Condition \
        * pacman_processing.FilterParams \
        & NeuronSpikeRaster

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
        spike_raster_keys = (NeuronSpikeRaster & key).fetch(as_dict=True)

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
                neuron_rate = fs_beh * filter_rel().filt(neuron_rate_key['neuron_spike_raster'], fs_beh)
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
    -> pacman_processing.AlignmentParams
    -> pacman_processing.BehaviorBlock
    -> pacman_processing.BehaviorQualityParams
    -> pacman_processing.FilterParams
    ---
    neuron_psth:     longblob # neuron trial-averaged firing rate (spikes/s)
    neuron_psth_sem: longblob # neuron firing rate standard error (spikes/s)
    """

    # limit conditions with good trials
    key_source = processing.Neuron \
        * pacman_processing.AlignmentParams \
        * pacman_processing.BehaviorBlock \
        * pacman_processing.BehaviorQualityParams \
        * pacman_processing.FilterParams \
        & NeuronRate \
        & (pacman_processing.GoodTrial & 'good_trial')


    def make(self, key):

        # fetch single-trial firing rates
        rates = (NeuronRate & key & (pacman_processing.GoodTrial & 'good_trial')).fetch('neuron_rate')
        rates = np.stack(rates)

        # update key with psth and standard error
        key.update(
            neuron_psth=rates.mean(axis=0),
            neuron_psth_sem=rates.std(axis=0, ddof=(1 if rates.shape[0] > 1 else 0))/np.sqrt(rates.shape[0])
        )

        # insert neuron PSTH
        self.insert1(key)


    def get_common_conditions(
        self,
        include_time: bool=False,
        include_force: bool=False,
        n_sigfigs: int=2,
    ) -> List[dict]:
        """Retrieves the greatest common set of condition attributes across the population.
        By default, returns the condition IDs along with their label and rank.
        Can optionally include the condition time and/or force vectors."""

        # identify common conditions across recording sessions
        neurons_per_condition = dj.U('condition_id').aggr(self, count='count(*)')
        common_conditions = (neurons_per_condition & 'count={}'.format(len(processing.Neuron & self))).proj()

        # condition table with labels and ranks
        condition_table = pacman_acquisition.ConditionParams().proj_rank() \
            * pacman_acquisition.ConditionParams().proj_label(n_sigfigs=n_sigfigs) \
            & common_conditions

        # aggregate secondary attributes to include
        secondary_attributes = []
        secondary_attributes.append('condition_time') if include_time else None
        secondary_attributes.append('condition_force') if include_force else None

        if any(secondary_attributes):

            # ensure matched sample rates across sessions
            behavior_recordings = acquisition.BehaviorRecording & self
            assert len(dj.U('behavior_recording_sample_rate') & behavior_recordings) == 1, 'Mismatched sample rates!'

            # join condition table with secondary attributes
            session_conditions = pacman_acquisition.Behavior.Condition & self
            condition_table = condition_table * (dj.U('condition_id', *secondary_attributes) & session_conditions)

        # fetch and return ordered condition attributes
        return condition_table.fetch('KEY', order_by='condition_rank')
        

    # =================
    def plot_heat(
        self, 
        soft_normalize: int=5, 
        z_score: bool=False, 
        condition_sort_idx=0,
        fig_size=(24, 14),
        color_map='plasma'
        ):

        # get common condition attributes
        condition_attributes = self.get_common_conditions(include_time=True, include_force=True)

        # fetch neuron keys
        unit_keys = (processing.Neuron & self).fetch('KEY', order_by=['session_date', 'neuron_id'])

        # fetch psths and stack across units
        psths = [np.stack((self & cond_attr).fetch('neuron_psth', order_by=['session_date', 'neuron_id'])) \
            for cond_attr in condition_attributes]

        # soft-normalize
        if soft_normalize is not None:

            psth_range = np.hstack(psths).ptp(axis=1, keepdims=True)
            psths = [X/(psth_range + 5) for X in psths]

        # z-score
        if z_score:

            psth_mean = np.hstack(psths).mean(axis=1, keepdims=True)
            psth_std = np.hstack(psths).std(axis=1, ddof=0, keepdims=True)
            psths = [(X-psth_mean)/psth_std for X in psths]

        # firing rate range across conditions
        psth_ranges = np.array([(X.min().min(), X.max().max()) for X in psths])
        psth_range_cross_cond = (int(np.floor(psth_ranges[:,0].min())), int(np.ceil(psth_ranges[:,1].max())))

        # plot order
        psth_amplitude = psths[condition_sort_idx].sum(axis=1)
        unit_idx = np.arange(psths[0].shape[0])
        unit_order = [iidx for x, iidx in sorted(zip(psth_amplitude, unit_idx), reverse=True)]

        # time range and width ratios
        cmap_prop = 0.25
        time_range = [x['condition_time'].ptp() for x in condition_attributes]
        time_range = time_range + [np.array(time_range).min() * cmap_prop]
        width_ratios = time_range / sum(time_range)

        # setup figure
        n_rows = 2
        n_columns = 1 + len(condition_attributes)

        fig = plt.figure(figsize=fig_size, constrained_layout=True)
        gs = fig.add_gridspec(ncols=n_columns, nrows=n_rows, width_ratios=width_ratios, height_ratios=[0.075, 0.925])

        # set colormap
        if isinstance(color_map, str):
            cmap = color_map
        else:
            cmap = mplcol.LinearSegmentedColormap.from_list('cmap', color_map)

        for nd_idx in np.ndindex(n_rows, n_columns):

            if nd_idx[1] < n_columns-1:

                # make subplot
                ax = fig.add_subplot(gs[nd_idx])
            
                # condition time vector
                t = condition_attributes[nd_idx[1]]['condition_time']

                # x-tick position indices
                x_tick_pos = [iidx for iidx, ti in enumerate(t) if ti==round(ti)]
                x_tick_vals = ['{}'.format(int(x)) for x in t[x_tick_pos]]

                if len(x_tick_pos) > 3:
                    x_tick_pos = x_tick_pos[1:-1]
                    x_tick_vals = x_tick_vals[1:-1]
                
                if nd_idx[0] == 0:

                    # plot condition force profile
                    ax.plot(t, condition_attributes[nd_idx[1]]['condition_force'], c='k')

                    # format axes
                    ax.set_xlim(t[[0, -1]])
                    ax.set_ylim([0, 20])
                    ax.set_xticks([])
                    ax.set_yticks([])
                    [ax.spines[edge].set_visible(False) for edge in ['top','right','bottom','left']]

                else:
                    # plot heatmap
                    im = ax.imshow(
                        psths[nd_idx[1]][unit_order,:], 
                        aspect='auto', 
                        cmap=cmap, 
                        vmin=(0 if soft_normalize is not None else psth_range_cross_cond[0]), 
                        vmax=(1 if soft_normalize is not None else psth_range_cross_cond[1])
                    )

                    # format axes
                    ax.set_xticks(x_tick_pos)
                    ax.set_xticklabels(x_tick_vals, fontsize='large')
                    ax.set_yticks([])
                    ax.set_xlabel('time (s)', fontsize='xx-large')

            elif nd_idx[1] == n_columns-1 and nd_idx[0] == 1:

                ax = fig.add_subplot(gs[nd_idx])

        cbar = fig.colorbar(im, cax=ax);
        cbar.ax.tick_params(labelsize='large')

        fig.show()

        # return sorted unit keys
        return fig, np.array(unit_keys)[unit_order]


