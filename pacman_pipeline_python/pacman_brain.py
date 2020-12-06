import datajoint as dj
import os, inspect, itertools
import pandas as pd
import numpy as np
import scipy
import neo
import matplotlib.pyplot as plt
import matplotlib.colors as mplcol
from churchland_pipeline_python import lab, acquisition, processing
from churchland_pipeline_python.utilities import datajointutils
from . import pacman_acquisition, pacman_processing
from sklearn import decomposition
from typing import Any, List, Tuple

import plotly.graph_objects as go
import dash_core_components as dcc
import dash_html_components as html
from jupyter_dash import JupyterDash
from dash.dependencies import Input, Output
from plotly.subplots import make_subplots

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
        

    def fetch_psths(
        self,
        fs: int=None,
        soft_normalize: int=None,
        mean_center: bool=False,
        output_format: str='array',
    ) -> (Any, Any, Any, List[dict], List[dict]):
        """Fetch PSTHs.

        Args:
            fs (int, optional): Sample rate. If not None, or if different sample rates across recordings, resamples PSTHs to new rate. Defaults to None.
            soft_normalize (int, optional): If not None, normalizes data with this value added to the firing rate range. Defaults to None.
            mean_center (bool, optional): Whether to subtract the cross-condition mean from the responses. Defaults to False.
            output_format (str, optional): Output data format. Options: 
                * 'array' (N x CT) [Default]
                * 'dict' (list of dictionaries per neuron/condition)
                * 'list' (list of N x T arrays, one per condition)

        Returns:
            psths (Any): PSTHs in specified output format
            condition_ids (Any): Condition IDs for each sample in X
            condition_times (Any): Condition time value for each sample in X
            condition_keys (List[dict]): List of condition keys in the dataset
            neuron_keys (List[dict]): List of neuron keys in the dataset
        """

        # ensure that there is one PSTH per neuron/condition
        neuron_condtion_keys = processing.Neuron.primary_key + pacman_acquisition.ConditionParams.primary_key
        remaining_keys = list(set(self.primary_key) - set(neuron_condtion_keys))
        
        n_psths_per_condition = dj.U(*neuron_condtion_keys).aggr(self, count='count(*)')
        assert not(n_psths_per_condition & 'count > 1'), 'More than one PSTH per neuron and condition. Check ' \
            + (', '.join(['{}'] * len(remaining_keys))).format(*remaining_keys)

        # get condition keys
        condition_keys = pacman_acquisition.ConditionParams().get_common_attributes(self, include=['label','rank','time','force'])

        # get neuron keys
        neuron_keys = (processing.Neuron & self).fetch('KEY')

        # remove standard errors from table
        self = self.proj('neuron_psth')

        # ensure matched sample rates across the population and with desired sample rate
        unique_sample_rates = (dj.U('behavior_recording_sample_rate') & (acquisition.BehaviorRecording & self)) \
            .fetch('behavior_recording_sample_rate')

        if len(unique_sample_rates) > 1 or (fs is not None and not all(unique_sample_rates == fs)):

            # use modal sample rate if multiple in dataset
            if fs is None:
                fs_mode, _ = scipy.stats.mode(unique_sample_rates)
                fs = fs_mode[0]

            # join psth table with condition table
            self *= pacman_acquisition.Behavior.Condition.proj(t_old='condition_time')

            psths = []
            for cond_key in condition_keys:

                # make new time vector
                t_new, _ = pacman_acquisition.ConditionParams.target_force_profile(cond_key['condition_id'], fs)
                cond_key.update(condition_time=t_new)

                # fetch psth data
                psth_data = [(self & cond_key & unit_key).fetch1() for unit_key in neuron_keys]

                # interpolate psths to new timebase as needed
                if fs is not None:
                    [X.update(neuron_psth=np.interp(t_new, X['t_old'], X['neuron_psth'])) for X in psth_data];

                # extract psths and append to list
                psths.append(np.array([X['neuron_psth'] for X in psth_data]))

        else:
            # fetch psths and stack across units
            psths = []
            for cond_key in condition_keys:
                psths.append(np.stack(
                    [(self & cond_key & unit_key).fetch1('neuron_psth') for unit_key in neuron_keys]
                ))

        # label each time step in concatenated population vector with condition index
        condition_ids = [(cond_key['condition_id'], ) * X.shape[1] for cond_key, X in zip(condition_keys, psths)]

        # extract condition times from keys
        condition_times = [cond_key['condition_time'] for cond_key in condition_keys]

        # soft normalize
        if soft_normalize is not None:
            rate_range = np.hstack(psths).ptp(axis=1, keepdims=True)
            psths = [X/(rate_range + soft_normalize) for X in psths]

        # mean-center
        if mean_center:
            rate_mean = np.hstack(psths).mean(axis=1, keepdims=True)
            psths = [X - rate_mean for X in psths]
        
        # format output
        if output_format == 'array':

            # stack output across conditions and times
            psths = np.hstack(psths)
            condition_ids = np.hstack(condition_ids)
            condition_times = np.hstack(condition_times)

        elif output_format == 'dict':

            # aggregate data into a dict
            psth_data = []
            for cond_key, X in zip(condition_keys, psths):
                for unit_key, Xi in zip(neuron_keys, X):
                    psth_data.append(dict(cond_key, **unit_key, neuron_psth=Xi))

            psths = psth_data

        return psths, condition_ids, condition_times, condition_keys, neuron_keys

        
    def plot_heat(
        self, 
        soft_normalize: int=5, 
        z_score: bool=False, 
        condition_sort_idx=0,
        fig_size=(24, 14),
        color_map='plasma'
        ):

        # get common condition attributes
        condition_attributes = pacman_acquisition.ConditionParams() \
            .get_common_attributes(self, n_sigfigs=2, include=['label','rank','time','force'])

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


    def dash_app(self, mode='inline'):

        # === SETUP ===

        # session dates
        session_dates = (acquisition.Session & self).fetch('session_date', as_dict=True)
        [date.update(session_date=date['session_date'].strftime('%Y-%m-%d')) for date in session_dates];

        # get common condition attributes
        condition_attributes = pacman_acquisition.ConditionParams() \
            .get_common_attributes(self, n_sigfigs=2, include=['label','rank','time','force'])

        # maximum force
        max_force = np.array([
            cond_attr['condition_force'].max() for cond_attr in condition_attributes
        ]).max()

        # === APP LAYOUT ===

        app = JupyterDash(__name__)
        app.layout = html.Div([
            html.H1('PSTH Explorer', style={'color':'white'}),
            dcc.Graph(id='graph'),
            html.Div([
                dcc.Dropdown(
                    id='condition-dropdown',
                    options=[
                        {'label': 'Condition {}: '.format(cond_idx) + cond_attr['condition_label'], 'value': cond_idx}
                        for cond_idx, cond_attr in enumerate(condition_attributes)
                    ],
                    value=0,
                    disabled=False,
                ),
                dcc.Slider(
                    id='session-slider',
                    min=0,
                    max=len(session_dates)-1,
                    value=0,
                    marks={i: str(i) for i in range(len(session_dates))},
                    disabled=False,
                ),
            ]),
        ])

        # === CALLBACKS ===

        # Update graph
        @app.callback(
            Output('graph', 'figure'),
            [
                Input('session-slider', 'value'),
                Input('condition-dropdown', 'value'),
            ]
        )
        def update_figure(session_idx, condition_idx):

            # fetch psth attributes
            cond_attr = condition_attributes[condition_idx]
            psth_attributes = (self & session_dates[session_idx] & cond_attr).fetch(as_dict=True)

            # maximum firing rate
            max_rate = np.array([
                (psth_attr['neuron_psth'] + psth_attr['neuron_psth_sem']).max() for psth_attr in psth_attributes
            ]).max()

            # make figure
            n_units = len(psth_attributes)
            n_columns = int(np.ceil(np.sqrt(n_units)))
            n_rows = int(np.ceil(n_units/n_columns))

            # add a row for the condition force profile
            n_rows += 1

            fig = make_subplots(
                rows=n_rows, 
                cols=n_columns, 
                shared_xaxes='all',
                shared_yaxes='rows',
            )

            # plot forces
            for nd_idx in np.ndindex((1, n_columns)):

                fig.add_trace(
                    go.Scatter(
                        x=cond_attr['condition_time'],
                        y=cond_attr['condition_force'],
                        showlegend=False,
                        line=dict(color='FireBrick'),
                        name='target force',
                    ),
                    row=1+nd_idx[0],
                    col=1+nd_idx[1],
                )

                # update axes
                fig.update_yaxes(
                    title_text=('force (N)' if nd_idx[1]==0 else None),
                    range=[-2, int(np.ceil(1.15*max_force))],
                    row=1+nd_idx[0],
                    col=1+nd_idx[1],
                )

            # plot PSTHs
            valid_nd_indices = list(np.ndindex((n_rows-1, n_columns)))[:n_units]

            for nd_idx, psth_attr in zip(np.ndindex((n_rows-1, n_columns)), psth_attributes):

                # labels
                fig.add_trace(
                    go.Scatter(
                        x=[cond_attr['condition_time'][-1] - cond_attr['condition_time'].ptp() * 0.05],
                        y=[max_rate],
                        legendgroup=0,
                        mode='text',
                        name='neuron IDs',
                        text='{}'.format(psth_attr['neuron_id']),
                        textposition='bottom left',
                        showlegend=(True if nd_idx==(0,0) else False),
                    ),
                    row=2+nd_idx[0],
                    col=1+nd_idx[1],
                )

                # standard error
                for a in [-1, 1]:

                    fig.add_trace(
                        go.Scatter(
                            x=cond_attr['condition_time'],
                            y=psth_attr['neuron_psth'] + a * psth_attr['neuron_psth_sem'],
                            legendgroup=1,
                            name='standard error',
                            showlegend=(True if a==-1 and nd_idx==(0,0) else False),
                            line=dict(color='LightSkyBlue'),
                            fill=(None if a==-1 else 'tonexty'),
                        ),
                        row=2+nd_idx[0],
                        col=1+nd_idx[1],
                    )

                # mean
                fig.add_trace(
                    go.Scatter(
                        x=cond_attr['condition_time'],
                        y=psth_attr['neuron_psth'],
                        legendgroup=2,
                        name='mean',
                        showlegend=(True if nd_idx==(0,0) else False),
                        line=dict(color='RoyalBlue'),
                    ),
                    row=2+nd_idx[0],
                    col=1+nd_idx[1],
                )

                # update axes
                fig.update_xaxes(
                    title_text=('time (s)' if (nd_idx[0]+1,nd_idx[1]) not in valid_nd_indices else None),
                    row=2+nd_idx[0], 
                    col=1+nd_idx[1],
                )
                fig.update_yaxes(
                    title_text=('spks/s' if nd_idx[1]==0 else None),
                    range=[0, int(np.ceil(max_rate/10)*10)],
                    row=2+nd_idx[0],
                    col=1+nd_idx[1],
                )

            # update figure layout
            fig.update_layout(
                height=800, 
                width=1200, 
                legend_traceorder='reversed', 
                title_text='{}'.format(session_dates[session_idx]['session_date'])
            )

            return fig

        # === RUN APP ===
        app.run_server(mode=mode)
