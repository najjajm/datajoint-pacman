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

schema = dj.schema(dj.config.get('database.prefix') + 'churchland_analyses_pacman_behavior')

# =======
# LEVEL 0
# =======

@schema
class Force(dj.Computed):
    definition = """
    # Single trial force
    -> pacman_processing.BehaviorBlock
    -> pacman_processing.TrialAlignment
    -> pacman_processing.FilterParams
    ---
    -> pacman_processing.GoodTrial
    force_raw:  longblob # raw (online), aligned force signal (V)
    force_filt: longblob # filtered, aligned, and calibrated force (N)
    """

    key_source = pacman_processing.BehaviorBlock \
        * (pacman_processing.TrialAlignment & 'valid_alignment') \
        * pacman_processing.FilterParams \
        & (pacman_acquisition.Behavior.Trial * pacman_processing.BehaviorBlock.SaveTag)

    def make(self, key):

        # convert raw force signal to Newtons
        trial_rel = pacman_acquisition.Behavior.Trial & key
        force = trial_rel.processforce(data_type='raw', filter=False)

        # get filter kernel
        filter_key = (processing.Filter & (pacman_processing.FilterParams & key)).fetch1('KEY')
        filter_parts = datajointutils.getparts(processing.Filter, context=inspect.currentframe())
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
            force_filt=force_filt_align,
            behavior_quality_params_id=(pacman_processing.BehaviorQualityParams & key).fetch1('behavior_quality_params_id'),
            good_trial=(pacman_processing.GoodTrial & key).fetch1('good_trial')
        )

        self.insert1(key)

    def plot(
        self, 
        group_by: List[str]=['behavior_block_id', 'condition_id'], 
        stack_by: List[str]=['trial'],
        plot_mean: bool = False,
        plot_ste: bool=False,
        plot_target: bool=False,
        trial_type: Tuple[str]='good',
        figsize: Tuple[int,int]=(12,8),
        n_rows: int=None,
        n_columns: int=None,
        y_tick_step: int=4,
        limit_figures: int=None,
        limit_subplots: int=None,
        limit_lines: int=None
    ) -> None:
        """Plot force trials."""

        # standardize input format
        if isinstance(trial_type, str):
            trial_type = (trial_type,)

        # check inputs
        assert set(group_by) <= set(self.primary_key), 'Group attribute {} not in primary key'.format(group_by)
        assert set(stack_by) <= set(self.primary_key), 'Stack attribute {} not in primary key'.format(group_by)
        assert set(trial_type) <= {'good','bad'},      'Unrecognized trial type {}'.format(trial_type)

        # filter by trial type
        if 'good' in trial_type and not 'bad' in trial_type:
            self = self & 'good_trial'
        
        elif 'bad' in trial_type and not 'good' in trial_type:
            self = self - 'good_trial'

        # get figure keys by non-grouping or stacking attributes
        separate_by = [attr for attr in self.primary_key if attr not in group_by + stack_by]
        figure_keys = (dj.U(*separate_by) & self).fetch('KEY')

        # downsample figure keys
        if limit_figures:
            figure_keys = figure_keys[:min(len(figure_keys),limit_figures)]

        # attribute title dictionary
        title_text = {
            'session_date': r'Session {}',
            'condition_id': r'Condition {}',
            'behavior_block_id': r'Block {}',
            'trial': r'Trial {}'
        }

        #== LOOP FIGURES ==
        for fig_key in figure_keys:
            
            # get subplot keys as unique grouping attributes 
            if group_by:
                subplot_keys = (dj.U(*group_by) & (self & fig_key)).fetch('KEY')
            else:
                subplot_keys = [fig_key]

            # downsample subplot keys
            if limit_subplots:
                subplot_keys = subplot_keys[:min(len(subplot_keys),limit_subplots)]

            n_subplots = len(subplot_keys)

            # setup page
            if not (n_columns or n_rows):
                n_columns = np.ceil(np.sqrt(n_subplots)).astype(int)
                n_rows = np.ceil(n_subplots/n_columns).astype(int)

            elif n_columns and not n_rows:
                n_rows = np.ceil(n_subplots/n_columns).astype(int)

            elif n_rows and not n_columns:
                n_columns = np.ceil(n_subplots/n_rows).astype(int)

            else:
                subplot_keys = subplot_keys[:min(len(subplot_keys), n_rows*n_columns)]

            # create axes handles and ensure indexable
            fig, axs = plt.subplots(n_rows, n_columns, figsize=figsize, sharey=True)

            if n_rows == 1 and n_columns == 1:
                axs = np.array(axs)
            
            axs = axs.reshape((n_rows, n_columns))

            #== LOOP SUBPLOTS ==
            for idx, plot_key in zip(np.ndindex((n_rows, n_columns)), subplot_keys):

                # get line keys as unique stacking attributes
                if stack_by:
                    line_keys = (dj.U(*stack_by) & (self & fig_key & plot_key)).fetch('KEY')
                else:
                    line_keys = {}

                # fetch condition time and target force
                t, target_force = (pacman_acquisition.Behavior.Condition & fig_key & plot_key)\
                    .fetch1('condition_time', 'condition_force')

                # get trial forces
                trial_forces = (self & fig_key & plot_key & line_keys).fetch('force_filt')
                trial_forces = np.stack(trial_forces)

                # plot trials
                axs[idx].plot(t, trial_forces.T, 'k');

                # plot mean
                if plot_mean:
                    axs[idx].plot(t, trial_forces.mean(axis=0), 'b');

                # plot standard error
                if plot_ste and len(line_keys) > 1:
                    mu = trial_forces.mean(axis=0)
                    std = trial_forces.std(axis=0, ddof=1)
                    ste = std / trial_forces.shape[0]
                    axs[idx].plot(t, mu + ste, 'b--');
                    axs[idx].plot(t, mu - ste, 'b--');

                # plot target force
                if plot_target:
                    axs[idx].plot(t, target_force, 'c');

                # format axes
                axs[idx].set_xlim(t[[0,-1]])
                y_lim = axs[idx].get_ylim()
                axs[idx].set_ylim([
                    min(0, min(y_lim[0], np.floor(trial_forces.min() / y_tick_step) * y_tick_step)), 
                    max(y_lim[1], np.ceil(trial_forces.max() / y_tick_step) * y_tick_step)
                ])
                y_lim = axs[idx].get_ylim()
                axs[idx].set_yticks(np.arange(y_lim[0], y_lim[1]+y_tick_step, y_tick_step))
                axs[idx].set_xlabel('time (s)');
                axs[idx].set_ylabel('force (N)');
                [axs[idx].spines[edge].set_visible(False) for edge in ['top','right']];

                # add subplot title
                axs[idx].set_title(
                    '. '.join([title_text[key].format(plot_key[key]) for key in plot_key.keys() if key in title_text.keys()])
                )

            # adjust subplot layout
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])

            # add figure title
            if n_subplots > 1:
                fig.suptitle(
                    '. '.join([title_text[key].format(fig_key[key]) for key in fig_key.keys() if key in title_text.keys()])
                )