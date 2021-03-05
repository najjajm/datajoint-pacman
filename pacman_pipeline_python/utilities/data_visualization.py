import re
import datajoint as dj
import numpy as np
from churchland_pipeline_python import acquisition
from njm2149_pipeline_python.pacman import preprocessing
from .. import pacman_acquisition
from .data_structures import TrialAveragedPopulationState
from typing import List

import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output
from jupyter_dash import JupyterDash
from plotly.subplots import make_subplots

def plot_state(
    population_state: TrialAveragedPopulationState,
    attribute_name: str,
    subspace_condition_set: str, 
    data_condition_sets: List[str], 
    condition_colors: dict,
    condition_set_ids: dict,
    condition_set_names: dict,
    eye_params: dict,
    component_numbers: list=[1,2,3], 
    show_legend: bool=False, 
    title_text: str=None,
    renderer: str='browser',
    ):

    # ===
    # n_rows = 1
    # n_cols = len(data_condition_sets)

    # project population_state into PC space
    population_state.project_pc_space(attribute_name, condition_set_ids[subspace_condition_set], soft_factor=5, max_components=10)
    coord_ranges = np.hstack((
        population_state.data_set[attribute_name + '_projection'].values.min(axis=1, keepdims=True), 
        population_state.data_set[attribute_name + '_projection'].values.max(axis=1, keepdims=True)
    ))
    coord_ratio = 3 * coord_ranges.ptp(axis=1) / coord_ranges.ptp(axis=1).sum()

    # axis label achors
    axis_anchor_coords = [
        coord_ranges[[0],[0]],
        coord_ranges[[1],[0]],
        coord_ranges[[2],[0]]
    ]
    axis_endpoint_coords = [
        np.array([0,1]) * 0.2 * coord_ranges[0,:].ptp(),
        np.array([0,1]) * 0.2 * coord_ranges[1,:].ptp(),
        np.array([0,1]) * 0.2 * coord_ranges[2,:].ptp(),
    ]

    # eye position
    eye_pos = eye_params[subspace_condition_set]['eye_dist'] \
        * np.array(eye_params[subspace_condition_set]['eye_angle']) \
        / np.linalg.norm(np.array(eye_params[subspace_condition_set]['eye_angle']), 2) ** 2

    # make figure
    # fig = make_subplots(
    #     rows=n_rows, 
    #     cols=n_cols, 
    #     subplot_titles=None,#[condition_set_names[cond_set] + ' conditions data' for cond_set in data_condition_sets],
    #     specs=[[{'type': 'scene'}] * n_cols] * n_rows
    # )

    fig = go.Figure()

    for data_set_idx, data_condition_set in enumerate(data_condition_sets):

        # overlay subspace condition set data in bw
        # if data_condition_set != subspace_condition_set:

        #     for cond_idx, cond_id in enumerate(condition_set_ids[subspace_condition_set]):

        #         X = population_state.data_set[attribute_name + '_projection'].sel(principal_component=component_numbers, condition_id=cond_id).values

        #         # trajectory
        #         fig.add_trace(
        #             go.Scatter3d(
        #                 x=X[0,:], 
        #                 y=X[1,:], 
        #                 z=X[2,:], 
        #                 marker=dict(color='rgba(0,0,0,0.25)', size=0.75), 
        #                 mode='markers', 
        #                 showlegend=False,
        #             ),
        #             # row=1, 
        #             # col=1+data_set_idx
        #         )

        # plot data sets in color
        for cond_idx, cond_id in enumerate(condition_set_ids[data_condition_set]):

            X = population_state.data_set[attribute_name + '_projection'].sel(principal_component=component_numbers, condition_id=cond_id).values

            # trajectory
            fig.add_trace(
                go.Scatter3d(
                    x=X[0,:], 
                    y=X[1,:], 
                    z=X[2,:], 
                    marker=dict(
                        color=np.arange(len(population_state.data_set.coords['time'].sel(condition_id=cond_id).values)),
                        colorscale=condition_colors[cond_id]['color_scale'],
                        size=2.5,
                    ), 
                    mode='markers', 
                    name=condition_colors[cond_id]['condition_label'],
                    legendgroup=str(cond_id),
                    showlegend=show_legend,
                ),
                # row=1, 
                # col=1+data_set_idx
            )

            # condition start
            fig.add_trace(
                go.Scatter3d(
                    x=X[[0],[0]],
                    y=X[[1],[0]],
                    z=X[[2],[0]],
                    marker=dict(
                        symbol='circle-open',
                        color=condition_colors[cond_id]['init_marker_color'],
                        size=10,
                    ),
                    mode='markers',
                    showlegend=False,
                    legendgroup=str(cond_id),
                ),
                # row=1, 
                # col=1+data_set_idx
            )

            # condition end
            fig.add_trace(
                go.Scatter3d(
                    x=X[[0],[-1]],
                    y=X[[1],[-1]],
                    z=X[[2],[-1]],
                    marker=dict(
                        symbol='square',
                        color=condition_colors[cond_id]['final_marker_color'],
                        size=10,
                    ),
                    mode='markers',
                    showlegend=False,
                    legendgroup=str(cond_id),
                ),
                # row=1, 
                # col=1+data_set_idx
            )

        for comp_idx, ax_comp in enumerate(component_numbers):

            fig.add_trace(
                go.Scatter3d(
                    x=(axis_anchor_coords[0] + axis_endpoint_coords[0] if comp_idx==0 else axis_anchor_coords[0] * np.array([1,1])),
                    y=(axis_anchor_coords[1] + axis_endpoint_coords[1] if comp_idx==1 else axis_anchor_coords[1] * np.array([1,1])),
                    z=(axis_anchor_coords[2] + axis_endpoint_coords[2] if comp_idx==2 else axis_anchor_coords[2] * np.array([1,1])),
                    mode='lines',
                    line=dict(
                        color='rgba(0,0,0,1)', #'rgba(255,255,255,1)',
                        width=7,
                    ),
                    showlegend=False,
                ),
                # row=1, 
                # col=1+data_set_idx
            )

            fig.add_trace(
                go.Scatter3d(
                    x=(axis_anchor_coords[0] + axis_endpoint_coords[0][1] if comp_idx==0 else axis_anchor_coords[0] * np.array([1,1])),
                    y=(axis_anchor_coords[1] + axis_endpoint_coords[1][1] if comp_idx==1 else axis_anchor_coords[1] * np.array([1,1])),
                    z=(axis_anchor_coords[2] + axis_endpoint_coords[2][1] if comp_idx==2 else axis_anchor_coords[2] * np.array([1,1])),
                    mode='text',
                    text='PC {}'.format(ax_comp),
                    textfont=dict(size=18),
                    showlegend=False,
                ),
                # row=1, 
                # col=1+data_set_idx
            )

        fig.update_scenes(
            xaxis=dict(range=1.15*coord_ranges[0,:], visible=False),
            yaxis=dict(range=1.15*coord_ranges[1,:], visible=False),
            zaxis=dict(range=1.15*coord_ranges[2,:], visible=False),
            camera=dict(eye=dict(x=eye_pos[0], y=eye_pos[1], z=eye_pos[2])),
            aspectratio=dict(x=coord_ratio[0], y=coord_ratio[1], z=coord_ratio[2]),
            # row=1,
            # col=1+data_set_idx
        )

    fig.update_layout(
        title=dict(text=title_text, font=dict(size=25)),
        legend=dict(font=dict(size=18)),
        template='simple_white',
        paper_bgcolor='rgba(0,0,0,0)',
    )

    if renderer is not None:
        fig.show(renderer=renderer)

    return fig

# class TrialAveragedPopulationStatePlotter:

#     def __init__(
#         self,
#         population_state: TrialAveragedPopulationState,
#         attribute_name: str,
#         theme: str='light',
#     ):
#         self.population_state = population_state
#         self.attribute_name = attribute_name
#         self.theme = theme
#         self._get_condition_set_data()
#         self._get_condition_set_colors()

#     @property
#     def theme(self):
#         return self._theme
#     @theme.setter
#     def theme(self, value):
#         color_params = {
#             'light': dict(alpha_range=(0.25,1), bw_range=(0.2,1), axis_color='rgba(0,0,0,1)', template='plotly_white'),
#             'dark': dict(alpha_range=(0,1), bw_range=(0,0.7), axis_color='rgba(255,255,255,1)', template='plotly_dark'),
#         }
#         assert value in color_params.keys(), 'Theme {} not recognized'.format(value)
#         self._alpha_range = color_params[value]['alpha_range']
#         self._bw_range = color_params[value]['bw_range']
#         self._axis_color = color_params[value]['axis_color']
#         self._template = color_params[value]['template']
#         self._theme = 'light'

#     @property
#     def alpha_range(self):
#         return self._alpha_range
#     @alpha_range.setter 
#     def alpha_range(self, value):
#         self._alpha_range = value
#         self._get_condition_set_colors()

#     @property
#     def bw_range(self):
#         return self._bw_range
#     @bw_range.setter 
#     def bw_range(self, value):
#         self._bw_range = value
#         self._get_condition_set_colors()

#     def _get_condition_set_data(self):
#         condition_set_table = self._get_condition_set_table()
#         self._condition_set_ids = (preprocessing.ConditionSet & condition_set_table).get_condition_set_ids()
#         self._condition_set_names = (preprocessing.ConditionSet & condition_set_table).get_condition_set_names()

#     def _get_condition_set_colors(self):
#         condition_set_table = self._get_condition_set_table()
#         self._condition_colors = (pacman_acquisition.ConditionParams & condition_set_table) \
#             .colorize_conditions(alpha_range=self.alpha_range, bw_range=self.bw_range)

#     def _get_condition_set_table(self):
#         condition_ids = np.unique(self.population_state.data_set['condition_id'].values)
#         return preprocessing.ConditionSet.Condition & [{'condition_id': cond_id} for cond_id in condition_ids]
        
#     def plot(
#         self,
#         subspace_condition_set: str,
#         data_condition_sets: list,
#         component_numbers: list=[1,2,3],
#         soft_normalization: int=0,
#         eye_angle: tuple=(1.25,1.25,1.25),
#         eye_distance: float=1,
#         orientation: str='horizontal',
#         subplot_titles: list=None,
#         figure_title: str=None,
#         show_legend: bool=True,
#         renderer: str='browser'
#         ):
#         self._subspace_condition_set = subspace_condition_set
#         self._data_condition_sets = data_condition_sets
#         self._component_numbers = component_numbers
#         self._soft_normalization = soft_normalization
#         self._eye_angle = eye_angle
#         self._eye_distance = eye_distance
#         self._orientation = orientation
#         self._subplot_titles = subplot_titles
#         self._figure_title = figure_title
#         self._show_legend = show_legend
#         self._renderer = renderer
#         self._project_population()
#         self._get_data_ranges()
#         self._get_axis_anchors()
#         self._update_eye_position()
#         self._set_layout_params()
#         self._make_figure()

#     def _project_population(self):
#         subspace_condition_ids = self._condition_set_ids[self._subspace_condition_set]
#         self.population_state.project_pc_space(
#             self.attribute_name, 
#             subspace_condition_ids, 
#             soft_factor=self._soft_normalization,
#             max_components=max(self._component_numbers)
#         )

#     def _get_data_ranges(self):
#         data_min = self.population_state.data_set[self.attribute_name + '_projection'].values.min(axis=1, keepdims=True)
#         data_max = self.population_state.data_set[self.attribute_name + '_projection'].values.max(axis=1, keepdims=True)
#         self._data_ranges = np.hstack((data_min, data_max))
#         self._coord_ratio = 3 * self._data_ranges.ptp(axis=1) / self._data_ranges.ptp(axis=1).sum()

#     def _get_axis_anchors(self):
#         self._axis_anchor_coords = [self._data_ranges[[i][0]] for i in range(3)]
#         self._axis_endpoint_coords = [np.array([0,1]) * 0.2 * self._data_ranges[i,:].ptp() for i in range(3)]

#     def _update_eye_position(self):
#         self._eye_pos = self._eye_distance * np.array(self._eye_angle) / np.linalg.norm(self._eye_angle, 2)**2

#     def _set_layout_params(self):
#         self._n_plots = len(self._data_condition_sets)
#         if self._orientation == 'horizontal':
#             self._n_rows = 1
#             self._n_columns = self._n_plots
#         elif self._orientation == 'vertical':
#             self._n_rows = self._n_plots
#             self._n_columns = 1
#         else:
#             self._n_columns = np.ceil(np.sqrt(self._n_plots)).astype(int)
#             self._n_rows = np.ceil(self._n_plots/self._n_columns).astype(int)

#     def _make_figure(self):

#         self._fig = make_subplots(
#             rows=self._n_rows,
#             cols=self._n_columns,
#             subplot_titles=self._subplot_titles,
#             specs=[[{'type': 'scene'}] * self._n_columns] * self._n_rows
#         )

#         for subplot_idx, data_condition_set in zip(np.ndindex((self._n_rows, self._n_columns)), self._data_condition_sets):
#             self._subplot_idx = subplot_idx
#             self._outline_subpace_conditions(data_condition_set)
#             self._plot_data_conditions(data_condition_set)

#         self._update_layout()
#         self._fig.show(renderer=self._renderer)

#         return self._fig

#     def _outline_subpace_conditions(self, condition_set: str):
#         if condition_set != self._subspace_condition_set:
#             self._plot_conditions(self._subspace_condition_set, outlines=True)

#     def _plot_data_conditions(self, condition_set: str):
#         self._plot_conditions(condition_set, outlines=False)

#     def _plot_conditions(self, condition_set: str, outlines: bool):
#         for condition_idx, condition_id in enumerate(self._condition_set_ids[condition_set]):
#             X = self.population_state.data_set[self.attribute_name + '_projection'] \
#                 .sel(principal_component=self._component_numbers, condition_id=condition_id).values
                
#             if outlines:
#                 marker_params = {
#                     'mode': 'markers',
#                     'marker': dict(
#                         color='rgba(0,0,0,0.25)', 
#                         size=0.5,
#                     ),
#                     'showlegend': False,
#                 }
#             else:
#                 t = self.population_state.data_set.coords['time'].sel(condition_id=cond_id).values
#                 marker_params = {
#                     'mode': 'markers',
#                     'marker': dict(
#                         color=np.arange(len(t)),
#                         colorscale=self._condition_colors[condition_id]['color_scale'],
#                         size=2,
#                     ),
#                     'name': self._condition_colors[condition_id]['condition_label'],
#                     'showlegend': True if self._show_legend else False,
#                     'legendgroup': str(condition_id),
#                 }

#             self._plot_trace_3d(X, marker_params)
#             self._plot_endpoint(condition_id, 'start')
#             self._plot_endpoint(condition_id, 'end')

#         self._plot_axes()
#         self._update_scene()

#     def _plot_endpoint(self, condition_id: int, endpoint_pos: str):
#         X = population_state.data_set[attribute_name + '_projection'] \
#             .sel(principal_component=self._component_numbers, condition_id=condition_id).values

#         if endpoint_pos == 'start':
#             X = X[:,[0]]
#             marker_symbol = 'circle-open'
#             marker_color = self._condition_colors[condition_id]['init_marker_color']
#         else:
#             X = X[:,[-1]]
#             marker_symbol = 'square'
#             marker_color = self._condition_colors[condition_id]['final_marker_color']

#         marker_params = {
#             'mode': 'markers',
#             'marker': dict(
#                 symbol=marker_symbol,
#                 color=marker_color,
#                 size=8,
#             ),
#             'showlegend': False,
#             'legendgroup': str(condition_id),
#         }

#         self._plot_trace_3d(X, marker_params)

#     def _plot_axes(self):
#         for comp_idx, comp_num in enumerate(self._component_numbers):
#             self._plot_component_axis(comp_idx)
#             self._label_component_axis(comp_idx, comp_num)

#     def _plot_component_axis(self, comp_idx):
#         X = []
#         for i in range(3):
#             if comp_idx == i:
#                 X.append(self._axis_anchor_coords[i] + self._axis_endpoint_coords[i])
#             else:
#                 X.append(self._axis_anchor_coords[i] * np.array([1,1]))
#         marker_params = {
#             'mode': 'lines',
#             'line': dict(
#                 color=self._axis_color,
#                 width=7,
#             ),
#             'showlegend': False,
#         }
#         self._plot_trace_3d(np.vstack(X), marker_params)

#     def _label_component_axis(self, comp_idx, comp_num):
#         X = []
#         for i in range(3):
#             if comp_idx == i:
#                 X.append(self._axis_anchor_coords[i][1] + self._axis_endpoint_coords[i])
#             else:
#                 X.append(self._axis_anchor_coords[i] * np.array([1,1]))
#         marker_params = {
#             'mode': 'text',
#             'text': 'PC {}'.format(comp_num),
#             'textfont': dict(size=16),
#             'showlegend': False
#         }
#         self._plot_trace_3d(np.vstack(X), marker_params)

#     def _plot_trace_3d(self, X, marker_params):
#         self._fig.add_trace(go.Scatter3d(x=X[0,:], y=X[1,:], z=X[2,:], **marker_params),
#             row=1+self._subplot_idx[0], col=1+self._subplot_idx[1])

#     def _update_scene(self):
#         self._fig.update_scenes(
#             xaxis=dict(range=1.15 * self._data_ranges[0,:], visible=False),
#             yaxis=dict(range=1.15 * self._data_ranges[1,:], visible=False),
#             zaxis=dict(range=1.15 * self._data_ranges[2,:], visible=False),
#             # camera=dict(eye=dict(x=self._eye_pos[0], y=self._eye_pos[1], z=self._eye_pos[2])),
#             aspectratio=dict(x=self._coord_ratio[0], y=self._coord_ratio[1], z=self._coord_ratio[2]),
#             row=1+self._subplot_idx[0],
#             col=1+self._subplot_idx[1],
#         )

#     def _update_layout(self):
#         self._fig.update_layout(
#             title=dict(text=self._figure_title, font=dict(size=25)),
#             legend=dict(font=dict(size=18)),
#             template=self._template,
#         )