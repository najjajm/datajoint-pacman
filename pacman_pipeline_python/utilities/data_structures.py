import itertools, re
import datajoint as dj
import numpy as np
import pandas as pd
import xarray as xr
from churchland_pipeline_python import acquisition
from .. import pacman_acquisition

class TrialAveragedPopulationState:

    def __init__(
        self,
        table,
        population_name: str,
        attribute_names: list,
        sample_rate: int=1000,
        condition_order: list=None,
        suppress_warnings: bool=False,
    ):
        self.population_name = population_name
        self.attribute_names = attribute_names
        self.sample_rate = sample_rate
        self.condition_order = condition_order
        self.suppress_warnings = suppress_warnings
        self._get_population_state(table)
        
    def _get_population_state(self, table):
        self._fetch_condition_attributes(table)
        self._fetch_population_keys(table)
        self._fetch_data_attributes(table)
        self._resample_data_attributes()
        self._get_condition_time_multi_index()
        self._get_data_keys()
        self._get_data_set()

    def _fetch_condition_attributes(self, table):
        self.condition_attributes = (pacman_acquisition.ConditionParams & table).fetch(as_dict=True)
        self._order_condition_attributes()
        for cond_idx, cond_attr in enumerate(self.condition_attributes):
            t, _ = pacman_acquisition.ConditionParams.target_force_profile(cond_attr['condition_id'], self.sample_rate)
            cond_attr.update(condition_index=cond_idx, condition_time=t)

    def _order_condition_attributes(self):
        if self.condition_order is not None:
            fetched_ids = {attr['condition_id'] for attr in self.condition_attributes}
            unorderd_ids = fetched_ids.difference(set(list(self.condition_order)))
            if any(unorderd_ids):
                print('Missing condition IDs in order. Using default ordering.') if not self.suppress_warnings else None
            else:
                attr_by_id = {attr['condition_id']: attr for attr in self.condition_attributes}
                self.condition_attributes = [attr_by_id[condition_id] for condition_id in self.condition_order]

    def _fetch_population_keys(self, table):
        self.population_keys = (dj.U(self.population_name) & table).fetch(as_dict=True)

    def _fetch_data_attributes(self, table):
        try:
            table.proj('old_sample_rate')
        except Exception:
            if not self.suppress_warnings:
                print(
                    'Assuming all entries were sampled at the "behavior recording" rate. ' +
                    'Project attribute onto table called "old_sample_rate" to override or set suppress_warnings=True.'
                )
            table *= acquisition.BehaviorRecording.proj(old_sample_rate='behavior_recording_sample_rate')
        self.data_attributes = table.proj(*(self.attribute_names + ['old_sample_rate'])).fetch(as_dict=True)

    def _resample_data_attributes(self):
        condition_times = {
            (attr['condition_id'], self.sample_rate): attr['condition_time'] for attr in self.condition_attributes
        }

        for attr in self.data_attributes:
            if attr['old_sample_rate'] != self.sample_rate:
                condition_time_key = (attr['condition_id'], attr['old_sample_rate'])
                try:
                    t_old = condition_times[condition_time_key]
                except KeyError:
                    t_old, _ = pacman_acquisition.ConditionParams.target_force_profile(attr['condition_id'], attr['old_sample_rate'])
                    condition_times.update({condition_time_key: t_old})
                finally:
                    t_new = condition_times[(attr['condition_id'], self.sample_rate)]
                    for name in self.attribute_names:
                        # slower than np.array_equal(x, x.astype(bool)), but more robust to different data types
                        is_boolean = all(x in [0,1] for x in attr[name].astype(float))

                        if is_boolean:
                            attr[name] = self._rebin(attr[name], t_old, t_new, self.sample_rate)
                        else:
                            attr[name] = np.interp(t_new, t_old, attr[name])

    def _rebin(self, raster, t_old, t_new, fs_new):
        t_bins = np.concatenate((t_new[:-1,np.newaxis], t_new[1:,np.newaxis]), axis=1).mean(axis=1)
        t_bins = np.append(np.insert(t_bins, 0, -np.Inf), np.Inf)

        new_spike_indices = np.digitize(t_old[raster], t_bins)
        return np.array([True if i in new_spike_indices else False for i in range(len(t_new))])

    def _get_condition_time_multi_index(self):
        times = [attr['condition_time'] for attr in self.condition_attributes]
        condition_ids = [(attr['condition_id'],)*len(t) for attr, t in zip(self.condition_attributes, times)]
        self.condition_times_index = pd.MultiIndex.from_arrays(
            [np.hstack(condition_ids), np.hstack(times)], names=('condition_id','time')
        )

    def _get_data_keys(self):
        self.data_keys = [(attr[self.population_name], attr['condition_id']) for attr in self.data_attributes]

    def _get_data_set(self):
        array_data = {(population_key[self.population_name], condition_attr['condition_id']): None \
            for population_key, condition_attr in itertools.product(self.population_keys, self.condition_attributes)}

        data_set = {}
        for attr_name in self.attribute_names:
            [array_data.update({data_key: data_attr[attr_name]}) for data_key, data_attr in zip(self.data_keys, self.data_attributes)];
            
            data_array = np.array([X for X in array_data.values()])
            data_array = data_array.reshape(len(self.population_keys), len(self.condition_attributes))
            data_array = np.vstack([np.hstack(X) for X in data_array])

            data_set.update({attr_name: ([self.population_name, 'condition_time'], data_array)})

        data_coords = {
            self.population_name: [k[self.population_name] for k in self.population_keys], 
            'condition_time': self.condition_times_index
        }
        self._raw_data_set = xr.Dataset(data_set, coords=data_coords)

        self.reset_data_set()
    
    def reset_data_set(self):
        self.data_set = self._raw_data_set.copy(deep=True)