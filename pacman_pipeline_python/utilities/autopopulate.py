"""Utilities for automatically populating manual tables for the Pac-Man Task."""

import datajoint as dj
import os, re, inspect, time, shutil
import neo
import progressbar
from churchland_pipeline_python import acquisition, action, equipment, lab, processing, reference
from churchland_pipeline_python.utilities import datajointutils as dju
from pacman_pipeline_python import pacman_acquisition, pacman_processing
from . import datasynthesis
from datetime import datetime
from typing import List, Tuple


# =======
# SESSION
# =======

def session(
        users: List[str],
        monkey: str='Cousteau',
        task_version: str='1.0',
        dates: List[str]=None,
        hardware: Tuple[str]=('Speedgoat', 'Cerebus', '5lb Load Cell'),
        software: Tuple[str]=('Simulink', 'Psychtoolbox')
    ) -> None:

        # input keys
        user_key =   (lab.User & [{'user_uni': uni} for uni in users]).fetch('KEY')
        monkey_key = (lab.Monkey & {'monkey': monkey}).fetch1('KEY')
        rig_key =    (lab.Rig & {'rig': 'Jumanji'}).fetch1('KEY')
        task_key =   (acquisition.Task & {'task': 'pacman', 'task_version': task_version}).fetch1('KEY')

        # get session dates and raw path
        session_dates, raw_path = datasynthesis.getsessions(monkey)
        
        # restrict dates based on user input
        if dates:
            session_dates = [d for d in session_dates if d in dates]            

        # remove sessions with inputs
        session_dates = [date for date in session_dates if not acquisition.Session & {'session_date': date}]

        for date in session_dates:

            session_path = os.path.sep.join([raw_path, date, ''])
            session_files = os.listdir(session_path)

            # ensure behavior directory exists
            try:
                if 'Speedgoat' in hardware:
                    behavior_dir = 'speedgoat'

                next(filter(lambda x: x==behavior_dir, session_files))

            except StopIteration:
                print('Missing behavior files for session {}'.format(date))

            else:         
                # ensure ephys directory exists
                try:
                    if 'Cerebus' in hardware:
                        ephys_dir = 'blackrock'

                    elif 'IMEC' in hardware:
                        ephys_dir = 'imec'

                    next(filter(lambda x: x==ephys_dir, session_files))

                except StopIteration:
                    print('Missing ephys files for session {}'.format(date))
                    
                else:
                    # session key
                    session_key = dict(session_date=date, **monkey_key)
                    
                    # insert session
                    acquisition.Session.insert1(dict(**session_key, **rig_key, **task_key))

                    # insert users
                    for user in user_key:
                        acquisition.Session.User.insert1(dict(**session_key, **user))

                    # insert notes
                    try:
                        notes_files = next(x for x in session_files if re.search('.*notes\.txt',x))
                    except StopIteration:
                        print('Missing notes for session {}'.format(date))
                    else:
                        with open(session_path + notes_files,'r') as f:
                            acquisition.Session.Notes.insert1(dict(**session_key, session_notes_id=0, session_notes=f.read()))

                    # insert hardware and software
                    for hardware_name in hardware:
                        acquisition.Session.Hardware.insert1(dict(
                            **session_key, 
                            **(equipment.Hardware & {'hardware': hardware_name}).fetch1('KEY')
                        ))

                    # software
                    for software_name in software:
                        acquisition.Session.Software.insert1(dict(
                            **session_key, 
                            **(equipment.Software & {'software': software_name}).fetch1('KEY')
                        ))


# ==================
# BEHAVIOR RECORDING
# ==================

def behaviorrecording(behavior_sample_rate: int=1e3, display_progress: bool=True):

    # remove problematic sessions and those with behavior recording entries
    key_source = acquisition.Session - 'session_problem' - acquisition.BehaviorRecording

    if display_progress:
        bar = progressbar.ProgressBar(max_value=len(key_source))
        bar.update(0)

    for key_idx, session_key in enumerate(key_source.fetch('KEY')):
        
        # path to raw data
        raw_path = datasynthesis.getdatapath(session_key['monkey'])

        # path to behavior files
        if (acquisition.Session.Hardware & session_key & {'hardware': 'Speedgoat'}):

            behavior_path = raw_path + os.path.sep.join([str(session_key['session_date']), 'speedgoat', ''])

        # get behavior files
        behavior_files = sorted(list(os.listdir(behavior_path)))

        # split files by extension
        behavior_file_parts = [x.split(sep='.') for x in behavior_files]

        # behavior file keys
        behavior_file_keys = [
            dict(**session_key, behavior_file_id=idx, behavior_file_prefix=x[0], behavior_file_extension=x[1]) \
                for idx, x in enumerate(behavior_file_parts)
            ]
        
        # insert behavior recording and files
        acquisition.BehaviorRecording.insert1(dict(
            **session_key, 
            behavior_recording_sample_rate=behavior_sample_rate,
            behavior_recording_path=behavior_path
        ))

        acquisition.BehaviorRecording.File.insert(behavior_file_keys)

        if display_progress:
            bar.update(1+key_idx)


# ===============
# EPHYS RECORDING
# ===============

def ephysrecording(display_progress: bool=True):

    # remove problematic sessions and those with ephys recording entries
    key_source = acquisition.Session - 'session_problem' - acquisition.EphysRecording

    if display_progress:
        bar = progressbar.ProgressBar(max_value=len(key_source))
        bar.update(0)

    for key_idx, session_key in enumerate(key_source.fetch('KEY')):

        # ephys file regular expression
        ephys_file_regexp = '_'.join([
            'pacman-task',
            session_key['monkey'][0].lower(),
            str(session_key['session_date']).replace('-','')[2:],
            '(emg|neu|neu_emg)',
            '\d{3}\.ns\d'
            ])
        
        # path to raw data
        raw_path = datasynthesis.getdatapath(session_key['monkey'])

        if (acquisition.Session.Hardware & session_key & {'hardware': 'Cerebus'}):

            # path to blackrock files
            blackrock_path = os.path.sep.join([raw_path[:-1], str(session_key['session_date']), 'blackrock', ''])
            blackrock_files = list(os.listdir(blackrock_path))

            # recording key
            ephys_recording_key = dict(**session_key,
                ephys_recording_path=(reference.EngramTier & {'engram_tier':'locker'}).ensureremote(blackrock_path), 
                ephys_recording_duration=0)

            # NSx files
            nsx_files = [f for f in blackrock_files if re.search(ephys_file_regexp,f)]

            # ephys file and channel keys
            ephys_file_keys = []
            ephys_channel_keys = []

            for i_file, file_name in enumerate(nsx_files):

                # behavior file keys
                file_parts = file_name.split(sep='.')
                ephys_file_keys.append(dict(
                    **session_key, 
                    ephys_file_id=i_file, 
                    ephys_file_prefix=file_parts[0], 
                    ephys_file_extension=file_parts[1]
                ))

                # read NSx file
                reader = neo.rawio.BlackrockRawIO(blackrock_path + file_name)
                reader.parse_header()

                # pull sample rate and update recording duration
                fs = int(next(iter(reader.sig_sampling_rates.values())))
                ephys_recording_key.update(ephys_recording_sample_rate = fs)
                ephys_recording_key['ephys_recording_duration'] += reader.get_signal_size(0,0) / fs

                # channel header name and ID indices
                name_idx, id_idx = [
                    idx for idx, name in enumerate(reader.header['signal_channels'].dtype.names) \
                    if name in ['name','id']
                ]

                chan_attr = []

                # //TODO for S-Probes, the name indicates the channel layout

                # insert channel header information //TODO double check the map files use the ID and not the label
                for j, chan in enumerate(reader.header['signal_channels']):

                    chan_name = chan[name_idx]

                    # read channel type
                    if re.search('^(\d|elec)', chan_name):
                        chan_type = 'brain'

                    elif re.search('ainp[1-8]$', chan_name):
                        chan_type = 'emg'

                    elif chan_name == 'ainp15':
                        chan_type = 'stim'

                    elif chan_name == 'ainp16':
                        chan_type = 'sync'

                    # write channel attributes
                    chan_attr.append(dict(
                        **session_key, 
                        ephys_file_id=i_file, 
                        ephys_channel_idx=j, 
                        ephys_channel_id=chan[id_idx], 
                        ephys_channel_type=chan_type
                    ))

                # write channel attributes
                ephys_channel_keys.extend(chan_attr)

            # insert data
            acquisition.EphysRecording.insert1(ephys_recording_key)
            acquisition.EphysRecording.File.insert(ephys_file_keys)
            acquisition.EphysRecording.Channel.insert(ephys_channel_keys)

            if display_progress:
                bar.update(1+key_idx)


# ===================
# BRAIN CHANNEL GROUP
# ===================

def brainchannelgroup(display_progress: bool=True):

    # remove files without brain channels or with pre-existing channel group entries
    key_source = (acquisition.EphysRecording.File \
        & (acquisition.EphysRecording.Channel & {'ephys_channel_type': 'brain'})) \
        - acquisition.BrainChannelGroup

    if display_progress:
        bar = progressbar.ProgressBar(max_value=len(key_source))
        bar.update(0)

    for key_idx, ephys_file_key in enumerate(key_source.fetch('KEY')):

        # read brain recording attributes from notes file
        brain_attr, _ = datasynthesis.parsenotes(ephys_file_key, read_type=('brain'))

        if brain_attr:

            # insert attributes to master table
            acquisition.BrainChannelGroup.insert(brain_attr)

            # get channel group primary keys
            brain_channel_group_keys = (acquisition.BrainChannelGroup \
                & ephys_file_key).fetch('KEY', order_by='brain_channel_group_id')

            # get ephys recording channel keys
            ephys_recording_chan_keys = (acquisition.EphysRecording.Channel \
                & ephys_file_key & {'ephys_channel_type': 'brain'}).fetch('KEY', order_by='ephys_channel_idx')

            num_brain_groups = len(brain_channel_group_keys)
            assert num_brain_groups == 1, '{} brain channel groups. Action unclear'.format(num_brain_groups)

            # replicate channel group key for each brain recording channel
            for idx, ephys_key in enumerate(ephys_recording_chan_keys):
                ephys_key.update(**brain_channel_group_keys[0], brain_channel_idx=idx)

            # insert channel group channel keys
            acquisition.BrainChannelGroup.Channel.insert(ephys_recording_chan_keys)

        else:
            print('Missing attributes for session {}'.format(ephys_file_key['session_date']))

        if display_progress:
            bar.update(1+key_idx)


# =================
# EMG CHANNEL GROUP
# =================

def emgchannelgroup(display_progress: bool=True):

    # remove files without EMG channels or with pre-existing channel group entries
    key_source = (acquisition.EphysRecording.File \
        & (acquisition.EphysRecording.Channel & {'ephys_channel_type': 'emg'})) \
        - acquisition.EmgChannelGroup

    if display_progress:
        bar = progressbar.ProgressBar(max_value=len(key_source))
        bar.update(0)

    for key_idx, ephys_file_key in enumerate(key_source.fetch('KEY')):

        # read emg recording attributes from notes file
        _, emg_attr = datasynthesis.parsenotes(ephys_file_key, read_type=('emg'))

        if emg_attr:

            # insert attributes to master table
            acquisition.EmgChannelGroup.insert(emg_attr)

            # get channel group primary keys
            emg_channel_group_keys = (acquisition.EmgChannelGroup \
                & ephys_file_key).fetch('KEY', order_by='emg_channel_group_id')

            # get ephys recording channel keys
            ephys_recording_chan_keys = (acquisition.EphysRecording.Channel \
                & ephys_file_key & {'ephys_channel_type': 'emg'}).fetch('KEY', order_by='ephys_channel_idx')

            num_muscle_groups = len(emg_channel_group_keys)
            num_emg_channels = len(ephys_recording_chan_keys)
            assert num_muscle_groups in (1, num_emg_channels), \
                '{} emg channel groups and {} emg recording channels. Action unclear'.format(num_muscle_groups, num_emg_channels)

            if num_muscle_groups == 1:

                # replicate channel group key for each emg recording channel
                for idx, ephys_chan_key in enumerate(ephys_recording_chan_keys):
                    ephys_chan_key.update(**emg_channel_group_keys[0], emg_channel_idx=idx)

            else:
                # copy emg group attributes to each emg recording channel
                for emg_group_key, ephys_chan_key in zip(emg_channel_group_keys, ephys_recording_chan_keys):
                    ephys_chan_key.update(**emg_group_key, emg_channel_idx=0)

            # insert channel group channel keys
            acquisition.EmgChannelGroup.Channel.insert(ephys_recording_chan_keys)

        else:
            print('Missing attributes for session {}'.format(ephys_file_key['session_date']))

        if display_progress:
            bar.update(1+key_idx)


# ==========
# BRAIN SORT
# ==========

def brainsort(monkey: str='Cousteau', spike_sorter: Tuple[str]=('Kilosort','2.0'), display_progress: bool=True):

    key_source = acquisition.BrainChannelGroup.fetch('KEY')

    # path to processed data
    processed_path = datasynthesis.getdatapath(monkey, data_type='processed')

    # match spike sorter input to software key
    software_key = next(iter(dju.matchfuzzykey({spike_sorter: equipment.Software}).values()))

    # path to spike sorter file
    if software_key['software'] == 'Kilosort':

        kilosort_path = [
            (key, processed_path + os.path.sep.join([str(key['session_date']), 'kilosort-manually-sorted', '']))
            for key in key_source]

    else:
        print('Spike sorter {} unrecognized'.format(software_key))
        return None

    # brain sort file regular expression
    brain_sort_file_regexp = '_'.join([
        'pacman-task',
        monkey[0].lower(),
        '\d{6}',
        '(neu|neu_emg)',
        '\d{3}'
        ])

    # full sort paths
    sort_path = []
    for key, pth in kilosort_path:
        sort_dir = [d for d in os.listdir(pth) if re.search(brain_sort_file_regexp, d)]
        sort_path.extend([(key, pth + os.path.sep.join([d, ''])) for d in sort_dir])

    # ensure remote path
    engram_rel = reference.EngramTier & {'engram_tier': 'locker'}
    sort_path = [(pth[0], engram_rel.ensureremote(pth[1])) for pth in sort_path]

    # remove paths already in table
    sort_path = [pth for pth in sort_path if not (processing.BrainSort & {'brain_sort_path': pth[1]})]

    if display_progress:
        bar = progressbar.ProgressBar(min_value=1, max_value=len(sort_path))

    for key_idx, (brain_sort_key, brain_sort_path) in enumerate(sort_path):

        # update key with spike sorter and sort path attributes
        brain_sort_key.update(**software_key, brain_sort_path=brain_sort_path)

        # increment sort ID number
        brain_sort_id = dju.nextuniqueint(processing.BrainSort, 'brain_sort_id', brain_sort_key)
        brain_sort_key.update(brain_sort_id=brain_sort_id)

        processing.BrainSort.insert1(brain_sort_key)

        if display_progress:
            time.sleep(0.1)
            bar.update(1+key_idx)


# ========
# EMG SORT
# ========

def emgsort(monkey: str='Cousteau', spike_sorter: Tuple[str]=('Myosort','1.0'), display_progress: bool=True):

    key_source = acquisition.EmgChannelGroup.fetch('KEY')

    # path to processed data
    processed_path = datasynthesis.getdatapath(monkey, data_type='processed')

    # match spike sorter input to software key
    software_key = next(iter(dju.matchfuzzykey({spike_sorter: equipment.Software}).values()))

    # path to spike sorter file
    if software_key['software'] == 'Myosort':

        kilosort_path = [
            (key, processed_path + os.path.sep.join([str(key['session_date']), 'kilosort-manually-sorted', '']))
            for key in key_source]

    else:
        print('Spike sorter {} unrecognized'.format(software_key))
        return None

    # emg sort file regular expression
    emg_sort_file_regexp = '_'.join([
        'pacman-task',
        monkey[0].lower(),
        '\d{6}',
        '(emg|neu_emg)',
        '\d{3}'
        ])

    # full sort paths
    sort_path = []
    for key, pth in kilosort_path:
        sort_dir = [d for d in os.listdir(pth) if re.search(emg_sort_file_regexp, d)]
        sort_path.extend([(key, pth + os.path.sep.join([d, ''])) for d in sort_dir])

    # ensure remote path
    engram_rel = reference.EngramTier & {'engram_tier': 'locker'}
    sort_path = [(pth[0], engram_rel.ensureremote(pth[1])) for pth in sort_path]

    # remove paths already in table
    sort_path = [pth for pth in sort_path if not (processing.EmgSort & {'emg_sort_path': pth[1]})]

    if display_progress:
        bar = progressbar.ProgressBar(min_value=1, max_value=len(sort_path))

    for key_idx, (emg_sort_key, emg_sort_path) in enumerate(sort_path):

        # update key with spike sorter and sort path attributes
        emg_sort_key.update(**software_key, emg_sort_path=emg_sort_path)

        # increment sort ID number
        emg_sort_id = dju.nextuniqueint(processing.EmgSort, 'emg_sort_id', emg_sort_key)
        emg_sort_key.update(emg_sort_id=emg_sort_id)

        processing.EmgSort.insert1(emg_sort_key)

        if display_progress:
            time.sleep(0.1)
            bar.update(1+key_idx)


# ========
# PIPELINE
# ========

def pipeline(display_progress: bool=True):

    # session descendants populate functions
    populate_functions = {
        acquisition.BehaviorRecording: (behaviorrecording,             {'display_progress': display_progress}),
        acquisition.EphysRecording:    (ephysrecording,                {'display_progress': display_progress}),
        acquisition.BrainChannelGroup: (brainchannelgroup,             {'display_progress': display_progress}),
        acquisition.EmgChannelGroup:   (emgchannelgroup,               {'display_progress': display_progress}),
        processing.SyncBlock:          (processing.SyncBlock.populate, {'display_progress': display_progress}),
        processing.BrainSort:          (brainsort,                     {'display_progress': display_progress}),
        # processing.EmgSort,
        # processing.Neuron,
        # processing.MotorUnit,
        ##pacman_acquisition.Behavior,
        ##pacman_processing.AlignmentParams,
        ##pacman_processing.EphysTrialStart,
        ##pacman_processing.TrialAlignment,
        # pacman_processing.BehaviorBlock,
        ##pacman_processing.FilterParams,
        ##pacman_processing.Force,
        # pacman_processing.BehaviorQuality,
        # pacman_processing.GoodTrial,
        # pacman_processing.NeuronSpikes,
        # pacman_processing.NeuronRate,
        # pacman_processing.NeuronPsth,
        # pacman_processing.MotorUnitSpikes,
        # pacman_processing.MotorUnitRate,
        # pacman_processing.MotorUnitPsth,
        # pacman_processing.Emg
    }

    for table in populate_functions.keys():

        # read table name and tier
        attributes = inspect.getmembers(table, lambda a:not(inspect.isroutine(a)))
        table_name = next(a[1] for a in attributes if a[0].startswith('full_table_name'))

        print('\n\nPopulating ' + table_name + '\n')
        time.sleep(0.2)

        func,  kwargs = populate_functions[table]
        func(**kwargs)