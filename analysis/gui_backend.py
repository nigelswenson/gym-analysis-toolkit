import sys
import os
import pickle as pkl
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QListWidget, QMessageBox, QFileDialog, QLineEdit
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import psutil
import glob
import multiprocessing
import copy

def total_pickle_size(folder_path):
    pickle_files = glob.glob(os.path.join(folder_path, '*.pkl'))  # or '*.pickle'
    total_size = sum(os.path.getsize(f) for f in pickle_files)
    return total_size

def get_available_memory():
    mem = psutil.virtual_memory()
    return mem.available

def getitem_for_folder(d, key, fcn):
    print(key)
    if isinstance(fcn, int):
        d=d[fcn]
        for level in key:
            d = d[level]
    elif fcn is None:
        values = []
        for tstep in d:
            for level in key:
                tstep = tstep[level]
            values.append(tstep)
        d = values
    else:
        values = []
        for tstep in d:
            for level in key:
                tstep = tstep[level]
            values.append(tstep)
        d=fcn(values)
    return d

def getitem_for(d, key):
    for level in key:
        d = d[level]
    return d

def setitem_for(d,key,val):
    for level in key[:-1]:
        d=d[level]
    d[key[-1]]=val

def clear_dict(dictionary):
    for k,v in dictionary.items():
        if isinstance(v,dict):
            dictionary[k] = clear_dict(v)
        else:
            dictionary[k] = None
    return dictionary

def pool_key_list(episode_file, key_tuples, functions, timestep_key='timsetep_data'):
    '''
    This takes a given episode file, tuple of timestep numbers and tuple of key tuples and returns
    a list with each key at its desired timestep. Designed to be used with Pool.starmap on a folder 
    full of data

    Unfortunately this doesnt currently support min/max
    '''
    # print(tsteps,key_tuples)
    with open(episode_file, 'rb') as ef:
        tempdata = pkl.load(ef)
    data = tempdata['timestep_list']
    return [getitem_for_folder(data,key,fcn) for key,fcn in zip(key_tuples,functions)]


# --- BACKEND CLASS ---
class DataPlotBackend:
    function_mapping = {'Min':np.min, 'Max':np.max, 'Average':np.mean}
    def __init__(self):
        self.data = {}
        self.structure_mapping = {}
        self.folder_data = {}

    def load_data(self, filename,timestep_key='timsetep_data'):
        # try:
        with open(filename, "rb") as f:
            data = pkl.load(f)
        if not isinstance(data, dict):
            raise ValueError("Pickle must contain a dictionary of lists.")
        data_dict = data
        # try:
        self.data = self.load_structured_data(data_dict,timestep_key)
        # self.folder_data = copy.deepcopy(self.data)
        # self.folder_data = clear_dict(self.folder_data)
        # except:
        #     print('data was not structured as expected, not modifying data dict before loading')
        #     self.data = data
        print('right after loading',self.data.keys())
        return self.data
        # except Exception as e:
        #     raise RuntimeError(f"Failed to load data: {e}")
    
    def load_folder(self, foldername, keylist):
        """
        Loads in the data associated with the keylist from all the files in the folder.
        If there is sufficient memory, we load in all the data associated with the keys provided.
        If not, we only load the specific data associated with the type of function in the key (min, max etc)
        """
        fcns = [k[0][1] if k[0][1] != None else DataPlotBackend.function_mapping[k[0][0]] for k in keylist]
        keys = [k[1] for k in keylist if getitem_for(self.folder_data,k[1]) is None]
        episode_files = [os.path.join(foldername, f) for f in os.listdir(foldername) if f.lower().endswith('.pkl')]
        
        
        pool = multiprocessing.Pool()
        # keys = (('state','obj_2','pose'),)
        cleaned_keys = []
        for k in keys:
            temp = k[:-1]
            mapping_thing = self.structure_mapping[k[-1]]
            if type(mapping_thing) is list:
                temp.extend(mapping_thing)
            else:
                temp.extend([mapping_thing])
            cleaned_keys.append(temp)
        thing = [[ef, cleaned_keys, fcns] for ef in episode_files]
        data_list = pool.starmap(pool_key_list,thing)
        pool.close()
        pool.join()
        
        data_list=np.transpose(data_list)
        for key_stuff, data in zip(keys,data_list):
            setitem_for(self.folder_data,key_stuff,data)
        # freemem = get_available_memory()
        # datasize = total_pickle_size(foldername)
        # with open(episode_files[0], 'rb') as ef:
        #     tempdata = pkl.load(ef)
        # data = tempdata['timestep_list']

        return data_list
    def build_initial_dict(self, final_dict, simple_dict, mapdict=None):
        for k,v in simple_dict.items():
            if isinstance(v,dict):
                final_dict[k] = {}
                self.build_initial_dict(final_dict[k],v)
            elif isinstance(v,list) | isinstance(v,tuple)| isinstance(v, np.ndarray):
                if mapdict:
                    new_mapdict = {k+'_'+str(subkey):[mapdict[k][i] if i!= len(mapdict[k]) else subkey for i in range(len(mapdict[k])+1)] for subkey,_ in enumerate(v)}
                else:
                    new_mapdict = {k+'_'+str(subkey):[k,subkey] for subkey,_ in enumerate(v)}
                v_dict = {k+'_'+str(subkey):val for subkey,val in enumerate(v)}
                self.build_initial_dict(final_dict,v_dict, new_mapdict)
            else:
                final_dict[k] = [v]
                if mapdict:
                    self.structure_mapping[k]=mapdict[k]
                else:
                    self.structure_mapping[k] = k
        return final_dict
    
    def expand_dict(self, final_dict, simple_dict):
        for k,v in simple_dict.items():
            if isinstance(v,dict):
                self.expand_dict(final_dict[k],v)
            elif isinstance(v,list) | isinstance(v,tuple)| isinstance(v, np.ndarray):
                v_dict = {k+'_'+str(subkey):val for subkey,val in enumerate(v)}
                self.expand_dict(final_dict,v_dict)
            else:
                final_dict[k].append(v)
        return final_dict 
    
    def insert_dict(self, final_dict, simple_dict):
        for k,v in simple_dict.items():
            if isinstance(v,dict):
                self.insert_dict(final_dict[k],v)
            elif isinstance(v,list) | isinstance(v,tuple)| isinstance(v, np.ndarray):
                v_dict = {k+'_'+str(subkey):val for subkey,val in enumerate(v)}
                self.insert_dict(final_dict,v_dict)
            else:
                final_dict[k].insert(0,v)
        return final_dict 
    
    def load_structured_data(self,data_holder,timestep_key='timsetep_data'):
        """
        loads data assuming the data dict has all timestep data saved in data_dict[timsetep_key]
        This processes the data into a new data dictionary that works with our plot function conventions"""
        final_dict = {}
        final_dict = self.build_initial_dict(final_dict, data_holder[timestep_key][0])
        for timestep in data_holder[timestep_key][1:]:
            final_dict = self.expand_dict(final_dict, timestep)
        try:
            addition_dict = {'state':data_holder['start_state'], 'info':data_holder['start_info']}
            final_dict = self.insert_dict(final_dict,addition_dict)
        except:
            print('no start state or info. Check that you arent losing data')
        
        return final_dict
    
    def prepare_data_singular(self, x_key, y_key):
        if x_key == ["Timesteps"]:
            y_data = getitem_for(self.data,y_key)
            x_data = list(range(len(y_data)))
        else:
            x_data = getitem_for(self.data,x_key)

        if y_key == ["Timesteps"]:
            x_data = getitem_for(self.data,x_key)
            y_data = list(range(len(x_data)))
        else:
            y_data = getitem_for(self.data,y_key)

        if len(x_data) != len(y_data):
            raise ValueError("X and Y data lengths do not match.")

        return x_data, y_data

    def prepare_data_multi(self, x_key, y_key):
        if x_key == ["Timesteps"]:
            y_data = getitem_for(self.data,y_key)
            x_data = list(range(len(y_data)))
        else:
            x_data = getitem_for(self.data,x_key)

        if y_key == ["Timesteps"]:
            x_data = getitem_for(self.data,x_key)
            y_data = list(range(len(x_data)))
        else:
            y_data = getitem_for(self.data,y_key)

        if len(x_data) != len(y_data):
            raise ValueError("X and Y data lengths do not match.")

        return x_data, y_data