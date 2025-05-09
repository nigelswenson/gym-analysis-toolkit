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
import re
def total_pickle_size(folder_path):
    pickle_files = glob.glob(os.path.join(folder_path, '*.pkl'))  # or '*.pickle'
    total_size = sum(os.path.getsize(f) for f in pickle_files)
    return total_size

def get_available_memory():
    mem = psutil.virtual_memory()
    return mem.available

def getitem_for_folder(d, key, fcn):
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

def estimate_memory_use(episode_data, keys):
    values  = []
    for key in keys:
        values.append(getitem_for(episode_data,key))
    return sum(sys.getsizeof(v) for v in values)

def setitem_for(d,key,val):
    for level in key[:-1]:
        d=d[level]
    d[key[-1]]=val

def clear_dict(dictionary):
    for k,v in dictionary.items():
        if isinstance(v,dict):
            dictionary[k] = clear_dict(v)
        else:
            dictionary[k] = {'Min':None,'Max':None,'Average':None,'Full Data':[]}
    return dictionary

def get_keylist(dictionary,current_branch):
    keylist = []
    for k,v in dictionary.items():
        temp = current_branch.copy()
        temp.append(k)
        if isinstance(v,dict):
            keylist.extend(get_keylist(v,temp))
        else:
            keylist.append(temp)
    return keylist

def pool_data_loading(episode_file, key_tuples, functions, timestep_key='timsetep_data'):
    '''
    This takes a given episode file, tuple of key tuples and a tuple of functions to apply and returns
    a list with each key and its desired function applied to it. Designed to be used with Pool.starmap on a folder 
    full of data
    '''
    # print(tsteps,key_tuples)
    with open(episode_file, 'rb') as ef:
        tempdata = pkl.load(ef)
    data = tempdata['timestep_list']
    return [getitem_for_folder(data,key,fcn) for key,fcn in zip(key_tuples,functions)]

def pool_full_values(episode_file, key_tuples, timestep_key='timsetep_data'):
    '''
    This takes a given episode file, tuple of key tuples and returns
    an np array of shape (n,m) where n is the number of timesteps and m is the number of key_tuples.
    Designed to be used with Pool.starmap on a folder full of data
    '''
    # print(tsteps,key_tuples)
    with open(episode_file, 'rb') as ef:
        tempdata = pkl.load(ef)
    data = tempdata['timestep_list']
    full_data = []
    for tstep in data:
        # datamess = [getitem_for(tstep,key) for key in key_tuples]
        # if any(np.array(datamess)==None):
        #     indexes = np.where(np.array(datamess)==None)
        #     print(indexes, key_tuples[indexes[0][0]])
        #     # print(datamess[indexes[0][0]])
        full_data.append([getitem_for(tstep,key) for key in key_tuples])
    return full_data

# --- BACKEND CLASS ---
class DataPlotBackend:
    function_mapping = {'Min':np.min, 'Max':np.max, 'Average':np.mean}
    def __init__(self):
        self.data = {}
        self.structure_mapping = {}
        self.folder_data = {}
        self.memory_usage_threshold = 0.75

    def load_data(self, filename,timestep_key='timsetep_data'):
        # try:
        with open(filename, "rb") as f:
            data = pkl.load(f)
        if not isinstance(data, dict):
            raise ValueError("Pickle must contain a dictionary of lists.")
        data_dict = data
        # try:
        self.data = self.load_structured_data(data_dict,timestep_key)
        self.folder_data = copy.deepcopy(self.data)
        self.folder_data = clear_dict(self.folder_data)
        # except:
        #     print('data was not structured as expected, not modifying data dict before loading')
        #     self.data = data
        return self.data
        # except Exception as e:
        #     raise RuntimeError(f"Failed to load data: {e}")
    
    def load_folder(self, foldername, key_function_list):
        """
        Loads in the data associated with the keylist from all the files in the folder.
        We check for 3 different memory conditions. If possible, we load ALL episode data
        If there is not sufficient memory for that, we load in all the data associated with the keys provided.
        If there isnt enough for that either, we only load the specific data associated with the type of function in the key (min, max etc)
        """

        freemem = get_available_memory()
        datasize = total_pickle_size(foldername)
        episode_files = []
        for fname in os.listdir(foldername):
            if fname.endswith(".pkl"):
                episode_files.append(fname)
        filenums = [re.findall('\d+',f) for f in episode_files]
        final_filenums = []
        for i in filenums:
            if len(i) > 0 :
                final_filenums.append(int(i[0]))
        
        sorted_inds = np.argsort(final_filenums)
        final_filenums = np.array(final_filenums)
        temp = final_filenums[sorted_inds]
        episode_files = np.array(episode_files)
        episode_files = episode_files[sorted_inds].tolist()
        episode_files = [os.path.join(foldername, f) for f in episode_files]
        if datasize*self.memory_usage_threshold < freemem:
            print('Enough space to load everything')
            # in this case we make a keylist with all leaves and pool load everything
            dictionary_only_keylist = get_keylist(self.data,[])
            backend_specific_key_list = copy.deepcopy(dictionary_only_keylist)
            for b in backend_specific_key_list:
                b.append('Full Data')          
            # print('length of dictionary keylist', len(dictionary_only_keylist))
            file_specific_key_list = []
            for k in dictionary_only_keylist:
                temp = k[:-1]
                mapping_thing = self.structure_mapping[k[-1]]
                if type(mapping_thing) is list:
                    temp.extend(mapping_thing)
                else:
                    temp.extend([mapping_thing])
                file_specific_key_list.append(temp)
            # assert False
            pool = multiprocessing.Pool()
            thing = [[ef, file_specific_key_list] for ef in episode_files]
            data_list = pool.starmap(pool_full_values,thing)
            pool.close()
            pool.join()
            datalist_shape = np.shape(data_list)
            data_list = np.array(data_list)
            data_list = data_list.transpose(2, 1, 0)
        else:
            fcns = [k[0][1] if k[0][1] != None else DataPlotBackend.function_mapping[k[0][0]] for k in key_function_list]
            dictionary_only_keylist = [k[1].copy() for k in key_function_list]
            # dictionary_only_keylist holds the keylist for simple dictionary exploration
            file_specific_key_list = []
            for k in dictionary_only_keylist:
                temp = k[:-1]
                mapping_thing = self.structure_mapping[k[-1]]
                if type(mapping_thing) is list:
                    temp.extend(mapping_thing)
                else:
                    temp.extend([mapping_thing])
                file_specific_key_list.append(temp)
            # file_specific_key_list holds the full path for arbitrary pickle files. Its used by the pool later for multiprocess loading
            backend_specific_key_list = copy.deepcopy(dictionary_only_keylist)
            for key_stuff,things in zip(backend_specific_key_list,key_function_list):
                key_stuff.append(things[0][0])
                if things[0][0] == 'Specific Timestep':
                    key_stuff.append(things[0][1])
            # backend_specific_key_list holds the keys for self.folder_data, including the type of data being saved
            # this allows us to search through each one and check if we already have the data before doing the pool
            pool_specific_key_list = []
            for backend_key, file_key in zip(backend_specific_key_list,file_specific_key_list):
                if getitem_for(self.folder_data,backend_key) is None:
                    pool_specific_key_list.append(file_key)
            with open(episode_files[0], 'rb') as ef:
                tempdata = pkl.load(ef)
            data = tempdata['timestep_list']
            mem_estimate = estimate_memory_use(data[0],pool_specific_key_list)*len(data)*len(episode_files)
            if  mem_estimate*self.memory_usage_threshold < freemem:
                print('not enough memory to load everything, but there is enough to load the full data we care about without agregation')
                backend_specific_key_list = copy.deepcopy(dictionary_only_keylist)
                for b in backend_specific_key_list:
                    b.append('Full Data')
                pool = multiprocessing.Pool()
                thing = [[ef, file_specific_key_list] for ef in episode_files]
                data_list = pool.starmap(pool_full_values,thing)
                pool.close()
                pool.join()
                datalist_shape = np.shape(data_list)
                data_list = np.reshape(data_list,(datalist_shape[2],datalist_shape[1],datalist_shape[0]))
            else:
                pool = multiprocessing.Pool()
                thing = [[ef, pool_specific_key_list, fcns] for ef in episode_files]
                data_list = pool.starmap(pool_data_loading,thing)
                pool.close()
                pool.join()

            data_list=np.transpose(data_list)
        for key_stuff, data in zip(backend_specific_key_list,data_list):
            setitem_for(self.folder_data,key_stuff,data)
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
        # i need to add a thing that checks to see if we have the data saved in folder data and, if not, fills it.
        if x_key == ["Timesteps"]:
            y_data = getitem_for(self.folder_data,y_key)
            if y_data is None:
                temp_y_key = y_key[:-1]
                aggregation_function = DataPlotBackend.function_mapping[y_key[-1]]
                temp_y_key.append('Full Data')
                full_y_data = getitem_for(self.folder_data,temp_y_key)
                y_data = aggregation_function(full_y_data,axis=0)
                setitem_for(self.folder_data,y_key,y_data)
            x_data = list(range(len(y_data)))
        else:
            x_data = getitem_for(self.folder_data,x_key)
            if x_data is None:
                temp_x_key = x_key[:-1]
                aggregation_function = DataPlotBackend.function_mapping[x_key[-1]]
                temp_x_key.append('Full Data')
                full_x_data = getitem_for(self.folder_data,temp_x_key)
                x_data = aggregation_function(full_x_data,axis=0)
                setitem_for(self.folder_data,x_key,x_data)

        if y_key == ["Timesteps"]:
            x_data = getitem_for(self.folder_data,x_key)
            if x_data is None:
                temp_x_key = x_key[:-1]
                aggregation_function = DataPlotBackend.function_mapping[x_key[-1]]
                temp_x_key.append('Full Data')
                full_x_data = getitem_for(self.folder_data,temp_x_key)
                x_data = aggregation_function(full_x_data,axis=0)
                setitem_for(self.folder_data,x_key,x_data)
            y_data = list(range(len(x_data)))
        else:
            y_data = getitem_for(self.folder_data,y_key)
            if y_data is None:
                temp_y_key = y_key[:-1]
                aggregation_function = DataPlotBackend.function_mapping[y_key[-1]]
                temp_y_key.append('Full Data')
                full_y_data = getitem_for(self.folder_data,temp_y_key)
                y_data = aggregation_function(full_y_data,axis=0)
                setitem_for(self.folder_data,y_key,y_data)

        if len(x_data) != len(y_data):
            raise ValueError("X and Y data lengths do not match.")

        return x_data, y_data