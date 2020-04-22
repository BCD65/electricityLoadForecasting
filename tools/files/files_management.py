#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import scipy as sp
import os
#import subprocess
#import json
#import socket
import pickle
#from termcolor  import colored
from scipy      import sparse
from subprocess import CalledProcessError, TimeoutExpired#, PIPE

import electricityLoadForecasting.tools as tools
import electricityLoadForecasting.forecasting.config     as config
#from cluster_cermics  import hive
#import str_make; importlib.reload(str_make)

#custex  = tools.custex
len_str = 25
max_len = 1e4

#
#
#dist__folder_data    =  hive + 'Saved/Data/'
#dist__folder_model   =  hive + 'Saved/models/'
#dist__folder_f_estim =  hive + 'Saved/Final_estimates/'
#dist__folder_weights =  hive + 'Saved/Weights/'
#dist__folder_perf    =  hive + 'Saved/Performances/'
#dist__folder_pred    =  hive + 'Saved/Predictions/'
#dist__folder_results =  hive + 'Results/'
###########################
t_out                = 30
###########################
#
tuple3errors = (TimeoutExpired, CalledProcessError, ValueError) 
#
#
def file_delete(path, param): 
    raise NotImplementedError('delete')
#    #to delete local copy after transfer
#    s =  os.path.basename(path)
#    msg = '{0:20.20}'.format('deleting') + '{0:20.20}'.format(s)
#    try:
#        if not config.delete']:
#            raise ValueError('on purpose ' + msg)
#        else:
#            print('{0:{len_str}}'.format(msg, len_str = len_str), end = ' - ')
#            subprocess.Popen(['rm', path], stderr = PIPE, stdout=PIPE, bufsize=4096)
#            print('deleted')
#    except IOError as e:
#        print(colored('fail', 'red'), colored(repr(e), 'red'))
#        raise e
#    except ValueError:
#            pass
#            
#  
##profile    
def file_download(files_to_include, to = t_out):
    raise NotImplementedError('download')
#    # transfer from origin to destination
#    msg = '{0:20.20}'.format('downloading') + '{0:20.20}'.format(', '.join([os.path.basename(k) for k in files_to_include]))
#    try:
    ##########################
#    if   'data'  in path.lower():
#        if not config.download_data:
#            raise ValueError('on purpose ' + msg)
#    elif 'model' in path.lower():
#        if not config.download_model:
#            raise ValueError('on purpose ' + msg)              
#    elif 'pred'  in path.lower():
#        if not config.download_pred:
#            raise ValueError('on purpose ' + msg)            
#    elif 'perf'  in path.lower():
#        if not config.download_perf:
#            raise ValueError('on purpose ' + msg)
#    else:
#        if not config.download: 
#            raise ValueError('on purpose ' + msg)            
    ##########################
#        else:
#            if socket.gethostname() in hive:
#                raise ValueError('no dl on hive')
#            command = ['rsync', '-ruvP']
#            for inc in files_to_include:
#                assert type(files_to_include[inc]) == str
#                assert files_to_include[inc] in {
#                                                 'npz', 
#                                                 'csv', 
#                                                 'txt',
#                                                 'dict_np', 
#                                                 'dict_sp', 
#                                                 'dict_tup', 
#                                                 'json', 
#                                                 'pickle', 
#                                                 }, ('otherwise need to adapt the conditions below and in the rest of this module', 
#                                                     '\n' + 'option was {0} in file_download'.format(files_to_include[inc]), 
#                                                     )
#                command += ['--include', os.path.basename(local_folder+inc)
#                                       + '.npz'*(files_to_include[inc] == 'npz')
#                                       + '.csv'*(files_to_include[inc] == 'csv')
#                                       + '.txt'*(files_to_include[inc] == 'txt')
#                                       + '**'  *(files_to_include[inc] in {
#                                                                           'dict_np', 
#                                                                           'dict_sp', 
#                                                                           'dict_tup', 
#                                                                           'json',
#                                                                           'pickle', 
#                                                                           })
#                                       ]
#                subprocess.check_output(['mkdir', '-p', os.path.dirname(local_folder+inc)], 
#                                         timeout = to, 
#                                         stderr=subprocess.STDOUT,
#                                         )
#            command += ['--exclude', '*',]
#            command += [os.path.dirname(dist_folder +inc)+'/', 
#                        os.path.dirname(local_folder+inc),
#                        ]
#            print('{0:{len_str}}'.format(msg, len_str = len_str), end = ' - ')
#            subprocess.check_output(command, timeout = to, stderr=subprocess.STDOUT)
#            print('{0:{len_str}}'.format('synced', len_str = len_str))
#    except tuple3errors as e:
#        print(colored('fail', 'red'), colored(repr(e)[:30], 'red'))
#    except Exception as e:
#        raise e


##profile    
def file_fetch(local_folder, files_to_fetch, t_o = t_out): #dist__folder, local_folder, param, 
    # Try to load, if fails try to transfer and load
    local_paths = {os.path.join(local_folder,
                                name,
                                ): v 
                   for name, v in files_to_fetch.items()
                   }
    data        = []
    try :
        data += [file_load(path, data_type = v) 
                 for path, v in local_paths.items()
                 ]
    except (*tuple3errors, FileNotFoundError, NotImplementedError):
        file_download(files_to_fetch, 
                      to = t_o,
                      )
        data += [file_load(path, data_type = v)
                 for path, v in local_paths.items()
                 ]
    except Exception as e:
        raise e
    if len(data) == 1: 
        data = data[0]
    return data


  
def file_load(path, data_type = None): 
    # load different formats
    msg = '{0:20.20}'.format('loading') + '{0:20.20}'.format(os.path.basename(path))
    ##########################
    if   'data'  in path.lower():
        if not config.load_data:
            raise ValueError('on purpose ' + msg)
    elif 'model' in path.lower():
        if not config.load_model:
            raise ValueError('on purpose ' + msg)              
    elif 'pred'  in path.lower():
        if not config.load_predictions:
            raise ValueError('on purpose ' + msg)            
    elif 'perf'  in path.lower():
        if not config.load_performances:
            raise ValueError('on purpose ' + msg)
    else:
        if not config.load: 
            raise ValueError('on purpose ' + msg)            
    ##########################
    print('{0:{len_str}}'.format(msg, len_str = len_str), end = ' - ')
    if data_type == 'np':
        f     = open(path, 'rb')
        data  = np.load(f, allow_pickle = True)
        f.close()   
#    elif option == 'json':
#        with open(path, 'r') as f:
#            data  = json.load(f)
#            if 'keys_upd' in path:
#                data = [tuple(tuple(e) if type(e) == list else e for e in small_list) for small_list in data]
    elif data_type == 'pickle':
        with open(path+'.pkl', 'rb') as f:
            data  = pickle.load(f)
#    elif option == 'npz':
#        assert 0
#        data  = sparse.load_npz(path + '.npz')
    elif data_type == 'dict_np':
        print()
        with open(os.path.join(path, 'keys'), 'rb') as f:
            keys = pickle.load(f)
        data = {}
        #list_files = os.listdir(path)
        if len(keys) >= max_len:
            raise ValueError('on purpose because list of files too long')
        for ii, key in enumerate(keys):
            print('\r'+str(ii), '/', len(keys), end = '')
            fname = repr(key)
            with open(os.path.join(path, fname), 'rb') as f:
                data[key] = np.load(f, allow_pickle = True)
            assert type(data[key]) == np.ndarray
            assert 'orig_masks' not in fname
        print()
    elif data_type == 'dict_sp':
        print()
        with open(os.path.join(path, 'keys'), 'rb') as f:
            keys = pickle.load(f)
        data = {}
        #list_files = os.listdir(path)
        if len(keys) >= max_len:
            raise ValueError('on purpose because list of files too long')
        for ii, key in enumerate(keys):
            print('\r'+str(ii), '/', len(keys), end = '')
            fname = repr(key)
#            if ('keys' in fname
#                or tuple(fname.split('$')) not in keys
#                   # and fname not in keys
#                    #)
#                ):
#                continue
            data[key] = sparse.load_npz(os.path.join(path, fname) + '.npz')
            assert type(data[key]) in {sp.sparse.csr_matrix, sp.sparse.csc_matrix}
            assert 'orig_masks' not in fname
        print()
#    elif option == 'dict_tup':
#        with open(path + '/' + 'keys', 'rb') as f:
#            keys = np.load(f, allow_pickle = True)
#        data = {}
#        print()
#        list_files = os.listdir(path + '/')
#        if len(list_files) >= max_len:
#            raise ValueError('on purpose because list of files too long')
#        for ii, fname in enumerate(list_files):
#            print('\r'+str(ii), '/', len(list_files), end = '')
#            if 'keys' in fname\
#            or (fname not in keys):
#                continue
#            with open(path + '/' + fname, 'rb') as f:
#                res = np.load(f, allow_pickle = True)
#            assert type(res) == np.ndarray, 'pb array'
#            data[eval(fname)] = res
#        print()
#        for k in keys:
#            assert eval(k) in data, (k, 'pb with keys', option)
    else : 
        raise ValueError('wrong data_type')
    print('{0:{len_str}}'.format('loaded', len_str = len_str))
    return data


   
def file_save(path, data, data_type = None): 
    # save locally before transfer
    ##########################
    msg = '{0:20.20}'.format('saving') + '{0:20.20}'.format(path.replace('/', '_').split('_')[-2] + '_' + path.replace('/', '_').split('_')[-1])
    if   'data'  in path.lower():
        if not config.save_data:
            raise ValueError('on purpose ' + msg)
    elif 'model' in path.lower():
        if not config.save_model:
            raise ValueError('on purpose ' + msg)              
    elif 'pred'  in path.lower():
        if not config.save_predictions:
            raise ValueError('on purpose ' + msg)            
    elif 'perf'  in path.lower():
        if not config.save_performances:
            raise ValueError('on purpose ' + msg)
    else:
        if not config.save: 
            raise ValueError('on purpose ' + msg)            
    ##########################
    dirname = os.path.dirname(path)
    os.makedirs(dirname, exist_ok = True)
    print('{0:{len_str}}'.format(msg, len_str = len_str), end = ' - ')
    if   data_type == 'np':    
        with open(path, 'wb') as f:
            np.save(f,data)
#            elif data_type == 'json':
#                with open(path, 'w') as f:
#                    json.dump(data, f)
    elif data_type == 'pickle':
        with open(path+'.pkl', 'wb') as f:
            pickle.dump(data, f)
#            elif data_type == 'npz':
#                sparse.save_npz(path + '.npz', data)
#            elif data_type == 'csv':
#                data.to_csv(path + '.csv', sep=';', decimal=',')
    elif data_type == 'dict_np':
        print()
        if len(data) >= max_len:
            raise ValueError('on purpose because list of files too long')
        assert type(data) == dict, 'wrong data type'
        os.makedirs(path, exist_ok = True)
        with open(os.path.join(path, 'keys'), 'wb') as f:
            pickle.dump(list(data.keys()), f)
        for ii, (key, obj_to_save) in enumerate(data.items()):
            print('\r{0:6} / {1:6}'.format(ii, len(data)), end = '')
            assert type(obj_to_save) == np.ndarray
            assert 'orig_masks' not in path
            if len(data) > 10:
                print('\r{0:6}'.format(ii), '/', '{0:5}'.format(len(data)), end = '')
            if type(key) == tuple:
                fname = repr(key)
            elif type(key) == str:
                fname = key
            else:
                fname = '$'.join(list(key))
            with open(os.path.join(path, fname), 'wb') as f:
                np.save(f, obj_to_save)
    elif data_type == 'dict_sp':
        print()
        if len(data) >= max_len:
            raise ValueError('on purpose because list of files too long')
        assert type(data) == dict, 'wrong data type'
        os.makedirs(path, exist_ok = True)
        with open(os.path.join(path, 'keys'), 'wb') as f:
            pickle.dump(list(data.keys()), f)
        for ii, (key, obj_to_save) in enumerate(data.items()):
            print('\r{0:6} / {1:6}'.format(ii, len(data)), end = '')
            assert type(obj_to_save) in {sp.sparse.csr_matrix, sp.sparse.csc_matrix}
            assert 'orig_masks' not in path
            if len(data) > 10:
                print('\r{0:6}'.format(ii), '/', '{0:5}'.format(len(data)), end = '')
            if type(key) == tuple:
                fname = repr(key)
            elif type(key) == str:
                fname = key
            else:
                fname = '$'.join(list(key))
            sparse.save_npz(os.path.join(path, fname + '.npz'), obj_to_save)
#            elif data_type == 'dict_tup':
#                if len(data) >= max_len:
#                    raise ValueError('on purpose because list of files too long')
#                assert type(data) == dict, 'wrong data type'
#                os.makedirs(path, exist_ok = True)
#                with open(path + '/' + 'keys', 'wb') as f:
#                    np.save(f, np.array([str(k) for k in data.keys()]))
#                for ii, (k, v) in enumerate(data.items()):
#                    print('\r{0:6} / {1:6}'.format(ii, len(data)), end = '')
#                    fname = str(k)
#                    with open(path + '/' + fname, 'wb') as f:
#                        np.save(f, v)
#            elif data_type == 'txt':
#                assert type(data) == str
#                with open(path+'.txt', "w") as text_file:
#                   print(data, file=text_file)
    else:
        raise ValueError('wrong data_type')
    print('{0:{len_str}}'.format('saved', len_str = len_str))
    return data


##profile    
def file_upload(files_to_include, t_o = t_out): 
    raise NotImplementedError('upload')
    # transfer from origin to destination
#    msg = '{0:20.20}'.format('uploading') + '{0:20.20}'.format(', '.join([k.replace('/', '_').split('_')[-2] +'_'+ k.replace('/', '_').split('_')[-1] \
#                                                                for k in files_to_include]))
#    try:
#        if not config.upload']:
#            raise ValueError('on purpose ' + msg)
#        if (   ('data'  in dist_folder.lower() and not config.upload_data'])
#            or ('model' in dist_folder.lower() and not config.upload_model'])
#            or ('pred'  in dist_folder.lower() and not config.upload_pred'])
#            or ('perf'  in dist_folder.lower() and not config.upload_perf'])
#            ):
#            raise ValueError('on purpose ' + msg)
#        else:
#            if socket.gethostname() in hive:
#                pass
#            for inc in files_to_include:
#                command = ['rsync', '-ruvP']
#                assert type(files_to_include[inc][1]) == str
#                assert files_to_include[inc][1] in {
#                                                    'npz', 
#                                                    'csv', 
#                                                    'txt',
#                                                    'dict_np', 
#                                                    'dict_sp', 
#                                                    'dict_tup', 
#                                                    'json', 
#                                                    'pickle', 
#                                                    }, ('otherwise need to adapt the conditions below and in the rest of this module', 
#                                                        '\n' + 'option was {0} in file_download'.format(files_to_include[inc]), 
#                                                        )
#                command += ['--include', os.path.basename(local_folder+inc)
#                                       + '.npz'*(files_to_include[inc][1] == 'npz')
#                                       + '.csv'*(files_to_include[inc][1] == 'csv')
#                                       + '.txt'*(files_to_include[inc][1] == 'txt')
#                                       + '**'  *(files_to_include[inc][1] in {
#                                                                              'dict_np', 
#                                                                              'dict_sp', 
#                                                                              'dict_tup', 
#                                                                              'json', 
#                                                                              'pickle', 
#                                                                              })
#                                       ]
#                folder_remote_create(os.path.dirname(dist_folder+inc))
#                command += ['--exclude', '*', 
#                            os.path.dirname(local_folder+inc)+'/', 
#                            os.path.dirname(dist_folder +inc),
#                            ]
#                print('{0:{len_str}}'.format(msg, len_str = len_str), end = ' - ')
#                subprocess.check_output(command, timeout = t_o)
#                print('{0:{len_str}}'.format('uploaded', len_str = len_str))
#    except OSError as e:
#        print(colored('fail', 'red'), colored(repr(e), 'red'))
#        raise e
#    except tuple3errors as e:
#        print(colored('fail', 'red'), colored(repr(e), 'red'))
#        raise e
#    except Exception as e:
#        raise e
#
#    
#    
#def folder_remote_create(dist_folder):
#    # Create directory through ssh
#    user, server, folder = dist_folder.replace(':', '@').split('@')
#    try:
#        e = subprocess.Popen(
#                             ['ssh', user + '@' + server, 'mkdir -p ', folder], 
#                             bufsize=4096, stdout=subprocess.PIPE, stderr = subprocess.PIPE
#                             )
#        e.wait()
#        assert e.poll() != None
#        del e
#    except Exception as e:
#        print(colored('server : {0}\nfolder : {1}\n'.format(server, folder) + str(e), 'yellow', 'on_red'))
#        raise e

###############################################################################

##profile    
def batch_load(local_folder, prefix = None, data_name = None, data_type = None):
    fpath = os.path.join(prefix,
                         tools.format_file_names(data_name),
                         )
    files_to_fetch    = {
                         fpath    : data_type,
                         }
    #dist__folder = None#dist_from_local_folder(local_folder)
    file = file_fetch(local_folder,
                      files_to_fetch,
                      )#, dist__folder, local_folder, param)
    if data_type == 'np':
        file  = file[()]
    return file


def batch_save(local_folder, data = None, prefix = None, data_name = None, data_type = None): 
    fpath = os.path.join(prefix, 
                         tools.format_file_names(data_name),
                         )
    files_to_save     = {
                         fpath : (data, data_type), 
                         }
    for k, v in files_to_save.items():
        file_save(os.path.join(local_folder,
                               k,
                               ), 
                  v[0], 
                  data_type = v[1],
                  )
    #dist__folder = dist_from_local_folder(local_folder)
    #file_upload(local_folder, dist__folder, files_to_save, hprm)
        
        
def batch_sync(local_folder, prefixes, param, data_type = None):
    raise NotImplementedError
#    assert type(prefixes) == list
#    for fpath in prefixes:
#        files_to_fetch = {fpath : data_type}
#        distant_folder = local_to_distant_folder(local_folder)
#        file_download(dist__folder, local_folder, files_to_fetch, param, to = t_out)
#    return 0
    
        
#def local_to_distant_folder(local_folder):
#    if   'model' in local_folder: 
#        dist__folder = dist__folder_model
#    elif 'Data' in local_folder: 
#        dist__folder = dist__folder_data
#    elif 'Perf' in local_folder: 
#        dist__folder = dist__folder_perf
#    elif 'Pred' in local_folder: 
#        dist__folder = dist__folder_pred
#    elif 'Results' in local_folder: 
#        dist__folder = dist__folder_results
#    else:
#        raise ValueError('Incorrect Local Folder')
#    return dist__folder
#    

