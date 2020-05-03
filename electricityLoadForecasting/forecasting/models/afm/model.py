
import time
import os
import sys
import numpy as np
import copy as cp
import scipy as sp
import scipy.signal  as sig
import scipy.ndimage as spim
import numbers
from termcolor import colored
#
from electricityLoadForecasting import tools, paths
from . import lbfgs

###############################################################################

path_betas = os.path.join(paths.outputs, 'Betas')
path_data  = os.path.join(paths.outputs, 'Data')

EXTRA_CHECK = 1

dikt_var_temp = {
                 'low_rank_U'       : 'low_rank_UVt', 
                 'tensor_product_L' : 'tensor_product_LR',
                 'tensor_product_R' : 'tensor_product_LR',
                 #'sesquivariate_b'  : 'sesquivariate_b',
                 'sesquivariate_m'  : 'sesquivariate_bm',
                 }

class additive_features_model:
    # Defines a model that stores all the covariates and the XtX, XtY
    # with the methods for optimization (proximal-gradient descent or lbfgs), 
    # prediction and loss computation

    def __init__(self, 
                 hprm,             
                 dikt_file_names = None,          
                 ):
        # Sets all the parameters as attributes of the class
        self.hprm        = hprm
        self.dikt        = dikt_file_names
        self.max_iter    = int(self.hprm['afm.algorithm.first_order.max_iter'])
        self.formula     = self.hprm['afm.formula']
        assert not self.formula.empty
        self.alpha       = {var : {key : self.formula.xs((var,key))['regularization_coef']
                                   for key in self.formula.loc[var].index
                                   }
                            for var in self.formula.index.get_level_values('coefficient').unique()
                            }
        self.pen         = {var : {key : self.formula.xs((var,key))['regularization_func']
                                   for key in self.formula.loc[var].index
                                   }
                            for var in self.formula.index.get_level_values('coefficient').unique()
                            }
        self.lbfgs       = (self.hprm['afm.algorithm'] == 'L-BFGS')
        
        if (    'low_rank_UVt' in self.alpha
            and 'low_rank_UVt' in self.pen
            ): 
            # Low-rank (along the substations) coefficient matrix 
            self.alpha['low_rank_U'] = self.alpha['low_rank_UVt']
            self.pen  ['low_rank_U'] = self.pen  ['low_rank_UVt']
            del self.alpha['low_rank_UVt'], self.pen['low_rank_UVt']
            
        if (    'tensor_product_LR' in self.alpha
            and 'tensor_product_LR' in self.pen
            ):
            # Low-rank interaction (independently for each substation)
            self.alpha['tensor_product_L'] = self.alpha['tensor_product_LR']
            self.pen  ['tensor_product_L'] = self.pen  ['tensor_product_LR']
            self.alpha['tensor_product_R'] = self.alpha['tensor_product_LR']
            self.pen  ['tensor_product_R'] = self.pen  ['tensor_product_LR']
            del self.alpha['tensor_product_LR'], self.pen['tensor_product_LR']
        self.gp_pen      = self.hprm['afm.sum_consistent.gp_pen'] # matrix for the sum-consistent model
        
        # Freeze the univariate or the bivariate coefficient matrices (rarely used)        
        self.frozen_variables = self.hprm['afm.algorithm.first_order.frozen_variables']
        
        # Stopping criteria
        self.tol               = self.hprm['afm.algorithm.first_order.tol']
        self.norm_grad_min     = self.hprm['afm.algorithm.first_order.norm_grad_min']
        self.dist_min          = self.hprm['afm.algorithm.first_order.dist_min']
        del hprm, dikt_file_names
        
        # Descent method
        self.bcd             = self.hprm['afm.algorithm.first_order.bcd']
        self.batch_cd        = not self.bcd
        self.active_set      = self.hprm['afm.algorithm.first_order.active_set']
        self.prop_active_set = self.hprm['afm.algorithm.first_order.prop_active_set']
        self.col_upd         = self.hprm['afm.algorithm.first_order.column_update'] if self.bcd else {}
        
        # Line-Search parameters  
        self.eta                = {} if self.bcd else 1
        self.etaA               = 1
        self.eta1               = 1
        self.eta2               = 1
        self.theta              = 0.5
        self.proba_ls           = 0.2
        self.nb_steps           = 0

        print('Formula      :', str(self.formula), '\n')      
        print('batch' if self.batch_cd else 'bcd per col for ' + repr([k for k, v in self.col_upd.items() if v]))
        assert self.batch_cd + self.bcd == 1    

    def fit(self, 
            specs,
            Y_training, 
            X_training, 
            XtX_training   = None, 
            Y_validation   = None, 
            X_validation   = None, 
            XtX_validation = None, 
            given_warm     = None,
            ):
        # Fot the model given the training and validation matrices (used for stopping criteria)
        self.Y_training   = Y_training
        self.X_training   = X_training
        self.XtX_training = XtX_training
        del X_training, Y_training, XtX_training
        self.n_training   = self.Y_training.shape[0]
        self.k            = self.Y_training.shape[1]
        self.mask         = specs['mask_univariate']
        self.size         = specs['size_univariate']
        self.mask.update(specs.get('mask_bivariate', {}))
        self.size.update(specs.get('size_bivariate', {}))
        self.size_tensor_bivariate = specs.get('size_tensor_bivariate',{})
        print('masks examples : ')
        self.active_gp = (self.gp_pen > 0) and bool(self.hprm['afm.sum_consistent.gp_matrix']) # indicator of the sum-consistent model
        if self.active_gp: # Parameters for the sum-consistent model
            self.make_gp_matrix()
            self.partition_tuples = {tuple(self.gp_matrix[:,k].nonzero()[0])
                                     for k in range(self.k)
                                     }
            self.partition_posts_to_tuples = {k : tuple(self.gp_matrix[:,k].nonzero()[0])
                                              for k in range(self.k)
                                              }
            self.partition_tuples_to_posts = {tt : [k for k in range(self.k) if self.partition_posts_to_tuples[k] == tt]
                                              for tt in self.partition_posts_to_tuples.values()
                                              }
                
        if (self.hprm['plot.afm'] or self.hprm['afm.algorithm.first_order.early_stop_ind_validation']):
            self.M_mean    = np.ones((self.k, self.k))/self.k
            
        # Warm-start
        self.given_coef = {} if given_warm is None else given_warm
        if self.given_coef:
            print(colored('Coef given', 'green'))
        del given_warm
        self.print_fit_with_mean(dataset = 'training', mean = 'training')
        
        # Precomputations training
        if not self.lbfgs:    
            print('Precomputations training')
            self.Y_sqfrob_training, self.YtY_training, self.XtY_training = self.precomputationsY(self.X_training, 
                                                                                                 self.Y_training,
                                                                                                 dataset = 'training',
                                                                                                 )
            print('Precomputations training done')        

        # validation dataset
        self.compute_validation    = bool(X_validation)
        self.precompute_validation = self.compute_validation and (not self.lbfgs)
        if self.compute_validation:
            self.Y_validation    = Y_validation
            self.X_validation    = X_validation
            self.XtX_validation  = XtX_validation
            del X_validation, Y_validation, XtX_validation
            self.n_validation    = self.Y_validation.shape[0]
            self.print_fit_with_mean(dataset = 'validation', mean = 'training')
            self.print_fit_with_mean(dataset = 'validation', mean = 'validation')
            if self.precompute_validation:
                print('Precomputations validation')
                self.Y_sqfrob_validation, self.YtY_validation, self.XtY_validation = self.precomputationsY(self.X_validation, 
                                                                                                           self.Y_validation,
                                                                                                           dataset = 'validation',
                                                                                                           )
                print('Precomputations validation done')
        self.iteration = 0

        print('Initialize Coefficients')
        self.coef, self.keys_upd = self.initialize_coef()
        print('    ', len(self.coef),     'coef')
        print('    ', len(self.keys_upd), 'keys_upd')
        self.orig_masks = self.make_orig_masks()
        self.dikt_masks = {k:v
                           for k, v in self.orig_masks.items() 
                           if (   type(v) != type(slice(None))
                               or v!= slice(None)
                               )
                           }
        for (var, key), v in self.coef.items():
            if key in self.dikt_masks:
                self.coef[var,key] = sp.sparse.csc_matrix(v)
        if EXTRA_CHECK:
            self.coef_prec = cp.deepcopy(self.coef)

        """
        Set flags
        """
        if not self.lbfgs:
            if self.bcd: #Need to be after initialize coef for self.keys_upd
                self.epoch_stopping_criteria = int(len(self.keys_upd)*2)
                self.epoch_print_progress    = 1 if EXTRA_CHECK else 100
                self.epoch_check_pred        = 1 if EXTRA_CHECK else 100
            else:
                raise ValueError # Need two check stopping criteria epochs ...
                self.epoch_stopping_criteria = 100
                self.epoch_print_progress    = 2
                self.epoch_check_pred        = 1 if EXTRA_CHECK else 100
            # Epochs
            self.epoch_compute_validation     = 1
            self.epoch_compute_ind_validation = self.epoch_stopping_criteria
            self.epoch_print_info             = 1 if EXTRA_CHECK else int(    self.epoch_stopping_criteria) 
            self.flag_stopping_criteria       = int(2 * self.epoch_stopping_criteria) # Needs two periods cause two means are taken
            self.flag_print_info              = 0
            self.flag_check_fit               = 1e5
            # Flags
            self.flag_print_progress         = 0
            self.flag_show_prison            = 0
            self.flag_compute_validation     = -1 #self.flag_print_progress - 1
            self.flag_compute_ind_validation = int(5 * self.epoch_stopping_criteria)
            self.flag_check_pred             = -1
            # 
            self.print_len_epochs()
            self.begin_epoch            = time.time()

        try:
            """
            Try to load final model
            """
            keys_upd   = tools.batch_load(path_betas, 
                                          prefix    = self.dikt['experience.whole'], 
                                          data_name = 'keys_upd', 
                                          data_type = 'pickle',
                                          )
            assert set(self.keys_upd) == set(keys_upd), 'set(self.keys_upd) != set(keys_upd)'
            self.coef  = tools.batch_load(path_betas, 
                                          prefix    = self.dikt['experience.whole'], 
                                          data_name = 'coef', 
                                          data_type = 'dict_np',
                                          )
            if self.lbfgs:
                shared_keys = sorted([e 
                                      for e in self.X_training.keys() 
                                      if e not in self.mask
                                      ])
                owned_keys  = sorted([e 
                                      for e in self.X_training.keys() 
                                      if e in self.mask
                                      ])
                self.sorted_keys = shared_keys + owned_keys
                for var, key in self.coef:
                    if var in self.keys:
                        self.keys[var].append(key)
                    else:
                        self.keys[var] = [key]
            print(colored('Coef loaded', 'blue'))
            self.change_sparse_structure()
        except tools.loading_errors as e:
            ### Begin algorithm then
            print(colored(str(e), 'red'))
            print(colored('coef not loaded', 'red'))
            self.sorted_keys, self.cats_owned = self.sort_keys(self.X_training.keys(), 
                                                               self.mask, 
                                                               )
            self.compute_normalization()
            # Normalize data
            for key in self.sorted_keys:
                inpt, location = key
                self.X_training   [key] = self.X_training   [key] / self.normalization[inpt]
                self.X_validation [key] = self.X_validation [key] / self.normalization[inpt]
            if hasattr(self, 'XtX_training'):
                for keys in self.XtX_training.keys():
                    key1, key2 = keys
                    inpt1, location1 = key1
                    inpt2, location2 = key2
                    self.XtX_training[keys] = self.XtX_training[keys] / (self.normalization[inpt1]*self.normalization[inpt2])
                    if self.precompute_validation:
                        self.XtX_validation[keys] = self.XtX_validation[keys] / (self.normalization[inpt1]*self.normalization[inpt2])
                if not self.lbfgs:
                    for key in self.XtY_training.keys():
                        inpt, location = key
                        self.XtY_training[key] = self.XtY_training[key] / self.normalization[inpt]
                        if self.precompute_validation:
                            self.XtY_validation[key] = self.XtY_validation [key] / self.normalization[inpt]
            if self.lbfgs:
                lbfgs.start_lbfgs(self)
                # Reorganize the coefficients
                for ii, key in enumerate(self.X_training.keys()):
                    self.coef['unconstrained',key] = cp.deepcopy(self.bfgs_long_coef[slice(self.key_col_large_matching[key][0], 
                                                                                           self.key_col_large_matching[key][1],
                                                                                           )
                                                                                     ])
                    if type(self.coef['unconstrained',key]) in {sp.sparse.csc_matrix, sp.sparse.csr_matrix}:
                        self.coef['unconstrained',key] = self.coef['unconstrained',key].toarray()                       
                # Change data type for sparsity rules
                self.change_sparse_structure()
            else:
                # Use the first-order algorithm
                # Initialize arrays
                self.change_sparse_structure()
                self.fit_training     = np.zeros(self.max_iter)
                if self.hprm['plot.afm'] or self.hprm['afm.algorithm.first_order.early_stop_ind_validation']:
                    self.fit_ind_training  = np.zeros((self.max_iter, self.k))
                    self.fit_mean_training = np.zeros(self.max_iter)
                self.fit_gp_training = np.zeros(self.max_iter)
                self.reg             = np.zeros(self.max_iter)
                self.ridge           = np.zeros(self.max_iter)
                self.obj_training    = np.zeros(self.max_iter)
                self.etas            = np.zeros(self.max_iter)
                self.etaAs           = np.zeros(self.max_iter)
                self.eta1s           = np.zeros(self.max_iter)
                self.eta2s           = np.zeros(self.max_iter)
                self.gap_B, self.gap_U, self.gap_V, self.gap_D, self.gap_W, self.gap_Z = [[] for k in range(6)] 
                if self.compute_validation:
                    self.fit_validation     = np.zeros(self.max_iter)
                    if self.hprm['plot.afm'] or self.hprm['afm.algorithm.first_order.early_stop_ind_validation']:
                        self.fit_ind_validation  = np.zeros((self.max_iter, self.k))                
                        self.fit_mean_validation = np.zeros(self.max_iter)                
                    self.fit_gp_validation  = np.zeros(self.max_iter)
                    self.obj_validation     = np.zeros(self.max_iter)
                self.warm_start()
                
                ### First values
                print('Compute fit training')
                self.cur_fit_training    = self.evaluate_fit(self.coef, dataset = 'training')
                self.cur_fit_gp_training = self.evaluate_fit(self.coef, dataset = 'training', 
                                                             gp_matrix = self.gp_matrix, 
                                                             gp_pen = self.gp_pen,
                                                             ) if self.active_gp else 0
                print('Compute reg')
                self.slope_ind  = {}
                self.offset_ind = {}
                self.cur_ind_reg, self.slope_ind, self.offset_ind = self.evaluate_ind_reg({coor:(self.coef[coor[:2]][:,:,self.dikt_masks.get(coor, slice(None))] 
                                                                                                 if coor[0] in {'tensor_product_L','tensor_product_R'}
                                                                                                 else
                                                                                                 self.coef[coor[:2]][:,self.dikt_masks.get(coor, slice(None))]
                                                                                                 )
                                                                                           for coor in self.keys_upd
                                                                                           })
                self.cur_reg       = np.sum([v for k, v in self.cur_ind_reg.items()])
                print('Compute ridge')
                self.cur_ind_ridge = self.evaluate_ind_ridge({coor:(self.coef[coor[:2]][:,:,self.dikt_masks.get(coor, slice(None))]
                                                                    if coor[0] in {'tensor_product_L','tensor_product_R'}
                                                                    else
                                                                    self.coef[coor[:2]][:,self.dikt_masks.get(coor, slice(None))]
                                                                    )
                                                              for coor in self.keys_upd
                                                              })
                self.cur_ridge     = np.sum([v for k, v in self.cur_ind_ridge.items()])
                print('Compute obj')
                self.cur_obj_training = (  self.cur_fit_training
                                         + self.cur_fit_gp_training
                                         + self.cur_reg
                                         + self.cur_ridge
                                         )
                print('Update arrays')
                self.fit_training[self.iteration] = self.cur_fit_training
                if (   self.hprm['plot.afm'] 
                    or self.hprm['afm.algorithm.first_order.early_stop_ind_validation']
                    ):
                    self.fit_ind_training[self.iteration]  = self.evaluate_ind_fit(self.coef, dataset = 'training')
                    self.fit_mean_training[self.iteration] = self.evaluate_fit(self.coef, dataset = 'training', gp_matrix = self.M_mean, gp_pen = 1)
                if self.active_gp:
                    self.fit_gp_training[self.iteration] = self.cur_fit_gp_training
                self.reg         [self.iteration] = self.cur_reg      
                self.ridge       [self.iteration] = self.cur_ridge      
                self.obj_training[self.iteration] = self.cur_obj_training
                self.decr_obj_0    = 10**(np.ceil(np.log(self.cur_obj_training)/np.log(10)))
                self.decr_obj      = self.decr_obj_0
                if self.compute_validation : 
                    print('Compute fit validation')
                    self.cur_fit_validation = self.evaluate_fit(self.coef, dataset = 'validation')
                    if self.hprm['plot.afm'] or self.hprm['afm.algorithm.first_order.early_stop_ind_validation']:
                        self.fit_ind_validation[self.iteration]  = self.evaluate_ind_fit(self.coef, dataset = 'validation')
                        self.fit_mean_validation[self.iteration] = self.evaluate_fit(self.coef, dataset = 'validation', gp_matrix = self.M_mean, gp_pen = 1)
                    self.cur_fit_gp_validation = self.evaluate_fit(self.coef, dataset = 'validation', gp_matrix = self.gp_matrix, gp_pen = self.gp_pen) if self.active_gp else 0
                    print('Compute ridge validation')
                    self.fit_validation[self.iteration] = self.cur_fit_validation
                    if self.active_gp:
                        self.fit_gp_validation[self.iteration] = self.cur_fit_gp_validation
                    self.cur_obj_validation = (  self.fit_validation[self.iteration]
                                               +  self.cur_fit_gp_validation
                                               +  self.cur_reg
                                               +  self.cur_ridge
                                               )
                    self.obj_validation[self.iteration] = self.cur_obj_validation
                self.iteration += 1 
        
                if EXTRA_CHECK:
                    self.Y_training0 = self.Y_training.copy()
                    self.X_training0 = cp.deepcopy(self.X_training)
                
                ### Start first-order descent
                self.start_descent()
                self.last_iteration = self.iteration
                print('last iteration : ', self.last_iteration)
                if (    self.hprm['plot.afm']
                    and '_sv_' not in self.dikt['experience.whole']
                    ):
                    try:
                        tools.plotting.afm.plot_fit(self, io = 0, ltx = 1)
                    except Exception as e:
                        print(colored(' \n \n' + repr(e) + '\n \n ', 'red', 'on_cyan'))

            # Algorithm terminated, denormalize data
            for key in self.sorted_keys:
                inpt, location = key
                self.X_training  [key] *= self.normalization[inpt]
                self.X_validation[key] *= self.normalization[inpt]
            if hasattr(self, 'XtX_training'):
                for keys in self.XtX_training.keys():
                    key1, key2 = keys
                    inpt1, location1 = key1
                    inpt2, location2 = key2                    
                    self.XtX_training[keys] *= self.normalization[inpt1]*self.normalization[inpt2]
                    if self.precompute_validation:
                        self.XtX_validation[keys] *= self.normalization[inpt1]*self.normalization[inpt2]
                if not self.lbfgs:
                    for key in self.XtY_training.keys():
                        inpt, location = key
                        self.XtY_training[key] *= self.normalization[inpt]
                        if self.precompute_validation:
                            self.XtY_validation[key] *= self.normalization[inpt]
            # Change coef accordingly
            for ii, (var,key) in enumerate(self.coef.keys()):
                inpt, location = key
                if var not in {'low_rank_V', 'tensor_product_R', 'sesquivariate_m'}:
                    self.coef[var,key] /= self.normalization[inpt] 
            ### Save
            for obj, name, opt in [(self.coef,       'coef',       'dict_np'), 
                                   (self.keys_upd,   'keys_upd',   'pickle'), 
                                   ]:
                try:
                    tools.batch_save(path_betas, 
                                     data      = obj, 
                                     prefix    = self.dikt['experience.whole'], 
                                     data_name = name, 
                                     data_type = opt, 
                                     )
                except tools.saving_errors as e:
                    print(e)
        if hasattr(self, 'fit_gp_training') and not self.active_gp:
            del self.fit_gp_training
        if hasattr(self, 'fit_gp_validation' ) and not self.active_gp:
            del self.fit_gp_validation

    
    def start_descent(self):
        if hasattr(self, 'X_validation'):
            assert len(self.X_training) == len(self.X_validation), (len(self.X_training), len(self.X_validation))
        #assert not self.hprm['afm.algorithm.first_order.sparse_coef'], 'slower'
        print('Start Descent')
        self.converged = 0
        self.list_coor = []
        np.random.seed(0)
        import gc; gc.collect()
        while not self.converged:
            if EXTRA_CHECK:
                print('\n'+colored('New Iteration', 'green'))
                assert np.all(self.Y_training == self.Y_training0)
                for key in self.X_training:
                    if type(self.X_training[key]) == np.ndarray:
                        assert np.all(self.X_training[key] == self.X_training0[key])
                    elif type(self.X_training0[key]) == sp.sparse.csc_matrix:
                        assert np.all(self.X_training[key].data == self.X_training0[key].data)
            if self.iteration >= self.flag_print_progress:
                self.print_progress()
                self.flag_print_progress += self.epoch_print_progress
                prprog = 1
            if self.batch_cd:
                assert 0
                dikt_fit_grad = self.compute_grad(self.coef)
                if self.active_gp:
                    dikt_fit_gp_grad = self.compute_grad(self.coef, gp_matrix = self.gp_matrix, gp_pen = self.gp_pen)
                q_training            = None
                old_part_fit_training = None
            elif self.bcd:
                coor_upd  = self.pop_coordinates() # Select a covariate (or a block) to update
                #if coor_upd[0] == 'sesquivariate_m': continue
                mask_upd  = self.dikt_masks.get(coor_upd, slice(None))
                # Extra checks
                orig_mask = self.orig_masks.get(coor_upd, slice(None))
                orig_mask = np.arange(self.k) if type(orig_mask) == type(slice(None)) else orig_mask
                out_mask  = [k for k in orig_mask if k not in (mask_upd if hasattr(type(mask_upd), 'len') else np.arange(self.k))]
                if not self.life_sentences:
                    if type(self.coef[coor_upd[:2]]) in {np.ndarray, np.matrix}:
                        assert np.linalg.norm(self.coef[coor_upd[:2]][:,out_mask]) == 0
                    else:
                        assert sp.sparse.linalg.norm(self.coef[coor_upd[:2]][:,out_mask]) == 0
                if prprog:
                    prprog = 0
                    print('    - ', 
                          '{0:5.5}' .format(coor_upd[0]), 
                          '-',
                          '{0:20.20}'.format(repr(coor_upd[1])),
                          '-',
                          '{0:8.8}' .format(repr(coor_upd[2])),  
                          '-', 
                          '{0:25.25}'.format(repr(self.dikt_masks.get(coor_upd, 'no_mask'))),
                          end = '')
                if EXTRA_CHECK:
                    print(coor_upd)
                coef_old_masked = {}
                if coor_upd[0] == 'low_rank_U':
                    coef_old_masked[coor_upd] = self.coef[coor_upd[:2]][:,mask_upd]
                    coef_old_masked['low_rank_UVt',coor_upd[1]] = self.coef['low_rank_UVt',coor_upd[1]][:,mask_upd]
                elif coor_upd[0] in {'tensor_product_L', 'tensor_product_R'}:
                    coef_old_masked[coor_upd] = self.coef[coor_upd[:2]][:,:,mask_upd]
                elif coor_upd[0] in {'sesquivariate_m'}:
                    coef_old_masked[coor_upd] = self.coef[coor_upd[:2]][:,mask_upd]
                else:
                    coef_old_masked[coor_upd] = self.coef[coor_upd[:2]][:,mask_upd]
                q_training, extra_part_training, dikt_fit_grad = self.compute_part_grad(coor_upd, mask_upd, dataset = 'training')
                if self.precompute_validation and self.iteration >= self.flag_compute_validation :
                    q_validation, extra_part_validation = self.compute_part_grad(coor_upd, mask_upd, dataset = 'validation')                    
                    old_part_fit_validation       = self.part_fit(coef_old_masked, coor_upd, q_validation, extra_part_validation)
                if self.active_gp:
                    q_gp_training, extra_part_gp_training, dikt_fit_gp_grad = self.compute_part_grad(coor_upd, mask_upd, dataset = 'training',  MMt = self.MMt, gp_pen = self.gp_pen)
                    if self.precompute_validation and self.iteration >= self.flag_compute_validation :
                        q_gp_validation, extra_part_gp_validation = self.compute_part_grad(coor_upd, mask_upd, dataset = 'validation',  MMt = self.MMt, gp_pen = self.gp_pen)
                        old_part_fit_gp_validation = self.part_fit(coef_old_masked, coor_upd, q_gp_validation,  extra_part_gp_validation)
                else:
                    dikt_fit_gp_grad     = {}
                    q_gp_training           = {}
                    extra_part_gp_training  = {}
                    if self.precompute_validation and self.iteration >= self.flag_compute_validation :
                        q_gp_validation            = {}
                        old_part_fit_gp_validation = 0
                if self.hprm['plot.afm'] or self.hprm['afm.algorithm.first_order.early_stop_ind_validation']:
                    old_part_fit_ind_training  = self.part_fit(coef_old_masked,
                                                               coor_upd,
                                                               q_training,
                                                               extra_part_training,
                                                               dataset = 'training',
                                                               indi    = True,
                                                               )
                if self.hprm['plot.afm']:
                    q_mean_training, extra_part_mean_training, _ = self.compute_part_grad(coor_upd,
                                                                                          mask_upd,
                                                                                          dataset = 'training',
                                                                                          MMt     = self.M_mean,
                                                                                          gp_pen = 1,
                                                                                          )
                    old_part_fit_mean_training = self.part_fit(coef_old_masked, coor_upd, q_mean_training, extra_part_mean_training, dataset = 'training') 
                    if self.precompute_validation:
                        old_part_fit_ind_validation  = self.part_fit(coef_old_masked, coor_upd, q_validation,    extra_part_validation,    dataset = 'validation', indi = True)
                        q_mean_validation, extra_part_mean_validation = self.compute_part_grad(coor_upd, mask_upd, dataset = 'validation',  MMt = self.M_mean, gp_pen = 1)
                        old_part_fit_mean_validation = self.part_fit(coef_old_masked, coor_upd, q_mean_validation, extra_part_mean_validation, dataset = 'validation') 
                old_part_ridge           = self.cur_ind_ridge.get(coor_upd, 0)
                old_part_fit_training    = self.part_fit(coef_old_masked, coor_upd, q_training,    extra_part_training,    dataset = 'training')
                old_part_fit_gp_training = self.part_fit(coef_old_masked, coor_upd, q_gp_training, extra_part_gp_training, dataset = 'training') if self.active_gp else 0
            dikt_ridge_grad = self.compute_grad_ridge(list(dikt_fit_grad.keys()), self.dikt_masks)
            if EXTRA_CHECK:
                for key in set(dikt_fit_grad.keys()).union(dikt_fit_gp_grad.keys()).union(dikt_ridge_grad.keys()):
                    if key in dikt_fit_gp_grad:
                        assert dikt_fit_grad[key].shape == dikt_fit_gp_grad[key].shape, (dikt_fit_grad[key].shape, dikt_fit_gp_grad[key].shape)
                    if key in dikt_ridge_grad:
                        assert dikt_fit_grad[key].shape == dikt_ridge_grad[key].shape, (dikt_fit_grad[key].shape, dikt_ridge_grad[key].shape)
            # Backtracking step
            (tmp_fit_training, tmp_fit_gp_training, tmp_ind_ridge, fit_plus_gp_plus_ridge_tilde_tmp,
              coef_tmp,        nb_inner_iter,       eta,           condition_ls,
              ) = self.backtracking_ls(dikt_fit_grad, 
                                       dikt_fit_gp_grad, 
                                       dikt_ridge_grad, 
                                       old_fit_training         = self.cur_fit_training, 
                                       old_fit_gp_training      = self.cur_fit_gp_training, 
                                       old_part_fit_training    = old_part_fit_training, 
                                       old_part_fit_gp_training = old_part_fit_gp_training, 
                                       old_ridge                = self.cur_ridge, 
                                       old_part_ridge           = old_part_ridge, 
                                       quant_training           = q_training,
                                       quant_gp_training        = q_gp_training,
                                       cur_ind_reg              = self.cur_ind_reg,
                                       mask_upd                 = mask_upd,
                                       ) 
            new_part_ridge = np.sum([tmp_ind_ridge[key] for key in tmp_ind_ridge.keys()])
            tmp_ind_reg, tmp_ind_slope, tmp_ind_offset = self.evaluate_ind_reg(coef_tmp)
            new_part_reg   = np.sum([tmp_ind_reg[key]   for key in tmp_ind_reg.keys()])
            if   self.batch_cd:
                tmp_reg  = new_part_reg
            elif self.bcd:
                # update old_ridge and old_reg because coef_tmp might contain less keys than coor_upd id BLS fails
                old_part_reg   = np.sum([self.cur_ind_reg  [key] for key in tmp_ind_reg  .keys()])
                old_part_ridge = np.sum([self.cur_ind_ridge[key] for key in tmp_ind_ridge.keys()])
                diff_reg     = new_part_reg   - old_part_reg#np.sum([tmp_ind_reg[key] - self.cur_ind_reg[key] for key in tmp_ind_reg.keys()])
                tmp_reg      = self.cur_reg   + diff_reg
                diff_ridge   = new_part_ridge - old_part_ridge
                tmp_ridge    = self.cur_ridge + diff_ridge
                assert tmp_reg   >= -1e-12
                assert tmp_ridge >= 0
                if not condition_ls:
                    self.punished_coor.add(coor_upd)
            tmp_obj_training = tmp_fit_training + tmp_fit_gp_training + tmp_ridge + tmp_reg
            if EXTRA_CHECK:
                assert self.cur_fit_training + self.cur_fit_gp_training + self.cur_ridge + self.cur_reg >= fit_plus_gp_plus_ridge_tilde_tmp + new_part_reg
                if not( tmp_obj_training <= self.cur_obj_training + 1e-12  or (self.pen == 'trend_filtering' and np.abs(tmp_obj_training - self.cur_obj_training) <= 1e-6)):
                    print('\n\n',
                          self.pen, '\n',
                          np.abs(tmp_obj_training - self.cur_obj_training), '\n',
                          coor_upd, '\n',
                          '\n')
                    var, key, ind = coor_upd
                    pen, alpha    = self.get_pen_alpha(var, key)
                    other_reg     = np.sum([self.cur_ind_reg[key] for key in self.cur_ind_reg if key not in coef_tmp])
                    tmp_reg       = other_reg + new_part_reg 
                    assert tmp_fit_training + tmp_fit_gp_training + tmp_ridge + tmp_reg        <= self.cur_obj_training + 1e-12
                    assert tmp_fit_training + tmp_fit_gp_training + tmp_ridge + tmp_reg        <= fit_plus_gp_plus_ridge_tilde_tmp + tmp_reg+ 1e-12
                    assert fit_plus_gp_plus_ridge_tilde_tmp + tmp_reg <= self.cur_obj_training+ 1e-12                
                    

            if self.bcd:
                (s,key,ind) = coor_upd
                if self.cur_obj_training - tmp_obj_training <= 1e-14:
                    if self.active_set and self.col_upd.get((s,key)): 
                        self.punished_coor.add(coor_upd)   
                else: # eta won't be created/updated if decrease too small
                    if s in {'unconstrained',
                             'low_rank_U',
                             'sesquivariate_b',
                             }:
                        self.eta1 = eta/(self.theta**np.random.binomial(1, self.proba_ls)) 
                    elif s in {'tensor_product_L',
                               'tensor_product_R',
                               'sesquivariate_m',
                               }:
                        self.eta2 = eta/(self.theta**np.random.binomial(1, self.proba_ls)) 
                    self.eta[coor_upd] = eta/(self.theta**np.random.binomial(1, self.proba_ls))
            else:                
                self.eta = eta/(self.theta**np.random.binomial(1, self.proba_ls)) 
            self.nb_steps += condition_ls
                
            # Insert intermediate values
            slice_inner_iter = slice(self.iteration, self.iteration + nb_inner_iter - 1)
            self.etas        [slice_inner_iter] = self.etas        [self.iteration - 1]
            self.etaAs       [slice_inner_iter] = self.etaAs       [self.iteration - 1]
            self.eta1s       [slice_inner_iter] = self.eta1s       [self.iteration - 1]
            self.eta2s       [slice_inner_iter] = self.eta2s       [self.iteration - 1]
            self.fit_training   [slice_inner_iter] = self.fit_training   [self.iteration - 1]
            if self.hprm['plot.afm'] or self.hprm['afm.algorithm.first_order.early_stop_ind_validation']:
                self.fit_ind_training[slice_inner_iter] = self.fit_ind_training[self.iteration - 1]
                self.fit_mean_training[slice_inner_iter] = self.fit_mean_training[self.iteration - 1]
            if self.active_gp:
                self.fit_gp_training[slice_inner_iter] = self.fit_gp_training[self.iteration - 1]
            self.reg      [slice_inner_iter] = self.reg      [self.iteration - 1]
            self.ridge    [slice_inner_iter] = self.ridge    [self.iteration - 1]
            self.obj_training[slice_inner_iter] = self.obj_training[self.iteration - 1]
            if self.compute_validation:# and False: 
                self.fit_validation   [slice_inner_iter] = self.fit_validation[self.iteration - 1]
                if self.hprm['plot.afm'] or self.hprm['afm.algorithm.first_order.early_stop_ind_validation']:
                    self.fit_ind_validation[slice_inner_iter]  = self.fit_ind_validation[self.iteration - 1]
                    self.fit_mean_validation[slice_inner_iter] = self.fit_mean_validation[self.iteration - 1]
                if self.active_gp:
                    self.fit_gp_validation[slice_inner_iter] = self.fit_gp_validation[self.iteration - 1]
                self.obj_validation[slice_inner_iter] = self.obj_validation[self.iteration - 1]

            if EXTRA_CHECK:
                temp__, _, _ = self.evaluate_ind_reg({(var,key,ind):(self.coef[var,key][:,:,mask_upd] if var in {'tensor_product_L','tensor_product_R'} else self.coef[var,key][:,mask_upd]) for (var,key,ind) in self.keys_upd})
            # Update current variables
            self.iteration    += nb_inner_iter - 1 
            max_iter_reached   = (self.iteration >= self.max_iter)
            if not (self.bcd):
                self.decr_obj = self.cur_obj_training - tmp_obj_training
            self.cur_fit_training    = tmp_fit_training
            self.cur_fit_gp_training = tmp_fit_gp_training
            self.cur_ridge           = tmp_ridge                       
            self.cur_reg             = tmp_reg                        
            self.cur_obj_training    = tmp_obj_training              
            self.update_coef(coef_tmp,
                             self.dikt_masks,
                             new_ind_slope  = tmp_ind_slope,
                             new_ind_offset = tmp_ind_offset,
                             )
            self.cur_ind_reg.update(tmp_ind_reg)
            self.cur_ind_ridge.update(tmp_ind_ridge)
            
            if EXTRA_CHECK:
                for coor in self.coef:
                    assert coor == coor[:2]
                tmp2_ind_reg, _, _ = self.evaluate_ind_reg({(*coor,()):self.coef[coor[:2]] for coor in self.coef})
                tmp2_reg           = np.sum([v for k, v in tmp2_ind_reg.items()])
                mmmm, _, _          = self.evaluate_ind_reg({(var,key,ind):(self.coef[var,key][:,:,mask_upd] if var in {'tensor_product_L','tensor_product_R'} else self.coef[var,key][:,mask_upd]) for (var,key,ind) in self.keys_upd})
                if not np.abs(self.cur_reg - tmp2_reg) <= 1e-12:
                    out_mask = [k for k in range(self.k) if type(mask_upd)!= type(slice(None)) and k not in mask_upd]
                    out_coef = self.coef[coor_upd[:2]][:,:,out_mask] if coor_upd[0] in {'tensor_product_L','Cv '} else self.coef[coor_upd[:2]][:,out_mask]
                    assert np.linalg.norm(out_coef) == 0, 'outside of masks, coef must be zero'
                    A = [(k,v) for k,v in tmp2_ind_reg.items()     if v > 0]
                    B = [(k,v) for k,v in self.cur_ind_reg.items() if v > 0]
                    print()
                    print(coor_upd)
                    print(out_coef)
                    print(A)
                    print(B)
                    raise ValueError
            
            # Update arrays
            if not max_iter_reached:
                assert (    self.cur_obj_training <= self.obj_training[self.iteration - 1] + 1e-12 
                        or (    self.pen == 'trend_filtering'
                            and np.abs(tmp_obj_training - self.cur_obj_training) <= 1e-8
                            )
                        ), (coor_upd, self.cur_obj_training, self.obj_training[self.iteration - 1])
                if self.bcd:
                    (s,key,ind) = coor_upd
                    if s in {'unconstrained',
                             'low_rank_U',
                             'sesquivariate_b',
                             }:
                        self.etaAs[self.iteration] = self.etaAs[self.iteration - 1]
                        self.eta1s[self.iteration] = self.eta1
                        self.eta2s[self.iteration] = self.eta2s[self.iteration - 1]
                    elif s in {'tensor_product_L',
                               'tensor_product_R',
                               'sesquivariate_m',
                               }:
                        self.etaAs[self.iteration] = self.etaAs[self.iteration - 1]
                        self.eta1s[self.iteration] = self.eta1s[self.iteration - 1]
                        self.eta2s[self.iteration] = self.eta2
                else:                
                    self.etas     [self.iteration] = self.eta
                self.fit_training   [self.iteration] = self.cur_fit_training
                self.fit_gp_training[self.iteration] = self.cur_fit_gp_training
                self.reg         [self.iteration] = self.cur_reg      
                self.ridge       [self.iteration] = self.cur_ridge    
                self.obj_training   [self.iteration] = self.cur_obj_training
                if self.hprm['plot.afm'] or self.hprm['afm.algorithm.first_order.early_stop_ind_validation']:
                    if self.batch_cd:# or self.vr:
                        assert 0, 'not implemented'
                    else:
                        if coef_tmp:
                            new_part_fit_ind_training = self.evaluate_fit_bcd(coor_upd, coef_tmp, q_training, dataset = 'training', mask = mask_upd, indi = True)
                            self.fit_ind_training[self.iteration]  = self.fit_ind_training[self.iteration-1]
                            self.fit_ind_training[self.iteration][mask_upd] = self.fit_ind_training[self.iteration][mask_upd] - old_part_fit_ind_training + new_part_fit_ind_training
                            assert (self.fit_ind_training[self.iteration] > 0).all() 
                            if self.hprm['plot.afm']:
                                new_part_fit_mean_training             = self.evaluate_fit_bcd(coor_upd, coef_tmp, q_mean_training, dataset = 'training', mask = mask_upd, MMt = self.M_mean, gp_pen = 1)
                                self.fit_mean_training[self.iteration] = self.fit_mean_training[self.iteration-1] - old_part_fit_mean_training + new_part_fit_mean_training
                        else:
                            self.fit_ind_training[self.iteration]  = self.fit_ind_training [self.iteration-1]
                            if self.hprm['plot.afm']:
                                self.fit_mean_training[self.iteration] = self.fit_mean_training[self.iteration-1]
                if self.compute_validation and self.iteration >= self.flag_compute_validation: 
                    if self.batch_cd:# or self.vr:
                        self.fit_validation   [self.iteration] = self.evaluate_fit(self.coef, dataset = 'validation')
                        if self.active_gp:
                            self.fit_gp_validation[self.iteration] = self.evaluate_fit(self.coef, dataset = 'validation', gp_matrix = self.gp_matrix, gp_pen = self.gp_pen)
                    elif self.bcd:
                        if self.precompute_validation:
                            if coef_tmp:
                                assert self.obj_training[self.iteration] - self.obj_training[self.iteration-1] <= 1e-14
                                new_part_fit_validation             = self.evaluate_fit_bcd(coor_upd, coef_tmp, q_validation, dataset = 'validation', mask = mask_upd)
                                self.cur_fit_validation             = self.fit_validation[self.iteration-1] - old_part_fit_validation + new_part_fit_validation
                                self.fit_validation[self.iteration] = self.cur_fit_validation    
                                if self.active_gp:
                                    new_part_fit_gp_validation             = self.evaluate_fit_bcd(coor_upd, coef_tmp, q_gp_validation, dataset = 'validation', mask = mask_upd, MMt = self.MMt, gp_pen = self.gp_pen)
                                    self.cur_fit_gp_validation             = self.fit_gp_validation[self.iteration-1] - old_part_fit_gp_validation + new_part_fit_gp_validation
                                    self.fit_gp_validation[self.iteration] = self.cur_fit_gp_validation    
                                if self.hprm['plot.afm'] or self.hprm['afm.algorithm.first_order.early_stop_ind_validation']:
                                    new_part_fit_ind_validation = self.evaluate_fit_bcd(coor_upd, coef_tmp, q_validation, dataset = 'validation', mask = mask_upd, indi = True)
                                    self.fit_ind_validation[self.iteration]  = self.fit_ind_validation[self.iteration-1]
                                    self.fit_ind_validation[self.iteration][mask_upd] = self.fit_ind_validation[self.iteration][mask_upd] - old_part_fit_ind_validation + new_part_fit_ind_validation
                                    assert (self.fit_ind_validation[self.iteration] > 0).all() 
                                    new_part_fit_mean_validation             = self.evaluate_fit_bcd(coor_upd, coef_tmp, q_mean_validation, dataset = 'validation', mask = mask_upd, MMt = self.M_mean, gp_pen = 1)
                                    self.fit_mean_validation[self.iteration] = self.fit_mean_validation[self.iteration-1] - old_part_fit_mean_validation + new_part_fit_mean_validation
                            else:
                                self.fit_validation[self.iteration] = self.fit_validation[self.iteration-1]
                                if self.active_gp:
                                    self.fit_gp_validation[self.iteration] = self.fit_gp_validation[self.iteration-1]  
                                if self.hprm['plot.afm'] or self.hprm['afm.algorithm.first_order.early_stop_ind_validation']:
                                    self.fit_ind_validation [self.iteration] = self.fit_ind_validation[self.iteration-1]
                                    self.fit_mean_validation[self.iteration] = self.fit_mean_validation[self.iteration-1]
                    self.obj_validation[self.iteration] = self.fit_validation[self.iteration] + self.fit_gp_validation[self.iteration] + self.reg[self.iteration] + self.ridge[self.iteration]
                if EXTRA_CHECK:
                    if self.iteration >= self.flag_check_fit:
                        self.flag_check_fit *= 10
                        print('\nCheck fit')
                        assert np.abs(self.evaluate_fit(self.coef, dataset = 'training') - self.cur_fit_training) <= 1e-12, ('pb_ch1', self.evaluate_fit(self.coef, dataset = 'training'), self.cur_fit_training)
                        assert np.abs(self.evaluate_fit(self.coef, dataset = 'validation')  - self.cur_fit_validation)  <= 1e-12, ('pb_ch2', self.evaluate_fit(self.coef, dataset = 'validation'),  self.cur_fit_validation)
                    if self.iteration >= self.flag_check_fit and self.active_gp:
                        print('\nCheck fit gp')
                        assert np.abs(self.evaluate_fit(self.coef, dataset = 'training', gp_matrix = self.gp_matrix, gp_pen = self.gp_pen) - self.cur_fit_gp_training) <= 1e-12, ('pb_ch3', self.evaluate_fit(self.coef, dataset = 'training', gp_matrix = self.gp_matrix, gp_pen = self.gp_pen), self.cur_fit_gp_training)
                        assert np.abs(self.evaluate_fit(self.coef, dataset = 'validation',  gp_matrix = self.gp_matrix, gp_pen = self.gp_pen) - self.cur_fit_gp_validation ) <= 1e-12, ('pb_ch4', self.evaluate_fit(self.coef, dataset = 'validation',  gp_matrix = self.gp_matrix, gp_pen = self.gp_pen), self.cur_fit_gp_validation)
                    if self.iteration >= self.flag_check_fit and (self.hprm['plot.afm'] or self.hprm['afm.algorithm.first_order.early_stop_ind_validation']):  
                        print('\nCheck fit ind')
                        assert np.allclose(self.evaluate_ind_fit(self.coef, dataset = 'training'),   self.fit_ind_training[self.iteration]), ('pb_ch32', self.evaluate_ind_fit(self.coef, dataset = 'training'), self.fit_ind_training[self.iteration])
                        assert np.allclose(self.evaluate_ind_fit(self.coef, dataset = 'training').sum(), self.fit_training[self.iteration]),'incoherence between ind_fit and fit training'
                        assert np.allclose(self.evaluate_ind_fit(self.coef, dataset = 'validation' ),   self.fit_ind_validation [self.iteration]), ('pb_ch42', self.evaluate_ind_fit(self.coef, dataset = 'validation'),  self.fit_ind_validation [self.iteration]) 
                        assert np.allclose(self.evaluate_ind_fit(self.coef, dataset = 'validation' ).sum(), self.fit_validation [self.iteration]),'incoherence between ind_fit and fit validation'
                    if self.iteration >= self.flag_check_fit and (self.hprm['plot.afm'] or self.hprm['afm.algorithm.first_order.early_stop_ind_validation']):  
                        print('\nCheck fit mean')
                        assert np.abs(self.evaluate_fit(self.coef, dataset = 'training', gp_matrix = self.M_mean, gp_pen = 1) - self.fit_mean_training[self.iteration]) <= 1e-12, ('pb_ch32', self.evaluate_fit(self.coef, dataset = 'training', gp_matrix = self.M_mean, gp_pen = 1), self.fit_mean_training[self.iteration])
                        assert np.abs(self.evaluate_fit(self.coef, dataset = 'validation',  gp_matrix = self.M_mean, gp_pen = 1) - self.fit_mean_validation [self.iteration]) <= 1e-12, ('pb_ch42', self.evaluate_fit(self.coef, dataset = 'validation',  gp_matrix = self.M_mean, gp_pen = 1), self.fit_mean_validation [self.iteration])  
            if self.nb_steps >= self.flag_print_info:
                self.print_info()
                self.flag_print_info += self.epoch_print_info
                
            small_decrease, early_stop, norm_grad_small, all_dead = self.stopping_criteria({key:dikt_fit_grad[key] + dikt_fit_gp_grad.get(key, 0) 
                                                                                            for key in set(dikt_fit_grad.keys()).union(dikt_fit_gp_grad.keys())
                                                                                            })
            
            assert not np.isnan(self.cur_obj_training)
            
            # To next iteration
            self.iteration  += 1
            max_iter_reached = (self.iteration >= self.max_iter)
            self.converged   = max(small_decrease, early_stop, norm_grad_small, max_iter_reached, all_dead)                                

###############################################################################

    
    
    def backtracking_ls(self, 
                        grad,                    # Gradient of the dataset-fitting term
                        grad_gp,                 # Gradient of the sum-consistent-fitting term 
                        ridge_grad,              # Gradient of the ridge term
                        old_fit_training         = None, 
                        old_fit_gp_training      = None, 
                        old_part_fit_training    = None, 
                        old_part_fit_gp_training = None, 
                        old_ridge                = None, 
                        old_part_ridge           = None, 
                        quant_training           = None, 
                        quant_gp_training        = None, 
                        cur_ind_reg              = None, 
                        mask_upd                 = None,
                        ):
        # Backtracking line-search, given a gradient descent direction
        condition_ls  = 0
        nb_inner_iter = 0
        converged     = 0
        if self.bcd:
            coor_upd = next(iter(grad.keys()))
            (s,key,ind) = coor_upd
            if s in {'unconstrained',
                     'low_rank_U',
                     'sesquivariate_b',
                     }:
                eta = self.eta1
            elif s in {'tensor_product_L',
                       'tensor_product_R',
                       'sesquivariate_m',
                       }:
                eta = self.eta2
            assert eta
            eta = self.eta[coor_upd] if coor_upd in self.eta else 1
        else:                
            eta = self.eta

        while not (condition_ls or converged):
            # Loop for the backtracking line search (iteratively decreasing the step size)
            if eta <= 1e-16: 
                if not self.bcd:
                    assert 0
                else:
                    print(colored(' \n    eta too small for {0} - > end BLS   \n'.format(coor_upd if len(grad) == 1 else 'unknown'), 'red', 'on_cyan'))
                    return self.cur_fit_training, self.cur_fit_gp_training, {}, self.cur_fit_training + self.cur_ridge, {}, nb_inner_iter, eta/self.theta, False
            coef_tmp = self.foba_step(grad,
                                      grad_gp,
                                      ridge_grad,
                                      eta,
                                      self.dikt_masks,
                                      )
            if EXTRA_CHECK:
                for coor_upd in grad.keys():
                    assert coef_tmp[coor_upd].shape == grad[coor_upd].shape
                    if coor_upd in grad_gp.keys():
                        assert coef_tmp[coor_upd].shape == grad_gp[coor_upd].shape
            # LS condition    
            fit_plus_gp_plus_ridge_tilde_tmp = self.compute_surrogate(grad, 
                                                                      grad_gp, 
                                                                      ridge_grad, 
                                                                      coef_tmp, 
                                                                      eta, 
                                                                      old_fit_training + old_fit_gp_training + old_ridge, 
                                                                      self.dikt_masks,
                                                                      )
            if EXTRA_CHECK:
                fit_plus_gp_plus_ridge_tilde_old = self.compute_surrogate(grad, 
                                                                          grad_gp, 
                                                                          ridge_grad, 
                                                                          {(var,key,ind):(self.coef[var,key][:,:,mask_upd]
                                                                                          if var in {'tensor_product_L','tensor_product_R'}
                                                                                          else self.coef[var,key][:,mask_upd]
                                                                                          )
                                                                           for (var,key,ind) in coef_tmp
                                                                           }, 
                                                                          eta, 
                                                                          old_fit_training + old_fit_gp_training + old_ridge, 
                                                                          self.dikt_masks,
                                                                          )
                assert fit_plus_gp_plus_ridge_tilde_old == self.cur_fit_training + self.cur_ridge + self.cur_fit_gp_training
                coef_old = {}
                for coor_upd in grad.keys():
                    var, key, ind      = coor_upd
                    coef_old[coor_upd] = (self.coef[var,key][:,:,mask_upd]
                                          if var in {'tensor_product_L','tensor_product_R'}
                                          else
                                          self.coef[var,key][:,mask_upd]
                                          )
                ind_old_regbis, _, _ = self.evaluate_ind_reg(coef_old)
                tmp_ind_reg,    _, _ = self.evaluate_ind_reg(coef_tmp)
                new_part_reg = np.sum([tmp_ind_reg[key] for key in tmp_ind_reg])
                other_reg    = np.sum([cur_ind_reg[key] for key in cur_ind_reg if key not in tmp_ind_reg])
                old_part_reg = np.sum([cur_ind_reg[key] for key in tmp_ind_reg])
                
                assert np.abs(  fit_plus_gp_plus_ridge_tilde_old
                              + other_reg
                              + old_part_reg
                              - self.cur_obj_training
                              ) <= 1e-12
                assert np.abs(  self.cur_fit_training
                              + self.cur_fit_gp_training
                              + self.cur_reg
                              + self.cur_ridge
                              - self.cur_obj_training
                              ) <= 1e-12
                if not (  fit_plus_gp_plus_ridge_tilde_tmp
                        + other_reg
                        + new_part_reg
                        ) <= (  fit_plus_gp_plus_ridge_tilde_old
                              + other_reg
                              + old_part_reg
                              ) + 1e-12:
                    raise ValueError('Objective has increased')
            
            # # Update low-rank
            if 'low_rank_UVt' in self.formula.index.get_level_values('coefficient').unique() and 'low_rank_U' in [e[0] for e in grad.keys()] :
                coef_tmp  = self.update_VB(coef_tmp)
            if 'tensor_product_LR' in self.formula.index.get_level_values('coefficient').unique():
                coef_tmp  = self.update_Cuv(coef_tmp)
            if 'sesquivariate_bm' in self.formula.index.get_level_values('coefficient').unique():
                coef_tmp  = self.update_Cbm(coef_tmp)
            if self.batch_cd:
                fit_tmp   = self.evaluate_fit(coef_tmp,
                                              dataset = 'training',
                                              )
                assert 0, 'MUST ADD fit_gp_tmp'
                ridge_tmp = self.evaluate_ridge(coef_tmp) 
            elif self.bcd:
                assert len(coef_tmp) <= 3 or coor_upd[0] == 'sesquivariate_b'
                new_part_fit_training = self.evaluate_fit_bcd(coor_upd,
                                                              coef_tmp,
                                                              quant_training,
                                                              dataset = 'training',
                                                              mask    = mask_upd,
                                                              )
                if self.active_gp:
                    new_part_fit_gp_training = self.evaluate_fit_bcd(coor_upd,
                                                                     coef_tmp,
                                                                     quant_gp_training,
                                                                     dataset = 'training',
                                                                     mask    = mask_upd,
                                                                     MMt     = self.MMt,
                                                                     gp_pen  = self.gp_pen,
                                                                     )
                if EXTRA_CHECK:
                    check_old_part_fit_training = self.evaluate_fit_bcd(coor_upd,
                                                                        {(var,*specs):(self.coef[var,specs[0]][:,:,mask_upd]
                                                                                        if var in {'tensor_product_L','tensor_product_R'}
                                                                                        else
                                                                                        self.coef[var,specs[0]][:,mask_upd]
                                                                                        )
                                                                         for var,*specs in coef_tmp
                                                                         },
                                                                        quant_training,
                                                                        dataset = 'training',
                                                                        mask    = mask_upd,
                                                                        )
                    assert np.abs(  check_old_part_fit_training
                                  - old_part_fit_training
                                  ) < 1e-12, ('pb5', check_old_part_fit_training, old_part_fit_training)
                    if self.active_gp:
                        check_old_part_fit_gp_training = self.evaluate_fit_bcd(coor_upd,
                                                                               {(var,key,ind):(self.coef[var,key][:,:,mask_upd]
                                                                                               if var in {'tensor_product_L','tensor_product_R'}
                                                                                               else self.coef[var,key][:,mask_upd]
                                                                                               )
                                                                                for (var,key,ind) in coef_tmp
                                                                                },
                                                                               quant_gp_training,
                                                                               dataset = 'training',
                                                                               mask    = mask_upd,
                                                                               MMt     = self.MMt,
                                                                               gp_pen  = self.gp_pen,
                                                                               )
                        assert np.abs(check_old_part_fit_gp_training - old_part_fit_gp_training) < 1e-12, ('pb6', check_old_part_fit_gp_training, old_part_fit_gp_training)
                new_ind_ridge  = self.evaluate_ind_ridge(coef_tmp) 
                new_part_ridge = np.sum([v for k, v in new_ind_ridge.items()])
                ridge_tmp      = (  old_ridge
                                  - old_part_ridge
                                  + new_part_ridge
                                  )
                fit_tmp        = (  old_fit_training
                                  - old_part_fit_training
                                  + new_part_fit_training
                                  )
                fit_gp_tmp     = (  old_fit_gp_training
                                  - old_part_fit_gp_training
                                  + new_part_fit_gp_training
                                  ) if self.active_gp else 0
                if EXTRA_CHECK:
                    new_coef = cp.deepcopy(self.coef)
                    for coor in coef_tmp:
                        if coor[0] in {'tensor_product_L','tensor_product_R'}:
                            new_coef[coor[:2]][:,:,mask_upd] = coef_tmp[coor]
                        else:
                            new_coef[coor[:2]][:,  mask_upd] = coef_tmp[coor]
                    # control fitting term
                    assert old_fit_training == self.cur_fit_training
                    fit_ctrl = self.evaluate_fit(new_coef,
                                                 dataset = 'training',
                                                 ) 
                    assert np.abs(  fit_ctrl 
                                  - fit_tmp
                                  ) < 1e-10
                    if self.active_gp:
                        assert old_fit_gp_training == self.cur_fit_gp_training
                        fit_gp_ctrl = self.evaluate_fit(new_coef,
                                                        dataset   = 'training',
                                                        gp_matrix = self.gp_matrix,
                                                        gp_pen    = self.gp_pen
                                                        ) if self.active_gp else 0
                        assert np.abs(  fit_gp_ctrl
                                      - fit_gp_tmp
                                      ) < 1e-12
                assert new_part_fit_training != np.nan 
                if self.active_gp:
                    assert new_part_fit_gp_training != np.nan 
            if fit_tmp < 0 or fit_gp_tmp < 0:
                # Detection of an error
                len_str = 25
                print('\n', )
                for e in ['self.bcd',
                          'self.cur_fit_training', 
                          'fit_tmp', 
                          'fit_gp_tmp', 
                          'ridge_tmp',
                          'old_part_fit_training',
                          'old_ridge',
                          'new_part_fit_training',
                          'eta',
                          'nb_inner_iter',
                          'set([a for a, b in self.coef.keys()])',
                          'sorted(self.keys.get("Cbm"),[])',
                          'sorted(self.keys.get("Cb"),[])',
                          'sorted(self.keys.get("Csp"),[])',
                          'sorted(coef_tmp.keys())', 
                          'sorted(coef_tmp.keys())',
                          'self.iteration',
                          ]:
                    print('{0:{width}} : {1}'.format(e,
                                                     eval(e),
                                                     width = len_str,
                                                     )) 
                print('{0:{width}} : {1}'.format('shape' ,
                                                 [(k, v.shape) 
                                                  for k, v in coef_tmp.items()
                                                  ],
                                                 width = len_str))
            assert fit_tmp    >= 0
            assert fit_gp_tmp >= 0
            assert ridge_tmp  >= 0
            condition_ls   = (fit_plus_gp_plus_ridge_tilde_tmp >= fit_tmp + fit_gp_tmp + ridge_tmp) 
            nb_inner_iter += 1
            converged      = (self.iteration + nb_inner_iter >= self.max_iter)
            if not condition_ls : 
                eta *= self.theta
        if converged and not condition_ls:
            coef_tmp   = {}
            fit_tmp    = self.cur_fit_training 
            fit_gp_tmp = self.cur_fit_gp_training 
            new_ind_ridge                    = {}
            fit_plus_gp_plus_ridge_tilde_tmp = fit_tmp + fit_gp_tmp
            print(colored('    converged but not condition_ls    ', 'red', 'on_cyan'))
        if EXTRA_CHECK:
            tmp_reg      = other_reg + new_part_reg
            if not fit_plus_gp_plus_ridge_tilde_tmp + tmp_reg <= self.cur_obj_training + 1e-12:
                assert 0
            if not(fit_tmp + fit_gp_tmp + ridge_tmp + tmp_reg      <= self.cur_obj_training + 1e-12):
                assert 0
            if not(fit_tmp + fit_gp_tmp + ridge_tmp + tmp_reg      <= fit_plus_gp_plus_ridge_tilde_tmp + tmp_reg+ 1e-12):
                assert 0
        return fit_tmp, fit_gp_tmp, new_ind_ridge, fit_plus_gp_plus_ridge_tilde_tmp, coef_tmp, nb_inner_iter, eta, condition_ls


    
    def best_orth_bv(self, coef, mask = None):
        # Update the orthogonal matrix V in the low-rank optimization problem
        coor_upd      = list(coef.keys())[0]
        var, key, ind = coor_upd
        assert not self.active_gp, 'optimization wrt V is different'
        assert not ind, 'no partial/column update for low-rank coefficients'
        assert mask == slice(None) # No mask when Bv
        XtY = self.XtY_training
        M = (  XtY[key][:,mask]   
             - np.sum([self.custom_sum_einsum(self.XtX_training, 
                                              self.coef, 
                                              var2, 
                                              key, 
                                              slice(None), 
                                              var_temp = dikt_var_temp.get(var,var),
                                              )
                       for var2 in self.formula.index.get_level_values('coefficient').unique()
                       ], 
                      axis = 0,
                      )
             )[:,mask].T @ coef.get(('low_rank_U', key, ()), coef.get(('low_rank_U',key), 'error'))
        A, sig, Bt = np.linalg.svd(M, full_matrices=0)
        v = A @ Bt
        return v


    def change_sparse_structure(self, ):
        # Sparsify the design matrices
        print('change structure X sparse')
        for ii, X in enumerate([self.X_training] + ([self.X_validation] if self.compute_validation else [])):
            for jj, (k, v) in enumerate(X.items()):
                print('\r{0} / {1} - {2:5} / {3:5}'.format(ii,
                                                           1+self.compute_validation,
                                                           jj,
                                                           len(X)),
                      end = '')
                if type(v) == sp.sparse.csc_matrix:
                    X[k] = sp.sparse.csr_matrix(v)
        print('\ndone')

    def compute_grad_ridge(self, list_coor, d_masks):
        # Compute the gradient of the differentiable regularization (ridge or smoothing-spline)
        dikt_ridge_grad = {}
        for coor_upd in list_coor:
            assert len(coor_upd) == 3
            var, key, ind  = coor_upd
            inpt, location = key
            pen, alpha     = self.get_pen_alpha(var, key)
            mask = d_masks.get(coor_upd, slice(None))
            if pen == 'ridge':
                if coor_upd[0] in {'tensor_product_L', 'tensor_product_R'}:
                    dikt_ridge_grad[coor_upd] = alpha*self.coef[var,key][:,:,mask] if (pen == 'ridge') else np.zeros(self.coef[var,key][:,:,mask].shape)
                else:
                    dikt_ridge_grad[coor_upd] = alpha*self.coef[var,key][:,  mask] if (pen == 'ridge') else np.zeros(self.coef[var,key][:,mask].shape)
            elif pen == 'smoothing_reg':
                if coor_upd[0] in {'tensor_product_L', 'tensor_product_R'}:
                    raise NotImplementedError # coef has then 3 dimensions
                M   = self.coef[var,key][:,mask]
                reshape_tensor = (    type(inpt[0]) == tuple
                                  and var not in {'tensor_product_L', 'tensor_product_R', 'sesquivariate_b', 'sesquivariate_m'}
                                  )
                if not reshape_tensor:
                    cc = M
                    dikt_ridge_grad[coor_upd] = np.zeros(cc.shape)
                    if cc.shape[0] > 2:
                        if self.hprm['inputs.cyclic'].get(inpt):
                            ker  = np.array([[1],[-4],[6],[-4],[1]])
                            conv = spim.filters.convolve(cc,ker,mode = 'wrap')
                            dikt_ridge_grad[coor_upd] = alpha * conv
                        else:
                            ker  = np.array([[1],[-2],[1]])
                            conv = sig.convolve(cc, ker, mode = 'valid')
                            dikt_ridge_grad[coor_upd][ :-2] += alpha * conv
                            dikt_ridge_grad[coor_upd][1:-1] -= alpha * 2*conv
                            dikt_ridge_grad[coor_upd][2:  ] += alpha * conv
                else:
                    cc = M.reshape(*self.size_tensor_bivariate[key], -1)
                    dikt_ridge_grad[coor_upd] = np.zeros(cc.shape)
                    inpt1, inpt2 = inpt
                    if cc.shape[0] > 2:
                        if self.hprm['inputs.cyclic'].get(inpt1):
                            ker1  = np.array([[[1],[-4],[6],[-4],[1]]])
                            conv1 = spim.filters.convolve(cc,ker1,mode = 'wrap')
                            dikt_ridge_grad[coor_upd] += alpha * conv1
                        else:
                            ker1  = np.array([[[1]],[[-2]],[[ 1]]])
                            conv1     = sig.convolve(cc, ker1, mode = 'valid')
                            dikt_ridge_grad[coor_upd][ :-2] += alpha * conv1
                            dikt_ridge_grad[coor_upd][1:-1] -= alpha * 2*conv1
                            dikt_ridge_grad[coor_upd][2:  ] += alpha * conv1
                    if cc.shape[1] > 2: 
                        if self.hprm['inputs.cyclic'].get(inpt2):
                            ker2  = np.array([[[1],[-4],[6],[-4],[1]]])
                            conv2 = spim.filters.convolve(cc,ker2,mode = 'wrap')
                            dikt_ridge_grad[coor_upd] += alpha * conv2
                        else:
                            ker2  = np.array([[[1],  [-2],  [1]]])
                            conv2     = sig.convolve(cc, ker2, mode = 'valid')
                            dikt_ridge_grad[coor_upd][:, :-2] += alpha * conv2
                            dikt_ridge_grad[coor_upd][:,1:-1] -= alpha * 2*conv2
                            dikt_ridge_grad[coor_upd][:,2:  ] += alpha * conv2
                    dikt_ridge_grad[coor_upd] = dikt_ridge_grad[coor_upd].reshape(M.shape)
            elif pen == 'factor_smoothing_reg':
                if coor_upd[0] in {'tensor_product_L', 'tensor_product_R'}:
                    raise NotImplementedError # coef has then 3 dimensions
                M   = self.coef[var,key][:,mask]
                reshape_tensor = (    type(inpt[0]) == tuple 
                                  and var not in {'tensor_product_L', 'tensor_product_R', 'sesquivariate_b', 'sesquivariate_m'}
                                  )
                if not reshape_tensor:
                    cc = M
                    dikt_ridge_grad[coor_upd] = np.zeros(cc.shape)
                    if cc.shape[0] > 2:
                        if self.hprm['inputs.cyclic'].get(inpt):
                            ker  = np.array([[1],[-4],[6],[-4],[1]])
                            conv = spim.filters.convolve(cc,ker,mode = 'wrap')
                            dikt_ridge_grad[coor_upd] = alpha * conv
                        else:
                            ker  = np.array([[1],[-2],[1]])
                            conv = sig.convolve(cc, ker, mode = 'valid')
                            dikt_ridge_grad[coor_upd][ :-2] += alpha * conv
                            dikt_ridge_grad[coor_upd][1:-1] -= alpha * 2*conv
                            dikt_ridge_grad[coor_upd][2:  ] += alpha * conv
                else:
                    cc = M.reshape(*self.size_tensor_bivariate[key], -1)
                    dikt_ridge_grad[coor_upd] = np.zeros(cc.shape)
                    inpt1, inpt2 = inpt
                    if cc.shape[0] > 2:
                        if self.hprm['inputs.cyclic'].get(inpt1):
                            ker1  = np.array([[[1],[-4],[6],[-4],[1]]])
                            conv1 = spim.filters.convolve(cc,ker1,mode = 'wrap')
                            dikt_ridge_grad[coor_upd] += alpha * conv1
                        else:
                            ker1  = np.array([[[1]],[[-2]],[[ 1]]])
                            conv1     = sig.convolve(cc, ker1, mode = 'valid')
                            dikt_ridge_grad[coor_upd][ :-2] += alpha * conv1
                            dikt_ridge_grad[coor_upd][1:-1] -= alpha * 2*conv1
                            dikt_ridge_grad[coor_upd][2:  ] += alpha * conv1
                    dikt_ridge_grad[coor_upd] = dikt_ridge_grad[coor_upd].reshape(M.shape)
            else:
                assert pen not in {'ridge', 'smoothing_reg'}
                if coor_upd[0] in {'tensor_product_L', 'tensor_product_R'}:
                    dikt_ridge_grad[coor_upd] = np.zeros(self.coef[var,key][:,:,mask].shape)
                else:
                    dikt_ridge_grad[coor_upd] = np.zeros(self.coef[var,key][:,mask].shape)
        return dikt_ridge_grad    


    
    def compute_part_grad(self, coor_upd, mask, dataset = None, MMt = None, gp_pen = 0):
        # Compute the gradient with respect to the block selected for the BCD algorithm
        # The computation is decomposed in different parts to avoid redundant computations
        if gp_pen:
            assert type(MMt) != type(None)
            mask_out   = np.array([k for k in     (self.orig_masks[coor_upd] 
                                                   if hasattr(self.orig_masks.get(coor_upd), 'len') and not self.col_upd.get(coor_upd[1],False) 
                                                   else range(self.k))
                                     if  k not in (mask  if type(mask) == np.ndarray else range(self.k))
                                   ]).astype(int)
            mask_Y = slice(None)
        else:
            mask_Y = mask
        assert len(coor_upd) == 3
        var, key, ind = coor_upd
        if var == 'low_rank_UVt':
            assert 0 # For now
            assert not ind
        extra_part = {}
        quant      = {}
        extra_part = {}
        if dataset == 'training':
            grad = {}
            n   = self.n_training
            X   = self.X_training
            Y   = self.Y_training
            XtX = self.XtX_training
            XtY = self.XtY_training
        else:
            assert dataset == 'validation'
            n   = self.n_validation
            X   = self.X_validation
            Y   = self.Y_validation
            XtX = self.XtX_validation
            XtY = self.XtY_validation
        
        if var in {'unconstrained',
                   'low_rank_U',
                   'sesquivariate_b',
                   'sesquivariate_m',
                   'tensor_product_L',
                   'tensor_product_R',
                   }:
            #Compute the normal part of quant[coor_upd]
            mmm = (1/n)*(- (XtY[key][:,mask_Y]
                            if type(XtY[key]) == np.ndarray
                            else
                            XtY[key][:,mask_Y].toarray()
                            )
                         + np.sum([self.custom_sum_einsum(XtX, 
                                                          self.coef, 
                                                          var2, 
                                                          key, 
                                                          'dyn' if gp_pen else mask, 
                                                          var_temp = dikt_var_temp.get(var, var),
                                                          ) 
                                    for var2 in self.formula.index.get_level_values('coefficient').unique()], 
                                   axis = 0,
                                   )
                         )

            if gp_pen:
                mmm = gp_pen * mmm @ MMt[:,mask]
                if len(mask_out):
                    mmm+= gp_pen * (1/n) * XtX[key,key] @ self.coef[dikt_var_temp.get(var,var),key][:,mask_out] @ MMt[mask_out][:,mask]
            
            # Check the computation of the normal part of quant[coor_upd]
            if EXTRA_CHECK:
            # The computation is different for Cb so the check is different too
                aaa =   Y[:,mask_Y].copy()
                for var2 in self.formula.index.get_level_values('coefficient').unique():   
                    for key2 in self.keys[var2]:
                        if (    (key,key2) in XtX
                            and (   (     var != 'sesquivariate_b'
                                     and (   key2 != key
                                          or var2 != dikt_var_temp.get(var, var)
                                          )
                                     )
                                 or (    var == 'sesquivariate_b'
                                     and (   var2 not in ('sesquivariate_b', 'sesquivariate_bm')
                                          or (    var2 == 'sesquivariate_b'
                                              and key  != key2
                                              )
                                          or (    var2 == 'sesquivariate_bm'
                                              and key  != (key2[0][0], (key2[1][0],))
                                              )
                                          )
                                     )
                                 )
                            ):                            
                            ss         = self.orig_masks.get((var2,key2,()), slice(None)) if gp_pen else mask
                            aaa[:,ss if gp_pen else slice(None)] += - X[key2] @ self.coef[var2,key2][:,ss] 
                mmm_check = -(1/n) * X[key].T @ aaa
                if gp_pen:
                    mmm_check = gp_pen * mmm_check @ MMt[:,mask]
                    if len(mask_out):
                        mmm_check+= gp_pen * (1/n) * XtX[key,key] @ self.coef[dikt_var_temp.get(var,var),key][:,mask_out] @ MMt[mask_out][:,mask]
                if not np.allclose(mmm, mmm_check):
                    import ipdb; ipdb.set_trace()
                assert mmm.shape == mmm_check.shape
                assert np.allclose(mmm, mmm_check)
            # Multiply accordingly when the updated variable intervenes as a product in the prediction BM, UV
            if var == 'low_rank_U':
                quant[coor_upd] = mmm # not multiplied by V since it will also be updated
            elif var == 'sesquivariate_b':
                quant[coor_upd] = mmm # Computation of quant[coor_upd] for Cb is not over
            elif var == 'sesquivariate_m':
                (inpt_b,inpt_m), (location_b,location_m) = key
                quant[coor_upd] = np.einsum('pqk,pk->qk', 
                                            mmm.reshape(( 
                                                         self.coef['sesquivariate_b', (inpt_b, (location_b,))][:,mask].shape[0], 
                                                         -1,
                                                         self.coef['sesquivariate_b', (inpt_b, (location_b,))][:,mask].shape[1],
                                                         )), 
                                            self.coef['sesquivariate_b', (inpt_b, (location_b,))][:,mask],
                                            )
                if EXTRA_CHECK:
                    mmm_check = np.einsum('pqk,pk->qk', 
                                          mmm_check.reshape((
                                                             self.coef['sesquivariate_b', (inpt_b, (location_b,))][:,mask].shape[0],
                                                             -1,
                                                             self.coef['sesquivariate_b', (inpt_b, (location_b,))][:,mask].shape[1],
                                                             )), 
                                          self.coef['sesquivariate_b', (inpt_b, (location_b,))][:,mask],
                                          )
            elif var == 'tensor_product_L':
                quant[coor_upd] = np.einsum('pqk,qrk->prk', 
                                            mmm.reshape((-1,
                                                         self.coef['tensor_product_R', key][:,:,mask].shape[0],
                                                         self.coef['tensor_product_R', key][:,:,mask].shape[2],
                                                         )),
                                            self.coef['tensor_product_R', key][:,:,mask],
                                            )
                if EXTRA_CHECK:
                    mmm_check = np.einsum('pqk,qrk->prk', 
                                          mmm_check.reshape((-1, 
                                                             self.coef['tensor_product_R', key][:,:,mask].shape[0], 
                                                             self.coef['tensor_product_R', key][:,:,mask].shape[2],
                                                             )), 
                                          self.coef['tensor_product_R', key][:,:,mask],
                                          )
            elif var == 'tensor_product_R':
                quant[coor_upd] = np.einsum('pqk,prk->qrk', 
                                            mmm.reshape((self.coef['tensor_product_L', key][:,:,mask].shape[0],
                                                         -1,
                                                         self.coef['tensor_product_L', key][:,:,mask].shape[2],
                                                         )),
                                            self.coef['tensor_product_L', key][:,:,mask],
                                            )
                if EXTRA_CHECK:
                    mmm_check = np.einsum('pqk,prk->qrk', 
                                          mmm_check.reshape((self.coef['tensor_product_L', key][:,:,mask].shape[0], 
                                                             -1, 
                                                             self.coef['tensor_product_L', key][:,:,mask].shape[2],
                                                             )), 
                                          self.coef['tensor_product_L', key][:,:,mask],
                                          )
            else:
                assert var == 'unconstrained'
                quant[coor_upd] = mmm
            # Add the part specific to Cb where Cb intervenes in Cbm
            if var == 'sesquivariate_b': 
                for keybm in self.keys['sesquivariate_bm']:
                    bbb       = {}
                    bbb_check = {}
                    assert type(keybm[0]) == tuple
                    if key == (keybm[0][0],(keybm[1][0],)):
                        bbb[keybm] = (1/n)*(- (XtY[keybm][:,mask_Y]
                                               if type(XtY[key]) == np.ndarray
                                               else
                                               XtY[keybm][:,mask_Y].toarray()
                                               )
                                            + np.sum([self.custom_sum_einsum(XtX, 
                                                                             self.coef, 
                                                                             var2, 
                                                                             keybm, 
                                                                             'dyn' if gp_pen else mask, 
                                                                             var_temp = 'sesquivariate_b',
                                                                             ) 
                                                       for var2 in self.formula.index.get_level_values('coefficient').unique()], 
                                                      axis = 0,
                                                     )
                                            )
                        if gp_pen:
                            bbb[keybm] = gp_pen * bbb[keybm] @ MMt[:,mask]
                            bbb[keybm]+= gp_pen * (1/n) * XtX[keybm,keybm] @ self.coef['sesquivariate_bm',keybm][:,mask_out] @ MMt[mask_out][:,mask]

                        quant[coor_upd] += np.einsum('pqk,qk->pk', 
                                                     bbb[keybm].reshape((-1, 
                                                                         self.coef['sesquivariate_m', keybm][:,mask].shape[0], 
                                                                         self.coef['sesquivariate_m', keybm][:,mask].shape[1],
                                                                         )), 
                                                     self.coef['sesquivariate_m', keybm][:,mask],
                                                     )
                        # Check this part of quant[coor_upd] that is specific to Cb
                        if EXTRA_CHECK:
                            aaa =   Y[:,mask_Y].copy()
                            for var2 in self.formula.index.get_level_values('coefficient').unique():
                                for key2 in self.keys[var2]: 
                                    if (    (keybm,key2) in XtX
                                        and (   var2 not in ('sesquivariate_b', 'sesquivariate_bm')
                                             or (    var2 == 'sesquivariate_b'
                                                 and key != key2
                                                 )
                                             or (    var2 == 'sesquivariate_bm'
                                                 and key != (key2[0][0],(key2[1][0],))
                                                 )
                                             )
                                        ):
                                        ss = self.orig_masks.get((var2,key2,()), slice(None)) if gp_pen else mask
                                        aaa[:,ss
                                            if gp_pen
                                            else
                                            slice(None)
                                            ] += - X[key2] @ self.coef[var2,key2][:,ss] 
                            bbb_check[keybm] = - (1/n) * X[keybm].T @ aaa
                            if gp_pen:
                                bbb_check[keybm] = gp_pen * bbb_check[keybm] @ MMt[:,mask]
                                if len(mask_out):
                                    bbb_check[keybm]+= gp_pen * (1/n) * XtX[keybm, keybm] @ self.coef['sesquivariate_bm',keybm][:,mask_out] @ MMt[mask_out][:,mask]
                            assert bbb[keybm].shape ==  bbb_check[keybm].shape
                            assert np.allclose(bbb[keybm], bbb_check[keybm])                           
                            mmm_check += np.einsum('pqk,qk->pk', 
                                                   bbb_check[keybm].reshape((-1, 
                                                                             self.coef['sesquivariate_m', keybm][:,mask].shape[0], 
                                                                             self.coef['sesquivariate_m', keybm][:,mask].shape[1],
                                                                             )), 
                                                   self.coef['sesquivariate_m', keybm][:,mask],
                                                   )

            if EXTRA_CHECK:
                assert quant[coor_upd].shape == mmm_check.shape, (coor_upd, quant[coor_upd].shape, mmm_check.shape)
                assert np.allclose(quant[coor_upd], mmm_check) 
            
            # Computations of extra_part
            if var == 'low_rank_U':
                # Very specific case of bu
                extra_part[coor_upd] = self.xtra_part_bu(n, XtX, self.coef, coor_upd, mask, gp_pen, MMt)
                if dataset == 'training': # Special case for bu because both bu and bv are updated at the same time
                    grad[coor_upd] =(quant[coor_upd] + extra_part[coor_upd]) @  self.coef[('low_rank_V',key)]#[mask]
            else:
                # Specific case of Cb
                if var == 'sesquivariate_b':
                    extra_part[coor_upd] = self.xtra_part_cb(n, XtX, self.coef, coor_upd, mask, gp_pen, MMt)
                # Specific case fo Cm
                elif var == 'sesquivariate_m':
                    extra_part[coor_upd] = self.xtra_part_cm(n, XtX, self.coef, coor_upd, mask, gp_pen, MMt)                                                
                # Specific case of Cu
                elif var == 'tensor_product_L':
                    extra_part[coor_upd] = self.xtra_part_cu(n, XtX, self.coef, coor_upd, mask, gp_pen, MMt)                                               
                # Specific case of Cu
                elif var == 'tensor_product_R':
                    extra_part[coor_upd] = self.xtra_part_cv(n, XtX, self.coef, coor_upd, mask, gp_pen, MMt)
                        
                else:
                    extra_part[coor_upd] = self.xtra_part_cl(n, XtX, self.coef, coor_upd, mask, gp_pen, MMt)
                if dataset == 'training':
                    assert quant[coor_upd].shape == extra_part[coor_upd].shape
                    grad[coor_upd] = quant[coor_upd] + extra_part[coor_upd]
            
            if EXTRA_CHECK:
                if var == 'low_rank_U':
                    assert self.coef[(var,key)][:,mask].shape ==(quant     [coor_upd]@self.coef[('low_rank_V',key)][mask]).shape 
                    assert self.coef[(var,key)][:,mask].shape ==(extra_part[coor_upd]@self.coef[('low_rank_V',key)][mask]).shape    
                elif var == 'tensor_product_L':
                    assert self.coef[(var,key)][:,:,mask].shape == quant     [coor_upd].shape 
                    assert self.coef[(var,key)][:,:,mask].shape == extra_part[coor_upd].shape 
                elif var == 'tensor_product_R':
                    assert self.coef[(var,key)][:,:,mask].shape == quant     [coor_upd].shape 
                    assert self.coef[(var,key)][:,:,mask].shape == extra_part[coor_upd].shape 
                else:
                    assert self.coef[(var,key)][:,mask].shape == quant     [coor_upd].shape, (self.coef[(var,key)][:,mask].shape, quant     [coor_upd].shape) 
                    assert self.coef[(var,key)][:,mask].shape == extra_part[coor_upd].shape, (self.coef[(var,key)][:,mask].shape, extra_part[coor_upd].shape)
        else:
            raise ValueError
  
     
        if EXTRA_CHECK:
            assert extra_part.keys() == quant.keys()  
            for k in extra_part.keys():
                assert extra_part[k].shape == quant[k].shape
        if dataset == 'training':
            if EXTRA_CHECK:
                assert grad
                assert extra_part.keys() == grad.keys() 
            return quant, extra_part, grad
        else:
            return quant, extra_part,
        
    
    def compute_grad_norm(self, dikt):
        sq  = 0
        for coor, M in dikt.items():
            sq += np.linalg.norm(M)**2
        return np.sqrt(sq)
    

    def compute_normalization(self, ):
        print('data normalization')    
        self.normalization     = {}
        self.normalized_alphas = {}
        cats_in_alpha = [(var,cat) 
                         for var in self.alpha 
                         for cat in self.alpha[var].keys()
                         ]
        for cat in set([e[0] for e in self.sorted_keys]):
            alpha_cat = 0
            for var in sorted(self.alpha):
                if cat in self.alpha[var]:
                    if var == 'low_rank_U':
                        var_norm = 'low_rank_UVt'
                    elif var in ('tensor_product_L,' 'tensor_product_R'):
                        var_norm = 'tensor_product_LR'
                    else:
                        var_norm = var
                    if cat in self.formula.loc[var_norm].index:
                        if (    self.pen[var][cat] in {'ridge','smoothing_reg'}
                            and var != 'tensor_product_R'
                            ):
                            assert not alpha_cat, 'it should not have already been found for the same cat and a penalization in ridge, r2sm, unless it is approximately lowrank but this is not implemented yet'
                            alpha_cat = self.alpha[var][cat]
            keys_for_cat = [(inpt, location) 
                            for (inpt, location) in self.sorted_keys 
                            if inpt == cat
                            ]
            if len(keys_for_cat) == 1:
                X_cat = self.X_training[keys_for_cat[0]]
            else:
                X_cat = sp.sparse.hstack([sp.sparse.csc_matrix(self.X_training[key]) 
                                          for key in keys_for_cat
                                          ], 
                                         format = 'csc',
                                         )
            if alpha_cat: # Take into account the smooth regularization for the normalizations
                X_tilde_cat = sp.sparse.vstack([X_cat,  
                                                np.sqrt(alpha_cat)*sp.sparse.eye(X_cat.shape[1], format = 'csc')
                                                ], 
                                               format = 'csc',
                                               )
            else:
                X_tilde_cat = X_cat
            if type(X_tilde_cat) == np.ndarray:
                if np.linalg.norm(X_tilde_cat) == 0:
                    self.normalization[cat] = 1
                else:
                    self.normalization[cat] = np.sqrt( (X_tilde_cat**2).mean()
                                                      - X_tilde_cat.mean()**2
                                                      )
            else:
                if X_tilde_cat.data.shape[0] == 0:
                    self.normalization[cat] = 1
                else:
                    self.normalization[cat] = np.sqrt(  (X_tilde_cat.data**2).sum() / np.prod(X_tilde_cat.shape) 
                                                      - (X_tilde_cat.data.sum() / np.prod(X_tilde_cat.shape))**2
                                                      )
            self.normalization[cat] += (self.normalization[cat] == 0)
            assert self.normalization[cat] > 0
        del X_cat, X_tilde_cat, cats_in_alpha
        self.normalization['ones'] = 1
        self.normalized_alphas = {}
        for var in self.alpha:
            for cat in self.alpha[var].keys():
                if  cat in self.normalization:
                    if var not in self.normalized_alphas:
                        self.normalized_alphas[var] = {}
                    if isinstance(self.alpha[var][cat], numbers.Number):
                        self.normalized_alphas[var][cat] = self.alpha[var][cat]/self.normalization[cat]**2
                    else:
                        assert type(self.alpha[var][cat]) == tuple
                        assert len(self.alpha[var][cat])  == 2
                        assert self.pen[var][cat]         == 'block_smoothing_reg'
                        self.normalized_alphas[var][cat] = (self.alpha[var][cat][0]/self.normalization[cat]**2, 
                                                            self.alpha[var][cat][1], 
                                                            )                      
        for var, dikt_cat_alpha in self.normalized_alphas.items():
            assert len(self.alpha[var]) == len(dikt_cat_alpha)
            for cat in self.alpha[var].keys():
                if isinstance(self.alpha[var][cat], numbers.Number):
                    assert isinstance(self.normalized_alphas[var][cat], numbers.Number)
                else:
                    assert type(self.alpha[var][cat]) == type(self.normalized_alphas[var][cat])



    def compute_surrogate(self, grad, grad_gp, ridge_grad, coef_tmp, eta, cur_fit, d_masks):
        # Compute the value of the smooth surrogate in the line-search procedure
        f    = cur_fit
        for coor_upd in grad.keys():
            var,key,ind = coor_upd
            mask =  d_masks.get(coor_upd, slice(None))
            if var in {'tensor_product_L',
                       'tensor_product_R',
                       }:
                assert coef_tmp[coor_upd].shape == self.coef[coor_upd[:2]][:,:,mask].shape, (coef_tmp[coor_upd].shape,
                                                                                             self.coef[coor_upd[:2]][:,:,mask].shape,
                                                                                             )
                if type(grad[coor_upd]) == np.matrix:
                    f += (  (grad[coor_upd]
                             + grad_gp.get(coor_upd,0)
                             + ridge_grad[coor_upd]
                             ).reshape(-1).dot((  coef_tmp[coor_upd] 
                                                - self.coef[coor_upd[:2]][:,mask]
                                                ).reshape(-1).T).sum()
                          + (1/(2*eta))*np.linalg.norm(  coef_tmp[coor_upd]
                                                       - self.coef[coor_upd[:2]][:,:,mask]
                                                       )**2
                          )
                else:
                    f += ( (grad[coor_upd] 
                            + grad_gp.get(coor_upd,0)
                            + ridge_grad[coor_upd]
                            ).reshape(-1).T@(coef_tmp[coor_upd] 
                                             - self.coef[coor_upd[:2]][:,:,mask]
                                             ).reshape(-1)
                          + (1/(2*eta))*np.linalg.norm(  coef_tmp[coor_upd]
                                                       - self.coef[coor_upd[:2]][:,:,mask]
                                                       )**2
                          )
            else:
                if type(grad[coor_upd]) == np.matrix:
                    f += ( (  grad[coor_upd] 
                            + grad_gp.get(coor_upd,0) 
                            + ridge_grad[coor_upd]
                            ).reshape(-1).dot((  coef_tmp[coor_upd] 
                                               - self.coef[coor_upd[:2]][:,mask]
                                               ).reshape(-1).T
                                               ).sum()
                          + (1/(2*eta))*np.linalg.norm(  coef_tmp[coor_upd] 
                                                       - self.coef[coor_upd[:2]][:,mask]
                                                       )**2
                          )
                else:
                    f += ( (  grad[coor_upd] 
                            + grad_gp.get(coor_upd,0)
                            + ridge_grad[coor_upd]
                            ).reshape(-1).T@(  coef_tmp[coor_upd]
                                             - self.coef[coor_upd[:2]][:,mask]
                                             ).reshape(-1)
                          + (1/(2*eta))*np.linalg.norm(  coef_tmp[coor_upd] 
                                                       - self.coef[coor_upd[:2]][:,mask]
                                                       )**2
                          )
            assert coef_tmp[coor_upd].shape == grad[coor_upd].shape,               ('pb_1', coef_tmp[coor_upd].shape, grad[coor_upd].shape)
            assert coef_tmp[coor_upd].shape == ridge_grad[coor_upd].shape,         ('pb_2', coef_tmp[coor_upd].shape, ridge_grad[coor_upd].shape)
            if var in {'tensor_product_L',
                       'tensor_product_R',
                       }:
                assert coef_tmp[coor_upd].shape == self.coef[(var,key)][:,:,mask].shape, ('pb_35', coef_tmp[coor_upd].shape, self.coef[(var,key)][:,:,mask].shape)
            else:
                assert coef_tmp[coor_upd].shape == self.coef[(var,key)][:,mask].shape, ('pb_3', coef_tmp[coor_upd].shape, self.coef[(var,key)][:,mask].shape)
            if self.active_gp:
                assert coef_tmp[coor_upd].shape == grad_gp[coor_upd].shape,          ('pb_4', coef_tmp[coor_upd].shape, grad_gp[coor_upd].shape)
            assert type(f) != np.matrix
        return f
    

    # Decomposition of the gradient computations 
    # because of the restrictions due to the sparse structure
    # and the use of the masks
    def custom_sum_einsum(self, XtX, coef, var2, key, mask, var_temp = None):
        var_upd    = (var_temp == var2)
        cbcbm_upd  = (var_temp == 'sesquivariate_b' and var2 == 'sesquivariate_bm')
        cbmcb_upd  = (var_temp == 'sesquivariate_b' and var2 == 'sesquivariate_b'  and type(key[0][0]) == tuple)
        cbmcbm_upd = (var_temp == 'sesquivariate_b' and var2 == 'sesquivariate_bm' and type(key[0][0]) == tuple)
        if EXTRA_CHECK:
            for key2 in self.keys[var2]:
                if self.active_gp:
                    if (var_temp,key) in coef and (var2,key2) in coef:
                        assert key,key2 in XtX # Probably need this check only when gp_pen > 0
                if (    (key,key2) in XtX
                    and not(   (var_upd    and key == key2)
                            or (cbcbm_upd  and key == (key2[0][0],(key2[1][0],)))
                            or (cbmcb_upd  and (key[0][0],(key[1][0],)) == key2)
                            or (cbmcbm_upd and (key[0][0],(key[1][0],)) == (key2[0][0],(key2[1][0],)))
                            )
                    ):                 
                    ss = self.orig_masks.get((var2,key2,()), slice(None))
                    if not (   mask == 'dyn'
                            or (    var_temp == 'sesquivariate_b'
                                and type(key[0][0]) == tuple
                                )
                            ):
                        assert (XtX[key,key2] @ (coef[var2,key2][:,mask])).shape == (XtX[key,key] @ (coef[var_temp,key][:,mask])).shape
        if mask == 'dyn':
            assert self.gp_pen
            ans = np.zeros((self.size[key], self.k))
            for key2 in self.keys[var2]:
                if (    (key,key2) in XtX
                    and not(   (var_upd    and key == key2)
                            or (cbcbm_upd  and key == (key2[0][0],(key2[1][0],)))
                            or (cbmcb_upd  and (key[0][0],(key[1][0],)) == key2)
                            or (cbmcbm_upd and (key[0][0],(key[1][0],)) == (key2[0][0],(key2[1][0],)))
                            )
                    ):
                    ss = self.orig_masks.get((var2,key2,()), slice(None))
                    ans[:,ss] += XtX[key,key2] @ coef[var2,key2][:,ss]
        else:
            ans = np.zeros((self.X_training[key].shape[1], coef[var2,next(iter(self.keys[var2]))][:,mask].shape[1]))
            for key2 in self.keys[var2]:
                if (    (key,key2) in XtX
                    and not(   (var_upd    and key == key2)
                            or (cbcbm_upd  and key == (key2[0][0],(key2[1][0],)))
                            or (cbmcb_upd  and (key[0][0],(key[1][0],)) == key2)
                            or (cbmcbm_upd and (key[0][0],(key[1][0],)) == (key2[0][0],(key2[1][0],)))
                            )
                    ):
                    ans += XtX[key,key2] @ (coef[var2,key2][:,mask])
        return ans


    def evaluate_fit(self, coef, dataset = None, gp_matrix = None, gp_pen = None):
        assert dataset
        if (type(gp_matrix)!=type(None)) and not gp_pen:
            return 0
        w_gp = (type(gp_matrix)!=type(None)) and gp_pen
        n    = self.n_training if dataset == 'training' else self.n_validation
        Y    = self.Y_training if dataset == 'training' else self.Y_validation
        if not w_gp:
            mse =        (0.5/n)*np.linalg.norm(Y 
                                                - self.predict(coef    = coef,
                                                               dataset = dataset,
                                                               )
                                                )**2
        else:
            mse = gp_pen*(0.5/n)*np.linalg.norm((Y
                                                 - self.predict(coef    = coef,
                                                                dataset = dataset,
                                                                )
                                                 ) @ gp_matrix
                                                )**2
        return mse


    def evaluate_ind_fit(self, coef, dataset = None):
        assert dataset
        n       = self.n_training    if dataset == 'training' else self.n_validation
        Y       = self.Y_training    if dataset == 'training' else self.Y_validation
        if self.iteration > self.flag_check_pred:
            self.flag_check_pred += self.epoch_check_pred
        ind_mse = (0.5/n)*np.linalg.norm((Y - self.predict(coef = coef, dataset = dataset)), axis = 0)**2
        return ind_mse


    def evaluate_fit_bcd(self, coor_upd, coef_tmp, quant, mask = None, dataset = None, MMt = None, gp_pen = 0, **kwargs):
        if gp_pen:
            assert type(MMt) != type(None)
        assert dataset
        assert len(quant) <= 3
        n   = self.n_training   if dataset == 'training' else self.n_validation
        XtX = self.XtX_training if dataset == 'training' else self.XtX_validation
        new_extra_part = {}
        new_fit        = 0
        for coor_upd in quant.keys():
            if coor_upd[0] in {'tensor_product_LR','low_rank_V','low_rank_UVt', 'sesquivariate_bm'}:
                raise ValueError
                #continue
            var, key, ind = coor_upd
            if var in {'unconstrained',
                       'low_rank_U',
                       'tensor_product_L',
                       'tensor_product_R',
                       'sesquivariate_b',
                       'sesquivariate_m',
                       }:
                if var == 'low_rank_U':
                    new_extra_part[coor_upd] = self.xtra_part_bu(n, XtX, coef_tmp, coor_upd, mask, gp_pen, MMt)
                elif var == 'sesquivariate_b':
                    new_extra_part[coor_upd] = self.xtra_part_cb(n, XtX, coef_tmp, coor_upd, mask, gp_pen, MMt) 
                elif var == 'sesquivariate_m':
                    new_extra_part[coor_upd] = self.xtra_part_cm(n, XtX, coef_tmp, coor_upd, mask, gp_pen, MMt)
                elif var == 'tensor_product_L':
                    new_extra_part[coor_upd] = self.xtra_part_cu(n, XtX, coef_tmp, coor_upd, mask, gp_pen, MMt)
                elif var == 'tensor_product_R':
                    new_extra_part[coor_upd] = self.xtra_part_cv(n, XtX, coef_tmp, coor_upd, mask, gp_pen, MMt)
                else:
                    new_extra_part[coor_upd] = self.xtra_part_cl(n, XtX, coef_tmp, coor_upd, mask, gp_pen, MMt)
                if EXTRA_CHECK:
                    try:
                        if var == 'low_rank_U':
                            assert coef_tmp['low_rank_UVt',coor_upd[1]].shape == new_extra_part[coor_upd].shape
                            assert coef_tmp[coor_upd].shape == (quant[coor_upd][:,self.dikt_masks.get(('low_rank_UVt', coor_upd[1]), slice(None))] @ coef_tmp['low_rank_V', key]).shape
                        else:
                            assert coef_tmp[coor_upd].shape == quant         [coor_upd].shape, (coef_tmp[coor_upd].shape, quant[coor_upd].shape)  
                            assert coef_tmp[coor_upd].shape == new_extra_part[coor_upd].shape, (coef_tmp[coor_upd].shape, quant[coor_upd].shape)
                    except Exception as e:
                        print(e)
                        raise e
                new_fit += self.part_fit(coef_tmp,
                                         coor_upd,
                                         quant,
                                         new_extra_part,
                                         dataset = dataset,
                                         **kwargs,
                                         )
            elif var in {'tensor_product_LR', 'low_rank_UVt', 'sesquivariate_bm'}:
                pass
            else:
                raise ValueError
        return new_fit 


    def evaluate_ind_reg(self, coef):
        reg    =  {}
        slope  =  {}
        offset =  {}
        cond   = (np.sum([e[0] == 'sesquivariate_b' for e in coef]) != 1) and len(coef) >= 10 and not EXTRA_CHECK
        for i, (coor, M) in enumerate(coef.items()):
            if   coor[0] in {'low_rank_UVt',
                             'low_rank_V',
                             'tensor_product_LR',
                             'sesquivariate_bm',
                             }:
                continue
            elif coor[0] in {'unconstrained',
                             'low_rank_U',
                             'tensor_product_L',
                             'tensor_product_R',
                             'sesquivariate_b',
                             'sesquivariate_m',
                             }:
                if cond:
                    print('\r'+'evaluate_ind_reg', i, '/', len(coef), end = '')
                var, key, ind = coor
                pen, alpha    = self.get_pen_alpha(var, key)
                res           = self.penalization(M, pen, alpha, key) # Compute regularization of one column
                if pen == 'clipped_abs_deviation':
                    reg[coor], slope[coor] ,offset[coor] = res
                else:
                    reg[coor] = res
            else:
                raise ValueError
        if cond:
            print()
        return reg, slope, offset


    def evaluate_ind_ridge(self, coef):
        ridge = {}
        for coor, M in coef.items():
            if coor[0] in {'low_rank_UVt', 'low_rank_V', 'tensor_product_LR', 'sesquivariate_bm'}:
                continue
            var, key, ind  = coor
            inpt, location = key
            pen, alpha     = self.get_pen_alpha(var, key)
            reshape_tensor = (    type(inpt[0]) == tuple
                              and var not in {'tensor_product_L', 'tensor_product_R', 'sesquivariate_b', 'sesquivariate_m'}
                              )
            if pen == '':
                pass
            elif pen == 'ridge':
                if type(M) in {np.ndarray, np.matrix}:
                    ridge[var,key,ind] = (alpha/2)*np.linalg.norm(M)**2 
                else:
                    ridge[var,key,ind] = (alpha/2)*sp.sparse.linalg.norm(M)**2
            elif pen == 'smoothing_reg':
                if type(M) not in {np.ndarray, np.matrix}:
                    raise NotImplementedError
                else:
                    if not reshape_tensor:
                        cc = M
                        if cc.shape[0] > 2:
                            ker  = np.array([[1],[-2],[1]])
                            if self.hprm['inputs.cyclic'].get(inpt):
                                conv = spim.filters.convolve(cc,
                                                             ker,
                                                             mode = 'wrap',
                                                             )
                            else:
                                conv = sig.convolve(cc, 
                                                    ker, 
                                                    mode = 'valid',
                                                    )
                            ridge[var,key,ind] = 0.5 * alpha * np.linalg.norm(conv)**2
                    else:
                        cc = M.reshape(*self.size_tensor_bivariate[key], -1)  
                        inpt1, inpt2 = inpt
                        if cc.shape[0] > 2:
                            ker1  = np.array([[[1]],[[-2]],[[1]]]) 
                            if self.hprm['inputs.cyclic'].get(inpt1):
                                conv1 = spim.filters.convolve(cc,
                                                              ker1,
                                                              mode = 'wrap',
                                                              )
                            else:
                                conv1 = sig.convolve(cc, 
                                                     ker1, 
                                                     mode = 'valid',
                                                     )
                            ridge[var,key,ind] = 0.5 * alpha * np.linalg.norm(conv1)**2
                        if var in {'sesquivariate_bm', 'sesquivariate_b', 'sesquivariate_m', 'low_rank_UVt', 'low_rank_V'}:
                            raise ValueError
                        if cc.shape[1] > 2:
                            ker2  = np.array([[[1 ],[ -2 ],[ 1]]])
                            if self.hprm['inputs.cyclic'].get(inpt2):
                                conv2 = spim.filters.convolve(cc,
                                                              ker2,
                                                              mode = 'wrap',
                                                              )
                            else:
                                conv2 = sig.convolve(cc, 
                                                     ker2, 
                                                     mode = 'valid',
                                                     )
                            ridge[var,key,ind] = 0.5 * alpha * np.linalg.norm(conv2)**2 
                            
            elif pen == 'factor_smoothing_reg':
                if type(M) not in {np.ndarray, np.matrix}:
                    raise NotImplementedError
                else:
                    if not reshape_tensor:
                        cc = M
                        if cc.shape[0] > 2:
                            ker  = np.array([[1],[-2],[1]])
                            if self.hprm['inputs.cyclic'].get(inpt):
                                conv = spim.filters.convolve(cc,
                                                             ker,
                                                             mode = 'wrap',
                                                             )
                            else:
                                conv = sig.convolve(cc, 
                                                    ker, 
                                                    mode = 'valid',
                                                    )
                            ridge[var,key,ind] = 0.5 * alpha * np.linalg.norm(conv)**2
                    else:
                        cc = M.reshape(*self.size_tensor_bivariate[key], -1)  
                        inpt1, inpt2 = inpt
                        if cc.shape[0] > 2:
                            ker1  = np.array([[[1]],[[-2]],[[1]]]) 
                            if self.hprm['inputs.cyclic'].get(inpt1):
                                conv1 = spim.filters.convolve(cc,
                                                              ker1,
                                                              mode = 'wrap',
                                                              )
                            else:
                                conv1 = sig.convolve(cc, 
                                                     ker1, 
                                                     mode = 'valid',
                                                     )
                            ridge[var,key,ind] = 0.5 * alpha * np.linalg.norm(conv1)**2
            else:
                raise NotImplementedError('Incorrect panalization : {0}'.format(pen))
        return ridge
    
    

    def foba_step(self, grad, grad_gp, ridge_grad, eta, d_masks):
        # Forward-backward step
        # ie gradient step and proximal operator
        coef_tmp= {}
        for coor_upd in grad.keys():
            var, key, ind = coor_upd
            mask          = d_masks.get(coor_upd, slice(None))
            pen, alpha    = self.get_pen_alpha(var, key)
            if pen == 'clipped_abs_deviation_v2':
                orig_mask  = self.orig_masks[coor_upd]
                same_mask  = (    type(mask) == type(orig_mask) 
                              and (mask == orig_mask
                                   if type(mask) == type(slice(None))
                                   else
                                   np.allclose(mask, orig_mask)
                                   )
                              )
                slope_mask = (   slice(None)
                              if same_mask
                              else
                              [k
                               for k in (orig_mask 
                                         if type(orig_mask) == np.ndarray
                                         else 
                                         np.arange(self.coef[coor_upd[:2]].shape[1])
                                         )
                               if k in mask
                               ]
                              )
                alpha = self.slope_ind[coor_upd][slope_mask]
            if EXTRA_CHECK:
                if var in {'tensor_product_L', 'tensor_product_R'}:
                    assert self.coef[var,key][:,:,mask].shape == grad[coor_upd].shape
                    assert self.coef[var,key][:,:,mask].shape == ridge_grad[coor_upd].shape
                else:
                    assert self.coef[var,key][:,mask].shape == grad[coor_upd].shape
                    assert self.coef[var,key][:,mask].shape == ridge_grad[coor_upd].shape
                if self.active_gp:
                    assert coor_upd in grad_gp
            if var in {'tensor_product_L', 'tensor_product_R'}:
                coef_tmp[coor_upd] = self.prox_operator(self.coef[(var,key)][:,:,mask]- eta*(grad[coor_upd]
                                                                                             + grad_gp.get(coor_upd,0)
                                                                                             + ridge_grad[coor_upd]
                                                                                             ),
                                                        pen,
                                                        eta,
                                                        alpha,
                                                        coef_zero = self.coef[(var,key)][:,:,mask],
                                                        )
                assert coef_tmp[coor_upd].shape == self.coef[(var,key)][:,:,mask].shape 
            else:    
                coef_tmp[coor_upd] = self.prox_operator(self.coef[(var,key)][:,mask] - eta*(  grad[coor_upd]
                                                                                            + grad_gp.get(coor_upd,0)
                                                                                            + ridge_grad[coor_upd]
                                                                                            ),
                                                        pen,
                                                        eta,
                                                        alpha,
                                                        coef_zero = self.coef[(var,key)][:,mask]
                                                        )
            if EXTRA_CHECK and alpha == 0 and var not in {'tensor_product_L', 'tensor_product_R'}:
                aa = self.coef[(var,key)][:,mask] - eta*(  grad[coor_upd]
                                                         + grad_gp.get(coor_upd,0)
                                                         + ridge_grad[coor_upd]
                                                         )
                bb = self.prox_operator(self.coef[(var,key)][:,mask] - eta*(  grad[coor_upd]
                                                                            + grad_gp.get(coor_upd,0)
                                                                            + ridge_grad[coor_upd]
                                                                            ),
                                        pen,
                                        eta,
                                        0,
                                        coef_zero = self.coef[(var,key)][:,mask]
                                        )
                assert np.all(aa == bb)
        return coef_tmp


        
    def get_pen_alpha(self, var, key):
        # For one coefficient matrix and one covariate, fetch the regularization and the coefficient
        inpt, location = key
        if var == 'A':
            pen   = self.pen.get(var,'')
            alpha = self.normalized_alphas.get(var,0)
        elif key == 'ones':
            pen   = ''
            alpha = 0
        else:
            pen   = self.pen  .get(var,{}).get(inpt,'')
            alpha = self.normalized_alphas.get(var,{}).get(inpt,0)
        return pen, alpha
    
    
    def initialize_coef(self, ):
        coef                = cp.deepcopy(self.given_coef)
        keys_upd            = []
        self.punished_coor  = set()
        self.prison_coor    = {}
        self.life_sentences = set()
        self.nb_sentences   = {}
        self.keys = {key : []
                     for key in self.formula.index.get_level_values('coefficient').unique()
                     }
                            
        if 'unconstrained' in self.keys.keys():
            # All 'unconstrained' coefficients
            for key in self.X_training.keys():
                inpt,location = key
                if inpt in self.formula.loc['unconstrained'].index:
                    if not (    key in self.mask and type(self.mask[key]) == np.ndarray 
                            and self.mask[key].shape[0] == 0
                            ):
                        self.keys['unconstrained'].append(key)
                    if 'unconstrained' not in self.frozen_variables:
                        if (    self.pen['unconstrained'].get(inpt) != 'row_group_lasso'
                            and self.col_upd.get(('unconstrained', inpt))
                            ):
                            keys_upd += [('unconstrained',key,(int(r),)) for r in self.mask.get(key, range(self.k))]
                        else:
                            keys_upd += [('unconstrained',key,())]
                    if ('unconstrained', key) not in coef:
                        coef['unconstrained',key] = np.zeros((self.size[key], self.k))
                        
        if 'low_rank_UVt' in self.keys.keys():   
            for key in list(filter(lambda x : x not in self.mask, self.X_training.keys())):
                inpt, location = key
                if inpt in self.formula.loc['low_rank_UVt'].index:
                    self.keys['low_rank_UVt'].append(key)
                    if 'low_rank_UVt' not in self.frozen_variables:
                        if (    self.pen['low_rank_U'].get(inpt) != 'row_group_lasso'
                            and self.col_upd.get(('low_rank_U', inpt))
                            ):
                            raise ValueError('we do not want column update for Blr')
                        else:
                            keys_upd += [('low_rank_U',key,())]
                    if ('low_rank_U', key) not in coef:
                        mm   = slice(None)
                        lenV = self.k
                        rk   = int(min(self.formula.xs(key = ['low_rank_UVt',inpt])['structure'],
                                       self.size[key],
                                       ))
                        coef[('low_rank_U',(inpt,location))] = 1e-4*np.random.randn(self.size[(inpt,location)], min(rk, self.k))
                        if lenV > 0:
                            coef['low_rank_V',(inpt,location)]        = np.zeros((self.k, min(rk, self.k)))
                            coef['low_rank_V',(inpt,location)][mm], _ = np.linalg.qr(np.random.randn(self.k, min(rk, self.k)))
                            coef['low_rank_UVt',(inpt,location)]      = coef[('low_rank_U',(inpt,location))] @ coef[('low_rank_V',(inpt,location))].T
                        
        if 'tensor_product_LR' in self.keys.keys():
            for key_uv in list(filter(lambda x : type(x[0][0]) == tuple, self.X_training.keys())):
                inpts,locations = key_uv
                if inpts in self.formula.loc['tensor_product_LR'].index:
                    if not (    key_uv in self.mask 
                            and type(self.mask[key_uv]) == np.ndarray 
                            and self.mask[key_uv].shape[0] == 0
                            ):
                        self.keys['tensor_product_LR'].append(key_uv)
                    if 'tensor_product_LR' not in self.frozen_variables:
                        if (    self.pen['tensor_product_L'].get(inpts) != 'row_group_lasso'
                            and self.col_upd.get(('tensor_product_L', inpts))
                            ):
                            keys_upd +=   [('tensor_product_L',key_uv,(int(r),),)
                                           for r in self.mask.get(key_uv, range(self.k))
                                           ] 
                        else:
                            keys_upd +=   [('tensor_product_L',key_uv,())]
                        if (    self.pen['tensor_product_R'].get(inpts) != 'row_group_lasso'
                            and self.col_upd.get(('tensor_product_R', inpts))
                            ):
                            keys_upd +=   [('tensor_product_R',key_uv,(int(r),),)
                                           for r in self.mask.get(key_uv, range(self.k))
                                           ]
                        else:
                            keys_upd +=   [('tensor_product_R',(inpts,locations),())]
                    if (   ('tensor_product_L', (inpts,(locations,))) not in coef 
                        or ('tensor_product_R', (inpts,(locations,))) not in coef
                        ):
                        rk = int(min(self.formula.xs(key = ['tensor_product_LR',inpts])['structure'],
                                     self.size_tensor_bivariate[key_uv][0],
                                     self.size_tensor_bivariate[key_uv][1],
                                     ))
                        coef['tensor_product_L', key_uv]  = np.zeros((self.size_tensor_bivariate[key_uv][0], rk, self.k))
                        coef['tensor_product_R', key_uv]  = np.zeros((self.size_tensor_bivariate[key_uv][1], rk, self.k))
                        coef['tensor_product_L', key_uv][:,:,self.mask.get(key_uv,None)] = 1e-3*np.random.randn(*coef['tensor_product_L',key_uv][:,:,self.mask.get(key_uv,None)].shape) 
                        coef['tensor_product_R', key_uv][:,:,self.mask.get(key_uv,None)] = 1e-3*np.random.randn(*coef['tensor_product_R',key_uv][:,:,self.mask.get(key_uv,None)].shape)
                        coef['tensor_product_LR',key_uv] = np.einsum('prk,qrk->pqk',
                                                                     coef['tensor_product_L',key_uv], 
                                                                     coef['tensor_product_R',key_uv], 
                                                                     ).reshape((-1, self.k)) 

        if 'sesquivariate_bm' in self.keys.keys():
            # 'sesquivariate_b' coefficients
            for key_b in list(filter(lambda x : type(x[0][0]) != tuple, self.X_training.keys())):
                (inpt_b,location_b) = key_b
                if inpt_b in self.formula.loc['sesquivariate_b'].index:
                    if not (key_b in self.mask and type(self.mask[key_b]) == np.ndarray and self.mask[key_b].shape[0] == 0):
                        self.keys['sesquivariate_b'].append(key_b)
                    if 'sesquivariate_b' not in self.frozen_variables:
                        if (    self.pen['sesquivariate_b'].get(inpt_b) != 'row_group_lasso'
                            and self.col_upd.get(('sesquivariate_b', inpt_b))
                            ):
                            for r in self.mask.get(key_b, range(self.k)):
                                if ('sesquivariate_b',key_b,(int(r),)) not in keys_upd:
                                    keys_upd += [('sesquivariate_b',key_b,(int(r),),) ]
                        else:
                            if ('sesquivariate_b',key_b,()) not in keys_upd:
                                keys_upd += [('sesquivariate_b',key_b,())]
                    if ('sesquivariate_b', key_b) not in coef:
                        coef['sesquivariate_b',key_b] = np.zeros((self.size[key_b], self.k))
            # 'sesquivariate_m' and 'sesquivariate_bm' coefficients
            for key_bm in list(filter(lambda x : type(x[0][0]) == tuple, self.X_training.keys())):
                (inpt_b,inpt_m),(location_b, location_m) = key_bm
                if (   (inpt_b,inpt_m) in self.formula.loc['sesquivariate_bm'].index 
                    or (inpt_m,inpt_b) in self.formula.loc['sesquivariate_bm'].index
                    ):
                    if not (    key_bm in self.mask 
                            and type(self.mask[key_bm]) == np.ndarray 
                            and self.mask[key_bm].shape[0] == 0
                            ):
                        self.keys['sesquivariate_bm'].append(key_bm)
                    if 'sesquivariate_bm' not in self.frozen_variables:
                        if (    self.pen['sesquivariate_bm'].get((inpt_b,inpt_m)) != 'row_group_lasso'
                            and self.col_upd.get(('sesquivariate_m', (inpt_b,inpt_m)))
                            ):
                            for r in self.mask.get(key_bm, range(self.k)):
                                if ('sesquivariate_m',key_bm,(int(r),),) not in keys_upd:
                                    keys_upd += [('sesquivariate_m',key_bm,(int(r),),) ]
                        else:
                            if ('sesquivariate_m',key_bm,()) not in keys_upd:
                                keys_upd += [('sesquivariate_m',key_bm,())]  
                    if ('sesquivariate_m', key_bm) not in coef:
                        assert self.size_tensor_bivariate[key_bm][0] == self.size[inpt_b,(location_b,)]
                        coef['sesquivariate_m', key_bm] = np.zeros((self.size_tensor_bivariate[key_bm][1], self.k))
                        coef['sesquivariate_bm',key_bm] = np.einsum('pk,qk->pqk',
                                                                    coef['sesquivariate_b',(inpt_b,(location_b,))], 
                                                                    coef['sesquivariate_m',key_bm], 
                                                                    ).reshape((-1, self.k))
                        
        assert len(keys_upd) == len(set(keys_upd)), (len(keys_upd), len(set(keys_upd)))
        print('Finished initialization')
        return coef, keys_upd
    
    
    def make_gp_matrix(self, ):
        # Compute the matrix for the sum-consistent model
        assert self.hprm['gp_matrix'] in {'cst', None} or self.lbfgs
        if self.hprm['gp_matrix'] in {'cst', 'nat', 'rteReg', 'admReg', 'districts', 'eye'}:
            if self.hprm['gp_matrix'] == 'cst':
                self.post_to_reg = {pp:'nat' for k, pp in enumerate(self.hprm['posts_names'])}
            elif self.hprm['gp_matrix'] in {'nat', 'rteReg', 'admReg', 'districts'}:
                assert self.hprm['zone'] == 'full', 'zone is not full - should think about it differently'
                raise NotImplementedError
            elif self.hprm['gp_matrix'] == 'eye':
                self.post_to_reg = {pp:pp for k, pp in enumerate(self.hprm['posts_names'])}
            else:
                raise ValueError
            self.idxPost_to_reg = {k : self.post_to_reg[self.hprm['posts_names'][k]]
                                   for k in range(len(self.hprm['posts_names']))
                                   }
            self.reg_to_tuples = {reg : tuple(sorted([ii 
                                                      for ii, post in enumerate(self.hprm['posts_names'])
                                                      if self.idxPost_to_reg[ii] == reg
                                                      ])
                                              )
                                  for reg in set(self.post_to_reg.values())
                                  }
            self.tuples_to_reg = {v:k
                                  for k, v in self.reg_to_tuples.items()
                                  }
            self.gp_matrix = np.zeros((self.k, self.k))
            for reg, indices in self.reg_to_tuples.items():
                for k in indices: 
                    for j in indices:
                        self.gp_matrix[k,j] = 1
        elif type(self.hprm['gp_matrix']) == int and self.hprm['gp_matrix'] > 0:
            assert len(self.hprm['coor_posts']) == self.k
            self.rolling_away, self.rolling_away_ind = tools.geography.rolling_away(self.hprm['coor_posts'],
                                                                                    self.hprm['coor_posts'],
                                                                                    )
            self.gp_matrix = np.zeros((self.k, self.k))
            for k in range(self.k):
                self.gp_matrix[self.rolling_away_ind[self.hprm['posts_names'][k]][:self.hprm['gp_matrix']],k] = 1
        else:
            raise ValueError('Incorrect gp_matrix : {0}'.format(self.hprm['gp_matrix']))     


    def make_orig_masks(self):
        d_masks = {}
        for coor_upd in self.keys_upd.copy():
            var, key, ind = coor_upd
            if ind == ():
                if key in self.mask:
                    d_masks[coor_upd] = np.array(self.mask[key])
                elif key[::-1] in self.mask:
                    d_masks[coor_upd] = np.array(self.mask[key[::-1]])
                else:
                    d_masks[coor_upd] = slice(None)
            else:
                assert type(ind) == tuple
                assert len (ind) == 1
                assert type(ind[0]) == int
                d_masks[coor_upd] = np.array(ind)
            if type(d_masks[coor_upd]) == np.ndarray and len(d_masks[coor_upd]) == 0:
                self.keys_upd.remove(coor_upd)
            if var == 'low_rank_U': 
                del d_masks[coor_upd]
        return d_masks
    
    
    def part_fit(self, coef, coor_upd, quant, extra_part, mask = slice(None), dataset = None, indi = False):
        # Decomposed computation of the data-fitting term
        var, key = coor_upd[:2]
        if var == 'low_rank_U':
            cof = coef['low_rank_UVt', key]
        elif var in {'tensor_product_L',
                     'tensor_product_R',
                     }:
            cof = coef[coor_upd] if coor_upd in coef else coef[var, key][:,:,mask]
        else:
            cof = coef[coor_upd] if coor_upd in coef else coef[var, key][:,mask]
        if EXTRA_CHECK:
            assert cof.shape == quant[coor_upd].shape, (cof.shape, quant[coor_upd].shape)
            assert cof.shape == extra_part[coor_upd].shape, (cof.shape, extra_part[coor_upd].shape)
        if type(cof) in {np.ndarray, np.matrix}:
            if EXTRA_CHECK:
                assert np.multiply(cof, extra_part[coor_upd]).sum()>= 0
            ans = np.multiply(cof, quant[coor_upd] + 0.5*extra_part[coor_upd])
        else:
            if EXTRA_CHECK:
                assert (cof.multiply(extra_part[coor_upd])).sum()>= 0
            ans  = cof.multiply(quant[coor_upd] + 0.5*extra_part[coor_upd])
        if indi:
            part_fit = ans.sum(axis = 0)
        else:
            part_fit = ans.sum()
        return part_fit            


    def penalization(self, M, pen, mu, key):
            # Compute the penalizations
            if pen == '':  
                return 0
            elif pen in {'ridge',
                         'smoothing_reg',
                         'factor_smoothing_reg',
                         }:  
                return 0
            elif pen == 'lasso':
                return mu*np.sum(np.abs(M))                 
            elif pen == 'row_group_lasso':
                assert M.ndim == 2
                if type(M) in {np.ndarray, np.matrix}:
                    return mu*np.sum([np.linalg.norm(M[i])        for i in range(M.shape[0])]) 
                else:
                    return mu*np.sum([sp.sparse.linalg.norm(M[i]) for i in range(M.shape[0])])
            elif pen == 'col_group_lasso':
                if type(M) in {np.ndarray, np.matrix}:
                    return mu*np.sum([np.linalg.norm(M[:,i])        for i in range(M.shape[1])])    
                else:
                    return mu*np.sum([sp.sparse.linalg.norm(M[:,i]) for i in range(M.shape[1])])
            elif pen == 'clipped_abs_deviation' :
                alpha, beta, gamma = mu
                p, rcol  = M.shape
                reg    = 0
                slope  = np.zeros(rcol)
                offset = np.zeros(rcol)
                for j in range(rcol):
                    if type(M) in {np.ndarray, np.matrix}:
                        c_norm = np.linalg.norm(M[:,j])
                    else:
                        c_norm = sp.sparse.linalg.norm(M[:,j])
                    pen1   = alpha*c_norm
                    pen2   = beta *c_norm + gamma
                    reg += min(pen1,pen2)
                    slope [j] = alpha if pen1 < pen2 else beta
                    offset[j] = 0     if pen1 < pen2 else gamma
                return reg, slope, offset                
            elif pen == 'elastic_net':
                alpha, theta = mu
                if type(M) in {np.ndarray, np.matrix}:
                    return theta*(alpha*np.sum(np.abs(M))     + ((1 - alpha)/2)*np.linalg.norm(M)**2)
                else:
                    return theta*(alpha*np.sum(np.abs(M))     + ((1 - alpha)/2)*sp.sparse.linalg.norm(M)**2)
            elif pen == 'total_variation':
                return mu*np.sum(np.abs(M[1:] - M[:-1]))  
            elif pen == 'trend_filtering':
                return mu*np.sum(np.abs(M[2:] - 2*M[1:-1] + M[:-2]))
            else:
                raise ValueError('Bad Penalization')

        
    def pop_coordinates(self):
        # Get a coordinate for the BCD algorithm
        # and manage the coordinates that lead to zero improvements
        # ie put them in prison for exponential times (if active_set is activated)
        while not self.list_coor:
            if self.active_set:
                # Exponential sentences
                assert len(self.prison_coor.keys())   == len(set(self.prison_coor.keys()))
                if bool(set(self.prison_coor.keys()) & set(self.punished_coor)):
                    assert 0
                self.nb_sentences.update({(var,key,post) : self.nb_sentences.get((var,key,post),0)+1
                                          for (var,key,post) in self.punished_coor
                                          }) 
                # Send the punished ones from police station to county prison 
                self.prison_coor.update( {(var,key,post) : min(2**self.nb_sentences[var,key,post], 1e4)
                                          for (var,key,post) in self.punished_coor
                                          })
                # Check 
                if EXTRA_CHECK and 0:
                    for (var,(inpt,location),post) in self.punished_coor:
                        if not self.col_upd.get(inpt):
                            if not np.linalg.norm(self.coef[var,(inpt,location)][:,post]) == 0: # Pas besoin d'être nuls pour être en prison
                                assert 0
                # Clear police station
                self.punished_coor = set()
                # Prison 
                self.prison_coor = {(var,(inpt,location),post) : v-1
                                    for (var,(inpt,location),post), v
                                    in self.prison_coor.items()
                                    if v-1 > 0
                                    }
                if EXTRA_CHECK and 0:
                    assert len(self.prison_coor) <= len(self.keys_upd)*self.k
                    for (var,(inpt,location),post) in self.prison_coor:
                        if not self.col_upd.get((var,inpt)):
                            if not np.linalg.norm(self.coef[var,(inpt,location)][:,post]) == 0:
                                assert 0 # A column can already be in prison but have moved during recent iterations because of a too small mask
                self.dikt_masks = {coor:v 
                                   for coor,v in {(var,(inpt,location),post) : self.prison_masks((var,(inpt,location),post),self.prison_coor) 
                                                  for (var,(inpt,location),post) in self.keys_upd
                                                  }.items() 
                                   if type(v) != type(slice(None))
                                   or v       !=      slice(None)
                                   }
                self.list_coor = [coor 
                                  for coor in self.keys_upd
                                  if (    (coor not in self.prison_coor) 
                                      and not (    type(self.dikt_masks.get(coor)) == np.ndarray
                                               and self.dikt_masks.get(coor).ndim == 1 
                                               and self.dikt_masks.get(coor).shape[0] == 0
                                               )
                                      )
                                  ]
                self.fully_out = [coor
                                  for coor in self.keys_upd
                                  if coor not in self.list_coor
                                  ]
                np.random.shuffle(self.list_coor)
            else:
                self.list_coor  = self.keys_upd.copy()
                if self.iteration == 1:
                    assert self.list_coor
        return self.list_coor.pop()
    
    
    def prison_masks(self, coor, prison):
        # Change the masks depending on the coefficients put in prison
        orig_mask = self.orig_masks.get(coor, slice(None))
        if self.life_sentences:
            orig_mask = np.array([k for k in (orig_mask
                                              if type(orig_mask) == np.ndarray
                                              else np.arange(self.k)
                                              )
                                    if k not in self.life_sentences
                                   ])
        sub_prison = [col for col in prison if coor[:2] == col[:2]]
        if sub_prison:
            new_mask = np.array([k 
                                 for k in (orig_mask
                                           if type(orig_mask) == np.ndarray
                                           else np.arange(self.k)
                                           ) 
                                 if (*coor[:2],k) not in sub_prison
                                 or np.linalg.norm(self.coef[coor[:2]][:,:,k] if coor[0] in {'tensor_product_L','tensor_product_R'} else self.coef[coor[:2]][:,k]) > 0  #  Do not give up column that moved while in prison  
                                 ]).astype(int)
            if   type(orig_mask) == type(slice(None)) and new_mask.shape[0]/self.k > self.prop_active_set:
                new_mask = orig_mask
            return new_mask
        else:
            return orig_mask
        
    
    def precomputationsY(self, X, Y, dataset = None):
        assert dataset in {'training', 'validation'}
        assert not self.active_gp
             
        # Compute everything with Y
        print(colored('compute YtY', 'red'))
        YtY      = Y.T.dot(Y)
        Y_sqfrob = np.trace(YtY)

        L = [YtY, Y_sqfrob]
        self.print_gso(L)
        print(colored('compute X1tY', 'red'))        
        try:
            XtY = tools.batch_load(path_data, 
                                    prefix    = self.dikt['features.{0}.univariate'.format(dataset)], 
                                    data_name = 'XtY_'+dataset,
                                    data_type = 'dict_sp',
                                    )
            for key in X: 
                assert key in XtY
        except Exception: 
            print(colored('compute XtY', 'red'))
            XtY  = {}
            for key in X.keys():
                mask = self.mask.get(key, slice(None))
                XtY[key] = np.zeros((X[key].shape[1], Y.shape[1]))
                XtY[key][:,mask] = X[key].T.dot(Y[:,mask]) 
            if len(XtY) < 1e3:
                try:
                    tools.batch_save(path_data,
                                     data      = XtY,
                                     prefix    = self.dikt['features.{0}.univariate'.format(dataset)],
                                     data_name = 'XtY_'+dataset,
                                     data_type = 'dict_sp',
                                     )
                except tools.saving_errors:
                    pass
                except Exception as e:
                    raise e
        L = XtY
        self.print_gso(L)
 
        ### Check
        if EXTRA_CHECK:
            key     = next(iter(XtY.keys()))
            mask    = self.mask.get(key, slice(None))
            control = np.zeros((X[key].shape[1], Y.shape[1]))
            control[:,mask] = X[key].T.dot(Y[:,mask]) 
            if type(XtY[key]) == sp.sparse.csr_matrix:
                control = sp.sparse.csr_matrix(control)
                assert np.linalg.norm(XtY[key].data - control.data)  < 1e-8
            else:
                assert np.allclose(XtY[key], control)#.all()          
        return Y_sqfrob, YtY, XtY

    
    
    def predict(self, coef = None, dataset = None, drop = {}, X_new = None, n_new = None):
        if hasattr(self, 'bfgs_long_coef'):
            return lbfgs.bfgs_pred(self, self.bfgs_long_coef, data = dataset)
        if   dataset == 'training':
            n = self.n_training
            X = self.X_training 
        elif dataset == 'validation':
            n = self.n_validation
            X = self.X_validation
        else:
            assert X_new
            assert n_new
            n = n_new
            X = X_new
        if type(coef) == type(None):
            coef = self.coef
        pred2 = np.zeros((n, self.k))
        for ii, coor in enumerate(coef):
            if len(coor) == 3:
                var, key, ind = coor
                cc            = coef[coor]
                if coor[0] == 'sesquivariate_bm':
                    if   ('sesquivariate_b',(coor[1][0][0],(coor[1][1][0],)),coor[2]) in coef:
                        mask = self.dikt_masks.get(('sesquivariate_b',coor[1][0],coor[2]), slice(None))
                    elif ('sesquivariate_m',)+coor[1:] in coef:
                        mask = self.dikt_masks.get(('sesquivariate_m',)+coor[1:], slice(None))
                    else:
                        raise KeyError                
                else:    
                    mask = self.dikt_masks.get((var,key,ind), slice(None))
            else:
                var, key = coor
                mask     = self.orig_masks.get((var,key), slice(None))
                cc       = coef[coor][:,mask]
            if var in self.keys and var not in drop:
                if key in self.keys[var]:
                    pred2[:,mask] += X[key] @ cc
        return pred2
    
    
    def print_fit_with_mean(self, dataset = 'training', mean = 'training'):
        if   dataset == 'training':
            Y = self.Y_training
            n = self.n_training
        elif dataset == 'validation':
            Y = self.Y_validation
            n = self.n_validation
        if   mean == 'training':
            Y_mean = self.Y_training.mean(axis = 0)
        elif mean == 'validation':
            Y_mean = self.Y_validation.mean(axis = 0)
        rmse = tools.format_nb.round_nb((0.5/n)*np.linalg.norm(Y - Y_mean)**2,3)
        print('fit', (dataset + ' ')[:5], 'with mean', (mean + ' ')[:5], ':', rmse)
        if dataset == 'training':
            self.normalized_variance_training = rmse
        elif dataset == 'validation':            
            self.normalized_variance_validation  = rmse


    def print_gso(self, lits)  :
        for e in lits:
            continue
            if type(e) == np.ndarray:
                print('var', ' - ', 'size', '{:.3e}'.format(sys.getsizeof(e)/1e6), 'MBytes', ', ', 'shape', e.shape)
            if type(e) == dict:
                print('var', ' - ', 'size', '{:.3e}'.format(sys.getsizeof(e)/1e6), 'MBytes')
                print([(k, v.shape) for k, v in e.items()])

        
    def print_info(self, ):
        print('\n'+
              'iter = ',  '{:<6}'.format(self.iteration),
              '& dec = ', '{:<6.6}'.format(tools.format_nb.round_nb(self.decr_obj)), 
              '& tol = ', '{:<6.6}'.format(self.tol), 
              '& obj = ', '{:<6.6}'.format(tools.format_nb.round_nb(self.cur_obj_training)), 
              end = '\n'
              )
        cat_of_prisoners = sorted(set([coor[1][0] for coor in self.prison_coor]))
        print('{0} prisoners'.format(len(self.prison_coor)), ' : ', cat_of_prisoners)
        self.end_epoch   = time.time()
        print('time for epoch : {0} seconds'.format(round(self.end_epoch - self.begin_epoch, 2)))
        self.begin_epoch = time.time()

   
    def print_len_epochs(self, ):
        print('Flags')
        for e in ['self.epoch_stopping_criteria', 'self.epoch_print_info']:
            print('   ', '{0:25}'.format(e[5:]), eval(e))
            
            
    def print_progress(self, ):
        n_app = 5
        info = [
                'iter = {:<6}'.format(self.iteration),
                'nfit_training  = {:10.10}'.format(tools.format_nb.round_nb(self.fit_training[self.iteration-1]/self.normalized_variance_training, nb_app = n_app)), 
                (   'dec  = {:10.10}'.format(tools.format_nb.round_nb(self.decr_obj, nb_app = n_app))
                 if self.decr_obj != self.decr_obj_0 
                 else 
                 'st/ch : {0}/{1}'.format(self.nb_steps, self.flag_stopping_criteria)
                 ),
                ]
        if self.compute_validation:
            info.append('nfit_validation = {:10.10}'.format(tools.format_nb.round_nb(self.fit_validation[self.iteration-1]/self.normalized_variance_validation, nb_app = n_app)))
        tools.progress.show([self.iteration, self.max_iter], *info)


    def prox_operator(self, X, pen, eta, mu, coef_zero = None):
        if pen == '':    
            X_tmp = X
        elif pen in {'ridge',
                     'smoothing_reg',
                     'factor_smoothing_reg',
                     }:    
            X_tmp = X
        elif pen == 'clipped_abs_deviation' :
            X_tmp     = tools.proximal.prox_clipped_abs_deviation(X, eta, mu, coef_zero = coef_zero)
        elif pen == 'col_group_lasso' :
            X_tmp     = tools.proximal.prox_col_group_lasso(X, eta*mu)
        elif pen == 'elastic_net' :
            X_tmp     = tools.proximal.prox_elastic_net(X, eta, mu)
        elif pen == 'lasso' :
            X_tmp     = tools.proximal.prox_lasso(X, eta*mu)
        elif pen == 'row_group_lasso' :
            X_tmp     = tools.proximal.prox_row_group_lasso(X, eta*mu)
        elif pen == 'total_variation':
            X_tmp = tools.proximal.prox_total_variation(X, eta*mu)
        elif pen == 'trend_filtering' :
            X_tmp = tools.proximal.prox_trend_filtering(X, eta*mu)
        else:
            raise ValueError('bad penalization : {0}'.format(pen))
        return X_tmp
                
                
    def size_coef(self, ):
        self.memory_size_coef = {}
        for (var,(inpt,location)), v in self.coef.items():
            self.memory_size_coef[inpt] = self.memory_size_coef.get(inpt,0) + sys.getsizeof(v)/1e6
        return self.memory_size_coef 
    
           
    def sort_keys(self, keys, masks):
        cat_owned   = {}
        keys_shared = []
        keys_owned  = []
        for inpt, location in keys: 
            cond = (    (inpt,location) in masks
                    and not (type(masks[inpt,location]) == type(slice(None)) and masks[inpt,location] == slice(None))
                    )
            if cond:
                keys_owned.append((inpt, location))
                cat_owned[inpt] = True
            else:
                keys_shared.append((inpt, location))
                if inpt not in cat_owned:
                    cat_owned[inpt] = False
        keys_shared = sorted(keys_shared, key = lambda x : str(x))
        keys_owned  = sorted(keys_owned,  key = lambda x : str(x))
        print('    {0} cats shared - {1} cats owned'.format(len(keys_shared), len(keys_owned)))
        # The shared variables are in the top rows of the design matrix
        # The individual covariates come after
        return keys_shared + keys_owned, cat_owned
    

    def stopping_criteria(self, dikt_fit_grad):
        # Computant the quantities relevant for the stopping criteria
        small_decrease, early_stop, norm_grad_small, max_iter_reached, all_dead = [0]*5
        if not self.bcd:
            small_decrease = (self.decr_obj < self.tol)            
        # Stopping criteria
        if self.nb_steps >= self.flag_stopping_criteria:
            self.flag_stopping_criteria += self.epoch_stopping_criteria
            # Evolution of obj validation
            if self.compute_validation and self.iteration >= self.flag_compute_validation :
                old_mean_ft      = self.fit_validation   [self.iteration - 2*self.epoch_stopping_criteria:self.iteration  - self.epoch_stopping_criteria].mean()
                new_mean_ft      = self.fit_validation   [self.iteration - self.epoch_stopping_criteria:self.iteration].mean()
                old_mean_gp_ft   = self.fit_gp_validation[self.iteration - 2*self.epoch_stopping_criteria:self.iteration  - self.epoch_stopping_criteria].mean()
                new_mean_gp_ft   = self.fit_gp_validation[self.iteration - self.epoch_stopping_criteria:self.iteration].mean()
                early_stop       = (old_mean_ft + old_mean_gp_ft < new_mean_ft + new_mean_gp_ft) and self.hprm['afm.algorithm.first_order.early_stop_validation']#and not (self.vr) 
            # Same thing individually
            if self.compute_validation and self.iteration >= self.flag_compute_ind_validation:
                if self.hprm['afm.algorithm.first_order.early_stop_ind_validation']:
                    self.flag_compute_ind_validation += self.epoch_compute_ind_validation
                    old_ind_ft      = self.fit_ind_validation[self.iteration - 2*self.epoch_compute_ind_validation:self.iteration  - self.epoch_compute_ind_validation].mean(axis = 0)
                    new_ind_ft      = self.fit_ind_validation[self.iteration -   self.epoch_compute_ind_validation:self.iteration].mean(axis = 0)
                    if not self.active_gp:
                        early_stop_ind  = (old_ind_ft < new_ind_ft)#and not (self.vr) 
                        for pp in range(self.k):
                            if early_stop_ind[pp]:
                                self.life_sentences.add(pp)
                        all_dead = (len(self.life_sentences) == self.k)
            # Small decrease
            if self.bcd:
                old_mean       = self.obj_training[self.iteration - 2*self.epoch_stopping_criteria:self.iteration  - self.epoch_stopping_criteria].mean()
                new_mean       = self.obj_training[self.iteration - self.epoch_stopping_criteria:self.iteration].mean()
                self.decr_obj  = old_mean - new_mean 
                small_decrease = (self.decr_obj < self.tol) and self.hprm['afm.algorithm.first_order.early_small_decrease']
            # Small grad
            norm_grad        = self.compute_grad_norm(dikt_fit_grad)
            norm_grad_small  = (norm_grad < self.norm_grad_min) and not (self.bcd ) and self.hprm['afm.algorithm.first_order.early_stop_small_grad']#or self.vr)
       
        if early_stop      : print('\nEARLY STOPPING : ' , 'old mean', old_mean_ft, ' - recent mean', new_mean_ft)
        if small_decrease  : print('\nSMALL DECREASE : ' , tools.format_nb.round_nb(self.decr_obj))
        if norm_grad_small : print('\nNORM GRAD SMALL : ', norm_grad)    
        return small_decrease, early_stop, norm_grad_small, all_dead
        

    def update_coef(self, coef_tmp, d_masks, new_ind_slope = {}, new_ind_offset = {}):
        for coor_upd in coef_tmp:
            if type(coef_tmp[coor_upd]) == np.matrix:
                coef_tmp[coor_upd] = np.asarray(coef_tmp[coor_upd])
            assert type(coef_tmp[coor_upd]) == np.ndarray
            if coor_upd[0] == 'sesquivariate_bm':
                if   ('sesquivariate_b',(coor_upd[1][0][0],(coor_upd[1][1][0],)),coor_upd[2]) in coef_tmp:
                    mask = d_masks.get(('sesquivariate_b',(coor_upd[1][0][0],(coor_upd[1][1][0],)),coor_upd[2]), slice(None))
                elif ('sesquivariate_m',)+coor_upd[1:] in coef_tmp:
                    mask = d_masks.get(('sesquivariate_m',)+coor_upd[1:], slice(None))
                else:
                    assert 0
            elif coor_upd[0] == 'tensor_product_LR':
                if   ('tensor_product_L',)+coor_upd[1:] in coef_tmp:
                    mask = d_masks.get(('tensor_product_L',)+coor_upd[1:], slice(None))
                elif ('tensor_product_R',)+coor_upd[1:] in coef_tmp:
                    mask = d_masks.get(('tensor_product_R',)+coor_upd[1:], slice(None))
                else:
                    raise KeyError
            else:
                mask = d_masks.get(coor_upd, slice(None))
            if type(new_ind_slope.get(coor_upd, None)) != type(None):
                orig_mask  = self.orig_masks.get(coor_upd, slice(None))
                inner_mask = np.array([i for i, k in enumerate(orig_mask 
                                                               if type(orig_mask) == np.ndarray
                                                               else 
                                                               np.arange(self.k)
                                                               ) if k in (mask 
                                                                          if type(mask) == np.ndarray
                                                                          else 
                                                                          np.arange(self.k)
                                                                          )])
                self.slope_ind [coor_upd][inner_mask] = new_ind_slope [coor_upd]
                self.offset_ind[coor_upd][inner_mask] = new_ind_offset[coor_upd]
            var,(inpt,location) = coor_upd[:2]
            assert var in {'unconstrained',
                           'low_rank_UVt',
                           'low_rank_U',
                           'low_rank_V',
                           'tensor_product_LR',
                           'tensor_product_L',
                           'tensor_product_R',
                           'sesquivariate_b',
                           'sesquivariate_m',
                           'sesquivariate_bm',
                           }
            if self.active_set and not self.col_upd.get((var,inpt)):
                orig_mask = self.orig_masks.get(coor_upd, slice(None))
                ind_posts = mask if type(mask) == np.ndarray else (orig_mask if type(orig_mask) == np.ndarray else np.arange(self.k))
                if EXTRA_CHECK and 0: # with prop active set, some coor can be both in prisons and active
                    for jj in ind_posts:
                        if (var,(inpt,location),jj) in self.prison_coor:
                            assert 0
                norm_diff    = (  np.linalg.norm((coef_tmp[coor_upd]
                                                  - (self.coef[var,(inpt,location)][:,:,mask] 
                                                     if var in {'tensor_product_L',
                                                                'tensor_product_R',
                                                                }
                                                     else
                                                     self.coef[var,(inpt,location)][:,mask]
                                                     )
                                                  ),
                                                 axis = tuple(k
                                                              for k in range(coef_tmp[coor_upd].ndim - 1)
                                                              )
                                                 )
                                )
                # One could evaluate fit_validation for each post and jail accordingly but too expansive : rather do col_upd
                convicts     = ind_posts[norm_diff == 0]
                convicts_tmp = np.arange(ind_posts.shape[0])[norm_diff == 0]
                for i, r in enumerate(convicts):
                    if not np.linalg.norm(self.coef[var,(inpt,location)][:,:,r]
                                          if var in {'tensor_product_L','tensor_product_R'}
                                          else
                                          self.coef[var,(inpt,location)][:,r]
                                          ) < np.finfo(float).eps:
                        if np.linalg.norm(coef_tmp[coor_upd][:,:,convicts_tmp[i]] if var in {'tensor_product_L','tensor_product_R'} else coef_tmp[coor_upd][:,convicts_tmp[i]]) == 0:
                            assert 0
                    if (*coor_upd[:2], r) not in self.prison_coor:
                        self.punished_coor.add((*coor_upd[:2], r))
            if EXTRA_CHECK:
                if var == 'low_rank_V':
                    assert self.coef[var,(inpt,location)][mask].shape     == coef_tmp[coor_upd].shape
                elif var in {'tensor_product_L','tensor_product_R'}:
                    assert self.coef[var,(inpt,location)][:,:,mask].shape == coef_tmp[coor_upd].shape
                else:
                    assert self.coef[var,(inpt,location)][:,mask].shape   == coef_tmp[coor_upd].shape
                self.coef_prec[var,(inpt,location)] = self.coef[var,(inpt,location)].copy()
            shape_before = self.coef[var, (inpt,location)].shape
            if var == 'low_rank_V':
                self.coef[var,(inpt,location)][mask]      = coef_tmp[coor_upd]
            if var in {'tensor_product_L','tensor_product_R'}:
                self.coef[var, (inpt,location)][:,:,mask] = coef_tmp[coor_upd]
            else:
                self.coef[var, (inpt,location)][:,mask]   = coef_tmp[coor_upd]
            assert shape_before == self.coef[var, (inpt,location)].shape
            if EXTRA_CHECK:
                if   var == 'low_rank_U':
                    assert self.coef[var,(inpt,location)].shape == coef_tmp[coor_upd].shape
                    assert np.allclose(self.coef[var,(inpt,location)], coef_tmp[coor_upd])
                elif var == 'low_rank_V':
                    assert self.coef[var,(inpt,location)][mask].shape == coef_tmp[coor_upd].shape
                    assert np.allclose(self.coef[var,(inpt,location)][mask], coef_tmp[coor_upd])
                elif var in {'tensor_product_L','tensor_product_R'}:
                    assert self.coef[var,(inpt,location)][:,:,mask].shape == coef_tmp[coor_upd].shape
                    assert np.allclose(self.coef[var,(inpt,location)][:,:,mask], coef_tmp[coor_upd])
                else:
                    assert self.coef[var,(inpt,location)][:,mask].shape == coef_tmp[coor_upd].shape
                    assert np.allclose(self.coef[var,(inpt,location)][:,mask], coef_tmp[coor_upd])
                if self.active_set and not self.col_upd.get((var,inpt)):
                    for i, rr in enumerate(convicts):
                        if not np.allclose(self.coef     [var,(inpt,location)][:,:,rr] if var in {'tensor_product_L','tensor_product_R'} else self.coef[var,(inpt,location)][:,rr], 
                                           self.coef_prec[var,(inpt,location)][:,:,rr] if var in {'tensor_product_L','tensor_product_R'} else self.coef[var,(inpt,location)][:,rr],
                                           ):
                            print('mask', mask)
                            print('ind_post', ind_posts)
                            print('convicts', convicts)
                            print('convicts_tmp', convicts_tmp)
                            assert 0
                        
    
    def update_Cbm(self, coef):
        # Update the Cbm coefficient matrics
        # It corresponds to the sesquivariate structural constraint
        for var, key, ind in list(coef.keys()):
            mm = self.dikt_masks.get((var, key, ind), slice(None))
            if var == 'sesquivariate_b':
                inpt_b, location_b = key
                for var2, key2 in self.coef:
                    if var2 == 'sesquivariate_bm':
                        (inpt_b2,inpt_m), (location_b2, location_m) = key2
                        if (    inpt_b     == inpt_b2
                            and location_b == (location_b2,)
                            ):
                            if ('sesquivariate_bm', key2) in self.coef:
                                coef['sesquivariate_bm', key2, ind] = np.einsum('pk,qk->pqk',
                                                                                coef[var, key, ind],
                                                                                self.coef['sesquivariate_m', key2][:,mm],
                                                                                ).reshape((-1, coef[var, key, ind].shape[1]))
                                assert coef['sesquivariate_bm', key2, ind].shape == self.coef['sesquivariate_bm', key2][:,mm].shape
            elif var == 'sesquivariate_m':
                assert ('sesquivariate_bm',key) in self.coef
                if ('sesquivariate_bm',key) in self.coef:
                    (inpt_b,inpt_m), (location_b, location_m) = key
                    coef['sesquivariate_bm', key, ind] = np.einsum('pk,qk->pqk',
                                                                   self.coef['sesquivariate_b', (inpt_b, (location_b,))][:,mm],
                                                                   coef[var, key, ind],
                                                                   ).reshape((-1, coef[var, key, ind].shape[1]))
                    assert coef['sesquivariate_bm', key, ind].shape == self.coef['sesquivariate_bm', key][:,mm].shape
        return coef   
                        
    
    def update_Cuv(self, coef):
        # Update the Cuv coefficient matrics
        # It corresponds to the low-rank structural constraint
        for var, key, ind in list(coef.keys()):
            if var in {'tensor_product_L','tensor_product_R'}:
                mm = self.dikt_masks.get((var, key, ind), slice(None))
                if var == 'tensor_product_L':
                    coef['tensor_product_LR', key, ind] = np.einsum('prk,qrk->pqk',
                                                      coef[var, key, ind],
                                                      self.coef['tensor_product_R', key][:,:,mm],
                                                      ).reshape((-1, coef[var, key, ind].shape[2]))
                elif var == 'tensor_product_R':
                    coef['tensor_product_LR', key, ind] = np.einsum('prk,qrk->pqk',
                                                      self.coef['tensor_product_L', key][:,:,mm],
                                                      coef[var, key, ind],
                                                      ).reshape((-1, coef[var, key, ind].shape[2]))
        return coef  

    
    
    def update_VB(self, coef):
        # Update the Blr coefficient matrics
        # It corresponds to the low-rank (along the substations) structural constraint
        svd = 1
        if self.iteration <= 10:
            print('svd'*svd + 'qr'*(1-svd))
        for var, key, ind in list(coef.keys()):
            mm = self.dikt_masks.get(('low_rank_V', key), slice(None))
            if var == 'low_rank_U':
                assert not ind
                if svd or ('low_rank_V', None) not in coef:
                    coef['low_rank_V', key] = self.best_orth_bv(coef, mask = mm)
                else:
                    assert 0 # Should be ckecked before use
                    Q, R = self.best_qr(coef, mask = mm)
                    coef['low_rank_V', key] = Q
                coef['low_rank_UVt', key] = coef[var, key, ind] @ coef['low_rank_V', key].T
        return coef

    
    def warm_start(self, ):
        if (not self.given_coef) and self.hprm['afm.algorithm.first_order.try_warmstart'] and (not self.hprm['plot.afm']): # No coef has been directly given 
            wanted      = self.dikt['model_wo_hyperp']
            list_wanted = [e for a in wanted.split('/') for e in a.split('_')]
            try:
                FF_candidates  = sorted(os.listdir(path_betas))
            except FileNotFoundError:
                FF_candidates = []
            for ii, FF in enumerate(FF_candidates):
                try:
                    GG_candidates  = sorted(os.listdir(os.path.join(path_betas, FF)))
                except FileNotFoundError:
                    GG_candidates = []
                for jj, GG in enumerate(GG_candidates):
                    cand = FF + '/' + GG
                    if ('pone' in wanted and 'pall' in cand) or ('pall' in wanted and 'pone' in cand):
                        continue
                    list_wanted_copy = list_wanted.copy()
                    list_wanted_copy = [e for e in list_wanted_copy if 'pone' not in e]
                    list_cand        = [e for a in cand.split('/') for e in a.split('_')]
                    list_cand        = [e for e in list_cand        if 'pone' not in e]
                    if list_cand == list_wanted_copy or list_cand == [e for e in list_wanted_copy if e != 'apr']:
                        list_hyperprm = os.listdir(os.path.join(path_betas, 
                                                                cand,
                                                                ))
                        if not list_hyperprm:
                            continue
                        str_warm_model = os.path.join(cand,
                                                      list_hyperprm[0],
                                                      )
                        try:
                            coef       = tools.batch_load(path_betas, 
                                                          prefix    = str_warm_model, 
                                                          data_name = 'coef', 
                                                          data_type = 'dict_tup',
                                                          )
                            keys_upd   = tools.batch_load(path_betas, 
                                                          prefix    = str_warm_model, 
                                                          data_name = 'keys_upd', 
                                                          data_type = 'pickle',
                                                          )
                            if sorted(keys_upd) == sorted(self.keys_upd):
                                assert sorted(coef.keys()) == sorted(self.coef.keys()), set(coef.keys()).symmetric_difference(set(self.coef.keys()))
                                self.coef = coef
                            else:
                                raise ValueError
                                for k in coef:
                                    if k in self.coef:
                                        assert self.coef[k].shape == coef[k].shape
                                        self.coef[k] = coef[k]
                        except ValueError as e:
                            print(e)
                            assert 0
                        except Exception as e:
                            print(e)
                            continue
                        print(colored('Warm Start Found :', 'blue'), cand)
                        if self.bcd:
                            self.epoch_stopping_criteria = int(len(self.keys_upd)*2)
                        else:
                            self.epoch_stopping_criteria = 2
                        return 0 # break
        print(colored('No Warm Start', 'blue'))
        
        
    def xtra_part_bu(self, n, XtX, coef, coor_upd, mask, gp_pen, MMt):
        key = coor_upd[1]
        xxx = (1/n)*XtX[key,key] @ (coef[('low_rank_UVt',*coor_upd[1:])]
                                    if ('low_rank_UVt',*coor_upd[1:]) in coef
                                    else
                                    coef['low_rank_UVt',coor_upd[1]][:,mask]
                                    )
        if gp_pen:
                xxx = gp_pen * xxx @ MMt[mask][:,mask]
        return xxx
    
    
    def xtra_part_cb(self, n, XtX, coef, coor_upd, mask, gp_pen, MMt):
        cbb =   (1/n)*(  XtX[coor_upd[1],coor_upd[1]]
                       @ (coef[coor_upd]
                          if coor_upd in coef
                          else coef[coor_upd[:2]][:,mask]
                          )
                       )
        cbm = 2*(1/n)*np.sum([  XtX[coor_upd[1],keybm] 
                              @ (coef['sesquivariate_bm', keybm, coor_upd[-1]]
                                 if ('sesquivariate_bm', keybm, coor_upd[-1]) in coef
                                 else
                                 coef['sesquivariate_bm', keybm][:,mask]
                                 )
                              for keybm in self.keys['sesquivariate_bm']
                              if (keybm[0][0],(keybm[1][0],)) == coor_upd[1]
                              if (coor_upd[1],keybm) in XtX
                              ], 
                              axis = 0,
                              )
        if not gp_pen:
            cc = (coef[coor_upd] if coor_upd in coef else coef[coor_upd[:2]][:,mask])
            cmm = np.zeros(cc.shape)
            for keybm2 in self.keys['sesquivariate_bm']:
                if (keybm2[0][0],(keybm2[1][0],)) == coor_upd[1]:
                    for keybm1 in self.keys['sesquivariate_bm']:
                        if (keybm1[0][0],(keybm1[1][0],)) == coor_upd[1]:
                            if (keybm1,keybm2) in XtX:
                                cmm += (1/n)*np.einsum('pqk,qk->pk', 
                                                       (  XtX[keybm1,keybm2]
                                                        @ (coef['sesquivariate_bm', keybm2, coor_upd[-1]]
                                                           if ('sesquivariate_bm', keybm2, coor_upd[-1]) in coef 
                                                           else 
                                                           coef['sesquivariate_bm', keybm2][:,mask]
                                                           )
                                                        ).reshape((-1,
                                                                   coef.get(('sesquivariate_m', keybm1),
                                                                            self.coef['sesquivariate_m', keybm1],
                                                                            )[:,mask].shape[0], 
                                                                   coef.get(('sesquivariate_m', keybm1), 
                                                                            self.coef['sesquivariate_m', keybm1],
                                                                            )[:,mask].shape[1],
                                                                   )),
                                                       coef.get(('sesquivariate_m', keybm1),
                                                                self.coef['sesquivariate_m', keybm1],
                                                                )[:,mask],
                                                       )
        if gp_pen:
            cbb = gp_pen * cbb @ MMt[mask][:,mask]
            if cbm.ndim > 0: # Check that cbm is not zero ie the list was not empty
                cbm = gp_pen * cbm @ MMt[mask][:,mask]
            else:
                assert cbm == 0
            cmm =   (1/n)*np.sum([np.einsum('pqk,qk->pk', 
                                            (XtX[keybm1,keybm2] @ (coef['sesquivariate_bm',keybm2, coor_upd[-1]]
                                                                   if ('sesquivariate_bm', keybm2, coor_upd[-1]) in coef
                                                                   else
                                                                   coef['sesquivariate_bm', keybm2][:,mask])@ MMt[mask][:,mask]
                                             ).reshape((-1, 
                                                        coef.get(('sesquivariate_m', keybm1),
                                                                 self.coef['sesquivariate_m', keybm1],
                                                                 )[:,mask].shape[0], 
                                                        coef.get(('sesquivariate_m', keybm1), self.coef['sesquivariate_m', keybm1])[:,mask].shape[1],
                                                        )),
                                            coef.get(('sesquivariate_m', keybm1), self.coef['sesquivariate_m', keybm1])[:,mask],
                                            )
                                  for keybm2 in self.keys['sesquivariate_bm'] if keybm2[0] == coor_upd[1]
                                  for keybm1 in self.keys['sesquivariate_bm'] if keybm1[0] == coor_upd[1]
                                  if (keybm1,keybm2) in XtX
                                  ],
                                 axis = 0, 
                                )
        xxx = cbb + cbm + cmm
        return xxx
    
    
    def xtra_part_cm(self, n, XtX, coef, coor_upd, mask, gp_pen, MMt):
        ccc = (1/n)*XtX[coor_upd[1],coor_upd[1]] @ (coef[('sesquivariate_bm',*coor_upd[1:])]
                                                    if ('sesquivariate_bm',*coor_upd[1:]) in coef
                                                    else
                                                    coef['sesquivariate_bm',coor_upd[1]][:,mask]
                                                    )#[:,mask_right]
        if gp_pen:
            ccc = gp_pen * ccc @ MMt[mask][:,mask]
        (inpt_b,inpt_m),(location_b,location_m) = coor_upd[1]
        xxx = np.einsum('pqk,pk->qk',
                        ccc.reshape((coef.get(('sesquivariate_b', (inpt_b, (location_b,))),
                                              self.coef['sesquivariate_b', (inpt_b, (location_b,))],
                                              )[:,mask].shape[0],
                                     -1,
                                     coef.get(('sesquivariate_b', (inpt_b, (location_b,))),
                                              self.coef['sesquivariate_b', (inpt_b, (location_b,))],
                                              )[:,mask].shape[1],
                                     )),
                        coef.get(('sesquivariate_b', (inpt_b, (location_b,))),
                                 self.coef['sesquivariate_b', (inpt_b, (location_b,))],
                                 )[:,mask],
                        )
        return xxx
 
    
    def xtra_part_cl(self, n, XtX, coef, coor_upd, mask, gp_pen, MMt):
        xxx = (1/n)*XtX[coor_upd[1],coor_upd[1]] @ (coef[coor_upd] if coor_upd in coef else coef[coor_upd[:2]][:,mask])
        if gp_pen: 
            xxx = gp_pen * xxx @ MMt[mask][:,mask]
        assert xxx.shape == (coef[coor_upd] if coor_upd in coef else coef[coor_upd[:2]][:,mask]).shape, (xxx.shape, (coef[coor_upd] if coor_upd in coef else coef[coor_upd[:2]][:,mask]).shape, coor_upd, mask)
        return xxx
    
    
    def xtra_part_cu(self, n, XtX, coef, coor_upd, mask, gp_pen, MMt):
        ccc = (1/n)*XtX[coor_upd[1],coor_upd[1]] @ (coef[('tensor_product_LR',*coor_upd[1:])] if ('tensor_product_LR',*coor_upd[1:]) in coef else coef['tensor_product_LR',coor_upd[1]][:,mask])#[:,mask_right]
        if gp_pen:
                ccc = gp_pen * ccc @ MMt[mask][:,mask]
        xxx = np.einsum('pqk,qrk->prk',
                        ccc.reshape((-1,
                                     coef.get(('tensor_product_R', coor_upd[1]), self.coef[('tensor_product_R', coor_upd[1])])[:,:,mask].shape[0],
                                     coef.get(('tensor_product_R', coor_upd[1]), self.coef[('tensor_product_R', coor_upd[1])])[:,:,mask].shape[2],
                                     )),
                        coef.get(('tensor_product_R', coor_upd[1]), self.coef[('tensor_product_R', coor_upd[1])])[:,:,mask],
                        )
        return xxx
    
    
    def xtra_part_cv(self, n, XtX, coef, coor_upd, mask, gp_pen, MMt):
        ccc = (1/n)*XtX[coor_upd[1],coor_upd[1]] @ (coef[('tensor_product_LR',*coor_upd[1:])] if ('tensor_product_LR',*coor_upd[1:]) in coef else coef['tensor_product_LR',coor_upd[1]][:,mask])
        if gp_pen:
                ccc = gp_pen * ccc @ MMt[mask][:,mask]
        xxx = np.einsum('pqk,prk->qrk',
                        ccc.reshape((coef.get(('tensor_product_L', coor_upd[1]), self.coef[('tensor_product_L', coor_upd[1])])[:,:,mask].shape[0],
                                     -1,
                                     coef.get(('tensor_product_L', coor_upd[1]), self.coef[('tensor_product_L', coor_upd[1])])[:,:,mask].shape[2],
                                     )),
                        coef.get(('tensor_product_L', coor_upd[1]), self.coef[('tensor_product_L', coor_upd[1])])[:,:,mask],
                        )
        return xxx
                    