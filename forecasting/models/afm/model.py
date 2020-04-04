
try:
    from termcolor import colored
except ModuleNotFoundError:
    colored = lambda s, t : s
    
import time
import os
import numpy as np
import copy as cp
import scipy as sp
import scipy.signal  as sig
import scipy.ndimage as spim
from copy import deepcopy
from sys        import getsizeof as gso
from numbers    import Number
from subprocess import CalledProcessError, TimeoutExpired
#from pickle    import UnpicklingError
#
#import nb_format
#import progress
#import misc
#from prox import prox_L1, prox_enet, prox_TV, prox_TF, prox_L2, prox_row_lasso, prox_col_lasso, prox_ncvx_classo_v2, prox_ncvx_classo
#import primitives
#from custom_exception import custex
#import build_data
from .lbfgs import start_lbfgs, bfgs_pred, sort_keys # Use old if gp_pen because the structure shared/owned seems unadapted in this case


    
#from .approx_tf_plots import plot_fit



#tuple_Errors = (AssertionError, CalledProcessError, custex, EOFError, FileNotFoundError, 
#                OSError, OverflowError, TimeoutExpired, UnpicklingError, ValueError)

EXTRA_CHECK = 0

class afm:
    """
    Defines a model that stores all the covariates and the XtX, XtY
    with the methods for optimization (proximal-gradient descent or lbfgs), prediction and loss computation
    """
    dikt_var_temp = {
                     'bu' : 'Blr', 
                     'Cu' : 'Cuv',
                     'Cv' : 'Cuv',
                     'Cb' : 'Cb',
                     'Cm' : 'Cbm',
                     }

    def __init__(self, 
                 param,             
                 path_Saved  ='',    
                 dikt       = None,          
                 ):
        # Sets all the parameters as attributes of the class
        self.param       = param
        self.max_iter    = int(self.param['tf_max_iter'])
        self.dikt        = dikt
        self.path_betas  = path_Saved + 'Betas/'
        self.config_coef = self.param['tf_config_coef']
        assert self.config_coef
        self.alpha       = {var:{key:alp 
                                 for key, alp in dikt_key_alp.items() 
                                 if (   (var in self.config_coef and key in self.config_coef[var])
                                     or (var in {'Cb','Cm'} and 'Cbm' in self.config_coef)
                                     )
                                 }
                            for var,dikt_key_alp in self.param['tf_alpha'].items()
                            }
        self.pen         = {k:v 
                            for k,v in self.param['tf_pen'].items()   
                            if (   k in self.config_coef
                                or (k in {'Cb','Cm'} and 'Cbm' in self.config_coef)
                                )
                            }
        self.lbfgs       = np.sum(['lbfgs' in key 
                                   for key, value in self.config_coef.items() 
                                   if bool(value)
                                   ]) 
                        # If one of the coefficient matrix (there is one at most)
                        # contains the keyword lbfgs then the lbfgs algorithm will be used
                        # otherwise the proximal gradient with backtracking line search is used
        self.share_enet  = self.param['share_enet'] # elastic net parameter
        if 'Blr' in self.alpha and 'Blr' in self.pen: 
            # Low-rank (along the substations) coefficient matrix 
            self.alpha['bu'] = self.alpha['Blr']
            self.pen  ['bu'] = self.pen  ['Blr']
            del self.alpha['Blr'], self.pen['Blr']
        if 'Cuv' in self.alpha and 'Cuv' in self.pen:
            # Low-rank interaction (independently for each substation)
            self.alpha['Cu'] = self.alpha['Cuv']
            self.pen  ['Cu'] = self.pen  ['Cuv']
            self.alpha['Cv'] = self.alpha['Cuv']
            self.pen  ['Cv'] = self.pen  ['Cuv']
            del self.alpha['Cuv'], self.pen['Cuv']
        self.gp_pen      = self.param['gp_pen'] # matrix for the sum-consistent model
        
        # Freeze the univariate or the bivariate coefficient matrices (rarely used)        
        self.freeze_B          = self.param.get('freeze_B')
        self.freeze_C          = self.param.get('freeze_C')
        
        # Stopping criteria
        self.dual_gap          = False
        self.tol               = self.param.get('tf_tol')
        self.norm_grad_min     = self.param.get('norm_grad_min')
        self.dist_min          = self.param.get('dist_min')
        del param, dikt, path_Saved, 
        
        # Line-Search parameters  
        self.eta                = 1
        self.etaA               = 1
        self.eta1               = 1
        self.eta2               = 1
        self.theta              = 0.5
        self.proba_ls           = 0.2
        self.nb_steps = 0
        
        # Problem Form
        self.A_max    = self.param.get('tf_A_max')
        self.orth_J   = 0
        self.r_B      = self.param.get('tf_rk_B', {})
        self.r_UV     = self.param.get('tf_rk_UV')
        
        # VR parameters
        self.epoch = 0
        
        # Descent method
        self.bcd             = self.param['tf_bcd'] 
        self.batch_cd        = not self.bcd
        self.active_set      = self.param['tf_active_set']
        self.prop_active_set = self.param['tf_prop_active_set']
        self.col_upd         = self.param['tf_col_upd'] if self.bcd else {}

        print('Coef      : '+ str(self.config_coef))
        print('Pen       : ')
        for k, v in self.pen.items():
            print('    ', k, ' : ', v)
        print('alphas    : ')
        for k, v in self.alpha.items():
            print('    ', k, ' : ', v)            
        print('batch' if self.batch_cd else 'bcd per col for ' + repr([k for k, v in self.col_upd.items() if v]))
        print('sparse_coef' if self.param['tf_sparse_coef'] else 'not sparse coef')
        assert self.batch_cd + self.bcd == 1
        self.sparse   = 0
    

    def fit(self, config,
            X_train, Y_train, 
            X_test = None, Y_test = None, 
            given_warm = None,
            ):
        # Fot the model given the training and test matrices (used for stopping criteria)
        self.X_train    = X_train.get('X_train', {})
        self.mask       = config['mask1']
        self.size       = config['size1']
        self.Y_train    = Y_train.copy()
        self.n_train    = self.Y_train.shape[0]
        self.k          = self.Y_train.shape[1]
        self.factor_A   = bool(set(self.config_coef) & {'A'})
        self.mask.update(**config.get('mask2', {}))
        self.size.update(**config.get('size2', {}))
        self.size_tensor2 = config.get('size_tensor2',{})
        print('masks examples : ')
        if self.mask:
            for cat in sorted(set(self.param['data_cat'].values())):
                for key, w in self.mask.items():
                    if self.param['data_cat'].get(key) == cat:
                        print('    ', key, ' : ', repr(w)[:80]) 
                        break
        else:
            print('no mask')
        self.active_gp = (self.gp_pen > 0) and bool(self.param['gp_matrix']) # indicator of the sum-consistent model
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
                
        if ((self.param['tf_plot'] and self.param['any_plot']) or self.param['tf_early_stop_ind_test']):
            self.M_mean    = np.ones((self.k, self.k))/self.k
            
        # Warm-start
        self.given_coef = {} if given_warm is None else given_warm
        if self.given_coef:
            print(colored('Coef given', 'green'))
        del given_warm
        self.print_fit_with_mean(data = 'train', mean = 'train')
        
        # Precomputations train
        if not self.lbfgs:    
            print('Precomputations train')
            self.precomputationsY(X_train, data = 'train')
            print('Precomputations train done')        
        del X_train, Y_train

        # Test data
        self.compute_test    = (type(X_test) != int) 
        self.precompute_test = not self.lbfgs and (type(X_test) != int) and self.param.get('tf_precompute_test', True)
        if self.compute_test:
            self.X_test  = X_test.get('X_test', {})
            self.Y_test  = Y_test.copy()
            self.n_test  = self.Y_test.shape[0]
            self.X_test.update({**X_test.get('X2_test', {})})
            self.print_fit_with_mean(data = 'test', mean = 'train')
            self.print_fit_with_mean(data = 'test', mean = 'test')
            if self.precompute_test:
                print('Precomputations test')
                self.precomputationsY(X_test, data = 'test')
                print('Precomputations test done')
        del X_test, Y_test
        self.iteration     = 0


        """
        Initialize coefficients
        """
        print('Initialize Coefficients')
        self.coef, self.keys_upd = self.initialize_coef()
        print('    ', len(self.coef), 'coef')
        print('    ', len(self.keys_upd), 'keys_upd')
        self.orig_masks = self.make_orig_masks()
        self.dikt_masks = {k:v for k, v in self.orig_masks.items() 
                               if type(v) != type(slice(None))
                               or v!= slice(None)
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
            self.epoch_compute_test     = 1
            self.epoch_compute_ind_test = self.epoch_stopping_criteria
            self.epoch_print_info       = 1 if EXTRA_CHECK else int(    self.epoch_stopping_criteria) 
            self.flag_stopping_criteria = int(2 * self.epoch_stopping_criteria) # Needs two periods cause two means are taken
            self.flag_print_info        = 0
            self.flag_check_fit         = 1e5
            # Flags
            self.flag_print_progress    = 0
            self.flag_show_prison       = 0
            self.flag_compute_test      = -1 #self.flag_print_progress - 1
            self.flag_compute_ind_test  = int(5 * self.epoch_stopping_criteria)
            self.flag_check_pred        = -1
            # 
            self.print_len_epochs()
            self.begin_epoch            = time.time()
        
        if self.param.get('stop_before_coef'):
            assert 0
        try:
            """
            Try to load final model
            """
            assert self.param['load'] and self.param['load_model']
            keys_upd   = primitives.tf_data_load(self.path_betas, 
                                                 self.dikt['model_pred'], 
                                                 self.param, 
                                                 'keys_upd', 
                                                 mod = 'json',
                                                 )
            assert set(self.keys_upd) == set(keys_upd), 'set(self.keys_upd) != set(keys_upd)'
            self.coef  = primitives.tf_data_load(self.path_betas, 
                                                 self.dikt['model_pred'], 
                                                 self.param, 
                                                 'coef', 
                                                 mod = 'dict_tup',
                                                 )
            if self.lbfgs:
                aaa = sorted([e 
                              for e in self.X_train.keys() 
                              if e not in self.mask
                              ])
                bbb = sorted([e 
                              for e in self.X_train.keys() 
                              if e     in self.mask
                              ], 
                             key = lambda x : (self.param['data_cat'][x],x), 
                             )
                self.sorted_keys = aaa + bbb
                for var, key in self.coef:
                    self.keys[var].append(key)
            print(colored('Coef loaded', 'blue'))
            self.change_sparse_structure()
        except (*tuple_Errors, KeyError) as e:
            """
            Begin algorithm then
            """
            print(colored(str(e), 'red'))
            print(colored('coef not loaded', 'red'))
            self.sorted_keys, self.cat_owned = sort_keys(
                                                         self.X_train.keys(), 
                                                         self.mask, 
                                                         self.param['data_cat'],
                                                         )
            self.compute_normalization()
            # Normalize data
            for key in self.sorted_keys:
                self.X_train[key] /= self.normalization[self.param['data_cat'][key]]
                self.X_test [key] /= self.normalization[self.param['data_cat'][key]]
            if hasattr(self, 'XtX_train'):
                for key in self.XtX_train.keys():
                    key1, key2 = key.split('@')
                    self.XtX_train[key] /= self.normalization[self.param['data_cat'][key1]]*self.normalization[self.param['data_cat'][key2]]
                    if self.precompute_test:
                        self.XtX_test [key] /= self.normalization[self.param['data_cat'][key1]]*self.normalization[self.param['data_cat'][key2]]
                for key in self.XtY_train.keys():
                    self.XtY_train[key] /= self.normalization[self.param['data_cat'][key]]
                    if self.precompute_test:
                        self.XtY_test [key] /= self.normalization[self.param['data_cat'][key]]
            if self.lbfgs:
                if self.active_gp and False:#self.param['zone'] != 'full': # Profiling lead to the following choice of implementations for the two setups
                    self.use_lbfgs_old = True
                    start_lbfgs_old(self) # Not used anymore
                else:
                    self.use_lbfgs_old = False
                    start_lbfgs(self) # Launch the lbfgs algorithm
                # Reorganize
                for ii, key in enumerate(self.X_train.keys()):
                    var = 'lbfgs_coef'
                    self.coef[var,key] = cp.deepcopy(self.bfgs_long_coef[slice(self.key_col_large_matching[key][0], 
                                                                               self.key_col_large_matching[key][1],
                                                                               )
                                                                         ])#/self.normalization[self.param['data_cat'][key]]
                    if ~self.param['tf_sparse_coef'] and type(self.coef[var,key]) in {sp.sparse.csc_matrix, sp.sparse.csr_matrix}:
                        self.coef[var,key] = self.coef[var,key].toarray()                       
                # Change data type for sparsity rules
                self.change_sparse_structure()
            else:
                # Use the first-order algorithm
                # Initialize arrays
                self.change_sparse_structure()
                self.fit_train     = np.zeros(self.max_iter)
                if (self.param['tf_plot'] and self.param['any_plot']) or self.param['tf_early_stop_ind_test']:
                    self.fit_ind_train = np.zeros((self.max_iter, self.k))
                    self.fit_mean_train = np.zeros(self.max_iter)
                self.fit_gp_train  = np.zeros(self.max_iter)
                self.reg       = np.zeros(self.max_iter)
                self.ridge     = np.zeros(self.max_iter)
                self.obj_train = np.zeros(self.max_iter)
                self.etas      = np.zeros(self.max_iter)
                self.etaAs     = np.zeros(self.max_iter)
                self.eta1s     = np.zeros(self.max_iter)
                self.eta2s     = np.zeros(self.max_iter)
                self.gap_B, self.gap_U, self.gap_V, self.gap_D, self.gap_W, self.gap_Z = [[] for k in range(6)] 
                if self.compute_test:
                    self.fit_test     = np.zeros(self.max_iter)
                    if (self.param['tf_plot'] and self.param['any_plot']) or self.param['tf_early_stop_ind_test']:
                        self.fit_ind_test = np.zeros((self.max_iter, self.k))                
                        self.fit_mean_test = np.zeros(self.max_iter)                
                    self.fit_gp_test  = np.zeros(self.max_iter)
                    self.reg          = np.zeros(self.max_iter)
                    self.obj_test     = np.zeros(self.max_iter)
                self.warm_start()
                
                """
                First values
                """
                print('Compute fit train')
                self.cur_fit_train    = self.evaluate_fit(self.coef, data = 'train')
                self.cur_fit_gp_train = self.evaluate_fit(self.coef, data = 'train', 
                                                          gp_matrix = self.gp_matrix, 
                                                          gp_pen = self.gp_pen,
                                                          ) if self.active_gp else 0
                print('Compute reg')
                self.slope_ind  = {}
                self.offset_ind = {}
                self.cur_ind_reg, self.slope_ind, self.offset_ind = self.evaluate_ind_reg({coor:(self.coef[coor[:2]][:,:,self.dikt_masks.get(coor, slice(None))] if coor[0] in {'Cu','Cv'} else self.coef[coor[:2]][:,self.dikt_masks.get(coor, slice(None))]) for coor in self.keys_upd})
                self.cur_reg       = np.sum([v for k, v in self.cur_ind_reg.items()])
                print('Compute ridge')
                self.cur_ind_ridge = self.evaluate_ind_ridge({coor:(self.coef[coor[:2]][:,:,self.dikt_masks.get(coor, slice(None))] if coor[0] in {'Cu','Cv'} else self.coef[coor[:2]][:,self.dikt_masks.get(coor, slice(None))]) for coor in self.keys_upd})
                self.cur_ridge     = np.sum([v for k, v in self.cur_ind_ridge.items()])
                print('Compute obj')
                self.cur_obj_train =   self.cur_fit_train\
                                     + self.cur_fit_gp_train\
                                     + self.cur_reg\
                                     + self.cur_ridge    
                assert np.abs(self.cur_fit_train + self.cur_fit_gp_train + self.cur_reg + self.cur_ridge - self.cur_obj_train ) <= 1e-12
                print('Update arrays')
                self.fit_train[self.iteration]     = self.cur_fit_train
                if (   (self.param['tf_plot'] and self.param['any_plot']) 
                    or self.param['tf_early_stop_ind_test']
                    ):
                    self.fit_ind_train[self.iteration]  = self.evaluate_ind_fit(self.coef, data = 'train')
                    self.fit_mean_train[self.iteration] = self.evaluate_fit(self.coef, data = 'train', gp_matrix = self.M_mean, gp_pen = 1)
                if self.active_gp:
                    self.fit_gp_train[self.iteration] = self.cur_fit_gp_train
                self.reg      [self.iteration] = self.cur_reg      
                self.ridge    [self.iteration] = self.cur_ridge      
                self.obj_train[self.iteration] = self.cur_obj_train
                self.decr_obj_0    = 10**(np.ceil(np.log(self.cur_obj_train)/np.log(10)))
                self.decr_obj      = self.decr_obj_0
                if self.compute_test : 
                    print('Compute fit test')
                    self.cur_fit_test = self.evaluate_fit(self.coef, data = 'test')
                    if (self.param['tf_plot'] and self.param['any_plot']) or self.param['tf_early_stop_ind_test']:
                        self.fit_ind_test[self.iteration]  = self.evaluate_ind_fit(self.coef, data = 'test')
                        self.fit_mean_test[self.iteration] = self.evaluate_fit(self.coef, data = 'test', gp_matrix = self.M_mean, gp_pen = 1)
                    self.cur_fit_gp_test = self.evaluate_fit(self.coef, data = 'test', gp_matrix = self.gp_matrix, gp_pen = self.gp_pen) if self.active_gp else 0
                    print('Compute ridge test')
                    self.fit_test[self.iteration] = self.cur_fit_test
                    if self.active_gp:
                        self.fit_gp_test[self.iteration] = self.cur_fit_gp_test
                    self.cur_obj_test =   self.fit_test[self.iteration] \
                                       +  self.cur_fit_gp_test\
                                       +  self.cur_reg \
                                       +  self.cur_ridge 
                    self.obj_test[self.iteration]    = self.cur_obj_test
                self.iteration += 1 
        
                """
                Start Descent
                """
                if EXTRA_CHECK:
                    self.Y_train0 = self.Y_train.copy()
                    self.X_train0 = cp.deepcopy(self.X_train)
                
                self.start_descent()
                self.last_iteration = self.iteration
                print('last iteration : ', self.last_iteration)
                if (self.param['any_plot'] and self.param['tf_plot']) and '_sv_' not in self.dikt['model_pred']:
                    try:
                        plot_fit(self, io = 0, ltx = 1)
                    except Exception as e:
                        print(colored(' \n \n' + repr(e) + '\n \n ', 'red', 'on_cyan'))

            # Algorithm terminated, denormalize data
            for key in self.sorted_keys:
                self.X_train[key] *= self.normalization[self.param['data_cat'][key]]
                self.X_test [key] *= self.normalization[self.param['data_cat'][key]]
            if hasattr(self, 'XtX_train'):
                for key in self.XtX_train.keys():
                    key1, key2 = key.split('@')
                    self.XtX_train[key] *= self.normalization[self.param['data_cat'][key1]]*self.normalization[self.param['data_cat'][key2]]
                    if self.precompute_test:
                        self.XtX_test [key] *= self.normalization[self.param['data_cat'][key1]]*self.normalization[self.param['data_cat'][key2]]
                for key in self.XtY_train.keys():
                    self.XtY_train[key] *= self.normalization[self.param['data_cat'][key]]
                    if self.precompute_test:
                        self.XtY_test [key] *= self.normalization[self.param['data_cat'][key]]
            # Change coef accordingly
            for ii, (var,key) in enumerate(self.coef.keys()):
                if var not in {'bv', 'Cv', 'Cm'}:
                    self.coef[var,key] /= self.normalization[self.param['data_cat'][key]] 
            
            print()
            """
            Save
            """
            for obj, name, opt in [(self.coef,       'coef',       'dict_tup'), 
                                   (self.keys_upd,   'keys_upd',   'json'), 
                                   #(self.orig_masks, 'orig_masks', 'dict_np'), 
                                   ]:
                try:
                    primitives.tf_data_save(self.path_betas, 
                                            self.dikt['model_pred'], 
                                            self.param, 
                                            obj, 
                                            obj_name = name, 
                                            mod      = opt, 
                                            )
                except tuple_Errors as e:
                    print(e)
        if hasattr(self, 'fit_gp_train') and not self.active_gp:
            del self.fit_gp_train
        if hasattr(self, 'fit_gp_test' ) and not self.active_gp:
            del self.fit_gp_test
        if self.param.get('trunc_svd', 0) > 0:
            # Use a low-rank approximation of the coefficient matrix
            assert 0
            print(colored('Truncate coefficient matrix', 'green'))
            keys_coef = [(b, v) for (b,v) in self.coef.keys() if v in self.keys1sh]
            B         = np.concatenate([self.coef[k] for k in keys_coef])
            len_ind   = [0]
            for k in keys_coef:
                len_ind.append(len_ind[-1] + self.coef[k].shape[0])
            u, s, vt  = np.linalg.svd(B)
            B_trunc   = u[:,:self.param['trunc_svd']] @ np.diag(s[:self.param['trunc_svd']]) @ vt[:self.param['trunc_svd']]
            for i, k in enumerate(keys_coef):
                self.coef[k] = B_trunc[len_ind[i]:len_ind[i+1]]

    #profile
    def start_descent(self):
        if hasattr(self, 'X_test'):
            assert len(self.X_train) == len(self.X_test), (len(self.X_train), len(self.X_test))
        assert not self.param['tf_sparse_coef'], 'slower'
        print('Start Descent')
        self.converged = 0
        self.list_coor = []
        if self.bcd:
            self.eta = {}
        np.random.seed(0)
        import gc; gc.collect()
        """
        While loop used for the descent
        """
        while not self.converged:
            if EXTRA_CHECK:
                print('\n'+colored('New Iteration', 'green'))
                assert np.all(self.Y_train == self.Y_train0)
                for key in self.X_train:
                    if type(self.X_train[key]) == np.ndarray:
                        assert np.all(self.X_train[key] == self.X_train0[key])
                    elif type(self.X_train0[key]) == sp.sparse.csc_matrix:
                        assert np.all(self.X_train[key].data == self.X_train0[key].data)
            if self.iteration >= self.flag_print_progress:
                self.print_progress()
                self.flag_print_progress += self.epoch_print_progress
                prprog = 1
            if self.batch_cd:
                assert 0
                dikt_fit_grad = self.compute_grad(self.coef)
                if self.active_gp:
                    dikt_fit_gp_grad = self.compute_grad(self.coef, gp_matrix = self.gp_matrix, gp_pen = self.gp_pen)
                q_train            = None
                old_part_fit_train = None
            elif self.bcd:
                coor_upd  = self.pop_coordinates() # Select a covariate (or a block) to update
                mask_upd  = self.dikt_masks.get(coor_upd, slice(None))
                # Extra checks
                orig_mask = self.orig_masks.get(coor_upd, slice(None))
                orig_mask = np.arange(self.k) if type(orig_mask) == type(slice(None)) else orig_mask
                out_mask  = [k for k in orig_mask if k not in (mask_upd if type(mask_upd) == np.ndarray else np.arange(self.k))]
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
                          '{0:20.20}'.format(coor_upd[1]),
                          '-',
                          '{0:8.8}' .format(repr(coor_upd[2])),  
                          '-', 
                          '{0:25.25}'.format(repr(self.dikt_masks.get(coor_upd, 'no_mask'))),
                          end = '')
                if EXTRA_CHECK:
                    print(coor_upd)
                coef_old_masked = {}
                if coor_upd[0] == 'bu':
                    coef_old_masked[coor_upd] = self.coef[coor_upd[:2]][:,mask_upd]
                    coef_old_masked['Blr',coor_upd[1]] = self.coef['Blr',coor_upd[1]][:,mask_upd]
                elif coor_upd[0] in {'Cu', 'Cv'}:
                    coef_old_masked[coor_upd] = self.coef[coor_upd[:2]][:,:,mask_upd]
                elif coor_upd[0] in {'Cm'}:
                    coef_old_masked[coor_upd] = self.coef[coor_upd[:2]][:,mask_upd]
                else:
                    coef_old_masked[coor_upd] = self.coef[coor_upd[:2]][:,mask_upd]
                q_train, extra_part_train, dikt_fit_grad = self.compute_part_grad(coor_upd, mask_upd, data = 'train')
                if self.precompute_test and self.iteration >= self.flag_compute_test :
                    q_test, extra_part_test = self.compute_part_grad(coor_upd, mask_upd, data = 'test')                    
                    old_part_fit_test       = self.part_fit(coef_old_masked, coor_upd, q_test, extra_part_test)
                if self.active_gp:
                    q_gp_train, extra_part_gp_train, dikt_fit_gp_grad = self.compute_part_grad(coor_upd, mask_upd, data = 'train',  MMt = self.MMt, gp_pen = self.gp_pen)
                    if self.precompute_test and self.iteration >= self.flag_compute_test :
                        q_gp_test, extra_part_gp_test = self.compute_part_grad(coor_upd, mask_upd, data = 'test',  MMt = self.MMt, gp_pen = self.gp_pen)
                        old_part_fit_gp_test = self.part_fit(coef_old_masked, coor_upd, q_gp_test,  extra_part_gp_test)
                else:
                    dikt_fit_gp_grad     = {}
                    q_gp_train           = {}
                    extra_part_gp_train  = {}
                    if self.precompute_test and self.iteration >= self.flag_compute_test :
                        q_gp_test            = {}
                        old_part_fit_gp_test = 0
                if (self.param['tf_plot'] and self.param['any_plot']) or self.param['tf_early_stop_ind_test']:
                    old_part_fit_ind_train  = self.part_fit(coef_old_masked, coor_upd, q_train,    extra_part_train,    data = 'train', indi = True)
                if (self.param['tf_plot'] and self.param['any_plot']):
                    q_mean_train, extra_part_mean_train, _ = self.compute_part_grad(coor_upd, mask_upd, data = 'train',  MMt = self.M_mean, gp_pen = 1)
                    old_part_fit_mean_train = self.part_fit(coef_old_masked, coor_upd, q_mean_train, extra_part_mean_train, data = 'train') 
                    if self.precompute_test:
                        old_part_fit_ind_test  = self.part_fit(coef_old_masked, coor_upd, q_test,    extra_part_test,    data = 'test', indi = True)
                        q_mean_test, extra_part_mean_test = self.compute_part_grad(coor_upd, mask_upd, data = 'test',  MMt = self.M_mean, gp_pen = 1)
                        old_part_fit_mean_test = self.part_fit(coef_old_masked, coor_upd, q_mean_test, extra_part_mean_test, data = 'test') 
                old_part_ridge        = self.cur_ind_ridge.get(coor_upd, 0)
                old_part_fit_train    = self.part_fit(coef_old_masked, coor_upd, q_train,    extra_part_train,    data = 'train')
                old_part_fit_gp_train = self.part_fit(coef_old_masked, coor_upd, q_gp_train, extra_part_gp_train, data = 'train') if self.active_gp else 0
            dikt_ridge_grad = self.compute_grad_ridge(list(dikt_fit_grad.keys()), self.dikt_masks)
            if EXTRA_CHECK:
                for key in set(dikt_fit_grad.keys()).union(dikt_fit_gp_grad.keys()).union(dikt_ridge_grad.keys()):
                    if key in dikt_fit_gp_grad:
                        assert dikt_fit_grad[key].shape == dikt_fit_gp_grad[key].shape, (dikt_fit_grad[key].shape, dikt_fit_gp_grad[key].shape)
                    if key in dikt_ridge_grad:
                        assert dikt_fit_grad[key].shape == dikt_ridge_grad[key].shape, (dikt_fit_grad[key].shape, dikt_ridge_grad[key].shape)
            # Backtracking step
            tmp_fit_train, tmp_fit_gp_train, tmp_ind_ridge, fit_plus_gp_plus_ridge_tilde_tmp, \
                            coef_tmp, nb_inner_iter, eta, condition_ls = self.backtracking_ls(
                                                                                              dikt_fit_grad, 
                                                                                              dikt_fit_gp_grad, 
                                                                                              dikt_ridge_grad, 
                                                                                              old_fit_train         = self.cur_fit_train, 
                                                                                              old_fit_gp_train      = self.cur_fit_gp_train, 
                                                                                              old_part_fit_train    = old_part_fit_train, 
                                                                                              old_part_fit_gp_train = old_part_fit_gp_train, 
                                                                                              old_ridge             = self.cur_ridge, 
                                                                                              old_part_ridge        = old_part_ridge, 
                                                                                              quant_train           = q_train,
                                                                                              quant_gp_train        = q_gp_train,
                                                                                              cur_ind_reg           = self.cur_ind_reg,
                                                                                              mask_upd              = mask_upd,
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
            tmp_obj_train = tmp_fit_train + tmp_fit_gp_train + tmp_ridge + tmp_reg
            if EXTRA_CHECK:
                assert self.cur_fit_train + self.cur_fit_gp_train + self.cur_ridge + self.cur_reg >= fit_plus_gp_plus_ridge_tilde_tmp + new_part_reg
                if not( tmp_obj_train <= self.cur_obj_train + 1e-12  or (self.pen == 'tf' and np.abs(tmp_obj_train - self.cur_obj_train) <= 1e-6)):
                    print('\n\n',
                          self.pen, '\n',
                          np.abs(tmp_obj_train - self.cur_obj_train), '\n',
                          coor_upd, '\n',
                          '\n')
                    var, key, ind = coor_upd
                    pen, alpha    = self.get_pen_alpha(var, key)
                    other_reg     = np.sum([self.cur_ind_reg[key] for key in self.cur_ind_reg if key not in coef_tmp])
                    tmp_reg       = other_reg + new_part_reg 
                    assert tmp_fit_train + tmp_fit_gp_train + tmp_ridge + tmp_reg        <= self.cur_obj_train + 1e-12
                    assert tmp_fit_train + tmp_fit_gp_train + tmp_ridge + tmp_reg        <= fit_plus_gp_plus_ridge_tilde_tmp + tmp_reg+ 1e-12
                    assert fit_plus_gp_plus_ridge_tilde_tmp + tmp_reg <= self.cur_obj_train+ 1e-12                
                    

            if self.bcd:
                (s,key,ind) = coor_upd
                if self.cur_obj_train - tmp_obj_train <= 1e-14:
                    if self.active_set and self.param['tf_col_upd'].get(key): 
                        self.punished_coor.add(coor_upd)   
                else: # eta won't be created/updated if decrease too small
                    if s in {'A'}:
                        self.etaA = eta/(self.theta**np.random.binomial(1, self.proba_ls)) 
                    elif s in {'Bsp', 'bu', 'Cb'}:
                        self.eta1 = eta/(self.theta**np.random.binomial(1, self.proba_ls)) 
                    elif s in {'U', 'V', 'Csp', 'Cu', 'Cv', 'Cm'}:
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
            self.fit_train   [slice_inner_iter] = self.fit_train   [self.iteration - 1]
            if (self.param['tf_plot'] and self.param['any_plot']) or self.param['tf_early_stop_ind_test']:
                self.fit_ind_train[slice_inner_iter] = self.fit_ind_train[self.iteration - 1]
                self.fit_mean_train[slice_inner_iter] = self.fit_mean_train[self.iteration - 1]
            if self.active_gp:
                self.fit_gp_train[slice_inner_iter] = self.fit_gp_train[self.iteration - 1]
            self.reg      [slice_inner_iter] = self.reg      [self.iteration - 1]
            self.ridge    [slice_inner_iter] = self.ridge    [self.iteration - 1]
            self.obj_train[slice_inner_iter] = self.obj_train[self.iteration - 1]
            if self.compute_test:# and False: 
                self.fit_test   [slice_inner_iter] = self.fit_test[self.iteration - 1]
                if (self.param['tf_plot'] and self.param['any_plot']) or self.param['tf_early_stop_ind_test']:
                    self.fit_ind_test[slice_inner_iter]  = self.fit_ind_test[self.iteration - 1]
                    self.fit_mean_test[slice_inner_iter] = self.fit_mean_test[self.iteration - 1]
                if self.active_gp:
                    self.fit_gp_test[slice_inner_iter] = self.fit_gp_test[self.iteration - 1]
                self.obj_test[slice_inner_iter] = self.obj_test[self.iteration - 1]

            if EXTRA_CHECK:
                temp__, _, _ = self.evaluate_ind_reg({(var,key,ind):(self.coef[var,key][:,:,mask_upd] if var in {'Cu','Cv'} else self.coef[var,key][:,mask_upd]) for (var,key,ind) in self.keys_upd})
            # Update current variables
            self.iteration    += nb_inner_iter - 1 
            max_iter_reached   = (self.iteration >= self.max_iter)
            if not (self.bcd):
                self.decr_obj = self.cur_obj_train - tmp_obj_train
            self.cur_fit_train    = tmp_fit_train
            self.cur_fit_gp_train = tmp_fit_gp_train
            self.cur_ridge        = tmp_ridge                       
            self.cur_reg          = tmp_reg                        
            self.cur_obj_train    = tmp_obj_train              
            self.update_coef(coef_tmp, self.dikt_masks, new_ind_slope = tmp_ind_slope, new_ind_offset = tmp_ind_offset)
            self.cur_ind_reg.update(tmp_ind_reg)
            self.cur_ind_ridge.update(tmp_ind_ridge)
            
            if EXTRA_CHECK:
                for coor in self.coef:
                    assert coor == coor[:2]
                tmp2_ind_reg, _, _ = self.evaluate_ind_reg({(*coor,()):self.coef[coor[:2]] for coor in self.coef})
                tmp2_reg           = np.sum([v for k, v in tmp2_ind_reg.items()])
                mmmm, _, _          = self.evaluate_ind_reg({(var,key,ind):(self.coef[var,key][:,:,mask_upd] if var in {'Cu','Cv'} else self.coef[var,key][:,mask_upd]) for (var,key,ind) in self.keys_upd})
                if not np.abs(self.cur_reg - tmp2_reg) <= 1e-12:
                    out_mask = [k for k in range(self.k) if type(mask_upd)!= type(slice(None)) and k not in mask_upd]
                    out_coef = self.coef[coor_upd[:2]][:,:,out_mask] if coor_upd[0] in {'Cu','Cv '} else self.coef[coor_upd[:2]][:,out_mask]
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
                assert self.cur_obj_train <= self.obj_train[self.iteration - 1] + 1e-12  or (self.pen == 'tf' and np.abs(tmp_obj_train - self.cur_obj_train) <= 1e-8), (coor_upd, self.cur_obj_train, self.obj_train[self.iteration - 1])
                if self.bcd:
                    (s,key,ind) = coor_upd
                    if s in {'A'}:
                        self.etaAs[self.iteration] = self.etaA
                        self.eta1s[self.iteration] = self.eta1s[self.iteration - 1]
                        self.eta2s[self.iteration] = self.eta2s[self.iteration - 1]
                    elif s in {'Bsp', 'bu', 'Cb'}:
                        self.etaAs[self.iteration] = self.etaAs[self.iteration - 1]
                        self.eta1s[self.iteration] = self.eta1
                        self.eta2s[self.iteration] = self.eta2s[self.iteration - 1]
                    elif s in {'U', 'V', 'Csp', 'Cu', 'Cv', 'Cm'}:
                        self.etaAs[self.iteration] = self.etaAs[self.iteration - 1]
                        self.eta1s[self.iteration] = self.eta1s[self.iteration - 1]
                        self.eta2s[self.iteration] = self.eta2
                else:                
                    self.etas     [self.iteration] = self.eta
                self.fit_train   [self.iteration] = self.cur_fit_train
                self.fit_gp_train[self.iteration] = self.cur_fit_gp_train
                self.reg         [self.iteration] = self.cur_reg      
                self.ridge       [self.iteration] = self.cur_ridge    
                self.obj_train   [self.iteration] = self.cur_obj_train
                if (self.param['tf_plot'] and self.param['any_plot']) or self.param['tf_early_stop_ind_test']:
                    if self.batch_cd:# or self.vr:
                        assert 0, 'not implemented'
                    else:
                        if coef_tmp:
                            new_part_fit_ind_train = self.evaluate_fit_bcd(coor_upd, coef_tmp, q_train, data = 'train', mask = mask_upd, indi = True)
                            self.fit_ind_train[self.iteration]  = self.fit_ind_train[self.iteration-1]
                            self.fit_ind_train[self.iteration][mask_upd] = self.fit_ind_train[self.iteration][mask_upd] - old_part_fit_ind_train + new_part_fit_ind_train
                            assert (self.fit_ind_train[self.iteration] > 0).all() 
                            if (self.param['tf_plot'] and self.param['any_plot']):
                                new_part_fit_mean_train             = self.evaluate_fit_bcd(coor_upd, coef_tmp, q_mean_train, data = 'train', mask = mask_upd, MMt = self.M_mean, gp_pen = 1)
                                self.fit_mean_train[self.iteration] = self.fit_mean_train[self.iteration-1] - old_part_fit_mean_train + new_part_fit_mean_train
                        else:
                            self.fit_ind_train[self.iteration]  = self.fit_ind_train [self.iteration-1]
                            if (self.param['tf_plot'] and self.param['any_plot']):
                                self.fit_mean_train[self.iteration] = self.fit_mean_train[self.iteration-1]
                if self.compute_test and self.iteration >= self.flag_compute_test: 
                    if self.batch_cd:# or self.vr:
                        self.fit_test   [self.iteration] = self.evaluate_fit(self.coef, data = 'test')
                        if self.active_gp:
                            self.fit_gp_test[self.iteration] = self.evaluate_fit(self.coef, data = 'test', gp_matrix = self.gp_matrix, gp_pen = self.gp_pen)
                    elif self.bcd:
                        if self.precompute_test:
                            if coef_tmp:
                                assert self.obj_train[self.iteration] - self.obj_train[self.iteration-1] <= 1e-14
                                new_part_fit_test             = self.evaluate_fit_bcd(coor_upd, coef_tmp, q_test, data = 'test', mask = mask_upd)
                                self.cur_fit_test             = self.fit_test[self.iteration-1] - old_part_fit_test + new_part_fit_test
                                self.fit_test[self.iteration] = self.cur_fit_test    
                                if self.active_gp:
                                    new_part_fit_gp_test             = self.evaluate_fit_bcd(coor_upd, coef_tmp, q_gp_test, data = 'test', mask = mask_upd, MMt = self.MMt, gp_pen = self.gp_pen)
                                    self.cur_fit_gp_test             = self.fit_gp_test[self.iteration-1] - old_part_fit_gp_test + new_part_fit_gp_test
                                    self.fit_gp_test[self.iteration] = self.cur_fit_gp_test    
                                if (self.param['tf_plot'] and self.param['any_plot']) or self.param['tf_early_stop_ind_test']:
                                    new_part_fit_ind_test = self.evaluate_fit_bcd(coor_upd, coef_tmp, q_test, data = 'test', mask = mask_upd, indi = True)
                                    self.fit_ind_test[self.iteration]  = self.fit_ind_test[self.iteration-1]
                                    self.fit_ind_test[self.iteration][mask_upd] = self.fit_ind_test[self.iteration][mask_upd] - old_part_fit_ind_test + new_part_fit_ind_test
                                    assert (self.fit_ind_test[self.iteration] > 0).all() 
                                    new_part_fit_mean_test             = self.evaluate_fit_bcd(coor_upd, coef_tmp, q_mean_test, data = 'test', mask = mask_upd, MMt = self.M_mean, gp_pen = 1)
                                    self.fit_mean_test[self.iteration] = self.fit_mean_test[self.iteration-1] - old_part_fit_mean_test + new_part_fit_mean_test
                            else:
                                self.fit_test[self.iteration] = self.fit_test[self.iteration-1]
                                if self.active_gp:
                                    self.fit_gp_test[self.iteration] = self.fit_gp_test[self.iteration-1]  
                                if (self.param['tf_plot'] and self.param['any_plot']) or self.param['tf_early_stop_ind_test']:
                                    self.fit_ind_test [self.iteration] = self.fit_ind_test[self.iteration-1]
                                    self.fit_mean_test[self.iteration] = self.fit_mean_test[self.iteration-1]
                    self.obj_test[self.iteration] = self.fit_test[self.iteration] + self.fit_gp_test[self.iteration] + self.reg[self.iteration] + self.ridge[self.iteration]
                if EXTRA_CHECK:
                    if self.iteration >= self.flag_check_fit:
                        self.flag_check_fit *= 10
                        print('\nCheck fit')
                        assert np.abs(self.evaluate_fit(self.coef, data = 'train') - self.cur_fit_train) <= 1e-12, ('pb_ch1', self.evaluate_fit(self.coef, data = 'train'), self.cur_fit_train)
                        assert np.abs(self.evaluate_fit(self.coef, data = 'test')  - self.cur_fit_test)  <= 1e-12, ('pb_ch2', self.evaluate_fit(self.coef, data = 'test'),  self.cur_fit_test)
                    if self.iteration >= self.flag_check_fit and self.active_gp:
                        print('\nCheck fit gp')
                        assert np.abs(self.evaluate_fit(self.coef, data = 'train', gp_matrix = self.gp_matrix, gp_pen = self.gp_pen) - self.cur_fit_gp_train) <= 1e-12, ('pb_ch3', self.evaluate_fit(self.coef, data = 'train', gp_matrix = self.gp_matrix, gp_pen = self.gp_pen), self.cur_fit_gp_train)
                        assert np.abs(self.evaluate_fit(self.coef, data = 'test',  gp_matrix = self.gp_matrix, gp_pen = self.gp_pen) - self.cur_fit_gp_test ) <= 1e-12, ('pb_ch4', self.evaluate_fit(self.coef, data = 'test',  gp_matrix = self.gp_matrix, gp_pen = self.gp_pen), self.cur_fit_gp_test)
                    if self.iteration >= self.flag_check_fit and ((self.param['tf_plot'] and self.param['any_plot']) or self.param['tf_early_stop_ind_test']):  
                        print('\nCheck fit ind')
                        assert np.allclose(self.evaluate_ind_fit(self.coef, data = 'train'),   self.fit_ind_train[self.iteration]), ('pb_ch32', self.evaluate_ind_fit(self.coef, data = 'train'), self.fit_ind_train[self.iteration])
                        assert np.allclose(self.evaluate_ind_fit(self.coef, data = 'train').sum(), self.fit_train[self.iteration]),'incoherence between ind_fit and fit train'
                        assert np.allclose(self.evaluate_ind_fit(self.coef, data = 'test' ),   self.fit_ind_test [self.iteration]), ('pb_ch42', self.evaluate_ind_fit(self.coef, data = 'test'),  self.fit_ind_test [self.iteration]) 
                        assert np.allclose(self.evaluate_ind_fit(self.coef, data = 'test' ).sum(), self.fit_test [self.iteration]),'incoherence between ind_fit and fit test'
                    if self.iteration >= self.flag_check_fit and ((self.param['tf_plot'] and self.param['any_plot']) or self.param['tf_early_stop_ind_test']):  
                        print('\nCheck fit mean')
                        assert np.abs(self.evaluate_fit(self.coef, data = 'train', gp_matrix = self.M_mean, gp_pen = 1) - self.fit_mean_train[self.iteration]) <= 1e-12, ('pb_ch32', self.evaluate_fit(self.coef, data = 'train', gp_matrix = self.M_mean, gp_pen = 1), self.fit_mean_train[self.iteration])
                        assert np.abs(self.evaluate_fit(self.coef, data = 'test',  gp_matrix = self.M_mean, gp_pen = 1) - self.fit_mean_test [self.iteration]) <= 1e-12, ('pb_ch42', self.evaluate_fit(self.coef, data = 'test',  gp_matrix = self.M_mean, gp_pen = 1), self.fit_mean_test [self.iteration])  
            if self.nb_steps >= self.flag_print_info:
                self.print_info()
                self.flag_print_info += self.epoch_print_info
                
            small_decrease, early_stop, norm_grad_small, all_dead = self.stopping_criteria({key:dikt_fit_grad[key] + dikt_fit_gp_grad.get(key, 0) 
                                                                                            for key in set(dikt_fit_grad.keys()).union(dikt_fit_gp_grad.keys())
                                                                                            })
            
            assert not np.isnan(self.cur_obj_train)
            
            # To next iteration
            self.iteration  += 1
            max_iter_reached = (self.iteration >= self.max_iter)
            self.converged   = max(small_decrease, early_stop, norm_grad_small, max_iter_reached, all_dead)                                



        
        
        

###############################################################################
###############################################################################
###############################################################################

    
    #profile
    def backtracking_ls(self, 
                        grad,                          # Gradient of the data-fitting term
                        grad_gp,                       # Gradient of the sum-consistent-fitting term 
                        ridge_grad,                    # Gradient of the ridge term
                        old_fit_train         = None, 
                        old_fit_gp_train      = None, 
                        old_part_fit_train    = None, 
                        old_part_fit_gp_train = None, 
                        old_ridge             = None, 
                        old_part_ridge        = None, 
                        quant_train           = None, 
                        quant_gp_train        = None, 
                        cur_ind_reg           = None, 
                        mask_upd              = None,
                        ):
        # Backtracking line-search, given a gradient descent direction
        condition_ls  = 0
        nb_inner_iter = 0
        converged     = 0
        if self.bcd:
            coor_upd = next(iter(grad.keys()))
            (s,key,ind) = coor_upd
            if s in {'A'}:
                eta = self.etaA
            elif s in {'Bsp', 'bu', 'Cb'}:
                eta = self.eta1
            elif s in {'U', 'V', 'Csp', 'Cu', 'Cv', 'Cm'}:
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
                    return self.cur_fit_train, self.cur_fit_gp_train, {}, self.cur_fit_train + self.cur_ridge, {}, nb_inner_iter, eta/self.theta, False
            coef_tmp    = self.foba_step(grad, grad_gp, ridge_grad, eta, self.dikt_masks)
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
                                                                      old_fit_train + old_fit_gp_train + old_ridge, 
                                                                      self.dikt_masks,
                                                                      )
            if EXTRA_CHECK:
                fit_plus_gp_plus_ridge_tilde_old = self.compute_surrogate(grad, 
                                                                          grad_gp, 
                                                                          ridge_grad, 
                                                                          {(var,key,ind):(self.coef[var,key][:,:,mask_upd] if var in {'Cu','Cv'} else self.coef[var,key][:,mask_upd]) for (var,key,ind) in coef_tmp}, 
                                                                          eta, 
                                                                          old_fit_train + old_fit_gp_train + old_ridge, 
                                                                          self.dikt_masks,
                                                                          )
                assert fit_plus_gp_plus_ridge_tilde_old == self.cur_fit_train + self.cur_ridge + self.cur_fit_gp_train
                coef_old = {}
                for coor_upd in grad.keys():
                    var, key, ind      = coor_upd
                    coef_old[coor_upd] = (self.coef[var,key][:,:,mask_upd] if var in {'Cu','Cv'} else self.coef[var,key][:,mask_upd])
                ind_old_regbis, _, _ = self.evaluate_ind_reg(coef_old)
                tmp_ind_reg,    _, _ = self.evaluate_ind_reg(coef_tmp)
                new_part_reg = np.sum([tmp_ind_reg[key] for key in tmp_ind_reg])
                other_reg    = np.sum([cur_ind_reg[key] for key in cur_ind_reg if key not in tmp_ind_reg])
                old_part_reg = np.sum([cur_ind_reg[key] for key in tmp_ind_reg])
                
                assert np.abs(fit_plus_gp_plus_ridge_tilde_old + other_reg + old_part_reg - self.cur_obj_train  ) <= 1e-12
                assert np.abs(self.cur_fit_train + self.cur_fit_gp_train + self.cur_reg + self.cur_ridge - self.cur_obj_train ) <= 1e-12
                
                if not fit_plus_gp_plus_ridge_tilde_tmp + other_reg + new_part_reg <=  fit_plus_gp_plus_ridge_tilde_old + other_reg + old_part_reg + 1e-12:
                    raise ValueError('Objective has increased')
            
            # # Update low-rank
            if 'Blr' in self.config_coef and 'bu' in [e[0] for e in grad.keys()] :
                coef_tmp  = self.update_VB(coef_tmp)
            if 'Cuv' in self.config_coef:
                coef_tmp  = self.update_Cuv(coef_tmp)
            if 'Cbm' in self.config_coef:
                coef_tmp  = self.update_Cbm(coef_tmp)
            if self.batch_cd:
                fit_tmp   = self.evaluate_fit(coef_tmp) #+ self.evaluate_ridge(coef_tmp)
                assert 0, 'MUST ADD fit_gp_tmp'
                ridge_tmp = self.evaluate_ridge(coef_tmp) 
            elif self.bcd:
                assert len(coef_tmp) <= 3 or coor_upd[0] == 'Cb'
                new_part_fit_train = self.evaluate_fit_bcd(coor_upd, coef_tmp, quant_train, data = 'train', mask = mask_upd)
                if self.active_gp:
                    new_part_fit_gp_train = self.evaluate_fit_bcd(coor_upd, coef_tmp, quant_gp_train, data = 'train', mask = mask_upd, MMt = self.MMt, gp_pen = self.gp_pen)
                if EXTRA_CHECK:
                    check_old_part_fit_train = self.evaluate_fit_bcd(coor_upd, {(var,key,ind):(self.coef[var,key][:,:,mask_upd] if var in {'Cu','Cv'} else self.coef[var,key][:,mask_upd]) for (var,key,ind) in coef_tmp}, quant_train, data = 'train', mask = mask_upd)
                    assert np.abs(check_old_part_fit_train - old_part_fit_train) < 1e-12, ('pb5', check_old_part_fit_train, old_part_fit_train)
                    if self.active_gp:
                        check_old_part_fit_gp_train = self.evaluate_fit_bcd(coor_upd, {(var,key,ind):(self.coef[var,key][:,:,mask_upd] if var in {'Cu','Cv'} else self.coef[var,key][:,mask_upd]) for (var,key,ind) in coef_tmp}, quant_gp_train, data = 'train', mask = mask_upd, MMt = self.MMt, gp_pen = self.gp_pen)
                        assert np.abs(check_old_part_fit_gp_train - old_part_fit_gp_train) < 1e-12, ('pb6', check_old_part_fit_gp_train, old_part_fit_gp_train)
                new_ind_ridge         = self.evaluate_ind_ridge(coef_tmp) 
                new_part_ridge        = np.sum([v for k, v in new_ind_ridge.items()])
                ridge_tmp             = old_ridge        - old_part_ridge        + new_part_ridge
                fit_tmp               = old_fit_train    - old_part_fit_train    + new_part_fit_train
                fit_gp_tmp            =(old_fit_gp_train - old_part_fit_gp_train + new_part_fit_gp_train) if self.active_gp else 0
                if EXTRA_CHECK:
                    new_coef = cp.deepcopy(self.coef)
                    for coor in coef_tmp:
                        if coor[0] in {'Cu','Cv'}:
                            new_coef[coor[:2]][:,:,mask_upd] = coef_tmp[coor]
                        else:
                            new_coef[coor[:2]][:,  mask_upd] = coef_tmp[coor]
                    """
                    FIT CONTROL
                    """
                    assert old_fit_train == self.cur_fit_train
                    fit_ctrl = self.evaluate_fit(new_coef, data = 'train') 
                    assert np.abs(fit_ctrl - fit_tmp) < 1e-12
                    if self.active_gp:
                        assert old_fit_gp_train == self.cur_fit_gp_train
                        fit_gp_ctrl = self.evaluate_fit(new_coef, data = 'train', gp_matrix = self.gp_matrix, gp_pen = self.gp_pen) if self.active_gp else 0
                        assert np.abs(fit_gp_ctrl - fit_gp_tmp) < 1e-12
                assert new_part_fit_train    != np.nan 
                if self.active_gp:
                    assert new_part_fit_gp_train != np.nan 
            if fit_tmp < 0 or fit_gp_tmp < 0:
                # Detection of an error
                len_str = 25
                print('\n', )
                for e in [
                          'self.bcd',
                          'self.cur_fit_train', 
                          'fit_tmp', 
                          'fit_gp_tmp', 
                          'ridge_tmp',
                          'old_part_fit_train',
                          'old_ridge',
                          'new_part_fit_train',
                          'eta',
                          'nb_inner_iter',
                          'set([a for a, b in self.coef.keys()])',
                          'sorted(self.keys["Cbm"])',
                          'sorted(self.keys["Cb"])',
                          'sorted(self.keys["Csp"])',
                          'sorted(coef_tmp.keys())', 
                          'sorted(coef_tmp.keys())',
                          'self.iteration',
                          ]:
                    print('{0:{width}} : {1}'.format(e, eval(e), width = len_str)) 
                print('{0:{width}} : {1}'.format('shape' , [(k, v.shape) for k, v in coef_tmp.items()], width = len_str))
                assert 0
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
            fit_tmp    = self.cur_fit_train 
            fit_gp_tmp = self.cur_fit_gp_train 
            new_ind_ridge                    = {}
            fit_plus_gp_plus_ridge_tilde_tmp = fit_tmp + fit_gp_tmp
            print(colored('    converged but not condition_ls    ', 'red', 'on_cyan'))
        if EXTRA_CHECK:
            tmp_reg      = other_reg + new_part_reg
            if not fit_plus_gp_plus_ridge_tilde_tmp + tmp_reg <= self.cur_obj_train + 1e-12:
                assert 0
            if not(fit_tmp + fit_gp_tmp + ridge_tmp + tmp_reg      <= self.cur_obj_train + 1e-12):
                assert 0
            if not(fit_tmp + fit_gp_tmp + ridge_tmp + tmp_reg      <= fit_plus_gp_plus_ridge_tilde_tmp + tmp_reg+ 1e-12):
                assert 0
        return fit_tmp, fit_gp_tmp, new_ind_ridge, fit_plus_gp_plus_ridge_tilde_tmp, coef_tmp, nb_inner_iter, eta, condition_ls


    #profile
    def best_orth_bv(self, coef, mask = None):
        # Update the orthogonal matrix V in the low-rank optimization problem
        coor_upd      = list(coef.keys())[0]
        var, key, ind = coor_upd
        assert not self.active_gp, 'optimization wrt V is different'
        assert not ind, 'no partial/column update for low-rank coefficients'
        assert mask == slice(None) # No mask when Bv
        XtY = self.XtY_train
        M = (  ((1 - self.coef['A','']) if self.factor_A else 1)*XtY[key][:,mask]   
             - np.sum([self.custom_sum_einsum(self.XtX_train, 
                                              self.coef, 
                                              var2, 
                                              key, 
                                              slice(None), 
                                              var_temp = self.dikt_var_temp.get(var,var),
                                              ) 
                        for var2 in self.config_coef], 
                       axis = 0)
             )[:,mask].T @ coef.get(('bu', key, ()), coef.get(('bu',key), 'error'))
        A, sig, Bt = np.linalg.svd(M, full_matrices=0)
        v = A @ Bt
        return v


    def change_sparse_structure(self, ):
        # Sparsify the design matrices
        print('change structure X sparse')
        for ii, X in enumerate([self.X_train] + ([self.X_test] if self.compute_test else [])):
            for jj, (k, v) in enumerate(X.items()):
                print('\r{0} / {1} - {2:5} / {3:5}'.format(ii, 1+self.compute_test, jj, len(X)), end = '')
                if type(v) == sp.sparse.csc_matrix:
                    X[k] = sp.sparse.csr_matrix(v)
        print('done' + ' '*20)

    def compute_grad_ridge(self, list_coor, d_masks):
        # Computet the gradient of the differentiable regularization (ridge or smoothing-spline)
        dikt_ridge_grad = {}
        for coor_upd in list_coor:
            assert len(coor_upd) == 3
            var, key, ind = coor_upd
            pen, alpha = self.get_pen_alpha(var, key)
            mask = d_masks.get(coor_upd, slice(None))
            if pen == 'rsm':
                if coor_upd[0] in {'Cu', 'Cv'}:
                    dikt_ridge_grad[coor_upd] = alpha*self.coef[var,key][:,:,mask] if (pen == 'rsm') else np.zeros(self.coef[var,key][:,:,mask].shape)
                else:
                    dikt_ridge_grad[coor_upd] = alpha*self.coef[var,key][:,  mask] if (pen == 'rsm') else np.zeros(self.coef[var,key][:,mask].shape)
            elif pen == 'r2sm':
                cat = self.param['data_cat'][key]
                if coor_upd[0] in {'Cu', 'Cv'}:
                    raise NotImplementedError # coef has then 3 dimensions
                M   = self.coef[var,key][:,mask]
                reshape_tensor = '#' in cat and var not in {'Cu', 'Cv', 'Cb', 'Cm'}
                if not reshape_tensor:
                    cc = M
                    dikt_ridge_grad[coor_upd] = np.zeros(cc.shape)
                    if cc.shape[0] > 2:
                        if self.param['qo_modulo'].get(cat):
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
                    cc = M.reshape(*self.size_tensor2[key], -1)
                    dikt_ridge_grad[coor_upd] = np.zeros(cc.shape)
                    cat1, cat2 = cat.split('#')
                    if cc.shape[0] > 2:
                        if self.param['qo_modulo'].get(cat1):
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
                        if self.param['qo_modulo'].get(cat2):
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
            elif pen == '0r2sm':
                cat = self.param['data_cat'][key]
                if coor_upd[0] in {'Cu', 'Cv'}:
                    raise NotImplementedError # coef has then 3 dimensions
                M   = self.coef[var,key][:,mask]
                reshape_tensor = '#' in cat and var not in {'Cu', 'Cv', 'Cb', 'Cm'}
                if not reshape_tensor:
                    cc = M
                    dikt_ridge_grad[coor_upd] = np.zeros(cc.shape)
                    if cc.shape[0] > 2:
                        if self.param['qo_modulo'].get(cat):
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
                    cc = M.reshape(*self.size_tensor2[key], -1)
                    dikt_ridge_grad[coor_upd] = np.zeros(cc.shape)
                    cat1, cat2 = cat.split('#')
                    if cc.shape[0] > 2:
                        if self.param['qo_modulo'].get(cat1):
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
                assert pen not in {'rsm', 'r2sm'}
                if coor_upd[0] in {'Cu', 'Cv'}:
                    dikt_ridge_grad[coor_upd] = np.zeros(self.coef[var,key][:,:,mask].shape)
                else:
                    dikt_ridge_grad[coor_upd] = np.zeros(self.coef[var,key][:,mask].shape)
        return dikt_ridge_grad    


    #profile
    def compute_part_grad(self, coor_upd, mask, data = None, MMt = None, gp_pen = 0):
        # Compute the gradient with respect to the block selected for the BCD algorithm
        # The computation is decomposed in different parts to avoid redundant computations
        if gp_pen:
            assert type(MMt) != type(None)
            mask_out   = np.array([k for k in     (self.orig_masks[coor_upd] 
                                                   if type(self.orig_masks.get(coor_upd)) == np.ndarray and not self.col_upd.get(coor_upd[1],False) 
                                                   else range(self.k))
                                     if  k not in (mask  if type(mask)  == np.ndarray else range(self.k))
                                   ]).astype(int)
            mask_Y = slice(None)
        else:
            mask_Y = mask
        assert len(coor_upd) == 3
        var, key, ind = coor_upd
        if var == 'Blr':
            assert 0 # For now
            assert not ind
        extra_part = {}
        quant      = {}
        extra_part = {}
        #q_test           = {}
        if data == 'train':
            grad = {}
            n   = self.n_train
            X   = self.X_train
            Y   = self.Y_train
            XtX = self.XtX_train
            XtY = self.XtY_train
        else:
            assert data == 'test'
            n   = self.n_test
            X   = self.X_test
            Y   = self.Y_test
            XtX = self.XtX_test
            XtY = self.XtY_test
        
        if var in {'Bsp','Csp','bu','Cb', 'Cm', 'Cu', 'Cv'}:
            
            ############# Compute the normal part of quant[coor_upd]
            
            mmm = (1/n)*(\
                         - ((1 - self.coef['A','']) if self.factor_A else 1)*(XtY[key][:,mask_Y] if type(XtY[key]) == np.ndarray else XtY[key][:,mask_Y].toarray())
                         + np.sum([self.custom_sum_einsum(XtX, 
                                                          self.coef, 
                                                          var2, 
                                                          key, 
                                                          'dyn' if gp_pen else mask, 
                                                          var_temp = self.dikt_var_temp.get(var, var),
                                                          ) 
                                    for var2 in self.config_coef], 
                                   axis = 0)
                                  )

            if gp_pen:
                st = time.time()
                mmm1  = gp_pen * mmm @ MMt[:,mask]
                bb = time.time()
                if 0:
                    #print('This computation can be acccelerated since MMt is just a sum and repeat')
                    width = len(mask) if type(mask) == np.ndarray else self.k
                    aa = time.time()
                    mmm2 = np.repeat((gp_pen/self.k)*mmm.sum(axis = 1)[:,np.newaxis], width, axis = 1)
                    c = time.time()
                    assert np.allclose(mmm1, mmm2)
                    print('Should simplify computations : ', np.round(bb - st, 8), 's', ' VS', np.round(c - aa, 8), 's', ' - ratio : ', np.round((bb - st)/(c - aa), 8), ' - width : ', width)
                mmm = mmm1
                if len(mask_out):
                    mmm+= gp_pen * (1/n) * XtX[key+'@'+key] @ self.coef[self.dikt_var_temp.get(var,var),key][:,mask_out] @ MMt[mask_out][:,mask]
            
            ############# Check the computation of the normal part of quant[coor_upd]
            if EXTRA_CHECK:
            ############# The computation is different for Cb so the check is different too

                aaa =   Y[:,mask_Y].copy()
                for var2 in self.config_coef:   
                    for key2 in self.keys[var2]: 
                        if key+'@'+key2 in XtX\
                        and (   (var != 'Cb' and (key2 != key or var2 != self.dikt_var_temp.get(var, var)))
                             or (var == 'Cb' and (key2.split('#')[0] != key or var2 not in ('Cb', 'Cbm') ))
                             ):
                            ss         = self.orig_masks.get((var2,key2,()), slice(None)) if gp_pen else mask
                            aaa[:,ss if gp_pen else slice(None)] += - X[key2] @ self.coef[var2,key2][:,ss] 
                mmm_check = -(1/n) * X[key].T @ aaa
                if gp_pen:
                    mmm_check = gp_pen * mmm_check @ MMt[:,mask]
                    if len(mask_out):
                        mmm_check+= gp_pen * (1/n) * XtX[key+'@'+key] @ self.coef[self.dikt_var_temp.get(var,var),key][:,mask_out] @ MMt[mask_out][:,mask]
                assert mmm.shape == mmm_check.shape
                assert np.allclose(mmm, mmm_check)

            ############# Multiply accordingly when the updated variable intervenes as a product in the prediction BM, UV

            if var == 'bu':
                quant[coor_upd] = mmm # not multiplied by V since it will also be updated
            elif var == 'Cb':
                quant[coor_upd] = mmm # Computation of quant[coor_upd] for Cb is not over
            elif var == 'Cm':
                quant[coor_upd] = np.einsum('pqk,pk->qk', 
                                            mmm.reshape(( 
                                                         self.coef['Cb', key.split('#')[0]][:,mask].shape[0], 
                                                         -1,
                                                         self.coef['Cb', key.split('#')[0]][:,mask].shape[1],
                                                         )), 
                                            self.coef['Cb', key.split('#')[0]][:,mask],
                                            )
                if EXTRA_CHECK:
                    mmm_check = np.einsum('pqk,pk->qk', 
                                          mmm_check.reshape((
                                                             self.coef['Cb', key.split('#')[0]][:,mask].shape[0],
                                                             -1,
                                                             self.coef['Cb', key.split('#')[0]][:,mask].shape[1],
                                                             )), 
                                          self.coef['Cb', key.split('#')[0]][:,mask],
                                          )
                    
            elif var == 'Cu':
                quant[coor_upd] = np.einsum('pqk,qrk->prk', 
                                            mmm.reshape((-1,
                                                         self.coef['Cv', key][:,:,mask].shape[0],
                                                         self.coef['Cv', key][:,:,mask].shape[2],
                                                         )),
                                            self.coef['Cv', key][:,:,mask],
                                            )
                if EXTRA_CHECK:
                    mmm_check = np.einsum('pqk,qrk->prk', 
                                          mmm_check.reshape((-1, 
                                                             self.coef['Cv', key][:,:,mask].shape[0], 
                                                             self.coef['Cv', key][:,:,mask].shape[2],
                                                             )), 
                                          self.coef['Cv', key][:,:,mask],
                                          )
            elif var == 'Cv':
                quant[coor_upd] = np.einsum('pqk,prk->qrk', 
                                            mmm.reshape((self.coef['Cu', key][:,:,mask].shape[0],
                                                         -1,
                                                         self.coef['Cu', key][:,:,mask].shape[2],
                                                         )),
                                            self.coef['Cu', key][:,:,mask],
                                            )
                if EXTRA_CHECK:
                    mmm_check = np.einsum('pqk,prk->qrk', 
                                          mmm_check.reshape((self.coef['Cu', key][:,:,mask].shape[0], 
                                                             -1, 
                                                             self.coef['Cu', key][:,:,mask].shape[2],
                                                             )), 
                                          self.coef['Cu', key][:,:,mask],
                                          )
            else:
                assert var in ['Bsp', 'Csp', ]
                quant[coor_upd] = mmm
            
            ############# Add the part specific to Cb where Cb intervenes in Cbm
            
            if var == 'Cb': 
                for keybm in self.keys['Cbm']:
                    bbb = {}
                    bbb_check = {}
                    assert '#' in keybm
                    if keybm.split('#')[0] == key:
                        bbb[keybm] = (1/n)*(- ((1 - self.coef['A','']) if self.factor_A else 1)*(XtY[keybm][:,mask_Y] if type(XtY[key]) == np.ndarray else XtY[keybm][:,mask_Y].toarray())
                                            + np.sum([self.custom_sum_einsum(XtX, 
                                                                             self.coef, 
                                                                             var2, 
                                                                             keybm, 
                                                                             'dyn' if gp_pen else mask, 
                                                                             var_temp = 'Cb',
                                                                             ) 
                                                       for var2 in self.config_coef], 
                                                      axis = 0,
                                                     )
                                            )
                        if gp_pen:
                            bbb[keybm] = gp_pen * bbb[keybm] @ MMt[:,mask]
                            bbb[keybm]+= gp_pen * (1/n) * XtX[keybm+'@'+keybm] @ self.coef['Cbm',keybm][:,mask_out] @ MMt[mask_out][:,mask]

                        quant[coor_upd] += np.einsum('pqk,qk->pk', 
                                                     bbb[keybm].reshape((-1, 
                                                                         self.coef['Cm', keybm][:,mask].shape[0], 
                                                                         self.coef['Cm', keybm][:,mask].shape[1],
                                                                         )), 
                                                     self.coef['Cm', keybm][:,mask],
                                                     )

                        ############# Check this part of quant[coor_upd] that is specific to Cb
                        
                        if EXTRA_CHECK:
                            aaa =   Y[:,mask_Y].copy()
                            for var2 in self.config_coef:
                                for key2 in self.keys[var2]: 
                                    if keybm+'@'+key2 in XtX\
                                    and (   key2.split('#')[0] != key
                                         or var2 not in ('Cb', 'Cbm')
                                         ):
                                        ss = self.orig_masks.get((var2,key2,()), slice(None)) if gp_pen else mask
                                        aaa[:,ss if gp_pen else slice(None)] += - X[key2] @ self.coef[var2,key2][:,ss] 
                            bbb_check[keybm] = - (1/n) * X[keybm].T @ aaa
                            if gp_pen:
                                bbb_check[keybm] = gp_pen * bbb_check[keybm] @ MMt[:,mask]
                                if len(mask_out):
                                    bbb_check[keybm]+= gp_pen * (1/n) * XtX[keybm, keybm] @ self.coef['Cbm',keybm][:,mask_out] @ MMt[mask_out][:,mask]
                            assert bbb[keybm].shape ==  bbb_check[keybm].shape
                            assert np.allclose(bbb[keybm], bbb_check[keybm])                           
                            mmm_check += np.einsum('pqk,qk->pk', 
                                                   bbb_check[keybm].reshape((-1, 
                                                                             self.coef['Cm', keybm][:,mask].shape[0], 
                                                                             self.coef['Cm', keybm][:,mask].shape[1],
                                                                             )), 
                                                   self.coef['Cm', keybm][:,mask],
                                                   )

            if EXTRA_CHECK:
                assert quant[coor_upd].shape == mmm_check.shape, (coor_upd, quant[coor_upd].shape, mmm_check.shape)
                assert np.allclose(quant[coor_upd], mmm_check) 
            
            ############# Computations of extra_part
            if var == 'bu':
                ############# Very specific case of bu
                extra_part[coor_upd] = self.xtra_part_bu(n, XtX, self.coef, coor_upd, mask, gp_pen, MMt)
                if data == 'train': # Special case for bu because both bu and bv are updated at the same time
                    grad[coor_upd] =(quant[coor_upd] + extra_part[coor_upd]) @  self.coef[('bv',key)]#[mask]
            else:
                ############# Specific case of Cb
                if var == 'Cb':
                    extra_part[coor_upd] = self.xtra_part_cb(n, XtX, self.coef, coor_upd, mask, gp_pen, MMt)
                ############# Specific case fo Cm
                elif var == 'Cm':
                    extra_part[coor_upd] = self.xtra_part_cm(n, XtX, self.coef, coor_upd, mask, gp_pen, MMt)                                                
                ############# Specific case of Cu
                elif var == 'Cu':
                    extra_part[coor_upd] = self.xtra_part_cu(n, XtX, self.coef, coor_upd, mask, gp_pen, MMt)                                               
                ############# Specific case of Cu
                elif var == 'Cv':
                    extra_part[coor_upd] = self.xtra_part_cv(n, XtX, self.coef, coor_upd, mask, gp_pen, MMt)
                        
                else:
                    extra_part[coor_upd] = self.xtra_part_cl(n, XtX, self.coef, coor_upd, mask, gp_pen, MMt)
                if data == 'train':
                    assert quant[coor_upd].shape == extra_part[coor_upd].shape
                    grad[coor_upd] = quant[coor_upd] + extra_part[coor_upd]
            
            if EXTRA_CHECK:
                if var == 'bu':
                    assert self.coef[(var,key)][:,mask].shape ==(quant     [coor_upd]@self.coef[('bv',key)][mask]).shape 
                    assert self.coef[(var,key)][:,mask].shape ==(extra_part[coor_upd]@self.coef[('bv',key)][mask]).shape    
                elif var == 'Cu':
                    assert self.coef[(var,key)][:,:,mask].shape == quant     [coor_upd].shape 
                    assert self.coef[(var,key)][:,:,mask].shape == extra_part[coor_upd].shape 
                elif var == 'Cv':
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
        if data == 'train':
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
        for cat in sorted(set([self.param['data_cat'][key] 
                               for key in self.sorted_keys
                               ])):
            alpha_cat = 0
            for var in sorted(self.alpha):
                if cat in self.alpha[var]:
                    if (var if var != 'bu' else 'Blr') in self.config_coef:
                        if cat in self.config_coef[var if var != 'bu' else 'Blr']:
                            if self.pen[var][cat] in {'rsm', 'r2sm'}:
                                assert not alpha_cat, 'it should not have already been found for the same cat and a penalization in rsm, r2sm, unless it is approximately lowrank but this is not implemented yet'
                                alpha_cat = self.alpha[var][cat]
            keys_for_cat = [key 
                            for key in self.sorted_keys 
                            if self.param['data_cat'][key] == cat
                            ]
            if len(keys_for_cat) == 1:
                X_cat = self.X_train[keys_for_cat[0]]
            else:
                X_cat = sp.sparse.hstack([sp.sparse.csc_matrix(self.X_train[key]) 
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
                    self.normalization[cat] = np.sqrt((X_tilde_cat**2).mean() - X_tilde_cat.mean()**2)
            else:
                if X_tilde_cat.data.shape[0] == 0:
                    self.normalization[cat] = 1
                else:
                    self.normalization[cat] = np.sqrt((X_tilde_cat.data**2).mean() - X_tilde_cat.data.mean()**2)
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
                    if isinstance(self.alpha[var][cat], Number):
                        self.normalized_alphas[var][cat] = self.alpha[var][cat]/self.normalization[cat]**2
                    else:
                        assert type(self.alpha[var][cat]) == tuple
                        assert len(self.alpha[var][cat])  == 2
                        assert self.pen[var][cat]         == 'row2sm'
                        self.normalized_alphas[var][cat] = (self.alpha[var][cat][0]/self.normalization[cat]**2, 
                                                            self.alpha[var][cat][1], 
                                                            )                      
        for var, dikt_cat_alpha in self.normalized_alphas.items():
            assert len(self.alpha[var]) == len(dikt_cat_alpha)
            for cat in self.alpha[var].keys():
                if isinstance(self.alpha[var][cat], Number):
                    assert isinstance(self.normalized_alphas[var][cat], Number)
                else:
                    assert type(self.alpha[var][cat]) == type(self.normalized_alphas[var][cat])



    def compute_surrogate(self, grad, grad_gp, ridge_grad, coef_tmp, eta, cur_fit, d_masks):
        # Compute the value of the smooth surrogate in the line-search procedure
        f    = cur_fit
        for coor_upd in grad.keys():
            var,key,ind = coor_upd
            mask =  d_masks.get(coor_upd, slice(None))
            if var in {'Cu', 'Cv'}:
                assert coef_tmp[coor_upd].shape == self.coef[coor_upd[:2]][:,:,mask].shape, (coef_tmp[coor_upd].shape, self.coef[coor_upd[:2]][:,:,mask].shape)
                if type(grad[coor_upd]) == np.matrix:
                    f +=   (grad[coor_upd] + grad_gp.get(coor_upd,0) + ridge_grad[coor_upd]).reshape(-1)\
                          .dot((coef_tmp[coor_upd] - self.coef[coor_upd[:2]][:,mask]).reshape(-1).T).sum()\
                          + (1/(2*eta))*np.linalg.norm(coef_tmp[coor_upd] - self.coef[coor_upd[:2]][:,:,mask])**2 
                else:
                    f +=   (grad[coor_upd] + grad_gp.get(coor_upd,0) + ridge_grad[coor_upd]).reshape(-1).T\
                          @(coef_tmp[coor_upd] - self.coef[coor_upd[:2]][:,:,mask]).reshape(-1)\
                          + (1/(2*eta))*np.linalg.norm(coef_tmp[coor_upd] - self.coef[coor_upd[:2]][:,:,mask])**2 
            else:
                if type(grad[coor_upd]) == np.matrix:
                    f +=   (grad[coor_upd] + grad_gp.get(coor_upd,0) + ridge_grad[coor_upd]).reshape(-1)\
                          .dot((coef_tmp[coor_upd] - self.coef[coor_upd[:2]][:,mask]).reshape(-1).T).sum()\
                          + (1/(2*eta))*np.linalg.norm(coef_tmp[coor_upd] - self.coef[coor_upd[:2]][:,mask])**2 
                else:
                    f +=   (grad[coor_upd] + grad_gp.get(coor_upd,0) + ridge_grad[coor_upd]).reshape(-1).T\
                          @(coef_tmp[coor_upd] - self.coef[coor_upd[:2]][:,mask]).reshape(-1)\
                          + (1/(2*eta))*np.linalg.norm(coef_tmp[coor_upd] - self.coef[coor_upd[:2]][:,mask])**2 
            assert coef_tmp[coor_upd].shape == grad[coor_upd].shape,               ('pb_1', coef_tmp[coor_upd].shape, grad[coor_upd].shape)
            assert coef_tmp[coor_upd].shape == ridge_grad[coor_upd].shape,         ('pb_2', coef_tmp[coor_upd].shape, ridge_grad[coor_upd].shape)
            if var in {'Cu', 'Cv'}:
                assert coef_tmp[coor_upd].shape == self.coef[(var,key)][:,:,mask].shape, ('pb_35', coef_tmp[coor_upd].shape, self.coef[(var,key)][:,:,mask].shape)
            else:
                assert coef_tmp[coor_upd].shape == self.coef[(var,key)][:,mask].shape, ('pb_3', coef_tmp[coor_upd].shape, self.coef[(var,key)][:,mask].shape)
            if self.active_gp:
                assert coef_tmp[coor_upd].shape == grad_gp[coor_upd].shape,          ('pb_4', coef_tmp[coor_upd].shape, grad_gp[coor_upd].shape)
            assert type(f) != np.matrix
        return f
    

    ##profile
    # Decomposition of the gradient computations 
    # because of the restrictions due to the sparse structure
    # and the use of the masks
    def custom_sum_einsum(self, XtX, coef, var2, key, mask, var_temp = None):
        var_upd    = (var_temp == var2)
        cbcbm_upd  = (var_temp == 'Cb' and var2 == 'Cbm')
        cbmcb_upd  = (var_temp == 'Cb' and var2 == 'Cb'  and '#' in key)
        cbmcbm_upd = (var_temp == 'Cb' and var2 == 'Cbm' and '#' in key)
        if EXTRA_CHECK:
            for key2 in self.keys[var2]:
                if self.active_gp:
                    if (var_temp,key) in coef and (var2,key2) in coef:
                        assert key+'@'+key2 in XtX # Probably need this check only when gp_pen > 0
                if key+'@'+key2 in XtX\
                and not(   (var_upd    and key2 == key)
                        or (cbcbm_upd  and key2.split('#')[0] == key)
                        or (cbmcb_upd  and key.split('#')[0]  == key2)
                        or (cbmcbm_upd and key.split('#')[0]  == key2.split('#')[0])
                        ):                 
                    ss = self.orig_masks.get((var2,key2,()), slice(None))
                    if not (mask == 'dyn' or cbmcb_upd or cbmcbm_upd):
                        assert (XtX[key+'@'+key2] @ (coef[var2,key2][:,mask])).shape == (XtX[key+'@'+key] @ (coef[var_temp,key][:,mask])).shape
        if mask == 'dyn':
            assert self.gp_pen
            ans = np.zeros((self.size[key], self.k))
            for key2 in self.keys[var2]:
                if key+'@'+key2 in XtX\
                and not(   (var_upd    and key2 == key)
                        or (cbcbm_upd  and key2.split('#')[0] == key)
                        or (cbmcb_upd  and key.split('#')[0]  == key2)
                        or (cbmcbm_upd and key.split('#')[0]  == key2.split('#')[0])
                        ):
                    ss = self.orig_masks.get((var2,key2,()), slice(None))
                    ans[:,ss] += XtX[key+'@'+key2] @ coef[var2,key2][:,ss]
        else:
            ans = np.zeros((self.X_train[key].shape[1], coef[var2,next(iter(self.keys[var2]))][:,mask].shape[1]))
            for key2 in self.keys[var2]:
                if (key+'@'+key2 in XtX
                    and not(   (var_upd    and key2 == key)
                            or (cbcbm_upd  and key2.split('#')[0] == key)
                            or (cbmcb_upd  and key.split('#')[0]  == key2)
                            or (cbmcbm_upd and key.split('#')[0]  == key2.split('#')[0])
                            )
                    ):
                    ans += XtX[key+'@'+key2] @ (coef[var2,key2][:,mask])
#            ans = np.sum([XtX[key+'@'+key2] @ (coef[var2,key2][:,mask]) 
#                          for key2 in self.keys[var2]
#                              if key+'@'+key2 in XtX
#                              and not(   (var_upd    and key2 == key)
#                                      or (cbcbm_upd  and key2.split('#')[0] == key)
#                                      or (cbmcb_upd  and key.split('#')[0]  == key2)
#                                      or (cbmcbm_upd and key.split('#')[0]  == key2.split('#')[0])
#                                      )
#                          ], 
#                          axis = 0)
##            assert ans.shape == ans_tmp.shape
        return ans
                 

    def duality_gap(self, pred, grad, var):
        raise NotImplementedError
        if self.pen[var] in {'', 'rsm'} or self.normalized_alphas[var] == 0:
            return self.cur_fit_train
        else:
            # On somme les duality gaps sur les différents postes
            assert self.pen[var] == 'lasso'            
            with_intercept = var in {'U', 'V'} and (self.UV_basis)
            if with_intercept:
                return self.cur_fit_train
            grad     = {(s,k):v for (s,k),v in grad.items() if s == var}
            # maximum coef in the gradient
            d        = np.array([max(1, np.max([np.max(np.abs(v[:,j] if v.ndim == 2 else v[:,:,j])) for k, v in grad.items()])/self.normalized_alphas[var]) for j in range(self.k)])
            c        = max(d)
            # dual variable
            dual_Z   = (1/self.n_train)*(pred - self.Y_train)/d
            # value of the primal function
            f        = self.evaluate_fit(self.coef) 
            f       += self.evaluate_ridge(self.coef)
            # value of the residuals
            if var == 'B':
                var_drop = 'B'
            elif var in {'U', 'V'}:
                var_drop = 'C'
            res      = self.Y_train - self.predict(coef = self.coef, drop = {var_drop}) 
            # value of the dual function
            g        = np.einsum('nk,nk->', dual_Z, res) + 0.5*self.n_train*np.linalg.norm(dual_Z)**2
            # scalar product
            if var in {'B'}:
                pr = np.sum([np.einsum('pk,pk->',   self.coef[(s,k)], grad[(s,k)]/d) for (s,k) in self.coef if s == var])
                #pp =         np.einsum('nk,nk->',   pred, dual_Z)                    
                #assert np.abs(pr - pp ) <= 1e-5
            elif var in {'U', 'V'}:
                pr = np.sum([np.einsum('prk,prk->', self.coef[(s,k)], grad[(s,k)]/d) for (s,k) in self.coef if s == var])
                #pp =         np.einsum('prk,prk->', pred, dual_Z)                
                #assert np.abs(pr - pp ) <= 1e-5
            else:
                raise ValueError
            # regularization
            reg      = np.sum([v for k, v in self.cur_ind_reg.items() if k[0] == var])
            #if var in {'U', 'V'}:
            ############
            ## Checks ##
            ############
            for k, v in grad.items():
                assert v.max()/c <= self.normalized_alphas[var] + 1e-14, (k, v.max(), c, self.normalized_alphas[var])
                for j in range(self.k):
                    if v.ndim == 2:
                        assert v[:,j].max()/d[j] <= self.normalized_alphas[var] + 1e-14
                    elif v.ndim == 3:
                        assert v[:,:,j].max()/d[j] <= self.normalized_alphas[var] + 1e-14
                    else: 
                        raise ValueError                
            P = f + g - pr
            assert P >= - 1e-14
            D = pr + reg
            assert D >= - 1e-14
#            print(P, D, P+D)
            return P + D

    #profile
    def evaluate_fit(self, coef, data = None, gp_matrix = None, gp_pen = None):
        assert data
        if (type(gp_matrix)!=type(None)) and not gp_pen:
            return 0
        w_gp = (type(gp_matrix)!=type(None)) and gp_pen
        n   = self.n_train if data == 'train' else self.n_test
        Y   = self.Y_train if data == 'train' else self.Y_test
        if not w_gp:
            mse =        (0.5/n)*np.linalg.norm((Y*((1 - coef['A','']) if self.factor_A else 1) - self.predict(coef = coef, data = data)))**2
        else:
            mse = gp_pen*(0.5/n)*np.linalg.norm((Y*((1 - coef['A','']) if self.factor_A else 1) - self.predict(coef = coef, data = data)) @ gp_matrix)**2
        return mse


    #profile
    def evaluate_ind_fit(self, coef, data = None):
        assert data
        n       = self.n_train    if data == 'train' else self.n_test
        Y       = self.Y_train    if data == 'train' else self.Y_test
        if self.iteration > self.flag_check_pred:
            self.flag_check_pred += self.epoch_check_pred
        ind_mse = (0.5/n)*np.linalg.norm((Y*((1 - coef['A','']) if self.factor_A else 1) - self.predict(coef = coef, data = data)), axis = 0)**2
        return ind_mse


    #profile
    def evaluate_fit_bcd(self, coor_upd, coef_tmp, quant, mask = None, data = None, MMt = None, gp_pen = 0, **kwargs):
        if gp_pen:
            assert type(MMt) != type(None)
        assert data
        assert len(quant) <= 3
        n   = self.n_train   if data == 'train' else self.n_test
        XtX = self.XtX_train if data == 'train' else self.XtX_test
        new_extra_part = {}
        new_fit        = 0
        for coor_upd in quant.keys():
            if coor_upd[0] in {'Cuv','bv','Blr', 'Cbm'}:
                raise ValueError
                #continue
            var, key, ind = coor_upd
            if var in {'Bsp', 'Csp', 'bu', 'Cu', 'Cv', 'Cb', 'Cm'}:
                if var == 'bu':
                    new_extra_part[coor_upd] = self.xtra_part_bu(n, XtX, coef_tmp, coor_upd, mask, gp_pen, MMt)
                elif var == 'Cb':
                    new_extra_part[coor_upd] = self.xtra_part_cb(n, XtX, coef_tmp, coor_upd, mask, gp_pen, MMt) 
                elif var == 'Cm':
                    new_extra_part[coor_upd] = self.xtra_part_cm(n, XtX, coef_tmp, coor_upd, mask, gp_pen, MMt)
                elif var == 'Cu':
                    new_extra_part[coor_upd] = self.xtra_part_cu(n, XtX, coef_tmp, coor_upd, mask, gp_pen, MMt)
                elif var == 'Cv':
                    new_extra_part[coor_upd] = self.xtra_part_cv(n, XtX, coef_tmp, coor_upd, mask, gp_pen, MMt)
                else:
                    new_extra_part[coor_upd] = self.xtra_part_cl(n, XtX, coef_tmp, coor_upd, mask, gp_pen, MMt)
                if EXTRA_CHECK:
                    try:
                        if var == 'bu':
                            assert coef_tmp['Blr',coor_upd[1]].shape == new_extra_part[coor_upd].shape
                            assert coef_tmp[coor_upd].shape == (quant[coor_upd][:,self.dikt_masks.get(('Blr', coor_upd[1]), slice(None))] @ coef_tmp['bv', key]).shape
                        else:
                            assert coef_tmp[coor_upd].shape == quant         [coor_upd].shape, (coef_tmp[coor_upd].shape, quant[coor_upd].shape)  
                            assert coef_tmp[coor_upd].shape == new_extra_part[coor_upd].shape, (coef_tmp[coor_upd].shape, quant[coor_upd].shape)
                    except Exception as e:
                        print(e)
                        raise e
                new_fit += self.part_fit(coef_tmp, coor_upd, quant, new_extra_part, data = data, **kwargs)
            elif var in {'Cuv', 'Blr', 'Cbm'}:
                pass
            else:
                raise ValueError
        return new_fit 


    def evaluate_ind_reg(self, coef):
        reg    = {}
        slope  = {}
        offset = {}
        cond =  (np.sum([e[0] == 'Cb' for e in coef]) != 1) and len(coef) >= 10 and not EXTRA_CHECK
        for i, (coor, M) in enumerate(coef.items()):
            if   coor[0] in {'Blr', 'bv', 'Cuv', 'Cbm'}:
                continue
            elif coor[0] in {'Bsp', 'Csp', 'bu', 'Cu', 'Cv', 'Cb', 'Cm'}:
                if cond:
                    print('\revaluate_ind_reg', i, '/', len(coef), end = '')
                var, key, ind = coor
                pen, alpha = self.get_pen_alpha(var, key)
                res        = self.penalization(M, pen, alpha, key) # Compute regularization of one column
                if pen == 'n2cvxclasso':
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
            if coor[0] in {'Blr', 'bv', 'Cuv', 'Cbm'}:
                continue
            var, key, ind = coor
            pen, alpha = self.get_pen_alpha(var, key)
            if pen == 'rsm':
                if type(M) in {np.ndarray, np.matrix}:
                    ridge[var,key,ind] = (alpha/2)*np.linalg.norm(M)**2 
                else:
                    ridge[var,key,ind] = (alpha/2)*sp.sparse.linalg.norm(M)**2
            elif pen == 'r2sm':
                if type(M) not in {np.ndarray, np.matrix}:
                    raise NotImplementedError
                else:
                    cat = self.param['data_cat'][key]
                    reshape_tensor = '#' in cat and var not in {'Cu', 'Cv', 'Cb', 'Cm'}
                    if not reshape_tensor:
                        cc = M
                        if cc.shape[0] > 2:
                            ker  = np.array([[1],[-2],[1]])
                            if self.param['qo_modulo'].get(cat):
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
                        cc = M.reshape(*self.size_tensor2[key], -1)  
                        cat1, cat2 = cat.split('#')
                        if cc.shape[0] > 2:
                            ker1  = np.array([[[1]],[[-2]],[[1]]]) 
                            if self.param['qo_modulo'].get(cat1):
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
                        if var in {'Cbm', 'Cb', 'Cm', 'Blr', 'bv'}:
                            raise ValueError
                        if cc.shape[1] > 2:
                            ker2  = np.array([[[1 ],[ -2 ],[ 1]]])
                            if self.param['qo_modulo'].get(cat2):
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
                            
            elif pen == '0r2sm':
                if type(M) not in {np.ndarray, np.matrix}:
                    raise NotImplementedError
                else:
                    cat = self.param['data_cat'][key]
                    reshape_tensor = '#' in cat
                    if not reshape_tensor:
                        cc = M
                        if cc.shape[0] > 2:
                            ker  = np.array([[1],[-2],[1]])
                            if self.param['qo_modulo'].get(cat):
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
                        cc = M.reshape(*self.size_tensor2[key], -1)  
                        cat1, cat2 = cat.split('#')
                        if cc.shape[0] > 2:
                            ker1  = np.array([[[1]],[[-2]],[[1]]]) 
                            if self.param['qo_modulo'].get(cat1):
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
        return ridge
    
    

    def foba_step(self, grad, grad_gp, ridge_grad, eta, d_masks):
        # Forward-backward step
        # ie gradient step and proximal operator
        coef_tmp= {}
        for coor_upd in grad.keys():
            var, key, ind = coor_upd
            mask          = d_masks.get(coor_upd, slice(None))
            pen, alpha    = self.get_pen_alpha(var, key)
            if pen == 'n2cvxclasso':
                orig_mask  = self.orig_masks[coor_upd]
                same_mask  = type(mask) == type(orig_mask) and (mask == orig_mask if type(mask) == type(slice(None)) else np.allclose(mask, orig_mask))
                slope_mask = slice(None) if same_mask else [k for k in (orig_mask 
                                                                        if type(orig_mask) == np.ndarray 
                                                                        else 
                                                                        np.arange(self.coef[coor_upd[:2]].shape[1])
                                                                        ) if k in mask]
                alpha = self.slope_ind[coor_upd][slope_mask]
            if EXTRA_CHECK:
                if var in {'Cu', 'Cv'}:
                    assert self.coef[var,key][:,:,mask].shape == grad[coor_upd].shape
                    assert self.coef[var,key][:,:,mask].shape == ridge_grad[coor_upd].shape
                else:
                    assert self.coef[var,key][:,mask].shape == grad[coor_upd].shape
                    assert self.coef[var,key][:,mask].shape == ridge_grad[coor_upd].shape
                if self.active_gp:
                    assert coor_upd in grad_gp
            if var in {'Cu', 'Cv'}:
                coef_tmp[coor_upd] = self.prox_op(self.coef[(var,key)][:,:,mask] - eta*(grad[coor_upd] + grad_gp.get(coor_upd,0) + ridge_grad[coor_upd]), pen, eta, alpha, coef_zero = self.coef[(var,key)][:,:,mask])
                assert coef_tmp[coor_upd].shape == self.coef[(var,key)][:,:,mask].shape 
            else:    
                coef_tmp[coor_upd] = self.prox_op(self.coef[(var,key)][:,mask] - eta*(grad[coor_upd] + grad_gp.get(coor_upd,0) + ridge_grad[coor_upd]), pen, eta, alpha, coef_zero = self.coef[(var,key)][:,mask])
            if EXTRA_CHECK and alpha == 0 and var not in {'Cu', 'Cv'}:
                aa = self.coef[(var,key)][:,mask] - eta*(grad[coor_upd] + grad_gp.get(coor_upd,0) + ridge_grad[coor_upd])
                bb = self.prox_op(self.coef[(var,key)][:,mask] - eta*(grad[coor_upd] + grad_gp.get(coor_upd,0) + ridge_grad[coor_upd]), pen, eta, alpha, coef_zero = self.coef[(var,key)][:,mask])
                assert np.all(aa == bb)
        return coef_tmp


        
    def get_pen_alpha(self, var, key):
        # For one coefficient matrix and one covariate, fetch the regularization and the coefficient
        if var == 'A':
            pen   = self.pen.get(var,'')
            alpha = self.normalized_alphas.get(var,0)
        elif key == 'ones':
            pen   = ''
            alpha = 0
        else:
            pen   = self.pen  .get(var,{}).get(self.param['data_cat'][key],'')
            alpha = self.normalized_alphas.get(var,{}).get(self.param['data_cat'][key],0)
        return pen, alpha
    
    #profile
    def initialize_coef(self, ):
        coef = deepcopy(self.given_coef)
        keys_upd = []
        self.punished_coor  = set()
        self.prison_coor    = {}
        self.life_sentences = set()
        self.nb_sentences   = {}
        method_init_UV = self.param.get('tf_method_init_UV')
        print('method_init_UV : ', method_init_UV)
        fac = 1e-4
        self.keys = {
                     key : []
                     for key in self.config_coef
                     }
        if self.factor_A:
            keys_upd     += ['A','',()]
            coef['A',''] = np.zeros(self.k)
        if 'Blr' in self.config_coef:   
            for key in list(filter(lambda x :                  x not in self.mask, self.X_train.keys())):
                if key in self.mask: 
                    print(colored('\n\nNo Mask when low-rank\n\n', 'red'))
                    raise ValueError
                    del self.mask[key]
                    print(colored('\n\nmask of {0} removed to include it in Blr\n\n'.format(key), 'red', 'on_cyan'))
                if self.param['data_cat'][key] in self.config_coef['Blr']:
                    self.keys['Blr'].append(key)
                    if not (hasattr(self, 'freeze_Blr') and self.freeze_Blr):
                        if self.pen['bu'].get(self.param['data_cat'][key]) != 'rlasso' and self.param['tf_col_upd'].get(self.param['data_cat'][key]):
                            raise ValueError('we do not want column update for Blr')
                            keys_upd += [('bu',key,(int(r),)) for r in self.mask.get(key, range(self.k))]
                        else:
                            keys_upd += [('bu',key,())]
                    if ('bu', key) not in coef:
                        mm   = slice(None)
                        lenV = self.k
                        coef[('bu',key)]    = np.random.randn(self.size[key], min(self.r_B[key], self.k))*fac
                        if lenV > 0:
                            coef[('bv',key)]    = np.zeros((self.k, min(self.r_B[key], self.k)))
                            coef[('bv',key)][mm], _ = np.linalg.qr(np.random.randn(self.k, min(self.r_B[key], self.k)))
                            coef[('Blr',key)]   = coef[('bu',key)] @ coef[('bv',key)].T
        if 'Bsp' in self.config_coef:
            for key in self.X_train.keys():
                if self.param['data_cat'][key] in self.config_coef['Bsp']:
                    if not (key in self.mask and type(self.mask[key])==np.ndarray and self.mask[key].shape[0] == 0):
                        self.keys['Bsp'].append(key)
                    if not self.freeze_B:
                        if self.pen['Bsp'].get(self.param['data_cat'][key]) != 'rlasso' and self.param['tf_col_upd'].get(self.param['data_cat'][key]):
                            keys_upd += [('Bsp',key,(int(r),)) for r in self.mask.get(key, range(self.k))]
                        else:
                            keys_upd += [('Bsp',key,())]
                    if ('Bsp', key) not in coef:
                        if   self.param['tf_sparse_coef']\
                        and (key in self.mask or 'classo' in self.pen.get('Bsp',{}).get(self.param['data_cat'][key])):
                            coef['Bsp',key] = sp.sparse.csc_matrix(np.zeros((self.size[key], self.k)))
                        else:
                            coef['Bsp',key] = np.zeros((self.size[key], self.k))
        if 'Cuv' in self.config_coef:
            for key in list(filter(lambda x : '#' in x, self.X_train.keys())):
                if self.param['data_cat'][key] in self.config_coef['Cuv']:
                    if key.split('#') != sorted(key.split('#')):
                       continue
                    if not (key in self.mask and self.mask[key].shape[0] == 0):
                        self.keys['Cuv'].append(key)
                    if not (hasattr(self, 'freeze_Cuv') and self.freeze_Cuv):
                        if self.pen['Cu'].get(self.param['data_cat'][key]) != 'rlasso' and self.param['tf_col_upd'].get(self.param['data_cat'][key]):
                            keys_upd +=   [('Cu',key,(int(r),),) for r in self.mask.get(key, range(self.k))] 
                        else:
                            keys_upd +=   [('Cu',key,())]
                        if self.pen['Cv'].get(self.param['data_cat'][key]) != 'rlasso' and self.param['tf_col_upd'].get(self.param['data_cat'][key]):
                            keys_upd +=   [('Cv',key,(int(r),),) for r in self.mask.get(key, range(self.k))]
                        else:
                            keys_upd +=   [('Cv',key,())]
                    if ('Cu', key) not in coef or ('Cv', key) not in coef:
                        if   self.param['tf_sparse_coef'] and False\
                        and (key in self.mask or 'classo' in self.pen.get('Cuv',{}).get(self.param['data_cat'][key])):
                            assert 0, 'no sparse and einsum'
                        else:
                            rk = min(self.r_UV, self.size_tensor2[key][0], self.size_tensor2[key][1])
                            coef['Cu',key ]  = np.zeros((self.size_tensor2[key][0], rk, self.k))
                            coef['Cu',key ][:,:,self.mask.get(key,None)] = np.random.randn(*coef['Cu',key][:,:,self.mask.get(key,None)].shape) 
                            coef['Cv',key ]  = np.zeros((self.size_tensor2[key][1], rk, self.k))
                            coef['Cv',key ][:,:,self.mask.get(key,None)] = np.random.randn(*coef['Cv',key][:,:,self.mask.get(key,None)].shape)
                            coef['Cuv',key] = np.einsum('prk,qrk->pqk',
                                                        coef['Cu',key], 
                                                        coef['Cv',key], 
                                                        ).reshape((-1, self.k))
        if 'Cbm' in self.config_coef:
            for key_b in (  list(filter(lambda x : '#' not in x, self.X_train.keys()))
                          + list(map(lambda x : x.split('#')[0], list(filter(lambda x : '#' in x, self.X_train.keys()))))
                          ):
                if self.param['data_cat'][key_b] in self.config_coef['Cb']:
                    if not (key_b in self.mask and self.mask[key_b].shape[0] == 0) and not (key_b in self.keys['Cb']):
                        self.keys['Cb' ].append(key_b)
                    if not (hasattr(self, 'freeze_Cb') and self.freeze_Cb):
                        if self.pen['Cb'].get(self.param['data_cat'][key_b]) != 'rlasso' and self.param['tf_col_upd'].get(self.param['data_cat'][key_b]):
                            for r in self.mask.get(key_b, range(self.k)):
                                if ('Cb',key_b,(int(r),),) not in keys_upd:
                                    keys_upd += [('Cb',key_b,(int(r),),) ]
                        else:
                            if ('Cb',key_b,()) not in keys_upd:
                                keys_upd += [('Cb',key_b,())]
    
                    if ('Cb', key_b) not in coef:
                        if  (self.param['tf_sparse_coef'] and False\
                             and (key_b in self.mask or 'classo' in self.pen.get('Cb',{}).get(self.param['data_cat'][key_b]))
                             ):
                            assert 0, 'no sparse and einsum'
                        else:
                            coef['Cb',key_b] = np.zeros((self.size[key_b], self.k))
            for key in list(filter(lambda x : '#' in x, self.X_train.keys())):
                key_b = key.split('#')[0]
                if self.param['data_cat'][key] in self.config_coef['Cbm']+tuple(['#'.join(e.split('#')[::-1]) for e in self.config_coef['Cbm']]):
                    if not (key in self.mask and self.mask[key].shape[0] == 0) and not (key in self.keys['Cbm']):
                        self.keys['Cbm'].append(key)
                    if not (hasattr(self, 'freeze_Cm') and self.freeze_Cm):
                        if self.pen['Cm'].get(self.param['data_cat'][key]) != 'rlasso' and self.param['tf_col_upd'].get(self.param['data_cat'][key]):
                            for r in self.mask.get(key, range(self.k)):
                                if ('Cm',key,(int(r),),) not in keys_upd:
                                    keys_upd += [('Cm',key,(int(r),),) ]
                        else:
                            if ('Cm',key,()) not in keys_upd:
                                keys_upd += [('Cm',key,())]  
                    if ('Cm', key) not in coef:
                        if  (self.param['tf_sparse_coef'] and False\
                             and (key in self.mask or 'classo' in self.pen.get('Cm',{}).get(self.param['data_cat'][key]))
                             ):
                            assert 0, 'no sparse and einsum'
                        else:
                            assert self.size_tensor2[key][0] == self.size[key_b], 'key : {0} - key_b : {1}'.format(key, key_b)
                            coef['Cm',key]   = np.zeros((self.size_tensor2[key][1], self.k))
                            coef['Cbm',key]  = np.einsum('pk,qk->pqk',
                                                        coef['Cb',key_b], 
                                                        coef['Cm',key], 
                                                        ).reshape((-1, self.k))                  
        assert len(keys_upd) == len(set(keys_upd)), (len(keys_upd), len(set(keys_upd)))
        print('Finished initialization')
        return coef, keys_upd
    
    
    def make_gp_matrix(self, ):
        # Compute the matrix for the sum-consistent model
        assert self.param['gp_matrix'] in {'cst', None} or self.lbfgs
        if self.param['gp_matrix'] in {'cst', 'nat', 'rteReg', 'admReg', 'districts', 'eye'}:
            if self.param['gp_matrix'] == 'cst':
                self.post_to_reg = {pp:'nat' for k, pp in enumerate(self.param['posts_names'])}
            elif self.param['gp_matrix'] in {'nat', 'rteReg', 'admReg', 'districts'}:
                assert self.param['zone'] == 'full', 'zone is not full - should thing about it otherwise'
                if  self.param['gp_matrix'] == 'nat':
                    self.post_to_reg = misc.get_dikt_nat(self.param) 
                elif self.param['gp_matrix'] == 'rteReg': 
                    self.post_to_reg = misc.get_dikt_reg_rte(self.param)
                elif self.param['gp_matrix'] == 'admReg':
                    self.post_to_reg = misc.get_dikt_reg_admin(self.param)
                elif self.param['gp_matrix'] == 'districts':
                    self.post_to_reg = misc.get_dikt_districts(self.param)
            elif self.param['gp_matrix'] == 'eye':
                self.post_to_reg = {pp:pp for k, pp in enumerate(self.param['posts_names'])}
            else:
                raise ValueError
            self.idxPost_to_reg = {
                                   k : self.post_to_reg[self.param['posts_names'][k]]
                                   for k in range(len(self.param['posts_names']))
                                   }
            self.reg_to_tuples = {
                                  reg : tuple(sorted([ii 
                                                      for ii, post in enumerate(self.param['posts_names'])
                                                      if self.idxPost_to_reg[ii] == reg
                                                      ])
                                              )
                                  for reg in set(self.post_to_reg.values())
                                  }
            self.tuples_to_reg = {v:k for k, v in self.reg_to_tuples.items()}
            self.gp_matrix = np.zeros((self.k, self.k))
            for reg, indices in self.reg_to_tuples.items():
                for k in indices: 
                    for j in indices:
                        self.gp_matrix[k,j] = 1
        elif type(self.param['gp_matrix']) == int and self.param['gp_matrix'] > 0:
            assert len(self.param['coor_posts']) == self.k
            self.rolling_away, self.rolling_away_ind = build_data.rolling_away(self.param['coor_posts'], self.param['coor_posts'])
            self.gp_matrix = np.zeros((self.k, self.k))
            for k in range(self.k):
                self.gp_matrix[self.rolling_away_ind[self.param['posts_names'][k]][:self.param['gp_matrix']],k] = 1
        else:
            raise NotImplementedError     


    def make_orig_masks(self):
        d_masks = {}
        for coor_upd in self.keys_upd.copy():
            var, key, ind = coor_upd
            if ind == ():
                if key in self.mask:
                    d_masks[coor_upd] = self.mask[key]
                elif '#'.join(key.split('#')[::-1]) in self.mask:
                    d_masks[coor_upd] = self.mask['#'.join(key.split('#')[::-1])]
                else:
                    d_masks[coor_upd] = slice(None)
            else:
                assert type(ind) == tuple
                assert len (ind) == 1
                assert type(ind[0]) == int
                d_masks[coor_upd] = np.array(ind)
            if type(d_masks[coor_upd]) == np.ndarray and d_masks[coor_upd].ndim == 1 and d_masks[coor_upd].shape[0] == 0:
                self.keys_upd.remove(coor_upd)
            if var == 'bu': 
                del d_masks[coor_upd]
        return d_masks
    
    
    def part_fit(self, coef, coor_upd, quant, extra_part, mask = slice(None), data = None, indi = False):
        # Decomposed computation of the data-fitting term
        var, key = coor_upd[:2]
        if var == 'bu':
            cof = coef['Blr', key]
        elif var in {'Cu', 'Cv'}:
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
            elif pen in {'rsm', 'r2sm', '0r2sm'}:  
                return 0
            elif pen == 'lasso':
                return mu*np.sum(np.abs(M))                 
            elif pen == 'rlasso':
                assert M.ndim == 2
                if type(M) in {np.ndarray, np.matrix}:
                    return mu*np.sum([np.linalg.norm(M[i])        for i in range(M.shape[0])]) 
                else:
                    return mu*np.sum([sp.sparse.linalg.norm(M[i]) for i in range(M.shape[0])])
            elif pen == 'classo':
                if type(M) in {np.ndarray, np.matrix}:
                    return mu*np.sum([np.linalg.norm(M[:,i])        for i in range(M.shape[1])])    
                else:
                    return mu*np.sum([sp.sparse.linalg.norm(M[:,i]) for i in range(M.shape[1])])
            elif pen == 'ncvxclasso' :
                alpha, beta, gamma = mu
                assert gamma > 0
                assert alpha > beta
                p, rcol  = M.shape
                reg   = 0
                for j in range(rcol):
                    if type(M) in {np.ndarray, np.matrix}:
                        c_norm = np.linalg.norm(M[:,j])
                    else:
                        c_norm = sp.sparse.linalg.norm(M[:,j])
                    pen1   = alpha*c_norm
                    pen2   = beta *c_norm + gamma
                    reg += min(pen1,pen2)
                return reg
            elif pen == 'n2cvxclasso' :
                alpha, beta, gamma = mu
                p, rcol  = M.shape
                reg   = 0
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
            elif pen == 'ridge':
                assert 0
                return (mu/2)*np.linalg.norm(M)**2                 
            elif pen == 'enet':
                if type(M) in {np.ndarray, np.matrix}:
                    return mu*(self.share_enet*np.sum(np.abs(M))     + ((1 - self.share_enet)/2)*np.linalg.norm(M)**2)
                else:
                    return mu*(self.share_enet*np.sum(np.abs(M))     + ((1 - self.share_enet)/2)*sp.sparse.linalg.norm(M)**2)
            elif pen == 'tv':
                return mu*np.sum(np.abs(M[1:] - M[:-1]))  
            elif pen == 'tf':
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
                self.nb_sentences.update({(var,key,post):self.nb_sentences.get((var,key,post),0)+1 for (var,key,post) in self.punished_coor}) 
                # Send the punished ones from police station to county prison 
                self.prison_coor.update( {(var,key,post):min(2**self.nb_sentences[var,key,post], 1e4)     for (var,key,post) in self.punished_coor})
                # Check 
                if EXTRA_CHECK and 0:
                    for (var,key,post) in self.punished_coor:
                        if not self.col_upd.get(self.param['data_cat'][key]):
                            if not np.linalg.norm(self.coef[var,key][:,post]) == 0: # Pas besoin d'être nuls pour être en prison
                                assert 0
                # Clear police station
                self.punished_coor = set()
                ### Prison 
                self.prison_coor = {(var,key,post):v-1 for (var,key,post),v in self.prison_coor.items() if v-1 >0}
                if EXTRA_CHECK and 0:
                    assert len(self.prison_coor) <= len(self.keys_upd)*self.k
                    for (var,key,post) in self.prison_coor:
                        if not self.param['tf_col_upd'].get(self.param['data_cat'][key]):
                            if not np.linalg.norm(self.coef[var,key][:,post]) == 0:
                                assert 0 # A column can already be in prison but have moved during recent iterations because of a too small mask
                self.dikt_masks = {coor:v for coor,v in {key:self.prison_masks(key,self.prison_coor) 
                                                         for key in self.keys_upd
                                                         }.items() 
                                                    if type(v) != type(slice(None))
                                                    or v       !=      slice(None)
                                                    }
                self.list_coor = [coor for coor in self.keys_upd if (coor not in self.prison_coor) 
                                                                 and not (       type(self.dikt_masks.get(coor)) == np.ndarray 
                                                                          and     self.dikt_masks.get(coor).ndim == 1 
                                                                          and self.dikt_masks.get(coor).shape[0] == 0
                                                                          ) 
                                                               ]
                self.fully_out = [coor for coor in self.keys_upd if coor not in self.list_coor]
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
                                 or np.linalg.norm(self.coef[coor[:2]][:,:,k] if coor[0] in {'Cu','Cv'} else self.coef[coor[:2]][:,k]) > 0  #  Do not give up column that moved while in prison  
                                 ]).astype(int)
            if   type(orig_mask) == type(slice(None)) and new_mask.shape[0]/self.k > self.prop_active_set:
                new_mask = orig_mask
            return new_mask
        else:
            return orig_mask
        
    #profile
    def precomputationsY(self, X, data = 'train'):
        assert data in {'train', 'test'}
        assert not self.active_gp
        X1 = {key:value for key, value in (self.X_train if data == 'train' else self.X_test).items() if '#' not in key}
        X1tX1 = X.get('X1tX1_'+data, {})
        X2 = {key:value for key, value in (self.X_train if data == 'train' else self.X_test).items() if '#' in key} 
        X1tX2 = X.get('X1tX2_'+data, {})
        X2tX1 = X.get('X2tX1_'+data, {})
        X2tX2 = X.get('X2tX2_'+data, {})
        for key, value in {**X1tX2, **X2tX1, **X2tX2}.items():
            assert type(value) in {np.ndarray, sp.sparse.csr_matrix}
            assert (key in X1tX2) + (key in X2tX1) + (key in X2tX2) == 1
        Y = self.Y_train if data == 'train' else self.Y_test
             
        # Compute everything with Y
        print(colored('compute YtY', 'red'))
        YtY      = Y.T.dot(Y)
        Y_sqfrob = np.trace(YtY)

        L = [YtY, Y_sqfrob]
        self.print_gso(L)
        path = self.param['path_outputs'] + 'Saved/Data/'

        print(colored('compute X1tY', 'red'))        
        try:
            X1tY = primitives.tf_data_load(path, self.dikt['primitives_'+data+'1'], self.param, 'X1tY_'+data, mod = 'dict_sp')# if bool(X2) else 'dict_np')
            for key in X1: 
                assert key in X1tY
        except Exception: 
            print(colored('compute X1tY', 'red'))
            X1tY  = {}
            for key in X1.keys():
                mask = self.mask.get(key, slice(None))
                X1tY[key] = sp.sparse.csc_matrix(np.zeros((X1[key].shape[1], Y.shape[1])))
                X1tY[key][:,mask] = X1[key].T.dot(Y[:,mask]) 
            if len(X1tY) < 1e3:
                try:
                    primitives.tf_data_save(path, self.dikt['primitives_'+data+'1'], self.param, X1tY, 'X1tY_'+data, mod = 'dict_sp')# if bool(X2) else 'dict_np')
                except (CalledProcessError, custex, TimeoutExpired, OSError):
                    pass
                except Exception as e:
                    raise e
        L = X1tY
        self.print_gso(L)
            
            
                    
        if bool(X2):
            print(colored('compute X2tY', 'red'))        
            try:
                X2tY = primitives.tf_data_load(path, self.dikt['primitives_'+data+'2'], self.param, 'X2tY_'+data, mod = 'dict_sp' if bool(X2) else 'dict_np')
            except Exception: 
                print(colored('compute X2tY', 'red'))
                X2tY = {}
                for key in X2.keys():
                    mask = self.mask.get(key, slice(None))
                    X2tY[key] = sp.sparse.csr_matrix(np.zeros((X2[key].shape[1], Y.shape[1])))
                    X2tY[key][:,mask] = X2[key].T.dot(Y[:,mask]) 
                if len(X2tY) < 1e3:
                    try:
                        primitives.tf_data_save(path, self.dikt['primitives_'+data+'2'], self.param, X2tY, 'X2tY_'+data, mod = 'dict_sp' if bool(X2) else 'dict_np')
                    except (CalledProcessError, custex, TimeoutExpired, OSError):
                        pass
                    except Exception as e:
                        raise e
            ### Check
            if EXTRA_CHECK:
                if len(X2tY) > 0:
                    key     = next(iter(X2tY.keys()))
                    mask    = self.mask.get(key, slice(None))
                    control = np.zeros((X2[key].shape[1], Y.shape[1]))
                    control[:,mask] = X2[key].T.dot(Y[:,mask]) 
                    if type(X2tY[key]) == sp.sparse.csr_matrix:
                        control = sp.sparse.csr_matrix(control)
                        assert np.linalg.norm(X2tY[key].data - control.data)  < 1e-8
                    else:
                        assert np.allclose(X2tY[key], control).all()           
            L = X2tY
            self.print_gso(L)

        if data == 'train':
            self.Y_train        = Y
            self.YtY_train      = YtY
            self.XtX_train      = {}
            self.XtY_train      = {}
            self.Y_sqfrob_train = Y_sqfrob
            if bool(X1tX1):
                self.XtX_train.update({**X1tX1})
                self.XtY_train.update({**X1tY})
            if bool(X2tX2):
                self.XtX_train.update({**X1tX2, **X2tX1, **X2tX2})
                self.XtY_train.update({**X2tY})

        elif data == 'test':
            self.Y_test        = Y
            self.YtY_test      = YtY
            self.XtX_test      = {}
            self.XtY_test      = {}
            self.Y_sqfrob_test = Y_sqfrob
            if bool(X1tX1):
                self.XtX_test.update({**X1tX1})
                self.XtY_test.update({**X1tY})
            if bool(X2tX2):
                self.XtX_test.update({**X1tX2, **X2tX1, **X2tX2})
                self.XtY_test.update({**X2tY})
    
    #profile
    def predict(self, coef = None, data = None, drop = {}, adjust_A = False, X_new = None, n_new = None, verbose = False):
        if verbose:
            print('predict {0}'.format(data))
        if hasattr(self, 'bfgs_long_coef'):
            if self.use_lbfgs_old: # Profiling lead to the following choice of implementations for the two setups
                assert 0, ('lbfgs_old not used anymore') # return bfgs_pred_old(self, self.bfgs_long_coef, data = data)
            else:
                return bfgs_pred(self, self.bfgs_long_coef, data = data)
        if   data == 'train':
            n = self.n_train
            X = self.X_train 
        elif data == 'test':
            n = self.n_test
            X = self.X_test
        else:
            assert X_new
            assert n_new
            n = n_new
            X = X_new
        if type(coef) == type(None):
            coef = self.coef
        pred2 = np.zeros((n, self.k))
        for ii, coor in enumerate(coef):
            if verbose:
                print('\r{0:5} / {1:5}'.format(ii, len(coef)), end = '')
            if len(coor) == 3:
                var, key, ind = coor
                cc            = coef[coor]
                if coor[0] == 'Cbm':
                    if   ('Cb',coor[1].split('#')[0],coor[2]) in coef:
                        mask = self.dikt_masks.get(('Cb',coor[1].split('#')[0],coor[2]), slice(None))
                    elif ('Cm',)+coor[1:] in coef:
                        mask = self.dikt_masks.get(('Cm',)+coor[1:], slice(None))
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
        if verbose:
            print('done' + ' '*20)
        if adjust_A and ('A','') in coef:
            pred2 /= (1 - coef['A',''])
        return pred2
    
    
    def print_fit_with_mean(self, data = 'train', mean = 'train'):
        if   data == 'train':
            Y = self.Y_train
            n = self.n_train
        elif data == 'test':
            Y = self.Y_test
            n = self.n_test
        if   mean == 'train':
            Y_mean = self.Y_train.mean(axis = 0)
        elif mean == 'test':
            Y_mean = self.Y_test.mean(axis = 0)
        rmse = nb_format.nb_round((0.5/n)*np.linalg.norm(Y - Y_mean)**2,3)
        print('fit', (data + ' ')[:5], 'with mean', (mean + ' ')[:5], ':', rmse)
        if data == 'train':
            self.normalized_variance_train = rmse
        elif data == 'test':            
            self.normalized_variance_test  = rmse


    def print_gso(self, lits)  :
        # gso = getsizeof
        # ie print the large parts of XtX
        for e in lits:
            continue
            if type(e) == np.ndarray:
                print('var', ' - ', 'size', '{:.3e}'.format(gso(e)/1e6), 'MBytes', ', ', 'shape', e.shape)
            if type(e) == dict:
                print('var', ' - ', 'size', '{:.3e}'.format(gso(e)/1e6), 'MBytes')
                print([(k, v.shape) for k, v in e.items()])

        
    def print_info(self, ):
        print('\n'+
              'iter = ',  '{:<6}'.format(self.iteration),
              '& dec = ', '{:<6.6}'.format(nb_format.nb_round(self.decr_obj)), 
              '& tol = ', '{:<6.6}'.format(self.tol), 
              '& obj = ', '{:<6.6}'.format(nb_format.nb_round(self.cur_obj_train)), 
              end = '\n'
              )
        cat_of_prisoners = sorted(set([self.param['data_cat'][coor[1]] for coor in self.prison_coor]))
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
                'iter = {:<6}' .format(self.iteration),
                'nfit_train  = {:10.10}'.format(nb_format.nb_round(self.fit_train[self.iteration-1]/self.normalized_variance_train, nb_app = n_app)), 
                'dec  = {:10.10}'.format(nb_format.nb_round(self.decr_obj,                    nb_app = n_app)) \
                if self.decr_obj != self.decr_obj_0 else 'st/ch : '+str(self.nb_steps)+'/'+str(self.flag_stopping_criteria),
                ]
        if self.compute_test:
            info.append('nfit_test = {:10.10}'.format(nb_format.nb_round(self.fit_test[self.iteration-1]/self.normalized_variance_test, nb_app = n_app)))
        progress.progress_print([self.iteration, self.max_iter], *info)


    def prox_op(self, X, pen, eta, mu, flag = False, coef_zero = None):
        # Proximal operator
        if pen == '':    
            X_tmp = X
        elif pen in {'rsm', 'r2sm', '0r2sm'}:    
            X_tmp = X
        elif pen == 'lasso' :
            X_tmp     = prox_L1(X, eta*mu, coef_zero = coef_zero)
        elif pen == 'rlasso' :
            X_tmp     = prox_row_lasso(X, eta*mu)
        elif pen == 'classo' :
            X_tmp     = prox_col_lasso(X, eta*mu)
        elif pen == 'ncvxclasso' :
            assert type(mu) == tuple
            X_tmp     = prox_ncvx_classo(X, tuple([eta*e for e in mu]), coef_zero = coef_zero)
        elif pen == 'n2cvxclasso' :
            X_tmp     = prox_ncvx_classo_v2(X, eta, mu, coef_zero = coef_zero)
        elif pen == 'ridge' :
                X_tmp     = prox_L2(X, mu)
        elif pen == 'enet' :
                X_tmp     = prox_enet(X, eta*mu, self.share_enet)
        elif pen == 'tv':
            X_tmp = prox_TV(X, eta*mu)
        elif pen == 'tf' :
            X_tmp = prox_TF(X, eta*mu)
        else:
            raise ValueError('bad penalization')
        return X_tmp
                
                
    def size_coef(self, ):
        self.memory_size_coef = {}
        for cat in list(self.param['data_cat'].values()):
            self.memory_size_coef[cat] = 0
            for (var,key), v in self.coef.items():
                if self.param['data_cat'][key] == cat:
                    self.memory_size_coef[cat] += gso(v)/1e6
        return self.memory_size_coef 
    
    
    def stopping_criteria(self, dikt_fit_grad):
        # Computant the quantities relevant for the stopping criteria
        small_decrease, early_stop, norm_grad_small, max_iter_reached, all_dead = [0]*5
        if not self.bcd:
            small_decrease = (self.decr_obj < self.tol)            
        # Stopping criteria
        if self.nb_steps >= self.flag_stopping_criteria:
            self.flag_stopping_criteria += self.epoch_stopping_criteria
            # Evolution of obj test
            if self.compute_test and self.iteration >= self.flag_compute_test :
                old_mean_ft      = self.fit_test   [self.iteration - 2*self.epoch_stopping_criteria:self.iteration  - self.epoch_stopping_criteria].mean()
                new_mean_ft      = self.fit_test   [self.iteration - self.epoch_stopping_criteria:self.iteration].mean()
                old_mean_gp_ft   = self.fit_gp_test[self.iteration - 2*self.epoch_stopping_criteria:self.iteration  - self.epoch_stopping_criteria].mean()
                new_mean_gp_ft   = self.fit_gp_test[self.iteration - self.epoch_stopping_criteria:self.iteration].mean()
                early_stop       = (old_mean_ft + old_mean_gp_ft < new_mean_ft + new_mean_gp_ft) and self.param['tf_early_stop_test']#and not (self.vr) 
            # Same thing individually
            if self.compute_test and self.iteration >= self.flag_compute_ind_test:
                if self.param.get('tf_early_stop_ind_test'):
                    self.flag_compute_ind_test += self.epoch_compute_ind_test
                    old_ind_ft      = self.fit_ind_test[self.iteration - 2*self.epoch_compute_ind_test:self.iteration  - self.epoch_compute_ind_test].mean(axis = 0)
                    new_ind_ft      = self.fit_ind_test[self.iteration -   self.epoch_compute_ind_test:self.iteration].mean(axis = 0)
                    if not self.active_gp:
                        early_stop_ind  = (old_ind_ft < new_ind_ft)#and not (self.vr) 
                        for pp in range(self.k):
                            if early_stop_ind[pp]:
                                self.life_sentences.add(pp)
                        all_dead = (len(self.life_sentences) == self.k)
            # Small decrease
            if self.bcd:
                old_mean       = self.obj_train[self.iteration - 2*self.epoch_stopping_criteria:self.iteration  - self.epoch_stopping_criteria].mean()
                new_mean       = self.obj_train[self.iteration - self.epoch_stopping_criteria:self.iteration].mean()
                self.decr_obj  = old_mean - new_mean 
                small_decrease = (self.decr_obj < self.tol) and self.param['tf_early_small_decrease']
            # Small grad
            norm_grad        = self.compute_grad_norm(dikt_fit_grad)
            norm_grad_small  = (norm_grad < self.norm_grad_min) and not (self.bcd ) and self.param['tf_early_stop_small_grad']#or self.vr)
            # Duality gap
            if self.dual_gap:
                assert 0
                grad = self.compute_grad(self.coef)
                pred = self.predict()
                assert pred.shape == self.Y_train.shape
                if 'Bsp' in self.config_coef:
                    self.gap_B.append(self.duality_gap(pred, grad, 'Bsp'))
       
        if early_stop      : print('\nEARLY STOPPING : ' , 'old mean', old_mean_ft, ' - recent mean', new_mean_ft)
        if small_decrease  : print('\nSMALL DECREASE : ' , nb_format.nb_round(self.decr_obj))
        if norm_grad_small : print('\nNORM GRAD SMALL : ', norm_grad)    
        return small_decrease, early_stop, norm_grad_small, all_dead
        

    def update_coef(self, coef_tmp, d_masks, new_ind_slope = {}, new_ind_offset = {}):
        for coor_upd in coef_tmp:
            if type(coef_tmp[coor_upd]) == np.matrix:
                coef_tmp[coor_upd] = np.asarray(coef_tmp[coor_upd])
            assert type(coef_tmp[coor_upd]) == np.ndarray
            if coor_upd[0] == 'Cbm':
                if   ('Cb',coor_upd[1].split('#')[0],coor_upd[2]) in coef_tmp:
                    mask = d_masks.get(('Cb',coor_upd[1].split('#')[0],coor_upd[2]), slice(None))
                elif ('Cm',)+coor_upd[1:] in coef_tmp:
                    mask = d_masks.get(('Cm',)+coor_upd[1:], slice(None))
                else:
                    assert 0
            elif coor_upd[0] == 'Cuv':
                mask = d_masks.get(('Cu',)+coor_upd[1:], slice(None))
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
            var,key = coor_upd[:2]
            assert var in {'A', 'Bsp', 'Blr', 'bu', 'bv', 'Csp', 'Cuv', 'Cu', 'Cv', 'Cb', 'Cm', 'Cbm'}
            if self.active_set and not self.param['tf_col_upd'].get(self.param['data_cat'][key]):
                orig_mask = self.orig_masks.get(coor_upd, slice(None))
                ind_posts = mask if type(mask) == np.ndarray else (orig_mask if type(orig_mask) == np.ndarray else np.arange(self.k))
                if EXTRA_CHECK and 0: # with prop active set, some coor can be both in prisons and active
                    for jj in ind_posts:
                        if (var,key,jj) in self.prison_coor:
                            assert 0
                norm_diff    = np.linalg.norm(coef_tmp[coor_upd] - (self.coef[var,key][:,:,mask] if var in {'Cu','Cv'} else self.coef[var,key][:,mask]), axis = tuple(k for k in range(coef_tmp[coor_upd].ndim - 1)))
                # One could evaluate fit_test for each post and jail accordingly but too expansive : rather do col_upd
                convicts     = ind_posts[norm_diff == 0]
                convicts_tmp = np.arange(ind_posts.shape[0])[norm_diff == 0]
                for i, r in enumerate(convicts):
                    if not np.linalg.norm(self.coef[var,key][:,:,r] if var in {'Cu','Cv'} else self.coef[var,key][:,r]) < np.finfo(float).eps:
                        if np.linalg.norm(coef_tmp[coor_upd][:,:,convicts_tmp[i]] if var in {'Cu','Cv'} else coef_tmp[coor_upd][:,convicts_tmp[i]]) == 0:
                            assert 0
                    if (*coor_upd[:2], r) not in self.prison_coor:
                        self.punished_coor.add((*coor_upd[:2], r))
                #berliners = ind_post[norm_diff != 0]
            if EXTRA_CHECK:
                if var == 'bv':
                    assert self.coef[var,key][mask].shape     == coef_tmp[coor_upd].shape
                elif var in {'Cu','Cv'}:
                    assert self.coef[var,key][:,:,mask].shape == coef_tmp[coor_upd].shape
                else:
                    assert self.coef[var,key][:,mask].shape   == coef_tmp[coor_upd].shape
                self.coef_prec[var,key] = self.coef[var,key].copy()
            shape_before = self.coef[var, key].shape
            if var == 'bv':
                self.coef[var,key][mask]      = coef_tmp[coor_upd]
            if var in {'Cu','Cv'}:
                self.coef[var, key][:,:,mask] = coef_tmp[coor_upd]
            else:
                self.coef[var, key][:,mask]   = coef_tmp[coor_upd]
            assert shape_before == self.coef[var, key].shape
            if EXTRA_CHECK:
                if   var == 'bu':
                    assert self.coef[var,key].shape == coef_tmp[coor_upd].shape
                    assert np.allclose(self.coef[var,key], coef_tmp[coor_upd])
                elif var == 'bv':
                    assert self.coef[var,key][mask].shape == coef_tmp[coor_upd].shape
                    assert np.allclose(self.coef[var,key][mask], coef_tmp[coor_upd])
                elif var in {'Cu','Cv'}:
                    assert self.coef[var,key][:,:,mask].shape == coef_tmp[coor_upd].shape
                    assert np.allclose(self.coef[var,key][:,:,mask], coef_tmp[coor_upd])
                else:
                    assert self.coef[var,key][:,mask].shape == coef_tmp[coor_upd].shape
                    assert np.allclose(self.coef[var,key][:,mask], coef_tmp[coor_upd])
                if self.active_set and not self.param['tf_col_upd'].get(self.param['data_cat'][key]):
                    for i, rr in enumerate(convicts):
                        if not np.allclose(self.coef     [var,key][:,:,rr] if var in {'Cu','Cv'} else self.coef[var,key][:,rr], 
                                           self.coef_prec[var,key][:,:,rr] if var in {'Cu','Cv'} else self.coef[var,key][:,rr],
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
            if var in {'Cb','Cm'}:
                mm = self.dikt_masks.get((var, key, ind), slice(None))
                if var == 'Cb':
                    for var2, key2 in self.coef:
                        if var2 == 'Cbm' and key2.split('#')[0] == key:
                            if ('Cbm', key2) in self.coef:
                                coef['Cbm', key2, ind] = np.einsum('pk,qk->pqk',
                                                                   coef[var, key, ind],
                                                                   self.coef['Cm', key2][:,mm],
                                                                   ).reshape((-1, coef[var, key, ind].shape[1]))
                                assert coef['Cbm', key2, ind].shape == self.coef['Cbm', key2][:,mm].shape
                elif var == 'Cm':
                    assert ('Cbm',key) in self.coef
                    if ('Cbm',key) in self.coef:
                        coef['Cbm', key, ind] = np.einsum('pk,qk->pqk',
                                                          self.coef['Cb', key.split('#')[0]][:,mm],
                                                          coef[var, key, ind],
                                                          ).reshape((-1, coef[var, key, ind].shape[1]))
                        assert coef['Cbm', key, ind].shape == self.coef['Cbm', key][:,mm].shape
        return coef   
                        
    
    def update_Cuv(self, coef):
        # Update the Cuv coefficient matrics
        # It corresponds to the low-rank structural constraint
        for var, key, ind in list(coef.keys()):
            if var in {'Cu','Cv'}:
                mm = self.dikt_masks.get((var, key, ind), slice(None))
                if var == 'Cu':
                    coef['Cuv', key, ind] = np.einsum('prk,qrk->pqk',
                                                      coef[var, key, ind],
                                                      self.coef['Cv', key][:,:,mm],
                                                      ).reshape((-1, coef[var, key, ind].shape[2]))
                elif var == 'Cv':
                    coef['Cuv', key, ind] = np.einsum('prk,qrk->pqk',
                                                      self.coef['Cu', key][:,:,mm],
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
            mm = self.dikt_masks.get(('bv', key), slice(None))
            if var == 'bu':
                assert not ind
                if svd or ('bv', None) not in coef:
                    coef['bv', key] = self.best_orth_bv(coef, mask = mm)
                else:
                    assert 0 # Should be ckecked before use
                    Q, R = self.best_qr(coef, mask = mm)
                    coef['bv', key] = Q
                coef['Blr', key] = coef[var, key, ind] @ coef['bv', key].T
        return coef

    
    def warm_start(self, ):
        if (not self.given_coef) and self.param['tf_try_ws'] and (not (self.param['tf_plot'] and self.param['any_plot'])): # No coef has been directly given 
            wanted      = self.dikt['model_wo_hyperp']
            list_wanted = [e for a in wanted.split('/') for e in a.split('_')]
            #try:
            if False:
                primitives.tf_data_sync(self.path_betas, 
                                        [self.dikt['model_wo_hyperp'].replace('_apr', ''), self.dikt['model_wo_hyperp']],
                                        self.param, 
                                        mod = 'dict_tup',
                                        )
            try:
                FF_candidates  = sorted(os.listdir(self.param['path_outputs'] + 'Saved/Betas/'))
            except FileNotFoundError:
                FF_candidates = []
            for ii, FF in enumerate(FF_candidates):
                try:
                    GG_candidates  = sorted(os.listdir(self.param['path_outputs'] + 'Saved/Betas/' + FF))
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
                        list_hyperprm = os.listdir(self.param['path_outputs'] + 'Saved/Betas/' + cand + '/')
                        if not list_hyperprm:
                            continue
                        str_warm_model = cand + '/' + list_hyperprm[0]
                        try:
                            coef       = primitives.tf_data_load(self.path_betas, 
                                                                 str_warm_model, 
                                                                 self.param, 
                                                                 'coef', 
                                                                 mod = 'dict_tup',
                                                                 )
                            keys_upd   = primitives.tf_data_load(self.path_betas, 
                                                                 str_warm_model, 
                                                                 self.param, 
                                                                 'keys_upd', 
                                                                 mod = 'json',
                                                                 )
                            if sorted(keys_upd) == sorted(self.keys_upd):
                                assert sorted(coef.keys()) == sorted(self.coef.keys()), set(coef.keys()).symmetric_difference(set(self.coef.keys()))
                                self.coef = coef
                            else:
                                raise custex
                                for k in coef:
                                    if k in self.coef:
                                        assert self.coef[k].shape == coef[k].shape
                                        self.coef[k] = coef[k]
                        except custex as e:
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
        xxx = (1/n)*XtX[key+'@'+key] @ (coef[('Blr',*coor_upd[1:])] if ('Blr',*coor_upd[1:]) in coef else coef['Blr',coor_upd[1]][:,mask])# VtV = I
        if gp_pen:
                xxx = gp_pen * xxx @ MMt[mask][:,mask]
        return xxx
    
    
    def xtra_part_cb(self, n, XtX, coef, coor_upd, mask, gp_pen, MMt):
        cbb =   (1/n)*XtX[coor_upd[1]+'@'+coor_upd[1]] @ (coef[coor_upd] if coor_upd in coef else coef[coor_upd[:2]][:,mask])
        cbm = 2*(1/n)*np.sum([XtX[coor_upd[1]+'@'+keybm] @ (coef['Cbm', keybm, coor_upd[-1]] if ('Cbm', keybm, coor_upd[-1]) in coef else coef['Cbm', keybm][:,mask])
                              for keybm in self.keys['Cbm'] if keybm.split('#')[0] == coor_upd[1]
                              if coor_upd[1]+'@'+keybm in XtX
                              ], 
                              axis = 0,
                              )
        if not gp_pen:
            cc = (coef[coor_upd] if coor_upd in coef else coef[coor_upd[:2]][:,mask])
            cmm = np.zeros(cc.shape)
            for keybm2 in self.keys['Cbm']:
                if keybm2.split('#')[0] == coor_upd[1]:
                    for keybm1 in self.keys['Cbm']:
                        if keybm1.split('#')[0] == coor_upd[1]:
                            if keybm1+'@'+keybm2 in XtX:
                                cmm += (1/n)*np.einsum('pqk,qk->pk', 
                                                       (XtX[keybm1+'@'+keybm2] @ (coef['Cbm', keybm2, coor_upd[-1]]
                                                        if ('Cbm', keybm2, coor_upd[-1]) in coef 
                                                        else 
                                                        coef['Cbm', keybm2][:,mask])).reshape((-1,
                                                                                               coef.get(('Cm', keybm1), self.coef['Cm', keybm1])[:,mask].shape[0], 
                                                                                               coef.get(('Cm', keybm1), self.coef['Cm', keybm1])[:,mask].shape[1],
                                                                                               )),
                                                       coef.get(('Cm', keybm1), self.coef['Cm', keybm1])[:,mask],
                                                       )
        if gp_pen:
            cbb = gp_pen * cbb @ MMt[mask][:,mask]
            if cbm.ndim > 0: # Check that cbm is not zero ie the list was not empty
                cbm = gp_pen * cbm @ MMt[mask][:,mask]
            else:
                assert cbm == 0
            cmm =   (1/n)*np.sum([np.einsum('pqk,qk->pk', 
                                            (XtX[keybm1+'@'+keybm2] @ (coef['Cbm', keybm2, coor_upd[-1]] if ('Cbm', keybm2, coor_upd[-1]) in coef else coef['Cbm', keybm2][:,mask])@ MMt[mask][:,mask]).reshape((-1, 
                                                                                                                                                                                                                 coef.get(('Cm', keybm1), self.coef['Cm', keybm1])[:,mask].shape[0], 
                                                                                                                                                                                                                 coef.get(('Cm', keybm1), self.coef['Cm', keybm1])[:,mask].shape[1],
                                                                                                                                                                                                                 )),
                                            coef.get(('Cm', keybm1), self.coef['Cm', keybm1])[:,mask],
                                            )
                                  for keybm2 in self.keys['Cbm'] if keybm2.split('#')[0] == coor_upd[1]
                                  for keybm1 in self.keys['Cbm'] if keybm1.split('#')[0] == coor_upd[1]
                                  if keybm1+'@'+keybm2 in XtX
                                  ],
                                 axis = 0, 
                                )
        xxx = cbb + cbm + cmm
        return xxx
    
    
    def xtra_part_cm(self, n, XtX, coef, coor_upd, mask, gp_pen, MMt):
        ccc = (1/n)*XtX[coor_upd[1]+'@'+coor_upd[1]] @ (coef[('Cbm',*coor_upd[1:])] if ('Cbm',*coor_upd[1:]) in coef else coef['Cbm',coor_upd[1]][:,mask])#[:,mask_right]
        if gp_pen:
                ccc = gp_pen * ccc @ MMt[mask][:,mask]
        xxx = np.einsum('pqk,pk->qk',
                        ccc.reshape((coef.get(('Cb', coor_upd[1].split('#')[0]), self.coef['Cb', coor_upd[1].split('#')[0]])[:,mask].shape[0],
                                     -1,
                                     coef.get(('Cb', coor_upd[1].split('#')[0]), self.coef['Cb', coor_upd[1].split('#')[0]])[:,mask].shape[1],
                                     )),
                        coef.get(('Cb', coor_upd[1].split('#')[0]), self.coef['Cb', coor_upd[1].split('#')[0]])[:,mask],
                        )
        return xxx
 
    
    def xtra_part_cl(self, n, XtX, coef, coor_upd, mask, gp_pen, MMt):
        xxx = (1/n)*XtX[coor_upd[1]+'@'+coor_upd[1]] @ (coef[coor_upd] if coor_upd in coef else coef[coor_upd[:2]][:,mask])
        if gp_pen: 
            xxx = gp_pen * xxx @ MMt[mask][:,mask]
        assert xxx.shape == (coef[coor_upd] if coor_upd in coef else coef[coor_upd[:2]][:,mask]).shape, (xxx.shape, (coef[coor_upd] if coor_upd in coef else coef[coor_upd[:2]][:,mask]).shape, coor_upd, mask)
        return xxx
    
    
    def xtra_part_cu(self, n, XtX, coef, coor_upd, mask, gp_pen, MMt):
        ccc = (1/n)*XtX[coor_upd[1]+'@'+coor_upd[1]] @ (coef[('Cuv',*coor_upd[1:])] if ('Cuv',*coor_upd[1:]) in coef else coef['Cuv',coor_upd[1]][:,mask])#[:,mask_right]
        if gp_pen:
                ccc = gp_pen * ccc @ MMt[mask][:,mask]
        xxx = np.einsum('pqk,qrk->prk',
                        ccc.reshape((-1,
                                     coef.get(('Cv', coor_upd[1]), self.coef[('Cv', coor_upd[1])])[:,:,mask].shape[0],
                                     coef.get(('Cv', coor_upd[1]), self.coef[('Cv', coor_upd[1])])[:,:,mask].shape[2],
                                     )),
                        coef.get(('Cv', coor_upd[1]), self.coef[('Cv', coor_upd[1])])[:,:,mask],
                        )
        return xxx
    
    
    def xtra_part_cv(self, n, XtX, coef, coor_upd, mask, gp_pen, MMt):
        ccc = (1/n)*XtX[coor_upd[1]+'@'+coor_upd[1]] @ (coef[('Cuv',*coor_upd[1:])] if ('Cuv',*coor_upd[1:]) in coef else coef['Cuv',coor_upd[1]][:,mask])
        if gp_pen:
                ccc = gp_pen * ccc @ MMt[mask][:,mask]
        xxx = np.einsum('pqk,prk->qrk',
                        ccc.reshape((coef.get(('Cu', coor_upd[1]), self.coef[('Cu', coor_upd[1])])[:,:,mask].shape[0],
                                     -1,
                                     coef.get(('Cu', coor_upd[1]), self.coef[('Cu', coor_upd[1])])[:,:,mask].shape[2],
                                     )),
                        coef.get(('Cu', coor_upd[1]), self.coef[('Cu', coor_upd[1])])[:,:,mask],
                        )
        return xxx
                    