'''
Created on 5.3.2021

@author: sofievm
'''

import numpy as np
import numpy.f2py
from scipy import optimize, signal
from toolbox import optimization_tools as opt, gridtools, stations
from toolbox import supplementary as spp, silamfile, drawer
from toolbox import MyTimeVars as MTV, structure_func
import os, shutil, glob
import datetime as dt
import netCDF4 as netcdf

#
# MPI may be tricky: in puhti it loads fine but does not work
# Therefore, first check for slurm loader environment. Use it if available
#
try:
    # slurm loader environment
    mpirank = int(os.getenv("SLURM_PROCID",None))
    mpisize = int(os.getenv("SLURM_NTASKS",None))
    chMPI = '_mpi%03g' % mpirank
    comm = None
    print('SLURM pseudo-MPI activated', chMPI, 'tasks: ', mpisize)
except:
    # not in pihti - try usual way
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        mpisize = comm.size
        mpirank = comm.Get_rank()
        chMPI = '_mpi%03g' % mpirank
        print ('MPI operation, mpisize=', mpisize, chMPI)
    except:
        print ("mpi4py failed, single-process operation")
        mpisize = 1
        mpirank = 0
        chMPI = ''
        comm = None




##############################################################################
#
# Class for the linear space-resolving model
# See Notebook 12, pp.4-12
# The brute-force approach when weighting coefs of models a_k = a_k(i,j,t) is impractical
# since optimization of the whole map of coefs for ~10 models is unfeasible.
# Two options exist:
# - multi-subsetting when regions is scanned by a small moving window. Inside the window,
#   we solve the uniform linear problem.
# - weights are assumed to be "some" prescribed functions (i.j), e.g. Fourier of Lagrange 
#   seriess, which parameters are optimized.   
#
##############################################################################

class linear_space_resolve():
    #
    # Class of space-resolving fusion methods based on linear regression.
    # The model weight vary in space.
    #
    def __init__(self, optim_descr, tStart, data_shape, log):
        self.optim_method = optim_descr[0] 
        self.constrain = optim_descr[1]
        self.abbrev = optim_descr[2]
        self.fit_function = {'unconstrained' : self.fit_unconstrained,
                             'nonneg' : self.fit_nonneg_weights} [self.constrain]
        self.nModels, self.nTimes, self.nStat = data_shape
        self.tStart = tStart
        self.log = log


    #==================================================================

    def fit_multisubset(self):
        #
        # Scans the domain with a moving window. For each window, a standard 
        # spatially-uniform problem is solved. After that, a unified map of 
        # coefficients is sreated by inetr/extrapolating the coefficients of 
        # the windows
        #
        











    #==================================================================
    
    def prepare(self, chStF_title, chMdlFNmTempl, mdlVariable, patch_zero, tsObs, StF_FNm, ifRefreshStF):
        #
        # Prepares the internal variables, first of all, the structure function
        # Note that the functions can be made only when the models are ranked by their 
        # information content and quality: the function is made using the rank-1 model
        #
        # Create the function instance
        self.StF = structure_func.structure_function(chStF_title, self.gridSTF_factor, 
                                                     self.filterRad, self.log)
        #
        # Compute it or read. Here we always need tsMatrix version
        # Also, we always need a single time step, so no issues with memory
        #
        if ifRefreshStF:
            # calculate...
            self.StF.make_StF_4_tsMatrix(chMdlFNmTempl, mdlVariable, patch_zero, tsObs)
            # store
            self.StF.to_netcdf(StF_FNm, True)  # close the file
        else:
            # read existing
            self.StF.from_netcdf(StF_FNm)
            if np.any(np.array(self.StF.stations) != np.array(tsObs.stations)):
                self.log.log('Old StFunction is made for different stations.')
                self.log.log('Stored statoins:' + '\n'.join(str(s) for s in self.StF.stations))
                self.log.log('tsObs statoins:' + '\n'.join(str(s) for s in tsObs.stations))
                raise ValueError
    

    #==================================================================

    def fit(self, poolMdl_flat, tsmObs_flat, kernel_flat,
            train_mdls_f, train_obs_f, train_krnl_f, 
            test_mdl_f, test_obs_f, test_krnl_f):  #tsmObs, poolMdl):
        #
        # Finds the coefficients for the spatially uniform linear fusion:
        # C_fused = a_0 + sum(a_i * C_n_i), i = 1..M
        # here C_m_i is map of i-th model, a_0, a_i are constant scalars
        # This sub finds a_0, a_i (i=1..M) minimising RMSE.
        #
        # We can require constrained minimization with non-negative 
        # weights - or let the models fly freely.
        #
        self.log.log('fit received %i models, %i times, %i stations' % 
                     (self.nModels, self.nTimes, self.nStat))
        return self.fit_function(self.optim_method, poolMdl_flat, tsmObs_flat, kernel_flat,
                           train_mdls_f, train_obs_f, train_krnl_f, 
                           test_mdl_f, test_obs_f, test_krnl_f)


    #==================================================================
    
    def fit_unconstrained(self, chFit, poolMdl_flat, tsmObs_flat, kernel_flat,
                          train_mdls_f, train_obs_f, train_krnl_f, 
                          test_mdl_f, test_obs_f, test_krnl_f):
        #
        # We can use several types of regularization. 
        # Two are simple: Ridge and Lasso, there is also 
        # unregularised MLR. This sub evaluates them all.
        # Two-step approach.
        # 1. Take some times for testng and identify the regularization weight.
        # 2. Having the weight fixed, identify the fitting for all possible dates
        #
        tsmObsOK = np.isfinite(tsmObs_flat)
        #
        # Initialize the fitting model
        #
        self.log.log(chFit)
        fuse_mdl = opt.linregr_regularised_model(chFit, opt.alpha_linreg_rmse4min, 100, self.log)
        #
        # Step.1. Identify the regularization weight
        #
        # The optimal criterion function: can be correlation (linreg_corr4min) or RMSE 
        # (linreg_rmse4min). 
        # Have to reshape arrays flattening time and stations and cutting out mean models
        #
        fuse_mdl.get_regulariser_weight_and_fit(train_mdls_f, train_obs_f, train_krnl_f, 
                                                test_mdl_f, test_obs_f, test_krnl_f)
#        self.log.log('%s regularization weight is %g' % (chFit, fuse_mdl.alpha))
        alpha_Tikhonov = fuse_mdl.verify_regularization(train_mdls_f, train_obs_f, train_krnl_f, 
                                                        test_mdl_f, test_obs_f, test_krnl_f)
        #
        # Step 2. Final fitting
        #
        # The model rgularization weight identified, if any, need to make the final fit
        # for the whole time period
        #
        self.out = fuse_mdl.fit(poolMdl_flat[:,tsmObsOK].T, 
                                tsmObs_flat[tsmObsOK].T, kernel_flat[tsmObsOK]) # predictors, answers
        self.out.x = np.zeros(shape=(self.out.coef_.shape[0]+1), dtype=np.float32)
        self.out.x[0] = self.out.intercept_
        self.out.x[1:] = self.out.coef_[0:]
        #
        # Whatever will be the model applicatoin, we will make use of the prediction for the 
        # same dataset as given here
        #
#        self.log.log(self.abbrev + ' verifies with ' +  str(fuse_mdl.clf.x))
#        for i in range(3):
#            self.log.log('Models for site ' + str(i) + str(poolMdl_flat[:,i]))
        
        fuse_predict = np.reshape(fuse_mdl.predict(poolMdl_flat.T), (self.nTimes, self.nStat))
        
#        self.log.log('Verification for 3 sites:' + str(fuse_predict[1,:3]))
        
        diff = np.subtract(np.ndarray.flatten(fuse_predict)[tsmObsOK], tsmObs_flat[tsmObsOK])
#        corr = np.corrcoef(np.ndarray.flatten(fuse_predict)[tsmObsOK], tsmObs_flat[tsmObsOK])[0,1]
        corr = spp.nanCorrCoef(np.ndarray.flatten(fuse_predict)[tsmObsOK], tsmObs_flat[tsmObsOK])
        bias = np.mean(diff * np.sqrt(kernel_flat[tsmObsOK]))
        rmse = np.sqrt((np.square(diff) * kernel_flat[tsmObsOK]).mean())
        self.log.log('Model %s regularization %g, rmse: %g, bias %g, corr %g' % 
                     (chFit, fuse_mdl.alpha, rmse, bias, corr))
        self.log.log('Coefficients %s: a0 %g, weights %s' % 
                     (chFit, fuse_mdl.clf.intercept_, ' '.join([str(v) for v in fuse_mdl.clf.coef_])))
    
        return (self, fuse_predict)
    

    #==================================================================
    
    def fit_nonneg_weights(self, chFit, poolMdl_flat, tsmObs_flat, kernel_flat,
                           train_mdls_f, train_obs_f, train_krnl_f, 
                           test_mdl_f, test_obs_f, test_krnl_f):
        #
        # Linear regression with constrained weights (non-negatives)
        # Cannot solve directly, have to iterate. scipy has nesessary tools but regularization
        # has to be made explicitly. Two options:
        # - introduce in-full regularization of Tikhonov and low-pass filter types
        # - use L-curve principle and truncated iterations
        # The explicit option requires reasonable algorithm to get the regularizatoin weight
        # The truncated iterations require full optimization trajectory 
        #
        # Eliminate occasoinal nans from obs
#        tsmObs_flat = np.ndarray.flatten(poolObs)
        tsmObsOK = np.isfinite(tsmObs_flat)
        #
        # Two ways of regularising: explicit applicatoin of Tikhonov and low-pas regurizations
        # or truncated iterations of the L-curve type. 
        #
        self.log.log(chFit)
        if chFit.upper() == 'MLR_REGEXPL'.upper():
            #
            # Explicit reguarization: split the set along time axis, find regularization coefs.,
            # make main fit
            arCoefs = np.ones(shape=(self.nModels+1)) * 1./self.nModels   # prepare the array
            arCoefs[0] = 0.0    # intercept
            #
            # The regularising weight will be determined for Ridge unconstrained minimization - 
            # the regulariser there is the same Tikhonov term with C0_Tikhonov=0. The difference
            # that the negative model weights are allowed but this should not be the problem: our
            # task is to eliminate the non-informative models.
            #
            fuse_mdl = opt.linregr_regularised_model('RIDGE', opt.alpha_linreg_rmse4min, 
                                                     100, self.log)
            fuse_mdl.get_regulariser_weight_and_fit(train_mdls_f, train_obs_f, train_krnl_f, 
                                                    test_mdl_f, test_obs_f, test_krnl_f)
            alpha_Tikhonov = fuse_mdl.verify_regularization(train_mdls_f, train_obs_f, train_krnl_f, 
                                                            test_mdl_f, test_obs_f, test_krnl_f)
            self.log.log('%s regularization weight is %g based on Ridge scheme' % 
                         (chFit, alpha_Tikhonov))
            C0_Tikhonov = 0.0
            alpha_low_pass = 0.0  # The low-pass term is null for a while 
            C_prev = 0.0
            #
            # Final fit for the whole dataset with identified regularization weights
            #
            self.out = optimize.minimize(opt.basic_rmse4min, arCoefs, 
                                         args=(alpha_Tikhonov, C0_Tikhonov, alpha_low_pass, C_prev,
                                               tsmObs_flat[tsmObsOK], 
                                               poolMdl_flat[:,tsmObsOK], 
                                               kernel_flat[tsmObsOK], False), 
                                         bounds = [(0.0,None)]*(self.nModels+1),
                                         method='L-BFGS-B')
                            # jac, hess, hessp, bounds, constraints, tol, callback, options)

        elif chFit.upper() == 'MLR_REGLCRV':
            self.log.log('TRUNCATED_ITERATIONS are not yet implemented')
            return None
        else:
            self.log.log('Unknown regularization_type: ' + chFit)
            return None
        #
        # Make prediction for all stations, compute and report scores
        #
        rmse_with_reg = opt.basic_rmse4min(self.out.x, 
                                           alpha_Tikhonov, C0_Tikhonov, alpha_low_pass, C_prev, 
                                           tsmObs_flat[tsmObsOK], 
                                           poolMdl_flat[:,tsmObsOK], 
                                           False)
        predict = self.predict(poolMdl_flat)   # average models out
        diff = np.subtract(predict[tsmObsOK], tsmObs_flat[tsmObsOK])
#        corr = np.corrcoef(predict[tsmObsOK], tsmObs_flat[tsmObsOK])[0,1]
        corr = spp.nanCorrCoef(predict[tsmObsOK], tsmObs_flat[tsmObsOK])
        bias = np.mean(diff)
        rmse = np.sqrt(np.square(diff).mean())
        
        self.log.log('Model nonneg_MLR regWeights: Tikhonov (%g,%g), low-pass %g; ' % 
                     (alpha_Tikhonov, C0_Tikhonov, alpha_low_pass) +
                     ' rmse %g, bias %g, corr %g, rmse_regul %g' % 
                     (rmse, bias,  corr, rmse_with_reg))
        self.log.log('Coefficients nonneg_MLR: a0 %g, weights %s' % 
                     (self.out.x[0], ' '.join([str(v) for v in self.out.x[1:]])))
        return (self, np.ndarray.reshape(predict, (self.nTimes,self.nStat)))
    
    
    #==================================================================

    def predict(self, poolMdl):
        #
        # Makes the prediction with the given set of models and weighting coefficients
        # that are supposed to be identified by now
        #
#        self.log.log(self.abbrev + ' predicts with ' + str(self.out.x))
#        self.log.log('Models for 3 sites ' + str(poolMdl[:,1,:3]))
#        self.log.log('Prediction ' + str((np.sum(np.array([poolMdl[iMdl] * self.out.x[iMdl+1] 
#                                                          for iMdl in range(self.nModels)]),axis=0) + 
#                                                          self.out.x[0])[:3]))
        pred = np.sum(np.array([poolMdl[iMdl] * self.out.x[iMdl+1] 
                                for iMdl in range(self.nModels)]),axis=0) + self.out.x[0]
        return np.sum(np.array([poolMdl[iMdl] * self.out.x[iMdl+1] 
                                for iMdl in range(self.nModels)]),axis=0) + self.out.x[0]



