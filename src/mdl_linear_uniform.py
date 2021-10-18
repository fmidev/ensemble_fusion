'''
Created on 5.3.2021

@author: sofievm
'''

import numpy as np
from scipy import optimize
from toolbox import optimization_tools as opt, supplementary as spp, MyTimeVars


##############################################################################
#
# Class for linear uniform in space and time models
#
##############################################################################

class linear_uniform():
    #
    # Class of fusion methods based on linear regression with various types
    # regularization and, possibly, constraints on the weight positiveness
    #
    def __init__(self, mdl_descr, data_shape, log):
        self.optim_method, self.constrain, self.abbrev = mdl_descr 
        self.fit_function = {'unconstrained' : self.fit_unconstrained,
                             'nonneg' : self.fit_nonneg_weights} [self.constrain]
        self.nModels, self.nTimes, self.nStat = data_shape
        self.log = log


    #==================================================================

    def reset_data_shape(self, nModels, nTimes, nStations):
        #
        # Redefines the data dimensions
        #
        self.nModels = nModels
        self.nTimes = nTimes
        self.nStat = nStations


    #==================================================================

    def fit(self, poolMdl_flat, tsmObs_flat, kernel_flat,
            train_mdls_f, train_obs_f, train_krnl_f, 
            test_mdl_f, test_obs_f, test_krnl_f,
            nModels, nTimes, nStat):  #tsmObs, poolMdl):
        #
        # Finds the coefficients for the spatially uniform linear fusion:
        # C_fused = a_0 + sum(a_i * C_n_i), i = 1..M
        # here C_m_i is map of i-th model, a_0, a_i are constant scalars
        # This sub finds a_0, a_i (i=1..M) minimising RMSE.
        #
        # We can require constrained minimization with non-negative 
        # weights - or let the models fly freely.
        #
        self.nModels, self.nTimes, self.nStat = (nModels, nTimes, nStat)
        
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
#        alpha_Tikhonov = fuse_mdl.verify_regularization(train_mdls_f, train_obs_f, train_krnl_f, 
#                                                        test_mdl_f, test_obs_f, test_krnl_f)
        #
        # Step 2. Final fitting
        #
        # The model rgularization weight identified, if any, need to make the final fit
        # for the whole time period
        #
        self.out = fuse_mdl.fit(poolMdl_flat[:,tsmObsOK].T, 
                                tsmObs_flat[tsmObsOK].T, kernel_flat[tsmObsOK]) # predictors, answers
        self.out.x = np.zeros(shape=(self.out.coef_.shape[0]+1))
        self.out.x[0] = self.out.intercept_
        self.out.x[1:] = self.out.coef_[0:]
        # If the case failed, i.e. no models are good for anything, put at random
        # something very small to avoid surprises in correlation
        if np.sum(np.abs(self.out.x)[1:]) < 1e-10:
            self.log.log('######### ERROR in fit_unconstrained: all zero weights') 
            self.out.x[1] = 1e-5
            self.out.coef_[0] = 1e-5
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
                     (chFit, fuse_mdl.clf.intercept_, ' '.join(['%g' % v for v in fuse_mdl.clf.coef_])))
    
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
#            alpha_Tikhonov = fuse_mdl.verify_regularization(train_mdls_f, train_obs_f, train_krnl_f, 
#                                                            test_mdl_f, test_obs_f, test_krnl_f)
            alpha_Tikhonov = fuse_mdl.alpha
            C0_Tikhonov = 0.0
            alpha_low_pass = 0.0  # The low-pass term is null for a while 
            C_prev = 0.0
            self.log.log('%s regularization weight is %g based on Ridge scheme' % 
                         (chFit, alpha_Tikhonov))
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
            # If the case failed, i.e. no models are good for anything, put at random
            # something very small to avoid surprises in correlation
            if np.sum(np.abs(self.out.x)[1:]) < 1e-10:
                self.log.log('######### ERROR in fit_nonneg_weights: all zero weights') 
                self.out.x[1] = 1e-5
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
        predict = self.predict(poolMdl_flat)
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
                     (self.out.x[0], ' '.join(['%g' % v for v in self.out.x[1:]])))
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
        if isinstance(poolMdl, np.ndarray):
            return np.sum(np.array([poolMdl[iMdl] * self.out.x[iMdl+1] 
                                    for iMdl in range(self.nModels)]),axis=0) + self.out.x[0]
        elif isinstance(poolMdl,MyTimeVars.TsMatrix):
            return np.sum(np.array([poolMdl[iMdl] * self.out.x[iMdl+1] 
                                    for iMdl in range(self.nModels)]),axis=0) + self.out.x[0]
        else:
            self.log.log('Unknown type of input poolMdl: ' + type(poolMdl))
            raise ValueError
        
