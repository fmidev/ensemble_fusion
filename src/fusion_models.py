'''
This module contains implementations of a variety of ensemble fusion models.
The input data are assumed to be tsMatrices, possibly, with additional metadata

Created on 22.3.2020

@author: sofievm
'''

import numpy as np
import numpy.f2py
import os, shutil, glob     #, random
from sklearn import linear_model
import mdl_linear_uniform as lin_uni
from toolbox import supplementary as spp, optimization_tools as opt, structure_func,\
    MyTimeVars
from toolbox import gridtools, drawer
import mdl_linear_uniform as MLU

#
# FORTRAN section
#
try:
    import linear_fusion_F
    ifFortranOK = True
except:
    # Attention: sizes of the arrays must be at the end
    fortran_code_linear_fusion = '''

subroutine merge_subsets(arHomogenCoefs, structFunWin, arWinStart, nWinSz, nWinOverlap, arCoefsOut, &
                       & nx, ny, nxWin, nyWin, nModels)
  !
  ! Merges the homogeneous coefficients from the subset-based fits into a map of coefficients
  ! Works in three steps
  ! 1. merges overlapping sections of the subsets windows
  ! 2. Regions not covered by the observations are extrapolated using the structure function as
  !    a proxy
  !
  implicit none
  
  ! Imported parameters
  real*4, dimension(0:nModels, 0:nxWin-1, 0:nyWin-1), intent(in) :: arHomogenCoefs
  real*4, dimension(0:nxWin-1, 0:nyWin-1, 0:nxWin-1, 0:nyWin-1), intent(in) :: structFunWin
  integer*4, dimension(0:nxWin-1, 0:nyWin-1, 0:1), intent(in) :: arWinStart
  integer, intent(in) :: nWinSz, nWinOverlap, nxWin, nyWin, nx, ny, nModels
  
  ! output maps of merger model weights
  real*4, dimension(0:nModels, 0:nx-1, 0:ny-1), intent(out) :: arCoefsOut
  
  ! Local variables
  real*4, dimension(0:nModels, 0:nxWin-1, 0:nyWin-1) :: arCoefsTmp
  real*4, dimension(0:nModels) :: vectCoefs_1d
  integer :: ix, iy, ixWin, iyWin, iMdl, ix1, iy1
  real*8 :: CntStF
  real*4, dimension(0:nx-1, 0:ny-1) :: arCntMain
  real*4, dimension(0:nWinSz-1, 0:nWinSz-1) :: kernelWin

  arCoefsTmp = arHomogenCoefs  ! cannot modify input but need to fill the gaps
  arCoefsOut = 0.0
  arCntMain = 0.0
  
!  print *, 'merge_subsets', 1, sum(arCoefsTmp)
  
  !
  ! Make the spatial kernel for the merging
  ! unity in the centre, linearly falling to 0 over teh overlap part
  !
  do ix = 0, nWinSz-1                ! x-axis fadeout
    if(ix < nWinOverlap-1)then
      kernelWin(ix,:) = real(ix+1) / real(nWinOverlap)
    elseif(ix > nWinSz - nWinOverlap)then
      kernelWin(ix,:) = real(nWinSz - ix) / real(nWinOverlap)
    else
      kernelWin(ix,:) = 1.0
    endif
  end do  ! ix
  do iy = 0, nWinSz-1           ! y-axis fadeout, on top to already existing x-axis falls
    if(iy < nWinOverlap-1)then
      kernelWin(:,iy) = kernelWin(:,iy) * real(iy+1) / real(nWinOverlap)
    elseif(iy > nWinSz - nWinOverlap)then
      kernelWin(:,iy) = kernelWin(:,iy) * real(nWinSz - iy) / real(nWinOverlap)
    endif
  end do  ! iy


!  do iy = 0, nWinSz-1           ! y-axis fadeout, on top to already existing x-axis falls
!    print *, kernelWin(:,iy)
!  end do  ! iy


  !
  ! Merge the overlapping sections of the windows filling-in the missing windows on the fly
  ! A missing window is the subdomain with no observations
  !
  do iyWin = 0, nyWin-1
    do ixWin = 0, nxWin-1
      !
      ! coefs available? If not (no observations in this window), fill from neighbours
      ! The missing value here is -999999
      !
      if(arCoefsTmp(0,ixWin,iyWin) < -999998)then
        !
        ! Extrapolate the coefficients to this cell using the structure function as a proxy. 
        ! All work is done in the subdoamins grid, i.e. ~10 times lower resolution than the main grid
        !
        vectCoefs_1d = 0.0
        CntStF = 0
        do iy = 0, nyWin-1
          do ix = 0, nxWin-1
          
            if(structFunWin(ixWin,iyWin,ix,iy) < 1e-10)then
              print *, "Strange structure function: ixWin,iyWin,ix,iy, StF", &
                                      & ixWin,iyWin,ix,iy, structFunWin(ixWin,iyWin,ix,iy)
            endif

            ! Sum only valid cells
            if(arCoefsTmp(0,ix,iy) > -999998)then
              vectCoefs_1d(:) = vectCoefs_1d(:) + &
                                      & arCoefsTmp(:,ix,iy) / structFunWin(ixWin,iyWin,ix,iy)
            endif
            CntStF = CntStF + 1./structFunWin(ixWin,iyWin,ix,iy)
          end do
        end do
        arCoefsTmp(:,ixWin,iyWin) = vectCoefs_1d(:) / CntStF
      endif  ! coefs are not available
      !
      ! Add the weighted coefficients to the output array
      !
      ix1 = arWinStart(ixWin,iyWin,0)
      iy1 = arWinStart(ixWin,iyWin,1)

!      print *, 'ix1-ix2, iy1-iy2', ix1, ix1+nWinSz-1, iy1, iy1+nWinSz-1 

      do iy =iy1, iy1+nWinSz-1
        do ix =ix1, ix1+nWinSz-1
          do iMdl = 0, nModels              ! 0 --> free term
            arCoefsOut(iMdl,ix,iy) = arCoefsOut(iMdl,ix,iy) + &
                                   & arCoefsTmp(iMdl,ixWin,iyWin) * kernelWin(ix-ix1,iy-iy1)
          end do
          arCntMain(ix,iy) = arCntMain(ix,iy) + kernelWin(ix-ix1,iy-iy1)
        end do
      end do
    end do  ! ixWin
  end do  ! iyWin
  
!  print *, 'merge_subsets', 3, sum(arCoefsTmp), sum(arCoefsOut), sum(arCntMain)
  
  !
  ! Average of the coefficients over the main grid
  !
  do iy = 0, ny-1
    do ix = 0, nx-1
      if(arCntMain(ix,iy) == 0) print *, 'ZERO', ix, iy, arCntMain(ix,iy)
      arCoefsOut(:,ix,iy) = arCoefsOut(:,ix,iy) / arCntMain(ix,iy)
    end do
  end do

!  print *, 'merge_subsets', 4, sum(arCoefsTmp), sum(arCoefsOut)

!  !
!  ! Possibility:
!  ! Extrapolate the coefficients using the structure function as a proxy in the main grid
!  ! Downsides: structure function nx*ny*nx*ny is enormous, so as time needed for its computation;
!  ! also, this function picks too many irrelevant details due to too high resolution.
!  ! Commented for now
!  !
!  do iy = 0,ny-1
!    do ix = 0, nx-1
!      if(arCnt(ix,iy) < 1e-10)then
!        do iyWin = 0, ny-1
!          do ixWin = 0, nx-1
!            arCoefsOut(ix,iy,:) = arCoefsOut(ix,iy,:) + &
!                                & arCoefsOut(ixWin,iyWin,:) / structFunWin(ix,iy,ixWin,iyWin)
!            arCntStF(ix,iy) = arCntStF(ix,iy) + 1./structFunWin(ix,iy,ixWin,iyWin)
!          end do
!        end do
!        arCoefsOut(ix,iy,:) = arCoefsOut(ix,iy,:) / arCntStF(ix,iy)
!      endif   arCnt is zero
!    end do  ! ix
!  end do  ! iy
  
end subroutine merge_subsets


'''

#    from numpy.distutils.fcompiler import new_fcompiler
#    compiler = new_fcompiler(compiler='intel')
#    compiler.dump_properties()

    # Compile the library and, if needed, copy it from the subdir where the compiler puts it
    # to the current directory
    #
    vCompiler = np.f2py.compile(fortran_code_linear_fusion, modulename='linear_fusion_F', 
                                extra_args = '--opt=-O3', verbose=1, extension='.f90')
#                                extra_args = '--opt=-O3 --debug --f90flags=-fcheck=all', verbose=1, extension='.f90')
    if vCompiler == 0:
        cwd = os.getcwd()     # current working directory
        if os.path.exists(os.path.join('linear_fusion_F','.libs')):
            list_of_files = glob.glob(os.path.join('linear_fusion_F','.libs','*'))
            latest_file = max(list_of_files, key=os.path.getctime)
            shutil.copyfile(latest_file, os.path.join(cwd, os.path.split(latest_file)[1]))
        try: 
            import linear_fusion_F
        except:
            print('>>>>>>> FORTRAN failed linear_fusion_F-2')
            raise
    else:
        print('>>>>>>> FORTRAN failed linear_fusion_F')
        raise

##############################################################################
#
# Data processor for the fusion models
#
##############################################################################

class fusion_mdl_general():

    def __init__(self, fusion_setup, log):
        self.fusion_setup = fusion_setup
        self.fusion_methods = fusion_setup['fusion_method']
        self.training_fraction = float(fusion_setup['training_fraction'][0])
#        self.fusion_grid = fusion_grid   # not here: too early
#        self.fusion_mdl = {}
#        self.fusion_verify = {}
        self.ifPreprocess_multifit = np.sum(list((ifSpaceResolving(m) for m in self.fusion_methods))) > 0
        self.log = log


    #========================================================================

    def start_new_dataset(self):
        #
        # Flushes the existing data structures to prepare the model to the new dataset, e.g. next day
        #
        self.fusion_mdl = {}
        self.fusion_verify = {}


    #========================================================================

    def ortogonalise_and_censor(self, obsMatr, mdlPool, kernel, idxSort, 
                                tc_indices_obs, tc_indices_mdl):
        #
        # Creates an orthogonal set of "models" of the same shape as the given pool
        # Procedure: 
        # 1. Make linear regression of unity and zero-mean best model to the second-best model,
        #    store residuals of regression as the added-value of this model
        # 2..N. Each next model is regressed to processed models leaving residual as added value
        # 3. The collection of residuals is stored as the new uncorrelated set of predictors.
        #
        poolMdlOrt, clf, v = opt.ortogonalise_TSM(mdlPool.vals, idxSort, kernel, tc_indices_mdl, self.log)
        self.log.log('Ordered models: ' + ' '.join(mdlPool.variables[im] for im in idxSort))
        self.log.log('Ordered model energy : ' + 
                     ' '.join('%g' % np.sqrt(np.square(poolMdlOrt[iMdl,tc_indices_mdl,:]).mean()) 
                              for iMdl in idxSort))
        # Information content for each model
        mdlInfoRanked = list((np.nanmean(poolMdlOrt[iP,tc_indices_mdl,:] * obsMatr.vals[tc_indices_obs,:]) 
                               for iP in idxSort))
        self.log.log('Ordered model useful information: ' + ' '.join('%g' % v for v in mdlInfoRanked))
        #
        # Eliminate the non-informative models
        #
        idxMdlInformative = idxSort[np.abs(mdlInfoRanked) > 0.05 * np.max(np.abs(mdlInfoRanked))]
        self.log.log('Informative models: %i: ' % len(idxMdlInformative) + 
                     ' '.join(mdlPool.variables[im] for im in idxMdlInformative))
        #
        # Final predictor dimension of the problem, still full time coverage: obs + fcst
        # Predictors are ordered and filtered
        #
        return (poolMdlOrt[idxMdlInformative,:,:], idxMdlInformative)


    #=========================================================================================
    
    def split_input_data(self, nValidTimesInDay, splitMethod, obsMatr, tc_indices_obs,
                         poolMdlOrt, tc_indices_mdl, times_kernel, times_common, randomGen):
        #
        # Split the datasets to training and test subsets
        # Makes use of optimization_tools.split_times
        # If there is not enough useful times due to hourly selection, use stations
        #
        nModelsWrk, nTimes, nStat = poolMdlOrt[:, tc_indices_mdl,:].shape
        kernel = np.repeat(times_kernel[:,np.newaxis], nStat, axis=1)
        splitMethod_local = splitMethod
        #
        # Try to make successful split up to 10 times
        #
        for i in range(10):
            # deterministic methods either work from the first try or fail forever,
            # therefore starting from the second try, should use random split
            if i > 0: splitMethod_local = 'native_random'
            if nValidTimesInDay == 1 and splitMethod_local == 'native_random':
                # Split will use stations and times
                # pick only data with kernel > 0
                PoolObs_f = obsMatr.vals[tc_indices_obs,:].ravel()
                PoolMdls_f = np.reshape(poolMdlOrt[:,tc_indices_mdl,:],
                                        (nModelsWrk, PoolObs_f.shape[0]))
                # make the split
                train_idx = sorted(randomGen.sample(range(PoolObs_f.shape[0]), 
                                                    max(1, int(PoolObs_f.shape[0] * 
                                                               self.training_fraction))))
                test_idx = sorted(list(set(range(PoolObs_f.shape[0])) - set(train_idx)))
                # filter out nans from observations
                trainIdxOK = np.isfinite(PoolObs_f[train_idx])
                testIdxOK = np.isfinite(PoolObs_f[test_idx])
                # final training
                train_obs_flat = PoolObs_f[train_idx][trainIdxOK]
                train_mdls_flat = PoolMdls_f[:,train_idx][:,trainIdxOK]
                train_kernel_flat = kernel.ravel()[train_idx][trainIdxOK]
                # ... and test
                test_obs_flat = PoolObs_f[test_idx][testIdxOK]
                test_mdls_flat = PoolMdls_f[:,test_idx][:,testIdxOK]
                test_kernel_flat = kernel.ravel()[test_idx][testIdxOK]
            else:
                # split along times
                train_idx, test_idx = opt.split_times(times_common, times_kernel,
                                                      self.training_fraction, splitMethod_local,
                                                      randomGen, self.log)
                # training
                train_obs_tmp = (obsMatr.vals[tc_indices_obs,:][train_idx,:]).ravel()
                trainIdxOK = np.isfinite(train_obs_tmp)
                train_obs_flat = train_obs_tmp[trainIdxOK]   # identifying individual missing observations
                train_kernel_flat = kernel[train_idx,:].ravel()[trainIdxOK]
                # test
                test_obs_tmp = (obsMatr.vals[tc_indices_obs,:][test_idx,:]).ravel()
                testIdxOK = np.isfinite(test_obs_tmp)
                test_obs_flat = test_obs_tmp[testIdxOK]     # identifying individual missing observations
                test_kernel_flat = kernel[test_idx,:].ravel()[testIdxOK]
                #
                # Split and flatten the orthgonalized models
                train_mdls_flat = np.reshape(poolMdlOrt[:,tc_indices_mdl,:][:,train_idx,:],    # nModels, nTimes, nStations
                                             (nModelsWrk, len(train_idx) * nStat))[:,trainIdxOK]
                test_mdls_flat = np.reshape(poolMdlOrt[:,tc_indices_mdl,:][:,test_idx,:],    # nModels, nTimes, nStations
                                            (nModelsWrk, len(test_idx) * nStat))[:,testIdxOK]
                # the whole set for .predict
                PoolObs_f = obsMatr.vals[tc_indices_obs,:].ravel()
                PoolMdls_f = np.reshape(poolMdlOrt[:,tc_indices_mdl,:], (nModelsWrk, nTimes * nStat))
            # Good split?
            if not(np.sum(train_obs_flat * train_kernel_flat) *
                   np.sum(test_obs_flat * test_kernel_flat) == 0):
                return (PoolMdls_f, PoolObs_f, kernel.ravel(),
                        train_mdls_flat.T, train_obs_flat, train_kernel_flat,
                        test_mdls_flat.T, test_obs_flat, test_kernel_flat,
                        nModelsWrk, nTimes, nStat)
        #
        # If reached here, nothing good was found
        self.log.error('Failed train-test split')
        self.log.log('train_obs_flat ' + ' '.join('%g' % v for v in train_obs_flat))
        self.log.log('train_kernel_flat ' + ' '.join('%g' % v for v in train_kernel_flat))
        self.log.log('test_obs_flat ' + ' '.join('%g' % v for v in test_obs_flat))
        self.log.log('test_kernel_flat ' + ' '.join('%g' % v for v in test_kernel_flat))
        return None


    #=========================================================================================
    
    def fusion_step(self, obsMatr, tc_indices_obs, poolMdlOrt, tc_indices_mdl, times_common, iStep,
                    nValidTimesInDay, splitMethod, times_kernel, fusion_grid, arSubgrids, arWinStart,
                    subdomain_struct_func, randomGen):
        #
        # Makes one step (either whole period or one of the 24 hourly steps) of the actual fusion: 
        # splits the times, fits the model and evaluates the results
        # There are two types of methods: spatially uniform and space-resolving.
        # The spatially-unifirm methods all operate with a single training-test split
        # The space-resolving multistep methods take nxWin x nyWin subsets of the obs sites and then
        # apply the homogeneous procedure to each subset
        #
        ifWholeArea_DataReady = False
        ifMultistep_DataReady = False
        inDat= {}
        for method in self.fusion_methods:
            fitMdl = self.initialise_fusion_model(method, fusion_grid, obsMatr, tc_indices_obs,
                                                  poolMdlOrt[:,tc_indices_mdl,:].shape,
                                                  arSubgrids, arWinStart, subdomain_struct_func)
            if not method in self.fusion_mdl.keys():
                self.fusion_mdl[method] = []
                self.fusion_verify[method] = np.zeros(poolMdlOrt[0,tc_indices_mdl,:].shape)
            #
            # input data for the whole domain will be needed for all fusion models
            # Are they ready?
            #
            if not ifWholeArea_DataReady:
                # training-test split
                inDat['whole_area'] = self.split_input_data(nValidTimesInDay, splitMethod, 
                                                            obsMatr, tc_indices_obs,
                                                            poolMdlOrt, tc_indices_mdl, 
                                                            times_kernel, times_common, randomGen)
                if inDat['whole_area'] is None:
                    self.log.error('Failed input data split for times\n' + 
                                   ' '.join((t.strftime('%Y%m%d_%H%M') for t in times_common)))
                    return None

                ifWholeArea_DataReady = True
            #
            # Space-resolving fusion will require a multi-domain input data
            #
            if ifSpaceResolving(method):
                # Space-resolving fusion model
                # input data ready?
                if not ifMultistep_DataReady:
#                    inDat['subgrids'] = {'filled':np.ones((len(fitMdl.get_subgrids())),
#                                                               dtype=np.bool) * False}
                    inDat['subgrids'] = {'filled':np.zeros((fitMdl.arGrids.shape), dtype=np.int16)}
                    for ixG in range(fitMdl.arGrids.shape[0]):
                        for iyG in range(fitMdl.arGrids.shape[1]):
                            # mask the stations outside this grid
                            stMask, obsMatrGrd = obsMatr.grid_mask(fitMdl.arGrids[ixG, iyG])
                            # Non-empty grid, i.e. at least one station has data
                            if len(obsMatrGrd.stations) > 0:
                                # Do the station prodide non-trivial data?
                                uniq, cntUniq = np.unique(obsMatrGrd.vals[tc_indices_obs]
                                                          [np.isfinite(obsMatrGrd.vals
                                                                       [tc_indices_obs])],
                                                          return_counts=True)
                                if uniq.shape[0] > 2 or (uniq.shape[0] == 2 and np.all(cntUniq > 1)):
                                    # Split the data
                                    inTmp = self.split_input_data(nValidTimesInDay, splitMethod,
                                                                  obsMatrGrd, tc_indices_obs,
                                                                  poolMdlOrt[:,:,stMask],
                                                                  tc_indices_mdl, times_kernel, 
                                                                  times_common, randomGen)
                                    if inTmp is None:
                                        self.log.log('Failed input split for subdomain (%g,%g)' %
                                                     (ixG,iyG))
                                        continue
                                    else:
                                        inDat['subgrids'][fitMdl.arGrids[ixG,iyG]] = inTmp
                                    # Store the number of stations
                                    inDat['subgrids']['filled'][ixG,iyG] = len(obsMatrGrd.stations)
                    ifMultistep_DataReady = True
                #
                # fitting the coefficients of the multistep space-resolving model
                # A complicated process - goes through the available windows and then
                # inter- / extra-polates the results
                #
                fMdl, fPred = fitMdl.fit(inDat)   #['subgrids'])
                
            else:
                # Homogeneous fusion model
                #
                # fitting the coefficients of the homogeneous model
                # The InDatSplit is a big tuple, open it via *
                #
                fMdl, fPred = fitMdl.fit(*inDat['whole_area'])
            #
            # Store the fusion model and prediction over the learning period for the folowup evaluation
            #
            if fMdl:
                # all hours = iHr in all days
                self.fusion_mdl[method].append(fMdl)
                if nValidTimesInDay == 1: 
                    # For hourly fusion: one fit is valid for one time step
                    idxAllHr = np.array(list( (t.hour == iStep for t in times_common)))
                    self.fusion_verify[method][idxAllHr,:] = fPred[idxAllHr, :]
                elif nValidTimesInDay == 24:
                    # For daily fusion one fit is valid over 24 hours
                    self.fusion_verify[method][:,:] = fPred[:, :]
                else:
                    self.log.error('Strange nValidTimesInDay %g' % nValidTimesInDay)
                    raise ValueError
            else:
                self.log.error('Failed fusion coefficients for ' + method)

        return self


    #=====================================================================================

    def fusion_metadata(self, methodName):
        #
        # Initialising the fusion models. Methods implemented this-far:
        #            Basic MLR, no regularization
        #        'LINEAR_UNIFORM_MLR' : ('MLR', 'unconstrained','LU_MLR'),
        #            MLR with RIDGE regularization
        #        'LINEAR_UNIFORM_RIDGE' : ('RIDGE', 'unconstrained','LU_RIDGE'),
        #                 MLR with LASSO regularization
        #        'LINEAR_UNIFORM_LASSO' : ('LASSO', 'unconstrained','LU_LASSO'),
        #                 MLR constrained to non-negative weights, explicit Tikhonov-type regularization
        #        'LINEAR_UNIFORM_MLRNONNEG_REGEXPL': ('MLR_regexpl', 'nonneg','LU_MLRnn_ex'),
        #                 MLR constrained to non-negative weights, 
        #                 implicit L-curve truncated iterations regularization
        #        'LINEAR_UNIFORM_MLRNONNEG_REGLCRV': ('MLR_regLcrv', 'nonneg','LU_MLRnn_L'),#
        #
        #                 Basic MLR, no regularization
        #        'LINEAR_SPACE_RESOLVING_MULTIFIT_MLR' : ('MLR', 'unconstrained','LU_MLR'),
        #                 MLR with RIDGE regularization
        #        'LINEAR_SPACE_RESOLVING_MULTIFIT_RIDGE' : ('RIDGE', 'unconstrained','LU_RIDGE'),
        #                 MLR with LASSO regularization
        #        'LINEAR_SPACE_RESOLVING_MULTIFIT_LASSO' : ('LASSO', 'unconstrained','LU_LASSO'),
        #                 MLR constrained to non-negative weights, explicit Tikhonov-type regularization
        #        'LINEAR_SPACE_RESOLVING_MULTIFIT_MLRNONNEG_REGEXPL': ('MLR_regexpl', 'nonneg','LU_MLRnn_ex'),
        #                 MLR constrained to non-negative weights, 
        #                 implicit L-curve truncated iterations regularization
        #        'LINEAR_SPACE_RESOLVING_MULTIFIT_MLRNONNEG_REGLCRV': ('MLR_regLcrv', 'nonneg','LU_MLRnn_L')
        #
        # Linear is so-far must
        if methodName.upper().startswith('LINEAR'): abbr = 'L'
        else:
            self.log.log('Unknown fusion method %s. Only LINEAR so far' % methodName)
            raise ValueError
        # Uniform or space-resolving?
        if '_UNIFORM_' in methodName.upper(): abbr += 'U_'
        elif '_SPACE_RESOLVING_' in methodName.upper():
            abbr += 'SR'
            if '_MULTIFIT_' in methodName.upper(): abbr += 'MF_'
            else:
                self.log.log('Strange fusion method %s. Only MULTIFIT for space-resolving methods' % methodName)
                raise ValueError
        else:
            self.log.log('Strange fusion method %s. Either UNIFORM or SPACE_RESOLVING' % methodName)
            raise ValueError
        # optimization method
        mthd = methodName.upper().split('_')[-1]
        if mthd in 'MLR RIDGE LASSO'.split():
            abbr += mthd
            constr = 'unconstrained'
        elif methodName.upper().endswith('MLRNONNEG_REGEXPL'):
            mthd = 'MLR_regexpl'
            constr = 'nonneg'
            abbr += 'MLRnn_ex'
        elif methodName.upper().endswith('MLRNONNEG_REGLCRV'):
            mthd = 'MLR_regLcrv'
            constr = 'nonneg'
            abbr += 'MLRnn_L'
        return (mthd, constr, abbr)
    

    #=========================================================================================
    
    def initialise_fusion_model(self, method_name, fusion_grid, obsTSM, tc_indices_obs, data_shape, 
                                arSubgrids, arWinStart, subdomain_struct_func):
        #
        # A wrapper for initialization of the fusion models. Not the most-elegant way but
        # it hides the classes of different fusion methods behind the strongs
        #
        # Models implemented this-far in this module
        #
        if method_name.upper().startswith('LINEAR_UNIFORM_'):
            # Linear uniform methods are in the dedicated module
            return lin_uni.linear_uniform(self.fusion_metadata(method_name), data_shape, self.log)
        
        elif method_name.upper().startswith('LINEAR_SPACE_RESOLVING_'):
            # The space-resolving methods are here because they will need functions in this module
            return linear_space_resolving(self.fusion_metadata(method_name), obsTSM, tc_indices_obs,
                                          data_shape, self.fusion_setup, fusion_grid, arSubgrids,
                                          arWinStart, subdomain_struct_func, self.log)
        else:
            self.log.log('initialise_fusion_model: Unknown fusion method name: %s' % method_name)
            return False



##########################################################################################
#
# Linear space-resolving fusion model
# Derived from the linear_uniform model and can heavily use it for multi-subsetting method
#
##########################################################################################

class linear_space_resolving(MLU.linear_uniform):

    def __init__(self, mdl_descr, obsTSM, tc_indices_obs, data_shape, fusion_setup, grid, arSubgrids,
                 arWinStart, subdomain_struct_func, log):
        #
        # Initialize the basics of the space-resolving and uniform components
        # Since the main grid is unknown, cannot finalize the definition
        #
        MLU.linear_uniform.__init__(self,mdl_descr, data_shape, log)
#        self.optim_method, self.constrain, self.abbrev = mdl_descr
#        self.fit_function = MLU.linear_uniform.fit_function
#        self.nModels, self.nTimes, self.nStat = data_shape
#        self.log = log
        # Multifit?
        if 'SRMF_' in self.abbrev: self.ifMultiFit = True
        else:
            self.log.log('Strange method abbreviation %s. Only SRMF_ sof far' % self.abbrev)
            raise ValueError
        #
        # Get the multifit parameters: window size and overlap of sequential window locations
        #
        self.main_grid = grid
        self.structFunWin = subdomain_struct_func
        self.obsTSM = obsTSM
        self.ref_time = obsTSM.times[tc_indices_obs][0]
        
        if self.ifMultiFit:
            self.iWinSize = int(fusion_setup['multifit_window_size_cells'][0])
            self.iWinOverlap = int(fusion_setup['multifit_window_overlap_cells'][0])
            self.arGrids = arSubgrids
            self.arWinStart = arWinStart
            #
            # geometry of the subsets
            self.nxWin, self.nyWin = arSubgrids.shape
            self.arValid = np.ones((self.nxWin, self.nyWin), dtype = np.bool) * False
            self.arMdl = np.empty((self.nxWin, self.nyWin), dtype = object)
            for ix in range(self.nxWin):
                for iy in range(self.nyWin):
                    self.arMdl[ix,iy] = lin_uni.linear_uniform(
                                             (self.optim_method, self.constrain, 
                                              self.abbrev.replace('SRMF_','U_%02g_%02g_' % (ix,iy))),
                                             data_shape, self.log)

    
    #==================================================================

    def fit(self, InDataAllDomains): #poolMdl_flat, tsmObs_flat, kernel_flat,
#            train_mdls_f, train_obs_f, train_krnl_f, 
#            test_mdl_f, test_obs_f, test_krnl_f):  #tsmObs, poolMdl):
        #
        # Scans the domain with a running window. Each window is a subarea of the main domain
        # considered independently. For each, we solve the homogenous fusion 
        # problem using the corresponding optimization method. After that, the coefficients are 
        # merged together via spatial inter- and extrapolation, finally coming to the main grid
        #
        self.log.log('Space-resolving fit goes along %g x %g subdomains %g size with %g overlap' % 
                     (self.nxWin, self.nyWin, self.iWinSize, self.iWinOverlap))
        self.log.log('fit received %i models, %i times, %i stations' % 
                     (self.nModels, self.nTimes, self.nStat))
        #
        # Fit all subdomain models
        #
        InData = InDataAllDomains['subgrids']
        for ix in range(self.nxWin):
            for iy in range(self.nyWin):
                if InData['filled'][ix,iy] > 0:  # subarea has stations
                    print('Fitting', ix, iy)
                    # Make the homogeneous fit
                    self.arMdl[ix,iy].fit(*InData[self.arGrids[ix,iy]])
        #
        # Make maps of coefficients using the overlapping subdomainss
        # Reserve the space and collect the homogeneous fits into a single temporary array
        #
        self.arCoefs = np.zeros((self.main_grid.nx, self.main_grid.ny, self.nModels+1),
                                dtype=np.float32)
        arHomogenCoefs = np.ones((self.nModels+1, self.nxWin, self.nyWin)) * (-999999)
        print('\nSubdomwin ixSub,iySub, lonCntr, latCntr, a0, w1...w_nMdls')
        for ix in range(self.nxWin):
            for iy in range(self.nyWin):
                if InData['filled'][ix,iy] > 0:  # subarea has stations
                    arHomogenCoefs[:,ix,iy] = self.arMdl[ix,iy].out.x[:]
                    grd = self.arGrids[ix,iy]
                    print('Homog coefs:', ix, iy, '(%g,%g)' % (grd.x0+grd.dx*grd.nx/2,
                                                               grd.y0+grd.dy*grd.ny/2),
                          '  '.join('%g' % v for v in arHomogenCoefs[:,ix,iy]))
        #
        # The actual coefficients maps are computed by FORTRAN sub
        #
        if np.any(self.structFunWin.structFunMap == 0):
            self.log.log('Structure function is zero!')
            raise ValueError

        self.arCoefs = linear_fusion_F.merge_subsets(arHomogenCoefs, 
                                                     self.structFunWin.structFunMap,
                                                     self.arWinStart, self.iWinSize,
                                                     self.iWinOverlap, 
                                                     self.main_grid.nx, 
                                                     self.main_grid.ny,
                                                     self.nxWin, self.nyWin, 
                                                     self.nModels)
        #
        # Fitting finished. Make prediction
        #
        # But first, we have to create the tsMatrix for new model weights: they now differ
        # for each station
        #
        tsCoefs_vals = np.zeros((self.nModels+1,1,len(self.obsTSM.stations))) * np.nan
        fX, fY = self.main_grid.geo_to_grid(np.array(list((s.lon for s in self.obsTSM.stations))),
                                            np.array(list((s.lat for s in self.obsTSM.stations))))
        coords = (np.round(np.array(fX)).astype(np.int), np.round(np.array(fY)).astype(np.int))
        for iMdl in range(self.nModels+1):
            tsCoefs_vals[iMdl,0,:] = self.arCoefs[iMdl,:,:][coords]
        self.tsMdlCoefs = MyTimeVars.TsMatrix([self.ref_time], self.obsTSM.stations,
                                              list(('Mdl_%02g' % i for i in range(self.nModels+1))),
                                              tsCoefs_vals, self.obsTSM.units * (self.nModels+1),
                                              -999999, self.obsTSM.timezone)
        # open 
        (poolMdl_f, tsmObs_f, krnl_f, trn_mdls_f, trn_obs_f, trn_krnl_f, 
         tst_mdl_f, tst_obs_f, tst_krnl_f, 
         self.nModels, self.nTimes, self.nStat) = InDataAllDomains['whole_area']        
        tsmObsOK = np.isfinite(tsmObs_f)
        #
        # Make the prediction for the stations
        #
        fuse_predict = self.predict(poolMdl_f)
        
#        self.log.log('Verification for 3 sites:' + str(fuse_predict[1,:3]))
        
        diff = np.subtract(np.ndarray.flatten(fuse_predict)[tsmObsOK], tsmObs_f[tsmObsOK])
#        corr = np.corrcoef(np.ndarray.flatten(fuse_predict)[tsmObsOK], tsmObs_flat[tsmObsOK])[0,1]
        corr = spp.nanCorrCoef(np.ndarray.flatten(fuse_predict)[tsmObsOK], tsmObs_f[tsmObsOK])
        bias = np.mean(diff * np.sqrt(krnl_f[tsmObsOK]))
        rmse = np.sqrt((np.square(diff) * krnl_f[tsmObsOK]).mean())
        self.log.log('Space-resolving model %s, rmse: %g, bias %g, corr %g' % 
                     (self.optim_method, rmse, bias, corr))

        return (self, fuse_predict)
        

    #==================================================================

    def predict(self, poolMdl):
        #
        # Makes the prediction with the identified set of models and weighting coefficients
        # Predictions can be made for stations or for maps
        # In both cases, poolMdl cpmes as a numpy array but its dimensions are different:
        # If maps are to be preicted, the poolMdl is (nModels, nx, ny) 
        # for tsMatrices, the last dimension is the nStations * nTimes
        #
        if len(poolMdl.shape) == 3:
            if (poolMdl.shape[0] == self.nModels and poolMdl.shape[1] == self.arCoefs.shape[1] and
                poolMdl.shape[2] == self.arCoefs.shape[2]):
                # we are predicting maps for a single time step
                return np.sum(poolMdl[:,:,:] * self.arCoefs[1:,:,:], axis=0) + self.arCoefs[0,:,:]
            else:
                # must be tsMatrix
                # get values from tsMatrix
                tsMdlCoefVals = self.tsMdlCoefs.vals.reshape(self.nModels+1, len(self.obsTSM.stations))
                try:
                    return np.sum(poolMdl[:,:,:] * tsMdlCoefVals[1:,np.newaxis,:],
                                  axis=0) + tsMdlCoefVals[0,np.newaxis,:]
                except:
                    print('Something funny, predict: poolMdl.shape, self.arCoefs.shape, self.nModels',
                          poolMdl.shape, self.arCoefs.shape, self.nModels)
                    return np.sum(poolMdl[:,:,:] * tsMdlCoefVals[1:,np.newaxis,:],
                                  axis=0) + tsMdlCoefVals[0,np.newaxis,:]
        else:
            # 2D poolMdl is tsMatrix
            # get values from tsMatrix
            tsMdlCoefVals = self.tsMdlCoefs.vals.reshape(self.nModels+1, len(self.obsTSM.stations))
            # compute the number of times requested
            nTimesRequested = np.round(poolMdl.shape[1] / self.nStat).astype(np.int)
            return np.sum(poolMdl[:,:].reshape(self.nModels, nTimesRequested, self.nStat) *
                          tsMdlCoefVals[1:,np.newaxis,:], axis=0) + tsMdlCoefVals[0,np.newaxis,:]


    #==================================================================

    def get_subgrids(self):
        return self.arGrids.ravel()


######################################################################################################
#
# Auxiliary subroutines
#
#========================================================================================

def ifSpaceResolving(fusion_method):
    #
    # Checks if any of the involved methods is space-resolving
    #
    return 'SPACE_RESOLVING' in fusion_method.upper()


#=========================================================================================

def create_subdomains(iWinSize, iWinOverlap, grid, log):
    #
    # Splits the given grid to a set of overlapping subgrids and makes sure that they
    # fill the whole grid: the last column and row of the subgrids can have larger overlap
    #
    if iWinSize <= iWinOverlap:
        log.log('Strange window size and overlap: %g %g' % (iWinSize, iWinOverlap))
        raise ValueError
    winSz = iWinSize - iWinOverlap
    print('Subdomain parameters:winSz, iWinSize, iWinOverlap', winSz, iWinSize, iWinOverlap)
    #
    # geometry of the subsets
    nxWin = int(grid.nx / winSz)
    nyWin = int(grid.ny / winSz)
    # last subdomains should end exactly at the end of the main domain
    if grid.nx > nxWin * winSz + iWinOverlap: nxWin += 1
    if grid.ny > nyWin * winSz + iWinOverlap: nyWin += 1
    # reserve space
    arGrids = np.empty((nxWin, nyWin), dtype = object)
    arWinStart = np.zeros((nxWin, nyWin, 2), dtype=np.int32)
    #
    # Create the 2D array of fusion models
    # 
    for ix in range(nxWin):
        xExtra = min(grid.nx - (ix * winSz + iWinSize), 0)
        arWinStart[ix,:,0] = ix * winSz + xExtra
        for iy in range(nyWin):
            yExtra = min(grid.ny - (iy * winSz + iWinSize), 0)
            arGrids[ix,iy] = gridtools.Grid(grid.x0 + (ix * winSz + xExtra) * grid.dx, 
                                            grid.dx, iWinSize,
                                            grid.y0 + (iy * winSz + yExtra) * grid.dy, 
                                            grid.dy, iWinSize,
                                            grid.proj)
            arWinStart[ix,iy,1] = iy * winSz + yExtra

#            print('Subdomain:', ix, iy, arWinStart[ix,iy,0], '-', arWinStart[ix,iy,0] + iWinSize,
#                  arWinStart[ix,iy,1], '-', arWinStart[ix,iy,1] + iWinSize)

    return (arGrids, arWinStart)


