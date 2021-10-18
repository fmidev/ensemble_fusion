'''
Created on Feb 16, 2020

This is the control module for the FUSE model 
It cnotains the high-level set of functions 
- creating the ensemble fusion model
- loading the model and observational data
- generating the fusiong models
- evalaution the fusion models
- reporting the fusion model fitting and skills

@author: Mikhail Sofiev, FMI
'''

import os
if os.path.exists('d:\\tools\\python_3'): os.environ['PYTHONPATH'] = 'd:\\tools\\python_3'
import numpy as np
from scipy import signal
import copy, pickle, psutil
from toolbox import supplementary as spp, stations, silamfile, drawer
from toolbox import timeseries as ts, MyTimeVars as tv
from toolbox import structure_func as StF
import fusion_models as fm
import post_process as pp
import datetime as dt
from zipfile import ZipFile

def above_threshold(x):
    return x > 0.01   # just something

minThreshold = {'CO':10.0, 'NO2' : 0.0, 'SO2': 0.0, 'O3' : 0.0, 'PM25': 0.0, 'PM10':0.0}
maxThreshold = {'CO':100000.0, 'NO2' : 10000.0, 'SO2': 10000.0, 'O3' : 10000.0, 
                'PM25': 10000.0, 'PM10':10000.0}

############################################################################################################
#
# Class for holding the results of the optimization and parameters of the optimal fusion
# The class is for one fitting exercise, a.c.a. day. The actual archive is a dictionary with
# days as keys
#
class FUSE_archive_day():
    #
    # One "day" of fusion 
    #
    def __init__(self, eval_stats, fusion_mdls, 
                 raw_models_sort, mean_models, raw_models_used_ranked):
        self.eval_stats = copy.deepcopy(eval_stats)
        self.fusion_mdls = copy.deepcopy(fusion_mdls)
        self.raw_models_sort = copy.copy(raw_models_sort)  # all models available for the day
        self.mean_models = copy.copy(mean_models)
        self.raw_models_used_ranked = copy.copy(raw_models_used_ranked)  # informative models used for that day
#
# Put the whole multi-day archive to pickle
#
def archive_to_pickle(chFNm, archive):
    with open(chFNm,'wb') as handle:
        pickle.dump(archive, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
# Get the whole multi-day archive from pickle
#
def archive_from_pickle(chFNm):
    with open(chFNm,'rb') as handle:
        return pickle.load(handle)


############################################################################################################
#
# Main model class FUSE
#
class FUSE():

    #=======================================================================

    def __init__(self, observations, models, today, mpirank, mpisize, chMPI, fusion_setup, chFLogNm):
        self.timer = spp.SFK_timer()
        self.timer.start_timer('00_overall')
        self.chMPI = chMPI
        self.mpirank = mpirank
        self.mpisize = mpisize
        self.Process = psutil.Process(os.getpid())
        self.fusion_setup = fusion_setup
        self.run_type = fusion_setup['run_type'][0]
        self.fusion_methods = fusion_setup['fusion_method']   # a bbunch of them
        self.today = today
        self.timestep_obs = spp.one_hour * float(fusion_setup['time_step_obs_hrs'][0])
        self.timestep_mdl = spp.one_hour * float(fusion_setup['time_step_models_hrs'][0])
        self.ifHourlyWeights = fusion_setup['time_step_model_weights'][0].upper() == 'HOURLY'
        self.learning_period_days = int(fusion_setup['learning_period_days'][0])
        self.output_directory = fusion_setup['output_directory'][0]
        self.ifPlotDraftPictures = fusion_setup['plot_draft_figures_too'][0].upper() == 'YES'
        self.ifPlotDailyPictures = fusion_setup['plot_daily_figures'][0].upper() == 'YES'
        self.if_orthogonalize = fusion_setup['if_orthogonalize'][0].upper() == 'YES'
#        if not os.path.exists(self.output_directory): os.makedirs(self.output_directory)
#        for subdir in 'daily models scores timeseries'.split() + '':
        for subdir in ['daily','tser_mdls','tser_skills','tser_weights','daily_weight_maps','fcst']:
            spp.ensure_directory_MPI(os.path.join(self.output_directory,subdir))

        self.rnGen = np.random.default_rng(2021)
        self.log = spp.log(os.path.join(self.output_directory, today.strftime(chFLogNm)),'w')
        # species
#        # multi-species stuff would be like this:
#        self.species = {}
#        for s in namelist_ini.get('species'):    # string: species_obs, species_mdl, threshold
#            self.species[s.split()[0]] = (s.split()[0], s.split()[1], float(s.split()[2]))
        # for single-species run
        self.species = fusion_setup['species'][0].split()  # 3 fields: name_obs, name_mdl, factor
        self.species[2] = float(self.species[2])   # conversion factor
        
        # models 
        self.modelsPreExtracted = fusion_setup['if_use_pre_extracted_fields'][0].upper() == 'YES'
        self.models = copy.copy(models)
        self.model_names_sort = sorted(list(self.models.keys()))
        self.raw_model_names_static = self.model_names_sort.copy()  # ALL individual models, fixed
        self.mean_models = ['average','median']
        self.fuseMdl_gen = fm.fusion_mdl_general(self.fusion_setup, self.log)
        if self.if_orthogonalize: self.log.log('Ortogonalization of model series is ON')
        else: self.log.log('Ortogonalization of model series is OFF')
        self.model_ranking = None

        # observations
        self.obsDataFormat = observations[0]    # netcdf or MMAS format
        self.obsCompletess = observations[1]    # requirement
        self.observation_FNm = observations[2]  # netcdf / MMAS main file
        self.station_FNm = observations[3]      # station file name for MMAS format
        if len(observations[4]) > 0:            # file name with stations to consider
#            stationsTmp = stations.readStations(observations[4],
#                                                columns = {'code':0,'longitude':1,'latitude':2,'name':3})
#            print(stationsTmp)
            self.stations2read = list(stations.readStations(observations[4][0],
                                                            columns = {'code':0,'longitude':1,
                                                                       'latitude':2,'name':3}))  # list of station codes
        else: self.stations2read = None

        self.archive = {}
        
    
    #=======================================================================

    def archive_last_day(self, today):
        #
        # Store the outcome of the previous day, with models active for that day
        #
        self.log.log(today.strftime('Archiving %Y%m%d_%H%M'))
        self.archive[self.today] = FUSE_archive_day(self.eval_stats,  
                                                    self.fuseMdl_gen, 
                                                    self.model_names_sort, self.mean_models,   # input + mean models
                                                    list(np.array(self.model_names_sort)
                                                         [self.idxMdlInformative])) # used models
        if self.ifPlotDraftPictures: self.plot_models_skills(today)   # a draft, without scales synchronization


    #=======================================================================

    def start_new_day(self, today, models):
        #
        # If the model is run in cycle over several fusion intervals,
        # it has to be partially reset, still maintaining the 
        # previous-days main outcome: fusion models coefficients and skills
        # They will be putshed to archiving dictionary and the current-round
        # structures will be reset
        self.today = today
        self.obsMatr = None  # if there is an overlap, one can try to re-use... eventually
        self.mdlMatr = {}
        self.mdlPool = None
        self.models = copy.copy(models)
        self.model_names_sort = sorted(list(self.models.keys()))
        self.fuseMdl_gen.start_new_dataset()
        self.log.log('\nStarting new time period for: ' + today.strftime('%Y%m%d_%H%M') + ' ' + 
                     ' '.join(list(self.models.keys())))
        return

    #=======================================================================

    
    def archive_to_pickle(self):
        #
        # Sends the whole archive to pickle binary
        #
        self.timer.start_timer('7_archive_to_pickle')
        self.log.log('Storing archive to pickle, ' + self.log.filename.replace('.log','.pickle'))
        archive_to_pickle(self.log.filename.replace('.log','.pickle'), self.archive)
        self.timer.stop_timer('7_archive_to_pickle')


    #=======================================================================

    def load_all_data_NC_input(self, learn_interval, fcst_interval):
        #
        # observational data straight from nc file
        # Do not forget: time tag is at the end of the interval, so 00:00 is previous-day-last-hour
        # We allo have two types of time intervals: learning and forecast periods
        # They can overlap. Forecast period, in case of evaluation regime, is also observed
        #
        self.timer.start_timer('0_load_data')
        self.learn_interval = (learn_interval[0] + spp.one_hour, learn_interval[1])  # cut yesterday-last-hour
        self.fcst_interval = (fcst_interval[0] + spp.one_hour, fcst_interval[1])
        if self.fusion_setup['if_evaluate_forecast']:
            toGet = (min(self.learn_interval[0], self.fcst_interval[0]),
                     max(self.learn_interval[1], self.fcst_interval[1]))
        else:
            toGet = self.learn_interval
        self.log.log('Total requested period %g days, %s - %s' % 
                     ((toGet[1] - toGet[0] + self.timestep_obs).days,
                      toGet[0].strftime('%Y%m%d_%H%M'), toGet[1].strftime('%Y%m%d_%H%M')))
        self.log.log('Getting the observations from: ' + self.observation_FNm)
        try:
            self.obsMatr = tv.TsMatrix.fromNC(self.observation_FNm, toGet[0], toGet[1], 
                                              self.stations2read)
        except:
            self.log.log('Failed to get observations. Exception is:')
            self.obsMatr = tv.TsMatrix.fromNC(self.observation_FNm, toGet[0], toGet[1], 
                                              self.stations2read)

        idxLearn = np.logical_and(self.obsMatr.times >= self.learn_interval[0],
                                  self.obsMatr.times <= self.learn_interval[1])
        self.learn_times = self.obsMatr.times[idxLearn] 
        self.log.log('Learning period %g days, %s - %s, %i stations' %
                     ((self.learn_times[-1] - self.learn_times[0] + self.timestep_obs).days,
                      self.learn_times[0].strftime('%Y%m%d_%H%M'), 
                      self.learn_times[-1].strftime('%Y%m%d_%H%M'), 
                      len(self.obsMatr.stations)))
        self.log.log('Forecast period: %s - %s' % 
                     (self.fcst_interval[0].strftime('%Y%m%d_%H%M'), 
                      self.fcst_interval[1].strftime('%Y%m%d_%H%M')))

#        for sts in self.obsMatr.stations:
#            self.log.log(str(sts))

        #
        # Model data, possibly, have to be extracted from the fields and only then consumed.
        # Alternatively, one can use pre-extracted fields, if available and allowed
        #
        # The model data must be loaded for both learning and forecast periods.
        # Compile the common time period. Note that the observational time step can be 
        # different from that of the models.
        #
        self.fcst_times = list((self.fcst_interval[0] + spp.one_hour * i 
                                for i in range(0, int(round((self.fcst_interval[1] - 
                                                             self.fcst_interval[0]) / 
                                                             self.timestep_mdl)) + 1, 1)))
        model_times = sorted(list(set(self.learn_times).union(set(self.fcst_times))))
        self.fcst_times = np.array(self.fcst_times)

        self.log.log('From models, request %g times, %s - %s' % 
                     (len(model_times), model_times[0].strftime('%Y%m%d_%H%M'), 
                      model_times[-1].strftime('%Y%m%d_%H%M')))
        for mdl in self.models.keys():
            if self.modelsPreExtracted:
                try:
                    self.mdlMatr[mdl] = tv.TsMatrix.fromNC(self.models[mdl][2] + '.extracted.nc',
                                                           model_times[0], model_times[-1], 
                                                           self.stations2read)
                    self.log.log('%s loaded %i times, %i stations' % 
                                 (mdl, len(self.mdlMatr[mdl].times), len(self.mdlMatr[mdl].stations)))
                    continue
                except: 
                    self.log.log('Failed model %s load from extracted, take main file' % str(mdl))
            # if we are here, extracted load has failed
            self.mdlMatr[mdl] = tv.TsMatrix.extract_from_fields(self.obsMatr.stations, 
                                                                model_times,
                                                                self.models[mdl][1],  #self.species[0], 
                                                                self.models[mdl][2]) 
        # A quick check for missing / empty models
        mdlMiss = []
        for mdl in self.models.keys():
            it=(np.any(np.isfinite(self.mdlMatr[mdl].vals),axis=1))
            if np.sum(it) < 1:
                self.log.log("No valid time steps. Remove the model %s" % (mdl))
                mdlMiss.append(mdl)
        for mdl in mdlMiss:
            self.mdlMatr.pop(mdl)
            self.models.pop(mdl)  # dictionary
            self.model_names_sort.remove(mdl)    # list
        
        self.timer.stop_timer('0_load_data')
        return len(self.mdlMatr.keys()) > 0


    #=======================================================================

    def load_all_data_MMAS_input(self, learning_days, unit):
        #
        # Get the stations 
        #
        self.timer.start_timer('0_load_data')
        columns={'latitude':4,'longitude':3,'code':2,'name':2}   # just for now, should be mde universal 
        sts = ts.readStations(self.station_FNm, columns=columns)
        self.log.log('Getting observation time series...')
        obs_series_lst = ts.fromFile(sts.values(), self.observation_FNm, timePos='start',
                                     columns = {'code': 0, 'quantity':1, 'year': 2, 'month': 3, 
                                                'day':4, 'hour':5, 'duration':6,'value':7},
                                     validator = ts.validIfNotNegative) #, quantity = chSpeciesNm)
        if len(obs_series_lst) < 1:
            self.log.log('Zero-length observation time series: %g stations, data file is %s' % 
                         (len(sts.values()), self.observation_FNm))
            raise
        # 
        # Adjust the parameters: need only stations and times available in observation set
        #
        tStart = min(list(min(obs_series.times()) for obs_series in obs_series_lst))
        tEnd = max(list(max(obs_series.times()) for obs_series in obs_series_lst))
        stCodes_ts = list(obs_series.station for obs_series in obs_series_lst)
        sts_keys_tmp = [k for k in sts.keys()]
        for stCode in sts_keys_tmp:
            if not sts[stCode] in stCodes_ts: sts.pop(stCode)
        #
        # Get the models time series using observation set as a driving
        # Note: timeseries are stored with the time stamp at the END of the time period
        # whereas the observations have it at the start of the period
        #
        mdl_series_set = {}
        for mdl in self.models.keys():
            self.log.log('Getting model time series:' + mdl)
            mdl_series_set[mdl] = ts.fromFile(sts.values(), self.models[mdl][2], 
                                              timePos='end', t1=tStart, t2=tEnd, 
                                              quantity = self.models[mdl][1],   # actually, species name 
                                              columns = {'code': 0, 'quantity':1, 'year': 2, 'month': 3, 
                                                         'day':4, 'hour':5, 'duration':6,'value':7},
                                                         validator = ts.validIfNotNegative)
            # ... adjust for the list of stations if models cannot do some
            stCodes_ts = list(mdl_series.station for mdl_series in mdl_series_set[mdl])
            for stCode in sts.keys():
                if not sts[stCode] in stCodes_ts: 
                    sts.pop(stCode)
                    for iTser in range(len(obs_series_lst)):
                        if obs_series_lst[iTser].station.code == stCode: 
                            obs_series_lst.pop(iTser)
                            break
        #
        # Convert time series to more convenient numpy structure: tsMatrix
        #
        self.obsMatr = tv.TsMatrix.fromTimeSeries(obs_series_lst, unit)
        self.mdlMatr = {}
        for mdl in mdl_series_set.keys():
            self.mdlMatr[mdl] = tv.TsMatrix.fromTimeSeries(mdl_series_set[mdl], unit)
        self.timer.stop_timer('0_load_data')

    
    #=======================================================================

    def clean_datasets(self):
        #
        # Finds the stations that have delivered data, were within the model domain and their
        # values were reasonable.
        # Similar for the models, which can be also strange
        #
        self.timer.start_timer('1_clean_datasets')
        # Start from the basic stupidity check
        stOK = np.any(np.isfinite(self.obsMatr.vals),axis=0)  # at least one valid observation
        if np.sum(stOK) != len(stOK):
            self.log.log('Removing stations with all nans: %g out of %g' % (np.sum(stOK), len(stOK)))
            self.obsMatr.vals = self.obsMatr.vals[:,stOK]
            self.obsMatr.stations = np.extract(stOK, self.obsMatr.stations) 
            for mdl in self.models.keys():
                self.mdlMatr[mdl].vals = self.mdlMatr[mdl].vals[:,stOK]
                self.mdlMatr[mdl].stations = np.extract(stOK, self.mdlMatr[mdl].stations) 
        # now all stations have at least something
        nTimes, nStat = self.obsMatr.vals.shape     # all times, all stations
        stOK = np.ones(shape=(nStat)) * True
        invalid_models = []
        for mdl in self.models.keys():  # some models might be without data or non-existent
#            try:
            stOK = np.logical_and(stOK, np.any(np.isfinite(self.mdlMatr[mdl].vals),axis=0))
#            except:
#                self.log.log('############ Problem with the model %s, removing ############' % mdl)
#                invalid_models.append(mdl)
        for mdl in invalid_models:
            self.models.pop(mdl)  # dictionary
            self.model_names_sort.remove(mdl)    # list
            try: self.mdlMatr.pop(mdl)  # dictionary, possibly without this model already
            except: pass
                
        nModelled = np.sum(stOK)
        # Eliminate the stations outside the modelling grids
        if nModelled < nStat:
            self.log.log('Stations inside the modelling grid: %i out of %i' % (nModelled,nStat))
            self.obsMatr.vals = self.obsMatr.vals[:,stOK]
            self.obsMatr.stations = np.extract(stOK, self.obsMatr.stations) 
            for mdl in self.models.keys():
                self.mdlMatr[mdl].vals = self.mdlMatr[mdl].vals[:,stOK]
                self.mdlMatr[mdl].stations = np.extract(stOK, self.mdlMatr[mdl].stations) 
        #
        # Identify the staions with strange values and remove these points
        # Check 1: 3 * standard deviation
        # Check 2: absolute limits
        obsMean = np.nanmean(self.obsMatr.vals)
        obsStd = np.nanstd(self.obsMatr.vals)
        ifBad = abs(np.where(np.isnan(self.obsMatr.vals), obsMean, self.obsMatr.vals) - 
                    obsMean) > 3. * obsStd
        ifBadStation = np.logical_or(np.nanmean(self.obsMatr.vals, 0) < minThreshold[self.species[0]],
                                     np.nanmean(self.obsMatr.vals, 0) > maxThreshold[self.species[0]])
        self.log.log('Obs Mean = %g and stdev = %g, values outside 3x std: %i' % 
                     (obsMean, obsStd, np.sum(np.where(ifBad, 1, 0))))
        self.obsMatr.vals[ifBad] = np.nan
        self.obsMatr.vals[:,ifBadStation] = np.nan
        self.log.log('New Obs Mean = %g and stdev = %g' % 
                     (np.nanmean(self.obsMatr.vals), np.nanstd(self.obsMatr.vals)))
        #
        # Find stations, which failed xx% of completeness threshold
        #
        stOK = np.any(np.isfinite(self.obsMatr.vals),axis=0)  # at least one valid observation
        stFull = np.all(np.isfinite(self.obsMatr.vals),axis=0) # all observation valid
        stThresh = np.sum(np.isfinite(self.obsMatr.vals),axis=0) > self.obsCompletess * nTimes
        nValid = np.sum(stOK)
        nFull = np.sum(stFull)
        nThreshOK = np.sum(stThresh)
        #
        # if majority of stations have full observed period, use them
        # Otherwise remove time slots that are poorly observed
        # but still require at least 75% of times observed and at least 75% 
        # of stations useful
        #
        self.log.log('Total %i, modelled %i; of these valid %i, full set %i, acceptable %i stations' %
                     (nStat, nModelled, nValid, nFull, nThreshOK))
#        print(stThresh, np.extract(stThresh, self.obsMatr.stations))
#        self.log.log('Above-threshold stations:' + 
#                     ' '.join(list((s.code for s in np.extract(stThresh,self.obsMatr.stations)))))

        if nFull > 0.9 * nValid:  # if most sites report 100% no need to compromise
            stOK = stFull
        elif nThreshOK > 0.5 * nValid: # can we get 50% of sites good completeness?
            stOK = stThresh
        else:
            self.log.log('No combination of times and stations: too poor observations')
            return False

        #
        # Final station list, only non-empty statoions
        #
        self.obsMatr.vals = self.obsMatr.vals[:,stOK]
        self.obsMatr.stations = np.extract(stOK, self.obsMatr.stations) 
        for mdl in self.models.keys():
            self.mdlMatr[mdl].vals = self.mdlMatr[mdl].vals[:,stOK]
            self.mdlMatr[mdl].stations = np.extract(stOK, self.mdlMatr[mdl].stations) 
        #
        # Final times: are there mad models?
        # Unlike stations, model is either mad or not - single point of nonsense should be
        # enough to remove the whole prediction. We shall use arithmetic average as a good
        # indicator of such nonsense: it is sensitive to even few mad points
        #
        avMdllst = []
        stdMdllst = []
        mdl_names = self.model_names_sort.copy()
        for mdl in mdl_names:
            if np.all(np.isfinite(self.mdlMatr[mdl].vals)):
                avMdllst.append(np.mean(self.mdlMatr[mdl].vals))
                stdMdllst.append(np.std(self.mdlMatr[mdl].vals))
            else:
                self.log.log('########## NaN in the model %s #########' % mdl)
                self.mdlMatr.pop(mdl)
                self.model_names_sort.remove(mdl)
                self.models.pop(mdl)
        avMdl = np.array(avMdllst)
        stdMdl = np.array(stdMdllst)
        avGlob = np.nanmedian(avMdl)
        stdGlob = np.nanmedian(stdMdl)
        # Now, check for reasnoble range of all values
        mdl_names = self.model_names_sort.copy()
        for imdl in range(len(mdl_names)):
            mdlOK = np.all(np.isfinite([avMdl[imdl], stdMdl[imdl]]))
            if mdlOK:
                mdlOK = (avMdl[imdl] < avGlob * 10 and avMdl[imdl] > avGlob * 0.1 and 
                         stdMdl[imdl] < stdGlob * 10 and stdMdl[imdl] > stdGlob * 0.1)
            if not mdlOK: #[imdl]:
                self.log.log('########## problem with %s predictions ######' % mdl_names[imdl])
                self.log.log('Global mean = %g, %s mean = %g, global stdDev = %g, %s stdDev = %g' %
                             (avGlob, mdl_names[imdl], avMdl[imdl], 
                              stdGlob, mdl_names[imdl], stdMdl[imdl]))
                self.mdlMatr.pop(mdl_names[imdl])
                self.model_names_sort.remove(mdl_names[imdl])
                self.models.pop(mdl_names[imdl])
            
        self.timer.stop_timer('1_clean_datasets')
        return len(self.model_names_sort) > 0   # let's require at least 1 model in the ensemble


    #=======================================================================

    def finalise_model_ensembles(self):
        #
        # Creates a unified matrix for al models together, alphbetic sorting
        # Adds two mean ensembles - arithmetic average and median
        #
        self.timer.start_timer('2_finalise_ensemble')
        # Model pool as a special class
#        self.mdlPool = FUSE_modelling_pool(self.model_names_sort + self.mean_models,
#                                           self.mdlMatr[self.model_names_sort[0]].times,
#                                           self.mdlMatr[self.model_names_sort[0]].stations)
        # Model pool as a multi-variable tsMatrix
        self.mdlPool = tv.TsMatrix(self.mdlMatr[self.model_names_sort[0]].times, 
                                   self.mdlMatr[self.model_names_sort[0]].stations, 
                                   self.model_names_sort,
                                   np.ones((len(self.model_names_sort),
                                            len(self.mdlMatr[self.model_names_sort[0]].times),
                                            len(self.mdlMatr[self.model_names_sort[0]].stations)),
                                           dtype=np.float32) * np.nan, 
                                   self.mdlMatr[self.model_names_sort[0]].units * 
                                                                len(self.model_names_sort),
                                   self.mdlMatr[self.model_names_sort[0]].fill_value,
                                   self.mdlMatr[self.model_names_sort[0]].timezone)
        for iMdl in range(len(self.model_names_sort)):
            self.mdlPool.vals[iMdl,:,:] = self.mdlMatr[self.model_names_sort[iMdl]].vals
        subst = self.models[self.model_names_sort[0]][1]
        #
        # Fitting has to go over the common time period whereas they are
        # different between the models and observations. Find the common one
        #
        self.times_common = sorted(list(set(self.learn_times).
                                        intersection(set(self.obsMatr.times)).
                                        intersection(set(self.mdlPool.times)))) 
        self.tc_indices_obs = np.searchsorted(self.obsMatr.times, self.times_common)
        self.tc_indices_mdl = np.searchsorted(self.mdlPool.times, self.times_common)
        #
        # Mean of the ensemble
        #
        self.MeanMdls = tv.TsMatrix(self.mdlMatr[self.model_names_sort[0]].times, 
                                    self.mdlMatr[self.model_names_sort[0]].stations, 
                                    self.mean_models,
                                    np.ones((len(self.mean_models),
                                             len(self.mdlMatr[self.model_names_sort[0]].times),
                                             len(self.mdlMatr[self.model_names_sort[0]].stations)),
                                            dtype=np.float32) * np.nan, 
                                    self.mdlMatr[self.model_names_sort[0]].units * 
                                                                            len(self.mean_models),
                                    self.mdlMatr[self.model_names_sort[0]].fill_value,
                                    self.mdlMatr[self.model_names_sort[0]].timezone)
        for avMdl in self.mean_models:
            self.models[avMdl] = [avMdl, subst,'dummy']
            if avMdl == 'average': 
                self.MeanMdls.vals[0,:,:] = np.average(self.mdlPool.vals[:,:,:], axis=0)
            elif avMdl == 'median':
                self.MeanMdls.vals[1,:,:] = np.median(self.mdlPool.vals[:,:,:], axis=0)
            else:
                self.log.log('Unknown mean model:' + avMdl)
                return False
#        self.model_names_sort += self.mean_models

        #
        # Having all models at hand, can compute the model ranking
        #
        # Shall we sort via RMSE or correlation? Keeping in mind that debiasing and scaling
        # are trivial for projection algorithm, one can argue for correlation. The diurnal
        # kernel is then a problem however. A paliative solution is to de-bias the models
        # and then take RMSE accounting for the diurnal summation kernel 
        #
        if self.fusion_setup['model_ranking_criterion'][0].upper() == 'RMSE':
            # For sorting along the RMSE with de-biasing
            bias = np.array(list((np.nanmean((self.mdlPool.vals[iMdl, self.tc_indices_mdl, :] -
                                              self.obsMatr.vals[self.tc_indices_obs, :]))  # * np.sqrt(kernel)) 
                                              for iMdl in range(self.mdlPool.vals.shape[0]))))
            MSE = list((np.nanmean(np.square(self.mdlPool.vals[iMdl,self.tc_indices_mdl,:] - 
                                             self.obsMatr.vals[self.tc_indices_obs,:] - 
                                             bias[iMdl])) # * times_kernel[:,None])
                        for iMdl in range(self.mdlPool.vals.shape[0])))
            self.model_ranking = np.argsort(np.array(MSE))   # indices that sort the RMSE array

        elif self.fusion_setup['model_ranking_criterion'][0].upper() == 'CORRELATION':
            # For sorting along the correlation
            corr = list((spp.nanCorrCoef(self.mdlPool.vals[iMdl, self.tc_indices_mdl, :],
                                         self.obsMatr.vals[self.tc_indices_obs,:])
                         for iMdl in range(self.mdlPool.vals.shape[0])))
            self.model_ranking = np.argsort(np.array(corr))[::-1]   # indices that sort corr array in DEscending order

        #
        # If space-resolving methods are used, we will need:
        #    - the main fusion grid
        #    - the set of subgrids
        #    - the structure function of the best model at the subgrid-split resolution level
        #
        if np.any(list((fm.ifSpaceResolving(m) for m in self.fusion_methods))):
            #
            # Open the best-model field file
            mdlTmp = self.models[self.model_names_sort[self.model_ranking[0]]]
            self.fusion_grid = silamfile.SilamNCFile(mdlTmp[2]).get_reader(mdlTmp[1]).grid
            #
            # Geometry of the subdomains
            #
            iWinSize = int(self.fusion_setup['multifit_window_size_cells'][0])
            self.arGrids, self.arWinStart = fm.create_subdomains(
                                                        iWinSize,
                                                        int(self.fusion_setup
                                                            ['multifit_window_overlap_cells'][0]),
                                                        self.fusion_grid, self.log)
            #
            # The structure function 
            #
            self.winSTF = StF.structure_function('fusion_subsets_StF', 
                                                 iWinSize, 0,   # gridSTF_factor, filterRad,
                                                 self.times_common[0], len(self.times_common),
                                                 self.log)
            # make it accounting for the overlap
#            self.winSTF.make_StF_4_subdomains_TSM(self.obsMatr.stations, 
#                                                  mdlTmp[2], self.species[0],
#                                                  self.arWinStart, patch_zero=1, 
#                                                  ifNormalize=False)
            self.winSTF.make_StF_4_subdomains_map(mdlTmp[2], mdlTmp[1],    #self.species[0],
                                                  self.arWinStart, patch_zero=3, 
                                                  ifNormalize=False)
            if np.any(self.winSTF.structFunMap == 0):
                self.log.log('Zero structure function ' + mdlTmp[2])
                raise ValueError
#            #
#            # TEST drawing
#            #
#            spp.ensure_directory_MPI(os.path.join(self.output_directory,'Struct_Fun_dump3'))
#            self.winSTF.draw(os.path.join(self.output_directory,'Struct_Fun_dump3'),
#                             'StF_%s' % mdlTmp[0], 0.01)
        else:
            self.fusion_grid = None

        self.timer.stop_timer('2_finalise_ensemble')
        return True

        
    #=======================================================================

    def fit_fusion_coefficients(self):
        #
        # A driver / selector for the fusion models
        # Note that fusion_coefs will have different meaning and dimensions
        # depending on the fusion method.
        # Prediction here is the prediction for the learning dataset
        #
        self.timer.start_timer('3_fit_coefficientss')
        #
        # sizes for work
        #
        nTimes = len(self.times_common)
        #
        # Diurnal kernel should be of size of observations
        # In generic case, it is just one. Will be redefined later if diurnal profile
        # is needed or if obs-related covariance will be involved
        #
        #
        # The temporal resolution of the coefficients can be daily and hourly
        if self.ifHourlyWeights:
            nSteps = 24
            # Kernel should be symmetrical with 1.0 for the hour in question and minimum at +12 hours
            kernel_gauss = np.zeros(shape=(24))
            kernel_gauss[:23] = signal.gaussian(23, float(self.fusion_setup
                                                          ['hourly_correlation_period_hrs'][0])) # symmetric, peak at 11
            kernel_gauss[23] = kernel_gauss[22]
            nValidTimesInDay = np.sum(kernel_gauss > 0)
        else:
            nSteps = 1
            nValidTimesInDay = 24

        self.log.log('Fitting common time period: ' + str(nTimes) + ' hours, %i steps, %s' % 
                     (nSteps, self.species[0]))
        #
        # How to split the datasets to tesst and training?
        # hourly_random; daily_random, daily_first_days_test, daily_last_days_test
        # For analysis, random hourly split is quite OK
        # For forecast and evaluation, whole days should be taken as tests
        #
        if self.run_type == 'FORECAST' or self.run_type == 'EVALUATION':
            if nTimes >= 48: splitMethod = 'daily_last_test'   #'daily_random'
            else: splitMethod = 'native_random'
        elif self.run_type == 'REANALYSIS':
            splitMethod = 'native_random'
        else:
            self.log.log('Unknown run_type ' + self.run_type)
        #
        # Ortogonalise the model dataset. Send only basic models, no mean models
        # Orthogonal model pool will come reordered, according to RMSE.
        # Split the datasets and flatten them
        #
        if self.if_orthogonalize:
            #
            # call the ortogonalization routine
            # It also censors the models preserving only informative ones.
            #
            kernelFlat = np.ones(shape = self.obsMatr.vals[self.tc_indices_obs,:].shape)
            self.poolMdlOrt, self.idxMdlInformative = self.fuseMdl_gen.ortogonalise_and_censor(
                                                            self.obsMatr, self.mdlPool, 
                                                            kernelFlat, self.model_ranking,
                                                            self.tc_indices_obs, self.tc_indices_mdl)
        else:
            # If orthogonalization is not required, set a view excluding the mean models
            self.poolMdlOrt = self.mdlPool.vals[:,:,:] 
            self.log.log('Models: ' + ' '.join(self.model_names_sort))
            self.model_ranking = list(range(len(self.model_names_sort)))
            self.idxMdlInformative = self.model_ranking
        #
        # The main step: either once for the whole period or 24 times for 24 hours of the day
        #
        for iStep in range(nSteps):
            if nSteps > 1: self.log.log('Fit step %i' % iStep)
            # For hourly resolution, kernel has to be made here
            if self.ifHourlyWeights:
                krnl = np.roll(kernel_gauss, iStep - 11)  # roll the array to put Gaussian max to the iHr
                times_kernel = np.array(list((krnl[self.times_common[i].hour] for i in range(nTimes))))
            else:
                times_kernel = np.ones(shape = (nTimes))
            #
            # Split times to training-test and perform fitting
            #
            self.fuseMdl_gen.fusion_step(self.obsMatr, self.tc_indices_obs,
                                         self.poolMdlOrt, self.tc_indices_mdl, 
                                         self.times_common, iStep, nValidTimesInDay, splitMethod,
                                         times_kernel, self.fusion_grid, self.arGrids,
                                         self.arWinStart, self.winSTF, self.rnGen)
        self.timer.stop_timer('3_fit_coefficientss')
        return True


    #=======================================================================

    def predict(self):
        #
        # Makes predictoin over the forecasting dates.
        # note that the modelling pool includes both learnig and forecast times
        #
        self.timer.start_timer('4_predict')
        #
        # select the dates we need to forecast
        #
        fcstIdxTmp = np.searchsorted(self.mdlMatr[self.model_names_sort[0]].times, self.fcst_times)
        idxValidTimes = fcstIdxTmp < self.mdlMatr[self.model_names_sort[0]].times.shape[0]
        fcstIdx = fcstIdxTmp[idxValidTimes]
#        print(self.fcst_times)
#        self.log.log('Predicting:' + ' '.join(list((str(t) for t in self.mdlMatr[self.model_names_sort[0]].times[fcstIdx]))))
        #
        # Make the forecast with all fusion models. The coefs are determined from the past,
        # fixed for the future, so just apply for the whole forecast horizon
        #
        self.FUSEpred = [{},      # place for forecasts
                         self.mdlMatr[self.model_names_sort[0]].times[fcstIdx]]  # times of forecasts
        #
        # The only complexity appears if a diurnal variation is accounted for in fusion weights
        # Those have to be scanned picking the right hour of prediction
        #
        if self.ifHourlyWeights:
            for fk in self.fusion_methods:
                # Create the space: (times, stations)
                self.FUSEpred[0][fk] = np.zeros(shape = (fcstIdx.shape[0], self.poolMdlOrt.shape[2]))
                # Now, fill other hours one by one
                for iHr in range(24):
                    # Filter for this hour
                    idxHr = np.array(list((t.hour == iHr for t in self.FUSEpred[1] )))
#                    print('Hour:',iHr,'idxHr', idxHr)
                    # pick the needed values
                    if self.if_orthogonalize:
                        self.FUSEpred[0][fk][idxHr,:] = (self.fuseMdl_gen.fusion_mdl[fk][iHr].
                                                            predict(self.poolMdlOrt[:,fcstIdx,:])[idxHr,:])

#                        self.log.log('Models for 3 sites, valid times at hour ' + str(iHr) + 
#                                     str(self.poolMdlOrt[:,fcstIdx,:][:,idxHr,:3]))
#                        self.log.log('Prediction for 3 sites: ' + str(self.FUSEpred[fk][idxHr,:3]))

                    else:
                        self.FUSEpred[0][fk][idxHr,:] = (self.fuseMdl_gen.fusion_mdl[fk][iHr].
                                                            predict(self.mdlPool.vals[:,fcstIdx,:])[idxHr,:])
        else:
            for fk in self.fusion_methods:
                if self.if_orthogonalize:
                    self.FUSEpred[0][fk] = self.fuseMdl_gen.fusion_mdl[fk][0].predict(
                                                                        self.poolMdlOrt[:,fcstIdx,:])
                else:
                    self.FUSEpred[0][fk] = self.fuseMdl_gen.fusion_mdl[fk][0].predict(
                                                                        self.mdlPool.vals[:,fcstIdx,:])


#        for iStation in range(len(self.obsMatr.stations)):
#            for imdl in range(len(self.raw_model_names_static)):
#                mdl = self.model_names_sort[imdl]
#                self.log.log('Predicting %s %s %s' % 
#                             (mdl, self.obsMatr.stations[iStation].code,
#                              ' '.join(list((str(v) for v in self.mdlPool.vals[imdl,fcstIdx,iStation])))))
#                if self.if_orthogonalize:
#                    self.log.log('Predicting %s_ort %s %s' % 
#                                 (mdl, self.obsMatr.stations[iStation].code,
#                                  ' '.join(list((str(v) for v in self.poolMdlOrt[imdl,fcstIdx,iStation])))))
#            for fk in self.fusion_methods:
#                self.log.log('Predicting %s %s %s' % 
#                             (fm, self.obsMatr.stations[iStation].code,
#                              ' '.join(list((str(v) for v in self.fusion_predict[fm][:,iStation])))))

        self.timer.stop_timer('4_predict')
        return


    #=======================================================================

    def apply_forecast(self, ifPrintMaps):
        #
        # Makes predictoin over the forecasting dates.
        # note that the modelling pool includes both learnig and forecast times
        # The difference from predict is just that it has to deal with maps rather than
        # with tsMatrices. The maps have to be read from disk and the forecasts have to be stored
        # at once: for long time periods it may be too much to keep in memory
        #
        self.timer.start_timer('8_apply_fcst')
        #
        # select the dates we need to forecast
        #
        fcstIdxTmp = np.searchsorted(self.mdlMatr[self.model_names_sort[0]].times, self.fcst_times)
        idxValidTimes = fcstIdxTmp < self.mdlMatr[self.model_names_sort[0]].times.shape[0]
        fcstIdx = fcstIdxTmp[idxValidTimes]
        fcst_times_loc = self.mdlMatr[self.model_names_sort[0]].times[fcstIdx]
        fusionMethods_abbr = list((self.fuseMdl_gen.fusion_mdl[fk][0].abbrev 
                                  for fk in self.fusion_methods))
#        print(self.fcst_times)
#        self.log.log('Predicting:' + ' '.join(list((str(t) for t in self.mdlMatr[self.model_names_sort[0]].times[fcstIdx]))))
        #
        # Make the forecast with all fusion models. The coefs are determined from the past,
        # fixed for the future, so just apply over the whole forecast horizon
        #
        # Input readers
        readers = {}
        for mdl in self.model_names_sort:
            readers[mdl] = silamfile.SilamNCFile(self.models[mdl][2]).get_reader(self.models[mdl][1])
        mapsIn = np.zeros(shape=(len(self.model_names_sort),
                                 self.fusion_grid.nx, self.fusion_grid.ny), dtype=np.float32)
        # output space:
        fcst_maps = np.zeros(shape=(len(self.fusion_methods), 
                                    self.fusion_grid.nx, self.fusion_grid.ny), dtype=np.float32)
        # output file:
        unitDic = {}
        for fk in fusionMethods_abbr:
            unitDic['cnc_%s_fm_%s' % (self.species[1], fk)] = self.mdlPool.units[0]

        outF = silamfile.open_ncF_out(os.path.join(self.output_directory,'fcst','fcst_%s_%s_%s.nc4' %
                                                   (self.species[1], fcst_times_loc[0].strftime('%Y%m%d'),
                                                    fcst_times_loc[-1].strftime('%Y%m%d'))),
                                      'NETCDF4', self.fusion_grid, silamfile.SilamSurfaceVertical(),
                                      self.learn_times[-1], fcst_times_loc, [], 
                                      list(('cnc_%s_fm_%s' % 
                                            (self.species[1], fk) for fk in fusionMethods_abbr)),
                                      unitDic, -999999, True, ppc=None, hst=None)
        if ifPrintMaps: 
            zipOut = ZipFile(os.path.join(self.output_directory,'fcst', 'fcst_%s_%s_%s.zip_tmp' %
                                          (self.species[1], fcst_times_loc[0].strftime('%Y%m%d'),
                                           fcst_times_loc[-1].strftime('%Y%m%d'))),'w')
        #
        # Read input and predict.
        # The only complexity appears if a diurnal variation is accounted for in fusion weights
        # Those have to be scanned picking the right hour of prediction
        #
        for it, t in enumerate(fcst_times_loc):
            #
            # Read the input maps for the given time
            for iMdl, mdl in enumerate(self.model_names_sort):
                readers[mdl].goto(t)
                try: mapsIn[iMdl,:,:] = readers[mdl].read(1).data
                except: mapsIn[iMdl,:,:] = readers[mdl].read(1)
            #
            # Apply the fusion methods one by one
            for ifm, fk in enumerate(self.fusion_methods):
                # make forecast map for this hour
                if self.ifHourlyWeights:  # hourly weights, each hour has own fusion model
                    fcst_maps[ifm,:,:] = self.fuseMdl_gen.fusion_mdl[fk][t.hour].predict(
                                                            mapsIn[self.idxMdlInformative,:,:])
                else:  # daily weights, same fusion model for all hours
                    fcst_maps[ifm,:,:] = self.fuseMdl_gen.fusion_mdl[fk][0].predict(
                                                            mapsIn[self.idxMdlInformative,:,:])
                # Store to output. Note that y is first, x is second
                outF.variables['cnc_%s_fm_%s' % 
                               (self.species[1], fusionMethods_abbr[ifm])][it,:,:] = fcst_maps[ifm,:,:].T
            # print?
            if ifPrintMaps:
                drawer.draw_map_with_points(list(('%s, %s forecast %s + %g hrs' % 
                                                  (self.species[0], 
                                                   self.fuseMdl_gen.fusion_metadata(fk)[2],  # abbreviation 
                                                   t.strftime('%Y%m%d'), 
                                                   (t-fcst_times_loc[0]).total_seconds() / 3600.) 
                                                  for fk in self.fusion_methods)),
                                             [], [], self.fusion_grid,  # lons, lats are used if points are active 
                                             os.path.join(self.output_directory,'fcst'),
                                             'fcst_%s_%s_%s_h%03g' % (self.species[0], 
                                                                     'fusion_%g_mdls' % 
                                                                     len(self.fusion_methods),
                                                                     t.strftime('%Y%m%d%H'),
                                                                     (t - fcst_times_loc[0]
                                                                      ).total_seconds() / 3600.),
                                             vals_points=None,
                                             vals_map = fcst_maps[:,:,::-1],
                                             chUnit=unitDic['cnc_%s_fm_%s' % 
                                                            (self.species[1],fusionMethods_abbr[ifm])],
                                             ifNegatives=True, #, numrange=(1e-2,100))
                                             zipOut = zipOut)
        outF.close()
        if ifPrintMaps: 
            nameTmp = zipOut.filename
            nameFinal = nameTmp.replace('_tmp','')    # final name for archive
            zipOut.close()   # nameTmp file ready
            if os.path.exists(nameFinal):          # already exists?
                if os.path.exists(nameFinal + '_prev'): os.remove(nameFinal + '_prev')  # ... previous one?
#                os.rename(nameFinal, nameFinal + '_prev')   # existing to backup. Takes space!!
                os.remove(nameFinal)   # remove existing
            os.rename(nameTmp, nameFinal) # rename to actual zip
        self.timer.stop_timer('8_apply_fcst')


    #=======================================================================

    def compute_fitting_skills(self):
        #
        # For the current set of stations, model data and fusion predictions for
        # the learning and forecasting time ranges, comptue the set of statistics
        #
        self.timer.start_timer('5_evaluate')
        nMdls = len(self.model_names_sort)
        nMeanMdls = len(self.mean_models)
        self.statNms = 'bias correlation RMSE stdevRatio'.split()
        self.statUnits = 'ug/m3 relative ug/m3 relative'.split()
        self.eval_stats = {'learn': np.zeros(shape=(len(self.statNms), 
                                                    nMdls + nMeanMdls + len(self.fusion_methods)))}
        #
        # Construct the evaluation time ranges and identify the indices in the tsMatrices
        # Forecast skills will be reported day by day
        #
        # For daily time step in drawing
        #
        sets4eval = [(self.learn_times,  # period to evaluate 
                      (self.fuseMdl_gen.fusion_verify, self.learn_times),  # fusion prediction over learning period
                      'learn')]
        #
        # Computing duration in days, account for averaging interval of the first timestep
        # the time tag is end-hour, so I need to leave out 00 hour and start from 01
        #
        dayStart = self.fcst_times[0].replace(hour=1, minute=0, second=0, microsecond=0)
        
        for iDay in range((self.fcst_times[-1] - self.fcst_times[0] + self.timestep_mdl).days):
            dayTimes = self.fcst_times[np.logical_and(self.fcst_times >= 
                                                      dayStart + iDay * spp.one_day,
                                                      self.fcst_times < 
                                                      dayStart + (iDay+1) * spp.one_day)]
            dayTimesOK = np.array(sorted(list(set(dayTimes).intersection(set(self.obsMatr.times)).
                                              intersection(set(self.mdlPool.times)))))
            if dayTimesOK.shape[0] < 1: continue
            
            tag = 'fcst_d_%g' % iDay
            self.eval_stats[tag] = np.zeros(shape=(len(self.statNms), 
                                                   nMdls + nMeanMdls + len(self.fusion_methods)))
            sets4eval.append((dayTimesOK, self.FUSEpred, tag))
#        for ev in sets4eval:
#            self.log.log('Evaluation set %s %s' % 
#                         (ev[2], ' '.join(list((t.strftime('%m%d-%H%M') for t in ev[0])))))
        #
        # For hourly time step in drawing
        #
        for it, t in enumerate(self.learn_times):
            tag = 'lrn_Hr%03i' % it
            sets4eval.append(([t],  # period to evaluate 
                              (self.fuseMdl_gen.fusion_verify, self.learn_times),  # fusion prediction over learning period
                              tag))
            self.eval_stats[tag] = np.zeros(shape=(len(self.statNms),
                                                   nMdls + nMeanMdls + len(self.fusion_methods)))
        #
        # Computing duration in days, account for averaging interval of the first timestep
        # the time tag is end-hour, so I need to leave out 00 hour and start from 01
        #
        dayStart = self.fcst_times[0].replace(hour=1, minute=0, second=0, microsecond=0)

        for iHr in range(np.int(np.round((self.fcst_times[-1] - self.fcst_times[0] + 
                                          self.timestep_mdl).total_seconds() / 3600))):
            dayTimes = self.fcst_times[np.logical_and(self.fcst_times >= 
                                                      dayStart + iHr * spp.one_hour,
                                                      self.fcst_times < 
                                                      dayStart + (iHr+1) * spp.one_hour)]
            dayTimesOK = np.array(sorted(list(set(dayTimes).intersection(set(self.obsMatr.times)).
                                              intersection(set(self.mdlPool.times)))))
            if dayTimesOK.shape[0] < 1: continue

            tag = 'fcst_hr%03g' % iHr
            self.eval_stats[tag] = np.zeros(shape=(len(self.statNms), 
                                                   nMdls + nMeanMdls + len(self.fusion_methods)))
            sets4eval.append((dayTimesOK, self.FUSEpred, tag))
        
        #
        # Scan the sets for the evaluation, evaluate and report the results
        #
        for timesOK, (prediction, predTimes), tag in sets4eval:
            # find times available from both observations and models
            obsT2take = np.searchsorted(self.obsMatr.times, timesOK)  # their indices
            obs = self.obsMatr.vals[obsT2take,:].T.flatten()               # corresponding data
            idxOK = np.isfinite(obs)    # bad stations have been eliminated but holes still there
            self.log.log('Reporting %s %s, (times,stations) = %s, total points = %g, period = %s - %s' % 
                         (tag, self.species[0], str(self.obsMatr.vals[obsT2take,:].shape), len(obs),
                          timesOK[0], timesOK[-1]))

#            self.log.log('Doing set %s %s' % 
#                         (tag, ' '.join(list((t.strftime('%m%d-%H%M') for t in times))))) 
#            self.log.log('Measurement obs %s' % (' '.join(list((str(v) for v in obs[idxOK])))))
            
            
            
            # get the model pool indices
            mdlT2take_ = np.searchsorted(self.mdlPool.times, timesOK)  # their indices
            mdlT2take = mdlT2take_[mdlT2take_ < self.mdlPool.times.shape[0]]
            iMdl = 0
            chMdlNms = []
            # Individual models
            for chMdlNm in self.model_names_sort:
                chMdlNms.append(chMdlNm)
                mdl = self.mdlPool.vals[iMdl,mdlT2take,:].T.flatten()
#                self.log.log('Prediction %s %s' % (chMdlNm, ' '.join(list((str(v) for v in mdl[idxOK])))))
                self.eval_stats[tag][0,iMdl] = (mdl[idxOK] - obs[idxOK]).mean()
                self.eval_stats[tag][1,iMdl] = spp.nanCorrCoef(mdl[idxOK], obs[idxOK])
                self.eval_stats[tag][2,iMdl] = np.sqrt(np.square(mdl[idxOK] - obs[idxOK]).mean())
                self.eval_stats[tag][3,iMdl] = np.std(mdl[idxOK]) / np.std(obs[idxOK])
                iMdl += 1
            # mean models
            for iMM, chMdlNm in enumerate(self.mean_models):
                chMdlNms.append(chMdlNm)
                mdl = self.MeanMdls.vals[iMM,mdlT2take,:].T.flatten()
#                self.log.log('Prediction %s %s' % (chMdlNm, ' '.join(list((str(v) for v in mdl[idxOK])))))
                self.eval_stats[tag][0,iMdl] = (mdl[idxOK] - obs[idxOK]).mean()
                self.eval_stats[tag][1,iMdl] = spp.nanCorrCoef(mdl[idxOK], obs[idxOK])
                self.eval_stats[tag][2,iMdl] = np.sqrt(np.square(mdl[idxOK] - obs[idxOK]).mean())
                self.eval_stats[tag][3,iMdl] = np.std(mdl[idxOK]) / np.std(obs[idxOK])
                iMdl += 1
            # fusion models
#            fusionT2take = np.searchsorted(self.fcst_times, timesOK)  # their indices
            fusionT2take = np.searchsorted(predTimes, timesOK)  # their indices
            fusionT2takeOK = fusionT2take[fusionT2take < self.mdlPool.times.shape[0]]
            for fk in self.fusion_methods:
                fkNm = self.fuseMdl_gen.fusion_mdl[fk][0].abbrev
                chMdlNms.append(fkNm)
                mdl = prediction[fk][fusionT2takeOK,:].T.flatten()
#                self.log.log('Prediction %s %s' % (fkNm, ' '.join(list((str(v) for v in mdl[idxOK])))))
                self.eval_stats[tag][0,iMdl] = (mdl[idxOK] - obs[idxOK]).mean()
                self.eval_stats[tag][1,iMdl] = spp.nanCorrCoef(mdl[idxOK], obs[idxOK])
                self.eval_stats[tag][2,iMdl] = np.sqrt(np.square(mdl[idxOK] - obs[idxOK]).mean())
                self.eval_stats[tag][3,iMdl] = np.std(mdl[idxOK]) / np.std(obs[idxOK])
                iMdl += 1
            for iMdl in range(len(chMdlNms)):
                self.log.log('%s bias %g, corr %g, rmse %g, stdevRatio %g' %
                             (chMdlNms[iMdl], self.eval_stats[tag][0,iMdl], self.eval_stats[tag][1,iMdl], 
                              self.eval_stats[tag][2,iMdl], self.eval_stats[tag][3,iMdl]))
        self.timer.stop_timer('5_evaluate')
        return

        
    #=======================================================================
    
    def plot_models_skills(self, day_to_draw=None, tag_filter=None):
        #
        # Prints a comparison table for the skils of the individual models and
        # fusion prediction
        # Draw the basic staistics: a bar chart, for both learning and forecasting statistics
        #
        self.timer.start_timer('6_plotting')
        fusionMethods = list((self.fuseMdl_gen.fusion_mdl[fk][0].abbrev for fk in self.fusion_methods))
        all_models = self.model_names_sort + self.mean_models + fusionMethods
        timeResolTag = {True:'hourly',False:'daily'}[self.ifHourlyWeights]
        if tag_filter is None: 
            tags2plot = list(self.eval_stats.keys())
        else:
            tags2plot = []
            for tag in self.eval_stats.keys():
                for tf in tag_filter:
                    if tf in tag: tags2plot.append(tag)
            
        if day_to_draw is not None:
            days2draw = [day_to_draw]
            self.log.log(day_to_draw.strftime('Fusion and individual model scores for %Y%m%d_%H%M'))
        else:
            days2draw = sorted(list(self.archive.keys()))
            self.log.log('Fusion and individual model scores for %s - %s' % 
                         (days2draw[0].strftime('%Y%m%d_%H%M'), days2draw[-1].strftime('%Y%m%d_%H%M')))
            mdl_scores = {}                # prepare for time series of skills
            for tag in tags2plot:   #['learn','fcst']:
                mdl_scores[tag] = {}
                for s in self.statNms:     # statistics
                    mdl_scores[tag][s] = {}
                    for mdl in self.raw_model_names_static + self.mean_models:  # all models
                        mdl_scores[tag][s][mdl] = np.ones(shape=(len(days2draw))) * np.nan
                    for fk in self.fuseMdl_gen.fusion_mdl.keys():                        # all fusion
                        mdl_scores[tag][s][self.fuseMdl_gen.fusion_mdl[fk][0].
                                           abbrev] = np.ones(shape=(len(days2draw))) * np.nan
            mdl_weights = {}
            for fk in fusionMethods:      # fusion methods
                mdl_weights[fk] = {}
                for mdl in self.raw_model_names_static + ['intercept']:   # raw models
                    mdl_weights[fk][mdl] = np.ones(shape=(len(days2draw))) * np.nan
        self.log.log('Time tags to plot: ' + ' '.join(tags2plot))
        #
        # If many days, find min-max values for all statistics, for the sake of unified axes
        #
        if len(days2draw) > 1:
            minmax_stat = {}
            for iStat in range(len(self.statNms)): 
                stNm = self.statNms[iStat]
                minmax_stat[stNm] = [1e20,-1e20]
                for day in days2draw:
#                    for stats in [self.archive[day].learn_stats, self.archive[day].fcst_stats]:
                    for tag in tags2plot:
                        stats = self.archive[day].eval_stats[tag]
                        minmax_stat[stNm] = [min(minmax_stat[stNm][0], min(stats[iStat,:])),
                                             max(minmax_stat[stNm][1], max(stats[iStat,:]))]
            #
            # ... same for weiguts
            # for space-resolving methods have to take a mean value over the grid
            #
            minmax_stat['weights'] = [1e20,-1e20]  # continue the same dictionary
            minmax_stat['intercept'] = [1e20,-1e20]  # continue the same dictionary
            for day in days2draw:
                for fk in self.fusion_methods:
                    for it in range(len(self.archive[day].fusion_mdls.fusion_mdl[fk])):
                        if fm.ifSpaceResolving(fk):
                            out_x = np.mean(self.archive[day].fusion_mdls.fusion_mdl[fk][it].
                                            tsMdlCoefs.vals[:,:,:],axis=2)
                            minmax_stat['weights'] = [min(minmax_stat['weights'][0], min(out_x[1:])),
                                                      max(minmax_stat['weights'][1], max(out_x[1:]))]
                            minmax_stat['intercept'] = [min(minmax_stat['intercept'][0], out_x[0]),
                                                        max(minmax_stat['intercept'][1], out_x[0])]
                        else:
                            minmax_stat['weights'] = [min(minmax_stat['weights'][0], 
                                                          min(self.archive[day].fusion_mdls.fusion_mdl[fk][it].
                                                              out.x[1:])),
                                                      max(minmax_stat['weights'][1], 
                                                          max(self.archive[day].fusion_mdls.fusion_mdl[fk][it].
                                                              out.x[1:]))]
                            minmax_stat['intercept'] = [min(minmax_stat['intercept'][0], 
                                                            self.archive[day].fusion_mdls.fusion_mdl[fk][it].
                                                            out.x[0]),
                                                        max(minmax_stat['intercept'][1], 
                                                            self.archive[day].fusion_mdls.fusion_mdl[fk][it].
                                                            out.x[0])]
            # 
            # For better appearance, widen the ranges and force zero if dynamic range > 3 times
            #
            for stNm in minmax_stat.keys():
                d = minmax_stat[stNm][1]-  minmax_stat[stNm][0]
                minmax_stat[stNm][0] = minmax_stat[stNm][0] - 0.15 * d  # margins
                minmax_stat[stNm][1] = minmax_stat[stNm][1] + 0.15 * d
                if minmax_stat[stNm][0] < 0.3 * minmax_stat[stNm][1]:   # zero bottom if range >3 times
                    minmax_stat[stNm][0] = min(minmax_stat[stNm][0], 0.)
        else:
            minmax_stat = None
        #
        # Now, draw individual daily pictures. Note that in some days some models might be missing
        # Meanwhile, prepare the time series of the statistics for the below plotting
        # informative models used for that day
        if self.ifPlotDailyPictures:
            #
            # Plot to zip. Careful: if program stopped before closing zip it will be unreadable
            # use zip_tmp to mark that
            zipOut_maps = ZipFile(os.path.join(self.output_directory,'daily_weight_maps', 
                                               'weight_intercept_maps_%s.zip_tmp' % self.species[0]),'w')
            zipOut_barcharts = ZipFile(os.path.join(self.output_directory,'daily', 
                                                    'daily_mdl_weights_%s.zip_tmp' % self.species[0]),'w')
            zipOut_skills = ZipFile(os.path.join(self.output_directory,'daily',
                                                 'daily_mdl_skills_%s.zip_tmp' % self.species[0]), 'w')

            for day in days2draw:
                print('Plotting ', day)
                for tag in tags2plot:
                    stats = self.archive[day].eval_stats[tag]
                    #
                    # Draw the daily skills, a multi-panel bar chart for all models and statistics
                    # It does not matter, of course, if some model was used or not
                    #
                    pp.draw_daily_skills(day,
                                         self.model_names_sort + self.mean_models + fusionMethods,
                                         self.statNms, tag, self.statUnits, 
                                         stats, self.species[0], timeResolTag,  # daily/hourly
                                         os.path.join(self.output_directory,'daily'), 
                                         minmax_stat, zipOut_skills)
                #
                # And the model weights and intercept. Need to pull them together first
                # They are the same for learning and forecast, of course
                #
                coefs = {}
                coefStd = {}
                for im, m in enumerate(self.archive[day].raw_models_used_ranked):
                    coefs[m] = []
                    coefStd[m] = []
                    for fk in self.fusion_methods:
                        if fm.ifSpaceResolving(fk):  # mean weight over stations and over time steps
                            coefs[m].append(np.mean(list((np.mean(self.archive[day].fusion_mdls.
                                                                  fusion_mdl[fk][it].
                                                                  tsMdlCoefs.vals[:,:,:],axis=2)[im+1] 
                                                      for it in 
                                                      range(len(self.archive[day].fusion_mdls.fusion_mdl[fk]))
                                                      ))))
                            coefStd[m].append(np.std(list((np.mean(self.archive[day].fusion_mdls.
                                                                  fusion_mdl[fk][it].
                                                                  tsMdlCoefs.vals[:,:,:],axis=2)[im+1] 
                                                       for it in 
                                                       range(len(self.archive[day].fusion_mdls.fusion_mdl[fk]))
                                                       )))) 
                        else:
                            coefs[m].append(np.mean(list((self.archive[day].fusion_mdls.fusion_mdl[fk][it].
                                                      out.x[im+1] for it in 
                                                      range(len(self.archive[day].fusion_mdls.fusion_mdl[fk]))
                                                      )))) 
                            coefStd[m].append(np.std(list((self.archive[day].fusion_mdls.fusion_mdl[fk][it].
                                                       out.x[im+1] for it in 
                                                       range(len(self.archive[day].fusion_mdls.fusion_mdl[fk]))
                                                       )))) 
                intercept = []
                interceptStd = []
                for fk in self.fusion_methods:
                    if fm.ifSpaceResolving(fk):
                        intercept.append(np.mean(list((np.mean(self.archive[day].fusion_mdls.
                                                               fusion_mdl[fk][it].
                                                               tsMdlCoefs.vals[:,:,:],axis=2)[0]
                                                   for it in 
                                                   range(len(self.archive[day].fusion_mdls.fusion_mdl[fk])) 
                                                   ))))
                        interceptStd.append(np.std(list((np.mean(self.archive[day].fusion_mdls.
                                                                 fusion_mdl[fk][it].
                                                                 tsMdlCoefs.vals[:,:,:],axis=2)[0]
                                                     for it in 
                                                     range(len(self.archive[day].fusion_mdls.fusion_mdl[fk])) 
                                                     ))))
                    else:
                        intercept.append(np.mean(list((self.archive[day].fusion_mdls.fusion_mdl[fk][it].
                                                       out.x[0]
                                                   for it in 
                                                   range(len(self.archive[day].fusion_mdls.fusion_mdl[fk])) 
                                                   ))))
                        interceptStd.append(np.std(list((self.archive[day].fusion_mdls.fusion_mdl[fk][it].
                                                         out.x[0]
                                                     for it in 
                                                     range(len(self.archive[day].fusion_mdls.fusion_mdl[fk])) 
                                                     ))))
                #
                # Draw the daily weights - bar charts
                #
                pp.draw_daily_model_weights(day, fusionMethods, coefs, coefStd, 
                                            intercept, interceptStd, self.species[0], timeResolTag,
                                            os.path.join(self.output_directory,'daily'), minmax_stat,
                                            zipOut_barcharts)
                #
                # Space-resolving methods have maps of weights,
                # att times within the archived day (1 for daily, 24 for hourly)
                #
                for fk in self.fusion_methods:
                    if fm.ifSpaceResolving(fk):
                        for it, mdlFuse in enumerate(self.archive[day].fusion_mdls.fusion_mdl[fk]): 
                            pp.draw_weight_maps(day, it, timeResolTag, mdlFuse.abbrev, mdlFuse.main_grid,
                                                self.archive[day].raw_models_used_ranked,
                                                mdlFuse.arCoefs, self.obsMatr.units[0], self.species[0],
                                                os.path.join(self.output_directory, 'daily_weight_maps'),
                                                zipOut_maps)
            #
            # Carefully close zip with pictures
            #
            for zipOut in [zipOut_barcharts, zipOut_maps, zipOut_skills]:
                chTmp = zipOut.filename
                nameFinal = zipOut.filename.replace('_tmp','')    # final name for archive
                if os.path.exists(nameFinal):          # already exists?
                    if os.path.exists(nameFinal + '_prev'): os.remove(nameFinal + '_prev')  # ... previous one?
                    os.rename(nameFinal, nameFinal + '_prev')   # existing to backup
                zipOut.close()   # Accurate closure of the zip file
                os.rename(chTmp, nameFinal)   # final name for the zip

        #
        # Draw time series if several days are available
        #
        if len(days2draw) > 1:
            #
            # Have to prepare the continious statitics different days have different sets 
            # of models
            #
            print('Plotting timeseries of skills')
            mdlAvailability = {}
            for iDay in range(len(days2draw)):
                day = days2draw[iDay]
                for tag in tags2plot:
                    stats = self.archive[day].eval_stats[tag]
                    for iStat, statNm in enumerate(self.statNms):
                        for iMdl, mdl in enumerate(all_models):
                            if mdl in self.archive[day].raw_models_sort:
                                mdl_scores[tag][statNm][mdl][iDay] = self.archive[day
                                                                    ].eval_stats[tag][iStat,iMdl]
                                mdlAvailability[mdl] = True
                            else:
                                mdl_scores[tag][statNm][mdl][iDay] = np.nan
                                mdlAvailability[mdl] = False
            # Some of the models might be not available at all
            available_raw_models = []
            for m in self.raw_model_names_static:
                try: 
                    if mdlAvailability[m]: available_raw_models.append(m)
                except: 
                    self.log.log('######### Cannot plot %s, no valid days #########' % m)
            #
            # Draw time series for all skills and all models, 
            # one skill per a single multimodel timeserie chart
            #
            zipOut_tser_skills = ZipFile(os.path.join(self.output_directory,'tser_skills', 
                                               'tser_skills_multimodel_%s.zip_tmp' % self.species[0]),'w')
            pp.draw_tseries_4_skills_multimodel(available_raw_models + self.mean_models + fusionMethods,
                                                self.statNms, days2draw, 
                                                mdl_scores, minmax_stat, 
                                                self.species[0], self.learning_period_days, timeResolTag,
                                                os.path.join(self.output_directory,'tser_skills'),
                                                zipOut_tser_skills) 
            chTmp = zipOut_tser_skills.filename
            nameFinal = zipOut_tser_skills.filename.replace('_tmp','')    # final name for archive
            if os.path.exists(nameFinal):          # already exists?
                if os.path.exists(nameFinal + '_prev'): os.remove(nameFinal + '_prev')  # ... previous one?
                os.rename(nameFinal, nameFinal + '_prev')   # existing to backup
            zipOut_tser_skills.close()   # Accurate closure of the zip file
            os.rename(chTmp, nameFinal)   # final name for the zip
            #
            # Time series for weights of individual models
            # one fusion model per a single multimodel timeserie chart
            #
            print('Plotting timeseries of model weights')
            nRawMdls = len(available_raw_models)
            coefs = {}
            for fk_full in self.fusion_methods:
                fk = self.fuseMdl_gen.fusion_mdl[fk_full][0].abbrev
                coefs[fk] = np.ones(shape=(nRawMdls+1,len(days2draw))) * np.nan
                for iDay, day in enumerate(days2draw):
                    # get the indices of the day-available models in the global static raw model list
                    idxAvail = np.searchsorted(available_raw_models, 
                                               self.archive[day].raw_models_used_ranked)
                    if fm.ifSpaceResolving(fk_full):
                        # Average the weights maps and them one by one
                        for idx in range(len(self.archive[day].raw_models_used_ranked)):
#                            if idxAvail[idx] < len(available_raw_models):
                            coefs[fk][idxAvail[idx]+1,iDay
                                      ] = np.mean(list((np.mean(self.archive[day].fusion_mdls.
                                                                fusion_mdl[fk_full][iC].
                                                                tsMdlCoefs.vals[:,:,:],axis=2)[idx+1] 
                                                        for iC in 
                                                        range(len(self.archive[day].
                                                                  fusion_mdls.fusion_mdl[fk_full]))))) 
                        # intercept
                        coefs[fk][0,iDay] = np.mean(list((np.mean(self.archive[day].fusion_mdls.
                                                                  fusion_mdl[fk_full][iC].
                                                                  tsMdlCoefs.vals[:,:,:],axis=2)[0] 
                                                          for iC in 
                                                          range(len(self.archive[day].
                                                                    fusion_mdls.fusion_mdl[fk_full])))))
                    else:
                        # Copy the weights one by one
                        for idx in range(len(self.archive[day].raw_models_used_ranked)):
#                            if idxAvail[idx] < len(available_raw_models):
                            coefs[fk][idxAvail[idx]+1,iDay
                                      ] = np.mean(list((self.archive[day].
                                                        fusion_mdls.fusion_mdl[fk_full][iC].out.x[idx+1]
                                                        for iC in 
                                                        range(len(self.archive[day].
                                                                  fusion_mdls.fusion_mdl[fk_full]))))) 
                        # intercept
                        coefs[fk][0,iDay] = np.mean(list((self.archive[day].
                                                          fusion_mdls.fusion_mdl[fk_full][iC].out.x[0]
                                                          for iC in 
                                                          range(len(self.archive[day].
                                                                    fusion_mdls.fusion_mdl[fk_full]))))) 
            # Next zip files
            zipOut_tser_weights = ZipFile(os.path.join(self.output_directory,'tser_weights', 
                                               'tser_model_weights_%s.zip_tmp' % self.species[0]),'w')
            zipOut_tser_model_skills = ZipFile(os.path.join(self.output_directory,'tser_mdls', 
                                               'tser_model_multiskill_%s.zip_tmp' % self.species[0]),'w')
            #
            # Plotting
            pp.draw_tseries_4_weights_multimodel(available_raw_models, fusionMethods, days2draw, 
                                                 coefs, self.species[0], self.learning_period_days,
                                                 timeResolTag,
                                                 os.path.join(self.output_directory,'tser_weights'), 
                                                 minmax_stat, zipOut_tser_weights)
            # Time series for each model, all skills in one multipanel chart
            #
            print('Plotting timeseies of individual model skills')
            pp.draw_tseries_4_models_multiskill(available_raw_models +
                                                self.mean_models + fusionMethods, self.statNms,
                                                self.statUnits, days2draw, mdl_scores, minmax_stat, 
                                                self.species[0], timeResolTag,
                                                os.path.join(self.output_directory,'tser_mdls'),
                                                self.log, zipOut_tser_model_skills)
            # Close zips
            for zipOut in [zipOut_tser_weights, zipOut_tser_model_skills]:
                chTmp = zipOut.filename
                nameFinal = zipOut.filename.replace('_tmp','')    # final name for archive
                if os.path.exists(nameFinal):          # already exists?
                    if os.path.exists(nameFinal + '_prev'): os.remove(nameFinal + '_prev')  # ... previous one?
                    os.rename(nameFinal, nameFinal + '_prev')   # existing to backup
                zipOut.close()   # Accurate closure of the zip file
                os.rename(chTmp, nameFinal)   # final name for the zip
            #
            # Forecast deterioration
            # Have to regroup the data along the forecast length. No zip - always needed
            #
            if len(mdl_scores.keys()) > 2:  # learn, plus >1 forecast days
                pp.draw_forecast_longevity(self.mean_models + fusionMethods, 
                                           self.statNms, self.statUnits, 
                                           mdl_scores, self.species[0],
                                           minmax_stat, self.learning_period_days, timeResolTag,
                                           os.path.join(self.output_directory,'tser_skills'))

        self.timer.stop_timer('6_plotting')


    #=======================================================================

    def report_content(self):
        # In-short, the content of the FUSE model
        self.log.log('\nInformation in FUSE, %s' % self.chMPI)
        self.log.log('%s Species to analyse: %s' % (self.chMPI, self.species[0]))
        self.log.log('%s Observation file: %s' % (self.chMPI, self.observation_FNm))
        self.log.log('%s Models: %s'  % (self.chMPI, ' '.join(self.models.keys())))
        self.log.log('%s Observed time period: %s - %s' % 
                     (self.chMPI, self.obsMatr.times[0].strftime('%Y%m%d_%H%M'), 
                      self.obsMatr.times[-1].strftime('%Y%m%d_%H%M')))
        self.log.log('%s Number stations for comparison: %i stations\n' % (self.chMPI, 
                                                                           len(self.obsMatr.stations)))
        self.log.log('%s memory use, %g MB, %g files open \n>>>>>%s\n' % 
                     (self.chMPI, 
                      self.Process.memory_info().rss / 1038336, len(self.Process.open_files()), 
                      '\n>>>>>'.join(str(f)  for f in self.Process.open_files())))
        


    #=======================================================================

    def salutation(self, info_string):
        # of the FUSE model
        self.log.log('\n\nFUSE mdoel v.1.0')
        self.log.log('%s Startng the run: %s UTC' % (self.chMPI, 
                                                     dt.datetime.utcnow().strftime('%Y4%m%d_%H%M')))
        if self.chMPI == '': self.log.log('Non-MPI run')
        else: self.log.log('MPI run, my process: %g out of %g' % (self.mpirank, self.mpisize))
        self.log.log(info_string)
        self.log.log('Fusion methods for the run:')
        for fusion_method in self.fusion_methods:
            self.log.log('%s = %s' % (fusion_method,
                                      self.fuseMdl_gen.fusion_metadata(fusion_method)[2]))
        self.log.log('%s memory use, %g MB, %g files open: \n>>>>>%s\n' % 
                     (self.chMPI, 
                      self.Process.memory_info().rss / 1038336, len(self.Process.open_files()), 
                      '\n>>>>>'.join(str(f)  for f in self.Process.open_files())))



#############################################################################################
#############################################################################################
#
# Supplementary routines
#
#

#=======================================================================

def pre_extract_data(observation_FNm, model_FNm, species, tStart, tEnd):
    #
    # Reads the model output datasets through and extracts all the data into tsMatraix netcdf
    #
    print('Reading observations ', observation_FNm)
    obsMatrTmp = tv.TsMatrix.fromNC(observation_FNm, tStart, tEnd) # we need stations
    print('Reading model ', model_FNm)
    mdlMatrTmp = tv.TsMatrix.extract_from_fields(obsMatrTmp.stations, obsMatrTmp.times, 
                                                 species, model_FNm) 
    print('Writing model ', model_FNm + '.extracted.nc')
    mdlMatrTmp.to_nc(model_FNm + '.extracted.nc')
        
    

