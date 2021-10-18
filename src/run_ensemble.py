'''
Created on Feb 16, 2020

@author: sofievm

This is the top-level module for running CAMS ensemble data fusion.
Its task is to initiate the computations with parameters set by external ini file

'''
import os
os.environ['PYTHONPATH'] = 'd:\\tools\\python_3'
import ensemble_driver
import datetime as dt
import sys, math
from toolbox import namelist as nl, supplementary as spp
import numpy as np

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    mpisize = comm.size
    mpirank = comm.Get_rank()
    print ('MPI operation, mpisize=', mpisize)
    chMPI = '_mpi%03g' % mpirank
except:
    comm = None
    mpisize = 1
    mpirank = 0
    print ('Single-process operation')
    chMPI = ''

#================================================================
def get_date(nlIn, chNamelistItem):
    line = Flds = nlIn.get(chNamelistItem)
    if len(line) > 0: 
        Flds = line[0].split()
    else: 
        print('No item:', chNamelistItem)
        raise
    return dt.datetime(int(Flds[0]), int(Flds[1]), int(Flds[2]))


#################################################################

def run_FUSE(namelist_ini): #, learning_days, forecast_days):
    #
    # Get the basic setup
    #
    fusion_setup = namelist_ini.todictionary()
    chFLogNm = namelist_ini.get_uniq('log_file_template') + chMPI + '.log'
        # species
#        # multi-species stuff would be like this:
#        self.species = {}
#        for s in namelist_ini.get('species'):    # string: species_obs, species_mdl, threshold
#            self.species[s.split()[0]] = (s.split()[0], s.split()[1], float(s.split()[2]))
    # for single-species run
    species = fusion_setup['species'][0].split()  # 3 fields: name_obs, name_mdl, factor
    # observations
    if 'stations_to_process' in fusion_setup: chFNmStat = fusion_setup['stations_to_process']
    else: chFNmStat = ''
        
    observations = [fusion_setup['obsdata_format'][0].upper(),
                    float(fusion_setup['obsdata_min_completeness_fraction'][0]),
                    fusion_setup['observation_file_name'][0],
                    '',
                    chFNmStat]  # many
    if observations[0].upper() == 'MMAS': 
        observations[3] = fusion_setup['station_file_name'][0]
    # models 
    models = {}
    for m in fusion_setup['model']:      # strings: model_name, model_file_template
        models[m.split()[0]] = m.split()
    
    start_date = get_date(namelist_ini,'start_day')  # start_date must be at the beginnign of learning
    
    #
    # what are we going to do? 
    # Notations:
    # start_date- the first date covered with observations and models, inclusive
    # end_date  - the last date covered by the models, inclusive
    # today     - the last day with observations
    # iLearnPeriod- the number of days from today backwards with observations, inclusive
    # iFcstPeriod - the number of days from today onwards covered by the forecast, inclusive
    #
    if fusion_setup['run_type'][0].upper() == 'FORECAST':
        #
        # learn over learning_period, then forecast over forecast_period
        # Note that start_date must be at the beginnign of learning
        # Also, today is both the last learning day and the first forecast day
        #
        fLearnBack = float(fusion_setup['learning_period_days'][0]) - 1 # apart from today
        fLearnFrw = 0.0                      # nothing beyond today
        iFcstLength = int(fusion_setup['forecast_period_days'][0]) # including today
        fusion_setup['if_evaluate_forecast'] = False
        today = start_date + spp.one_day * fLearnBack
        end_date = today + spp.one_day * iFcstLength
        
    elif fusion_setup['run_type'][0].upper() == 'REANALYSIS':
        #
        # Reanalysis is just the optimal fitting for each day. Learning is symmetric
        # Forecast is made for today whatever learning is
        #
        fLearnBack = 0.5 * (float(fusion_setup['learning_period_days'][0]) - 1) # apart from today
        fLearnFrw = fLearnBack           # beyond today forwards
        iFcstLength = 1                  # prediction only for today (whatever learning is)
        fusion_setup['if_evaluate_forecast'] = True
        today = start_date + spp.one_day * (math.ceil(fLearnBack) - 1)
        end_date = get_date(namelist_ini,'end_day') - spp.one_day * (math.ceil(fLearnFrw) - 1)

    elif fusion_setup['run_type'][0].upper() == 'EVALUATION':
        #
        # Evaluation is a series of forecasts made over the evaluation period 
        # Learning is asymmetric - backwards - while forecasts go forwards. Both
        # include today. Measurements have to cover both directions.
        #
        fLearnBack = float(fusion_setup['learning_period_days'][0]) - 1
        fLearnFrw = 0.0
        iFcstLength = int(fusion_setup['forecast_period_days'][0])  # including today
        fusion_setup['if_evaluate_forecast'] = True
        today = start_date + spp.one_day * (fLearnBack-1) 
        end_date = get_date(namelist_ini,'end_day')
            
    else:
        print('Unknown run_type: ' + fusion_setup['run_type'][0])
        print('Available only FORECAST, REANALYSIS and EVALUTION')
        sys.exit()
    #
    # create the FUSE instance and load the FUSE configuration and model data
    #
    theModel = ensemble_driver.FUSE(observations, models, today, mpirank, mpisize, chMPI, 
                                    fusion_setup, chFLogNm)
    theModel.salutation('Start_namelist:\n' + namelist_ini.tostr() + '\nEnd_namelist\n')
    #
    # The main working cycle. In operational forecast, will be just one time
    #
    if math.ceil(fLearnBack) >= (end_date-start_date).days-iFcstLength+1:
        print('Incompatible start and end dates or too long forecasting length')
        print('Learning-back, days:', fLearnBack, 'forecast, days:', iFcstLength)
        print('STart and end of the period:', start_date, end_date)
        print('(end-start).days-iFcstLength+1 - learning <= 0: ', 
              (end_date-start_date).days-iFcstLength+1 - math.ceil(fLearnBack))
        raise ValueError
    
    
    for now in (start_date + spp.one_day*i for i in range(math.ceil(fLearnBack), 
                                                          (end_date-start_date).days-iFcstLength+1)):
        #
        # Today starts from 00 but learning can be from mid-day yesterday till midday tomorrow
        # if the learning period is 2 days and it is symmetrical (i.e. reanalysis)
        #
        learn_range = (now - spp.one_hour * fLearnBack * 24,
                       now + spp.one_hour * (fLearnFrw + 1) * 24)
        fcst_range = (now, now + spp.one_hour * iFcstLength*24)   # today and forwards
        #
        # Since we are in the cycle, the model has to be partially reset
        # But not the whole model: we still like to keep skills and fits of the previous days
        #
        theModel.start_new_day(now, models)
        #
        # Run a bunch of forecasts evaluating each of them
        # Here I make them completely independent but this is evidently not the
        # optimal way: there is a huge overlap between the forecassts, so no real need to 
        # start from scratch each time. But for now it is easier this way
        #
        # Load the main datasets:
        # - today's model forecasts
        # - for models available today, get their previous forecasts for the training period 
        # - get the observational data for the training period
        #
        theModel.log.log('Reading the ensemble %s...' % species[0])
        if not theModel.load_all_data_NC_input(learn_range, fcst_range):
            theModel.log.log(now.strftime('###> Failed to get data for %Y%m%d'))
            continue  

        if not theModel.clean_datasets():       # sufficient input to proceed ?
            theModel.log.log(now.strftime('###> Failed cleaning the datasets for %Y%m%d'))
            continue

        if not theModel.finalise_model_ensembles():     # add average & median, unify models
            theModel.log.log(now.strftime('###> Failed adding mean ensemble for %Y%m%d'))
            continue

        theModel.report_content()
        #
        # Make the fusion: 
        # - identify the coefficients
        # - with these coefficients, compute the fusion forecast
        # the functions return the optimal models, RMSE, prediction for the training dataset
        #
        theModel.log.log('Getting the fusion coefficients %s...' % species[0])
        if not theModel.fit_fusion_coefficients():
            theModel.log.log('###> Failed fusion fitting. Stop')
            sys.exit()
        #
        # Learning period has been predicted as FusePredict. If we need future, 
        # need to make it. Here, predict is the prediction of the observations
        # whereas apply_forecast is actual generation of the forecasted maps.
        #
        if iFcstLength > 0:
            theModel.log.log('\nComputing the fusion forecast %s...' % species[0])
            theModel.predict()                        # making tsMatrices
            if fusion_setup['make_forecasts'][0].upper() == 'YES':
                theModel.apply_forecast(ifPrintMaps=fusion_setup['plot_forecasts'][0].upper() == 'YES')   # making maps
        #
        # Having the model fitted nd predictions available for the learning period and
        # for the forecasting period, compute both sets of skills
        #
        theModel.log.log(now.strftime('\nEnsemble skills, today = %Y%m%d_%H%M, ' + species[0]))
        theModel.compute_fitting_skills()  # also dumps a table to the log file
        #
        # Having skills and fitting coefficients computed, archive them
        #
        theModel.archive_last_day(now)
    #
    # After we finish all days, the collected models and skills should be stored and reported
    #
    theModel.archive_to_pickle()
    theModel.plot_models_skills(tag_filter = ['learn','fcst_d','lrn_Hr','fcst_Hr'])
#    #
#    # If MPI processes, need to gather the files together.
#    #
#    if mpisize > 1 and mpirank == 0: 
#        theModel.gather_output_MPI_files()
    
    theModel.log.log('End of the %s run: ' % species[0] + dt.datetime.utcnow().strftime('%Y4%m%d_%H%M') + ' UTC')
     
    theModel.timer.report_timers(theModel.log.logfile)

    theModel.log.close()


################################################################################################

if __name__ == '__main__':

    ifTry = False
    ifPreExtractModelData = False

    dirMain = 'd:\\project\\COPERNICUS\\CAMS_63_Ensemble\\FUSE_v1_1'
#    dirMain = '/fmi/projappl/project_2001411/CAMS_63_Ensemble/FUSE_v1_0'

    learning_times = [3, 5, 7] #, 10, 15, 1, 2] #, 1, 2, 30, 45, 60, 90, 120, 180, 350]
#    species_lst = 'CO SO2 PM25 NO2 O3 PM10'.split()
    species_lst = ['O3']
    diurnal_sets = [('DAILY', 0)] #, 
                    #('HOURLY', 0.01), ('HOURLY', 0.5), 
#                    ('HOURLY',1), ('HOURLY', 2), ('HOURLY', 5)]
    sizes = [(150,50)] #,(50,25),(25,10)]

    os.environ['FUSION_INI'] = 'f:\\project\\CAMS63_ensemble\\AQ_2018\\test_set\\CAMS_fusion_test.ini'

    #======================================================================
    #
    # Pre-extracting the model data
    
    if ifPreExtractModelData:
        for iSpecies, species in enumerate(species_lst):
            if np.mod(iSpecies, mpisize) != mpirank: continue
            namelist_ini = nl.NamelistGroup.fromfile(os.path.join(dirMain, 'ini',
                                                                  'CAMS_fusion_template_try.ini')
                                                     ).lists['main']
            namelist_ini.substitute({'$LEARNING_DAYS': 1, '$SPECIES': species})
            print ('Extracting the model data with ini file: %s, %g MPI processes' % 
                   ('CAMS_fusion_template_try.ini', mpisize))
            for m in namelist_ini.get('model'):      # string: model_name, model_file_template
                print('Reading the species & model: ', species, m.split()[0])
                ensemble_driver.pre_extract_data(namelist_ini.get_uniq('observation_file_name'),
                                                 m.split()[2], species, 
                                                 get_date(namelist_ini,'start_day'),
                                                 get_date(namelist_ini,'end_day'))
        sys.exit()
        
        
    #======================================================================
    #
    # Operational setup
    #
    # if ini file can be given as an argument, stored in environment variable or default
    if len(sys.argv) == 2:
        os.environ['FUSION_INI'] = sys.argv[1]
    elif os.getenv('FUSION_INI') is None:
#        os.environ['FUSION_INI'] = os.path.join(dirMain,'ini','CAMS_fusion_template.ini')
        os.environ['FUSION_INI'] = os.path.join(dirMain,'ini','CAMS_fusion_analysis.ini')
    else:
        print ('FUSION_INI = ', os.getenv('FUSION_INI'))
    #
    # Read the ini file and use only namelist further on 
    # Call the FUSE dispatcher.
    # Note the difference relevant for MPI:
    # you cannot set different environmental variables for different MPI threads
    # This is still a single run.
    # Therefore, namelist has 
    # - automatic handling of environment variables - ONLY in form ${var_key}
    # - namelist also has a substitute subroutine, which allows a multitude of 
    #   substitutes applied to the whole namelist. For MPI runs, this is the mechanism
    #
    iProcess = 0
    #
    # Sizes of subgrids
    #
    for multifit_win_size, multifit_win_overlap in sizes:
        os.environ['MULTIFIT_WINDOW_SIZE'] = str(multifit_win_size)
        os.environ['MULTIFIT_WINDOW_OVERLAP'] = str(multifit_win_overlap)
        #
        # Diurnal profile: daily or sigma-hourly
        #
        for Diurnal in diurnal_sets:
            os.environ['DIURNAL_TYPE'] = Diurnal[0]
            if Diurnal[0] == 'DAILY': 
                chDiurnal = 'DAILY'
            else:
                chDiurnal = ('%s_sigma_%g' % Diurnal).replace('.','_')
                os.environ['DIURNAL_SIGMA'] = str(Diurnal[1])
            #
            # Learning periods
            #
            for lt in learning_times:
                os.environ['LEARNING_DAYS'] = str(lt)
                #
                # Species
                #
                for species in species_lst:
                    iProcess += 1
                    if np.mod(iProcess-1, mpisize) != mpirank: continue
                    os.environ['SPECIES'] = species
                    os.environ['MULTIFIT_WINDOW_SIZE'] = str(multifit_win_size)
                    os.environ['MULTIFIT_WINDOW_OVERLAP'] = str(multifit_win_overlap)
                    namelist_ini = nl.NamelistGroup.fromfile(os.environ['FUSION_INI']).lists['main']
                    namelist_ini.substitute({'$LEARNING_DAYS': lt, '$SPECIES': species,
                                             '$DIURNAL_STR':chDiurnal, '$DIURNAL_TYPE':Diurnal[0],
                                             '$DIURNAL_SIGMA' : str(Diurnal[1]),
                                             '$MULTIFIT_WINDOW_SIZE':multifit_win_size,
                                             '$MULTIFIT_WINDOW_OVERLAP':multifit_win_overlap})
                    print ('Starting with ini file: %s, %g MPI processes' % 
                           (os.environ['FUSION_INI'], mpisize))
                    print ('namelist:\n' + namelist_ini.tostr())
                    
                    run_FUSE(namelist_ini)
        

