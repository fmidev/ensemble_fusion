'''
Created on 11.4.2020

This module deals with the outcoem of the fusion.
The instrumentation implemented is:
- analyse log file


@author: sofievm
'''

import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import datetime as dt
import os, io
import fusion_models as fm
from toolbox import drawer

dpi_def = 400

#==================================================================================

model_colors = {'CHIMERE':'purple',  'DEHM':'lime',       'EMEP':'orchid',
                'EURAD_IM':'red',    'GEM_AQ':'grey',     'LOTOS_EUROS':'orange',
                'MATCH':'saddlebrown','MOCAGE':'darkblue','SILAM':(0.15,0.57,0.15),  # 'forestgreen': (0.106,0.522,0.106)
                'SILAM_an':'black',
                'average':'paleturquoise','median':'cyan',
                'LU_MLR': 'blue',    'LU_RIDGE':'slateblue','LU_LASSO':'dodgerblue',
                'LU_MLRnn_ex':'lightslategrey','LU_MLRnn_L':'steelblue',
#                'LSRMF_MLR': 'darkgreen','LSRMF_RIDGE':'limegreen','LSRMF_LASSO':'lime',
#                'LSRMF_MLRnn_ex':'mediumseagreen','LSRMF_MLRnn_L':'aquamarine',
                'LSRMF_MLR': 'teal','LSRMF_RIDGE':'mediumseagreen','LSRMF_LASSO':'lime',
                'LSRMF_MLRnn_ex':'mediumturquoise','LSRMF_MLRnn_L':'aquamarine',
                }
def get_colors(mdlNMs):
    return list((model_colors[m] for m in mdlNMs))
def get_color(mdlNm):
    return model_colors[mdlNm]


#==================================================================================

def draw_daily_model_weights(day, fusion_mdlNMs, mdl_coefs, mdl_coefStd, intercept, interceptStd, 
                             species, chCoefTimeType, outDir, minmax_stat=None, zipOut=None):
    #
    # And the model weights and intercept
    #
    fig = plt.figure() #num, figsize, dpi, facecolor, edgecolor, frameon, FigureClass, clear)
    ax = fig.subplots()
    ax1 = ax.twinx()
    drawer.bar_plot(ax, mdl_coefs, data_stdev=mdl_coefStd, 
                 colors=list((get_color(mdlNm) for mdlNm in mdl_coefs)), 
                 legend=(True,4,6,None),  # (if needed, location, fontsize, anchor)
                 group_names=fusion_mdlNMs)
#    ax1.plot(np.arange(len(fusion_mdlNMs)), intercept, yerr=interceptStd, 
    ax1.errorbar(np.arange(len(fusion_mdlNMs)), intercept, yerr=interceptStd, 
                 marker='o', linestyle='', color='blue')
    ax.set_ylabel('model weights') #,color='red')
    if minmax_stat: 
        ax.set_ylim(minmax_stat['weights'][0], minmax_stat['weights'][1])
    ax.set_title(species + day.strftime(',  model weights & intercept, %Y%m%d'), fontsize=9)
#    ax1.set_xticks(range(len(fusion_mdlNMs)))
#    ax1.set_xticklabels(fusion_mdlNMs, fontsize=7, rotation=80)
    ax1.tick_params(axis='y',labelsize=7, color='blue')
    ax1.set_ylabel('intercept, ug/m3',color='blue')
    ax1.tick_params(axis='y', colors='blue')
    ax1.get_yaxis().get_offset_text().set_color('blue')
    ax1.yaxis.label.set_color('blue')
    if minmax_stat: 
        ax1.set_ylim(minmax_stat['intercept'][0], minmax_stat['intercept'][1])
#            d = max(intercept) - min(intercept)
#            ax1.set_ylim(min(intercept) - 0.15 * d, max(intercept) + 0.15 * d)
    if zipOut is None:
        fig.savefig(os.path.join(outDir, '%s_%s_model_weights_%s.png' % (species, chCoefTimeType,
                                                                         day.strftime('%Y%m%d'))), 
                    dpi=dpi_def) #, bbox_inches='tight')
    else:
        streamOut = io.BytesIO()
        plt.savefig(streamOut, dpi=dpi_def)
        with zipOut.open('%s_%s_model_weights_%s.png' % (species, chCoefTimeType,
                                                         day.strftime('%Y%m%d')), 'w') as pngOut:
            pngOut.write(streamOut.getbuffer())
        streamOut.close()
    fig.clf()
    plt.close()


#==================================================================================

def draw_daily_skills(day, mdl_names, stat_names, subset, stat_units, arStats, species, 
                      timeResolTag, outDir, minmax_stat=None, zipOut=None):
    #
    # Draws the model performance for the specific day, all skills in a single
    # multipanel chart
    #
    nMdlsTot = len(mdl_names)
    # Draw the bar charts
    fig = plt.figure() #num, figsize, dpi, facecolor, edgecolor, frameon, FigureClass, clear)
    axes = fig.subplots(2,2)
    iStat = 0
    for ax in axes.flatten():
        ax.bar(np.arange(nMdlsTot), arStats[iStat,:], color=get_colors(mdl_names))
        ax.set_xticklabels(mdl_names, fontsize=7, rotation=80)
        ax.set_xticks(np.arange(nMdlsTot))
        #            ax.set_xlabel('')
        ax.set_ylabel(species + ' ' + stat_units[iStat], fontsize=7)
        if minmax_stat: 
            ax.set_ylim(minmax_stat[stat_names[iStat]][0], minmax_stat[stat_names[iStat]][1])
        ax.tick_params(axis='y',labelsize=7)
        ax.set_title(species + '  ' + stat_names[iStat], fontsize=8)
        # prepaare the time series...
        iStat += 1
    fig.suptitle(subset + day.strftime(' %Y-%m-%d'))
    fig.tight_layout()
    if zipOut is None:
        fig.savefig(os.path.join(outDir, '%s_%s_scores_%s_%s' % (species, timeResolTag, subset,
                                                                 day.strftime('_%Y%m%d.png'))), 
                                 dpi=dpi_def) #, bbox_inches='tight')
    else:
        streamOut = io.BytesIO()
        plt.savefig(streamOut, dpi=dpi_def)
        with zipOut.open('%s_%s_scores_%s_%s' % (species, timeResolTag, subset,
                                                 day.strftime('_%Y%m%d.png')), 'w') as pngOut:
            pngOut.write(streamOut.getbuffer())
        streamOut.close()
    fig.clf()
    plt.close()


#==================================================================================

def draw_tseries_4_skills_multimodel(mdl_names, stat_names, days2draw, mdl_scores, 
                                     minmax_stat, species, learning_period_days, 
                                     chCoefTimeType, outDir, zipOut=None):
    # Draw time series for each skill for all models in a single chart
    #
    for subset in mdl_scores.keys():   # tags, e.g. learn, fcst_d, learn_hr, fcst_hr
        for s in stat_names:           # statistics
            fig = plt.figure() #num, figsize, dpi, facecolor, edgecolor, frameon, FigureClass, clear)
            ax = fig.subplots()
            for mdl in mdl_names:
                ax.plot_date(days2draw, mdl_scores[subset][s][mdl], marker='.', label=mdl, 
                             markersize=3, color=get_color(mdl))
            handles, labels = ax.get_legend_handles_labels()
            ax.set_xticklabels(days2draw, fontsize=7, rotation=20)
            if len(days2draw) < 15:
                ax.xaxis.set_major_locator(mpl.dates.DayLocator())
            else:
                ax.xaxis.set_major_locator(mpl.dates.DayLocator(interval = 
                                                                np.ceil(len(days2draw) / 15).
                                                                astype(np.int)))
#            if len(days2draw) > 6:
#                ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%m-%d"))
#                ax.xaxis.set_minor_formatter(mpl.dates.DateFormatter("%m-%d"))
#            else:
            ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%Y-%m-%d"))
            ax.xaxis.set_minor_formatter(mpl.dates.DateFormatter("%Y-%m-%d"))
            ax.set_ylabel(s, fontsize=7)
            ax.set_ylim(minmax_stat[s][0], minmax_stat[s][1])
            ax.tick_params(axis='y',labelsize=7)
            ax.set_title('%s %s, %s, learning %gd, %s' % 
                         (s, species, subset, learning_period_days, chCoefTimeType))
            fig.legend(handles, labels, fontsize=6, loc='upper right')
            if zipOut is None:
                fig.savefig(os.path.join(outDir, '%s_%s_%s_%s.png' % (species, chCoefTimeType, s, subset)), 
                            dpi=dpi_def)
            else:
                streamOut = io.BytesIO()
                plt.savefig(streamOut, dpi=dpi_def)
                with zipOut.open('%s_%s_%s_%s.png' % (species, chCoefTimeType, s, subset), 'w') as pngOut:
                    pngOut.write(streamOut.getbuffer())
                streamOut.close()
            fig.clf()
            plt.close()


#==================================================================================

def draw_tseries_4_weights_multimodel(individual_mdlNMs, fusion_mdlNMs, days2draw, 
                                      mdl_coefs, species, learning_period_days, 
                                      chCoefTimeType, outDir, minmax_weights, zipOut=None):
    #
    # Draw time series for the individual model weights, one chart for each fusion model
    #   coefs[fk][:,iDay] = self.archive[days2draw[iDay]].fusion_mdls[fk].out.x[:] 
    #
    for fusion_mdl in fusion_mdlNMs:
        fig = plt.figure() #num, figsize, dpi, facecolor, edgecolor, frameon, FigureClass, clear)
        ax = fig.subplots()
        ax1 = ax.twinx()     # for intercept
        for iMdl in range(len(individual_mdlNMs)):
            mdl = individual_mdlNMs[iMdl]
            ax.plot_date(days2draw, mdl_coefs[fusion_mdl][iMdl+1], marker='.', label=mdl, 
                         markersize=3, color=get_color(mdl))
        ax1.plot(days2draw, mdl_coefs[fusion_mdl][0], label='intercept', linewidth=1)
        handles, labels = ax.get_legend_handles_labels()
        handles1, labels1 = ax1.get_legend_handles_labels()
        ax.set_xticklabels(days2draw, fontsize=7, rotation=20)
        if len(days2draw) < 15:
            ax.xaxis.set_major_locator(mpl.dates.DayLocator())
        else:
            ax.xaxis.set_major_locator(mpl.dates.DayLocator(interval = 
                                                            np.ceil(len(days2draw) / 15).
                                                            astype(np.int)))
#        if len(days2draw) > 6:
#            ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%m-%d"))
#            ax.xaxis.set_minor_formatter(mpl.dates.DateFormatter("%m-%d"))
#        else:
        ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%Y-%m-%d"))
        ax.xaxis.set_minor_formatter(mpl.dates.DateFormatter("%Y-%m-%d"))
        ax.set_ylabel('model weights', fontsize=7)
        ax1.set_ylabel('intercept', fontsize=7)
        ax.set_ylim(minmax_weights['weights'][0], minmax_weights['weights'][1])
        ax1.set_ylim(minmax_weights['intercept'][0], minmax_weights['intercept'][1])
        ax.tick_params(axis='y',labelsize=7)
        ax1.tick_params(axis='y',labelsize=7)
        ax.set_title('model weights, %s, %s, learn %gd %s' % 
                     (fusion_mdl, species, learning_period_days, chCoefTimeType))
        fig.legend(handles, labels, fontsize=6, loc=2)
        fig.legend(handles1, labels1, fontsize=6, loc=1)
        if zipOut is None:
            fig.savefig(os.path.join(outDir, '%s_%s_model_weights_%s.png' % (species, chCoefTimeType, 
                                                                             fusion_mdl)), 
                        dpi=dpi_def)
        else:
            streamOut = io.BytesIO()
            plt.savefig(streamOut, dpi=dpi_def)
            with zipOut.open('%s_%s_model_weights_%s.png' % (species, chCoefTimeType, 
                                                             fusion_mdl), 'w') as pngOut:
                pngOut.write(streamOut.getbuffer())
            streamOut.close()
        fig.clf()
        plt.close()


#==================================================================================

def draw_tseries_4_models_multiskill(mdl_names, stat_names, stat_units, days2draw, mdl_scores, 
                                     minmax_stat, species, chCoefTimeType, outDir, log, zipOut=None):
    #
    # Time series of skills for each model, all scores in one multi-plot chart
    # learning and forecasting go together
    #
    for mdl in mdl_names:
        print(mdl)
        fig = plt.figure(figsize=(11, 6.5)) #num, figsize, dpi, facecolor, edgecolor, frameon, FigureClass, clear)
        gs = fig.add_gridspec(2, 15)
        ax_legend = fig.add_subplot(gs[0:1,14])
        ax_legend.set_axis_off()
        axes = np.empty(shape=(2,2), dtype=object)
        for ix in [0,1]:
            for iy in [0,1]: axes[ix,iy] = fig.add_subplot(gs[ix,7*iy:7*iy+6])

        for ax, s, su in zip(axes.flatten(), stat_names, stat_units):
            for tag in mdl_scores.keys():
                if not np.any(np.isfinite(mdl_scores[tag][s][mdl])): continue
                ax.plot_date(days2draw, mdl_scores[tag][s][mdl], marker='.', markersize=3, 
                             label = tag
#                             + ' min/max day ' + 
#                             days2draw[np.nanargmin(mdl_scores[tag][s][mdl])].strftime('%d.%m / ') +
#                             days2draw[np.nanargmax(mdl_scores[tag][s][mdl])].strftime('%d.%m')
                             ) 
#                            , color=get_color(mdl))   # if single color is needed
                log.log('Extreme %s %s min/max dates: %s / %s' % 
                        (mdl, tag, 
                         days2draw[np.nanargmin(mdl_scores[tag][s][mdl])].strftime('%Y%m%d / '),
                         days2draw[np.nanargmax(mdl_scores[tag][s][mdl])].strftime('%Y%m%d')))

            if len(days2draw) < 15:
                ax.xaxis.set_major_locator(mpl.dates.DayLocator())
            else:
                ax.xaxis.set_major_locator(mpl.dates.DayLocator(interval = 
                                                                np.ceil(len(days2draw) / 15).
                                                                astype(np.int)))

#            if len(days2draw) > 6:
#                ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%m-%d"))
#                ax.xaxis.set_minor_formatter(mpl.dates.DateFormatter("%m-%d"))
#            else:
            ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%Y-%m-%d"))
            ax.xaxis.set_minor_formatter(mpl.dates.DateFormatter("%Y-%m-%d"))
            ax.set_ylabel(s + ', ' + su, fontsize=7)
            ax.set_ylim(minmax_stat[s][0], minmax_stat[s][1])
            ax.tick_params(axis='x',labelsize=7, rotation=20)
            ax.tick_params(axis='y',labelsize=7)
            plt.subplots_adjust(hspace = 0.3)
            ax.set_title(s, fontsize=8)
        handles, labels = ax.get_legend_handles_labels()
        ax_legend.legend(handles, labels, loc='upper center',fontsize=8)
        if days2draw[0].year == days2draw[-1].year: 
            fig.suptitle('%s %s %s %g' % (mdl, species, chCoefTimeType, days2draw[0].year))
        else:
            fig.suptitle('%s %s %s %g-%g' % (mdl, species, chCoefTimeType, days2draw[0].year, 
                                             days2draw[-1].year))
#        fig.tight_layout()
        if zipOut is None:
            fig.savefig(os.path.join(outDir, '%s_%s_%s_scores.png' % 
                                     (species, chCoefTimeType, mdl)), dpi=dpi_def)
        else:
            streamOut = io.BytesIO()
            plt.savefig(streamOut, dpi=dpi_def)
            with zipOut.open('%s_%s_%s_scores.png' % (species, chCoefTimeType, mdl), 'w') as pngOut:
                pngOut.write(streamOut.getbuffer())
            streamOut.close()
        fig.clf()
        plt.close()
        print(mdl,'done')


#==================================================================================

def draw_regularization(days2draw, alpha, fusion_models, outDir):
    #
    # Dawrs the regularization coefficients for the fusion models
    #
    fig = plt.figure()
    ax = fig.subplots()
    print(alpha.keys(), fusion_models)
    for fm in alpha.keys():
        if fm == 'MLR': continue
        ax.plot_date(days2draw, alpha[fm], label=fm, marker='.', markersize=3)
    ax.set_xticklabels(days2draw, fontsize=7, rotation=20)
    if len(days2draw) > 6:
        ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%Y-%m-%d"))
        ax.xaxis.set_minor_formatter(mpl.dates.DateFormatter("%Y-%m-%d"))
    ax.legend(fontsize=7)
    ax.set_title('Regularization weights')
    fig.tight_layout()
    fig.savefig(os.path.join(outDir, 'regulaization_weights.png'), dpi=dpi_def)
    fig.clf()
    plt.close()


#==================================================================================

def draw_forecast_longevity(models2draw, statNms, statUnits, mdl_scores, species, minmax_stat, 
                            learning_period_days, chCoefTimeType, outDir):
    #
    # Draw the longevity of the forecasting skills, mean over the considered period
    # dimension: mdl_scores[tag][statNm][mdlNm][iDay]
    #
    # Prepare the arrays. Tags are for both hourly and daily time resolution. These
    # should be separated
    #
    tags_all = np.array(list(mdl_scores.keys()))
    tags_daily = []
    tags_hourly = []
    for tag in tags_all:
        if 'Hr' in tag: tags_hourly.append(tag)
        else: tags_daily.append(tag)
    for tags, tUnit in [(np.array(tags_daily),'day'),(np.array(tags_hourly),'hour')]:
        iStep = max(1, np.int(tags.shape[0] / 10))
        fcst_skills = {}
        fcst_diff = {}
        fcst_std = {}
        for s in statNms:
            fcst_skills[s] = {}
            fcst_diff[s] = {}
            fcst_std[s] = {}
            for im, m in enumerate(models2draw):
                fcst_skills[s][m] = np.array(list((np.mean(mdl_scores[tag][s][m]) for tag in tags)))
                if im > 0:
                    fcst_diff[s][m] = np.array(list((np.mean(mdl_scores[tag][s][m])
                                                     for tag in tags))) - fcst_skills[s][models2draw[0]]
                    fcst_std[s][m] = np.array(list((np.std(mdl_scores[tag][s][m] - 
                                                           mdl_scores[tag][s][models2draw[0]])
                                                     for tag in tags)))
        #
        # Statistics for different longevioty, outer cycle over statistics - a multiplot chart 
        #
        fig = plt.figure()
        axes = fig.subplots(2,2)
        for ax, s, su in zip(axes.flatten(), statNms, statUnits):
            handles, labels = drawer.bar_plot(ax, fcst_skills[s], 
                                              colors=list((get_color(mdlNm) for mdlNm in models2draw)), 
                                              legend=(False,1,7,None))  # (if needed, location, fontsize,anchor)
            ax.set_title(s, fontsize=8)
            ax.set_xticks(range(0,len(tags),iStep))
            ax.set_xticklabels(tags[::iStep], fontsize=7, rotation=30)
            ax.tick_params(axis="y", labelsize=7)
            ax.set_xlabel('forecast length, %s' % tUnit, fontsize=7)
            ax.set_ylabel(su, fontsize=7)
      #      ax.set_ylim(minmax_stat[s][0], minmax_stat[s][1])
        fig.suptitle('%s fcst longevity\nlearn %gd %s' % 
                     (species, learning_period_days, chCoefTimeType), fontsize=9)
        fig.legend(handles, labels, loc='center', fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(outDir, '_%s_%s_fit_fcst_longevity_%s_learn_%03g_d.png' % 
                                 (species, chCoefTimeType, tUnit, learning_period_days)), dpi=dpi_def)
        fig.clf()
        plt.close()
        #
        # Similar for the difference from average - a multiplot chart 
        #
        fig = plt.figure()
        axes = fig.subplots(2,2)
        for ax, s, su in zip(axes.flatten(), statNms, statUnits):
            handles, labels = drawer.bar_plot(ax, fcst_diff[s], fcst_std[s], 
                                              colors=list((get_color(mdlNm) for mdlNm in models2draw[1:])), 
                                              legend=(False,1,7,None))  # (if needed, location, fontsize,anchor)
            ax.set_title(s, fontsize=8)
            ax.set_xticks(range(0, len(tags), iStep))
            ax.set_xticklabels(tags, fontsize=7, rotation=30)
            ax.tick_params(axis="y", labelsize=7)
            ax.set_xlabel('forecast length, %s' % tUnit,fontsize=7)
            ax.set_ylabel('mdl-Average, ' + su, fontsize=7)
      #      ax.set_ylim(minmax_stat[s][0], minmax_stat[s][1])
        fig.suptitle('%s fcst longevity\nlearn %gd %s' % 
                     (species, learning_period_days, chCoefTimeType), fontsize=9)
        fig.legend(handles, labels, loc='center', fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(outDir, '_%s_%s_fcst_longevity_vs_average_%s_learn_%03g_d.png' % 
                                 (species, chCoefTimeType, tUnit, learning_period_days)), dpi=dpi_def)
        fig.clf()
        plt.close()
    
    
#==================================================================================

def draw_weight_maps(day, iTime, timeResolTag, chFusionNm, grid, arRawModelsNms, mapMdlWeights,
                     chInterceptUnit, species, outDir, zipOut=None):
    #
    # Draws a multiplot of maps of weights of the models for space-resolving methods
    #
    # The multi-model case means a multi-panel picture, control from here
    diurnalTag = {'daily':'daily_fit', 'hourly':'hr_%02g_fit' % iTime}[timeResolTag]
    
    # For daily fusio resolution, there will be one time step, for hourly - 24 steps
    # Intercept should be separate, otherwise a common scale does not work
    
    drawer.draw_map_with_points(list(('%s, %s %s %s %s weight' % 
                                      (species, chFusionNm, day.strftime('%Y%m%d'), diurnalTag, 
                                       mdlRaw) for mdlRaw in arRawModelsNms)),
                                [], [], grid,  # lons, lats are used if points are active 
                                outDir,'weight_maps_%s_%s_%s_%s' %
                                (species, chFusionNm, day.strftime('%Y%m%d'), diurnalTag),
                                vals_points=None,
                                vals_map = mapMdlWeights[1:,:,::-1],
                                chUnit='',ifNegatives=True, zipOut=zipOut) #, numrange=(1e-2,100))

    drawer.draw_map_with_points('%s, %s, %s %s intercept' % 
                                (species, chFusionNm, day.strftime('%Y%m%d'), diurnalTag),
                                [], [], grid,  # lons, lats are used if points are active 
                                outDir,'intercept_map_%s_%s_%s_%s' %
                                (species, chFusionNm, day.strftime('%Y%m%d'), diurnalTag),
                                vals_points=None,
                                vals_map = mapMdlWeights[0,:,::-1],
                                chUnit=chInterceptUnit, ifNegatives=True, zipOut=zipOut) #, numrange=(1e-2,100))


#==================================================================================

def analyse_log_file(chLogFNm, outDir):
    fIn = open(chLogFNm,'r')
    if not os.path.exists(outDir): os.makedirs(outDir)
    days = []
    skills = 'bias corr rmse stdevRatio'.split()
    units = ['ug/m3','','us/m3','']
    mdl_scores = {}
    mdl_weights = {}
    alpha = {}
    single_models = None
    all_models = None
    fusion_models = []
    ifReadingSkills = False
    learning_period_days = None
    for line in fIn:
        if line.startswith('Species to analyse'): species = line.split(':')[1].strip()
        if line.startswith('Learning'):                    # how many days? 
            if not learning_period_days:
                daystr = line[line.find(':')+1 : line.find(',')]
                d1  = dt.datetime.strptime(daystr.split(' - ')[0].strip(), '%Y-%m-%d %H:%M:%S')
                d2 = dt.datetime.strptime(daystr.split(' - ')[1].strip(), '%Y-%m-%d %H:%M:%S')
                learning_period_days = (d2-d1).days
        if line.startswith('Starting new time period for'):   # get the raw models
            if not single_models:
                single_models = line.split(':')[-1].strip().split()[1:]
                print('models', single_models)
                all_models = single_models.copy()
            continue
        if line.startswith('Model '):
            if 'Tikhonov' in line:
                try:
                    alpha[line.split()[1]].append(float(line.split('(')[1].split(',')[0]))
                except:
                    alpha[line.split()[1]] = [float(line.split('(')[1].split(',')[0])]
            else:
                try:
                    alpha[line.split()[1]].append(float(line.split()[3][:-1]))
                except:
                    alpha[line.split()[1]] = [float(line.split()[3][:-1])]
        if line.startswith('Coefficients'):
            fusion_mdl = line.split(':')[0].split()[1]
            mdl_weights[fusion_mdl] = [float(line.split(':')[1].split(',')[0].split()[1])]  # a0
            mdl_weights[fusion_mdl] += list((float(s) for s in line.split('weights')[1].split()))
        if line.startswith('Ensemble skills, today'):
            # do not forget to add ensemble models
            days.append(dt.datetime.strptime(line.split('=')[1].strip(), '%Y-%m-%d'))
            ifReadingSkills = True
            continue
        if line.startswith('Archiving'):  # end of the model skills lineset
            ifReadingSkills = False
            continue
        if ifReadingSkills:
            if 'Reporting' in line: 
                subset = line.split()[1]
                mdl_scores[subset] = {}
                for s in skills:
                    for mdl in all_models: 
                        try: mdl_scores[subset][s][mdl] = []    # scores for individual models
                        except: mdl_scores[subset][s] = {mdl:[]}    # scores for individual models
                continue
            if not 'bias' in line:
                print('Should be model-skill line but no bias:', line)
                raise
            # decode the line and store skills
            flds = line.split()
            if not flds[0] in all_models:                # if new model (i.e. ensemble fusion)
                all_models.append(flds[0])
                fusion_models.append(flds[0])
                for s in skills: mdl_scores[subset][s][flds[0]] = [np.nan]
            for iFld in range(1, len(flds), 2):
                for s in skills:
                    if flds[iFld] == s: 
                        mdl_scores[subset][s][flds[0]].append(float(flds[iFld+1].strip(',')))
    #
    # Reading is over, verify and compute rangs for each score
    #
    print('Days found:', len(days)) #, days)
    print('All models found:', all_models)
    minmax_stat = {}
    for subset in mdl_scores.keys():
        for s in skills:
            minmax_stat[s] = [1e20, -1e20]
            for mdl in all_models:
                minmax_stat[s] = [min(minmax_stat[s][0], np.nanmin(mdl_scores[subset][s][mdl])), 
                                  max(minmax_stat[s][1], np.nanmax(mdl_scores[subset][s][mdl]))]
    d = minmax_stat[s][1] - minmax_stat[s][0]
    minmax_stat[s][0] = minmax_stat[s][0] - 0.15 * d  # margins
    minmax_stat[s][1] = minmax_stat[s][1] + 0.15 * d
#    if minmax_stat[s][0] < 0.3 * minmax_stat[s][1]:   # zero bottom if range >3 times
#        minmax_stat[s][0] = min(minmax_stat[s][0], 0.)
    print('min-max for each parameters: ',minmax_stat)
    #
    # Draw the skills as time series
    #
    # Daily skills
    #
#    for day in days:
#        draw_daily_model_weights(day, single_models, fusion_models, mdl_weights[1:], mdl_weights[0], 
#                             species, outDir, minmax_stat)
#    
#        draw_daily_skills(day, all_models, skills, subset, units, 
#                      mdl_scores, species, outDir, minmax_stat)
#    #
#    # Draw time series
#    #
#    draw_tseries_4_skills_multimodel(all_models, skills, days, mdl_scores, 
#                                     minmax_stat, species, learning_period_days, outDir)
#    draw_tseries_4_models_multiskill(all_models, skills, days, mdl_scores, 
#                                     minmax_stat, species, outDir)
#    
#    draw_tseries_4_weights_multimodel(single_models, fusion_models, days,
#                                      mdl_coefs, 
#                                      species, learning_period_days, outDir, minmax_weights)
    
    #
    # Draw regularization
    #
    draw_regularization(days, alpha, fusion_models, outDir)


#######################################################################################

if __name__ == '__main__':

    logTempl = 'd:\\project\\COPERNICUS\\CAMS_63_Ensemble\\FUSE_v1_0\\output\\learn_1_days\\run_FUSE_%s_20180101.log'
#    logTempl = 'd:\\project\\COPERNICUS\\CAMS_63_Ensemble\\FUSE_v1_0\\output\\run_FUSE_SO2_20180101.log'
    outDir = 'd:\\project\\COPERNICUS\\CAMS_63_Ensemble\\FUSE_v1_0\\output\\learn_1_days\\log_analysis'
    
#    for species in 'NO2 O3 CO SO2 PM10 PM25'.split():
#        chLogFNm = logTempl % species
#        analyse_log_file(chLogFNm, outDir)

    analyse_log_file( 'd:\\project\\COPERNICUS\\CAMS_63_Ensemble\\FUSE_v1_0\\output\\learn_3\\_run_FUSE_MPI_NO2_20180101_mpi000.log',
                      'd:\\project\\COPERNICUS\\CAMS_63_Ensemble\\FUSE_v1_0\\output\\try_log_anal')


