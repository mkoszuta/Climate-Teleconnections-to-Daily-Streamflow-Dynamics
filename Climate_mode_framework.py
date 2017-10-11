# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 11:53:16 2017

Developed by Dr. Caitlin Spence (advisor) of The Pennsylvania State University
and Matthew Koszuta (student @ Northland College) during the 2017 Climate Science REU at The 
Pennsylvania University.

This is a collection of functions written to determine the parameters of a beta, a 
poisson, and a gdp-poisson model using the maximum likelihood estimate method. 
These functions make up a framework to optimize a streamflow generator model by
integrating non-stationarity into parameter estimates through the omnipresent 
North Atlantic Oscillation (NAO) climate mode index.

@author: Matthew Koszuta
"""
import os
import numpy as np
import pandas as pd
import datetime
from scipy.stats import exponweib
from scipy.stats import genextreme
from scipy.stats import gumbel_r
from scipy.stats import powerlaw
from scipy.optimize import fmin
from scipy.special import gammaln


os.chdir('/Users/alt_kazoo/Documents/Susquehanna') # Sets the working directory if needed

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

# This section of functions is not up to date and is used as the backbone of
# climate connections to daily streamflow dynamics (from functions fitweibull()
# to season_seperate())

# CHESSIE Streamflow Generator
# @author: Dr. Caitlin Spence

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def fitweibull(x):
    # from https://gist.github.com/plasmaman/5508278
   def optfun(theta):
      return -np.sum(np.log(exponweib.pdf(x, 1, theta[0], scale = theta[1], loc = 0)))
   logx = np.log(x)
   shape = 1.2 / np.std(logx)
   scale = np.exp(np.mean(logx) + (0.572 / shape))
   return fmin(optfun, [shape, scale], xtol = 0.01, ftol = 0.01, disp = 0)

def fitlognorm(x):
    logincs = np.log(x)
    mn = np.mean(logincs)
    sd = np.std(logincs)
    parms=((mn, sd))
    
    return(parms)

def findtroughs(Q):
    # "mindiff" is the fractional flow multiplier that the next day's Q must 
    # surpass to be a flow peak. An example would be 1.05, meaning Q[k] must
    # be 5% less than than Q[k+1} to be marked a trough]
    L = len(Q)
    if isinstance(Q, np.ndarray): 
        pass
    else:
        Q = pd.Series.as_matrix(Q)
        Q = np.asarray(Q)
    inds = list()
    for k in range(L):
        if k == 1:
            pass
        elif k < (L-1):
            # set so that trough happens before a long period of constant flow
            # Revised 7/6/17 so that trough happens at the end of a long period
            # of constant flow- increase after trough must be significant
            if np.logical_and((Q[k] <= Q[k-1]), (Q[k] < Q[k+1])):
                inds.append(k)
         
    inds = np.asarray(inds)
    return(inds)

def findpeaks(Q):
    # "mindiff" is the fractional flow multiplier that the next day's Q must 
    # surpass to be a flow peak. An example would be 1.05, meaning Q[k] must
    # be 5% greater than Q[k+1} to be marked a peak]
    L = len(Q)
    if isinstance(Q, np.ndarray): 
        pass
    else:
        Q = pd.Series.as_matrix(Q)
        Q = np.asarray(Q)
    inds = list()
    for k in range(L):
        if k == 1:
            pass
        elif k < (L-1):
            # set so that peak can ascend to constant flow, but ends when flow starts
            # to decrease
            # Revised 7/6 so that peak can ascend to near-constant flow, but ends
            # when flow has to decrease (decrease must be significant)
            if np.logical_and((Q[k] >= Q[k-1]), (Q[k] > Q[k+1])):
                inds.append(k)
         
    inds = np.asarray(inds)
    return(inds)

def filter_multiples(peaks, troughs, Q):
    N_events = np.min((len(peaks), len(troughs)))
    
    if (peaks[0] > troughs[0]):
    # Series begins with an decrease to trough
        d = 0
        while d < (N_events - 1): # 
            # Remove all but the minimum trough prior to first peak
            if d == 0: 
                troughinds = np.full(troughs.shape, True, dtype=bool)
                troughinds[troughs < peaks[d]] = False # Any troughs before first peak set to True
            else: 
                troughinds = np.full(troughs.shape, True, dtype=bool)
                troughinds[np.logical_and((troughs < peaks[d]), (troughs > peaks[d - 1]))] = False
            Qcands = Q[troughs[~troughinds]] # pick out flows on the trough indices
            troughinds[~troughinds] = (Qcands == np.min(Qcands)) # amend troughinds so that trough with minimum flow set to False
            troughs = troughs[troughinds] # remove any troughinds set to False
            
            # remove any peaks prior to the next trough
            peakinds = np.full(peaks.shape, True, dtype=bool)
            peakinds[np.logical_and((peaks < troughs[d+1]), (peaks > troughs[d]))] = False
            Qcands = Q[peaks[~peakinds]]
            peakinds[~peakinds] = (Qcands == np.max(Qcands))
            peaks = peaks[peakinds]   # remove any peaks set to False
            
            N_events = np.min((len(peaks), len(troughs)))
            d = d + 1
    else:
        d = 0
        # proceed as though series begins with ascension to a peak
        while d < (N_events - 1):
            if d == 0: 
                peakinds = np.full(peaks.shape, True, dtype=bool)
                peakinds[peaks < troughs[d]] = False # Any troughs before first peak set to True
            else: 
                peakinds = np.full(peaks.shape, True, dtype=bool)
                peakinds[np.logical_and((peaks < troughs[d]), (peaks > troughs[d - 1]))] = False
            Qcands = Q[peaks[~peakinds]] # pick out flows on the trough indices
            peakinds[~peakinds] = (Qcands == np.max(Qcands)) # amend troughinds so that trough with minimum flow set to False
            peaks = peaks[peakinds] # remove any troughinds set to False
            
            # remove any peaks prior to the next trough
            troughinds = np.full(troughs.shape, True, dtype=bool)
            troughinds[np.logical_and((troughs < peaks[d+1]), (troughs > peaks[d]))] = False
            Qcands = Q[troughs[~troughinds]]
            troughinds[~troughinds] = (Qcands == np.min(Qcands))
            troughs = troughs[troughinds]   # remove any peaks set to False
            
            N_events = np.min((len(peaks), len(troughs)))
            d = d + 1
    return(peaks, troughs)

def filter_blips(peaks, troughs, Q, mindiff):
    # Remove peaks that are less than 1.05*flow at the previous trough.
    d = 0
    N_events = np.min((len(peaks), len(troughs)))
    while d < N_events:
        if (peaks[0] > troughs[0]):
            # descend to a trough
            if (Q[peaks[d]] - Q[troughs[d]])/Q[troughs[d]] < mindiff:
                # Check whether current trough  is a blip on the way down
                if np.logical_and((Q[peaks[d]] < Q[peaks[d-1]]), (Q[troughs[d]] > Q[troughs[d+1]])):
                    # remove this trough and this peak
                    
                    peakinds = np.full(peaks.shape, True, dtype=bool)
                    peakinds[d] = False
                    peaks = peaks[peakinds]
                
                    # remove the trough that's higher between trough[d] and trough[d+1]
                    troughinds = np.full(troughs.shape, True, dtype=bool)
                    troughinds[d] = False
                    troughs = troughs[troughinds]
                # Or a blip on the way up
                elif np.logical_and((Q[peaks[d]] < Q[peaks[d+1]]), (Q[troughs[d]] > Q[troughs[d-1]])):
                    peakinds = np.full(peaks.shape, True, dtype=bool)
                    peakinds[d] = False
                    peaks = peaks[peakinds]
                
                    # remove the trough that's higher between trough[d] and trough[d+1]
                    troughinds = np.full(troughs.shape, True, dtype=bool)
                    troughinds[d] = False
                    troughs = troughs[troughinds]
            else:
                pass
            
            N_events = np.min((len(peaks), len(troughs)))
            d = d+1
        
        else:   # Ascend to a peak
            if (Q[peaks[d+1]] - Q[troughs[d]])/Q[troughs[d]] < mindiff:
                peakinds = np.full(peaks.shape, True, dtype=bool)
                peakinds[d+1] = False
                peaks = peaks[peakinds]
                
                # remove the trough that's higher between trough[d] and trough[d+1]
                troughinds = np.full(troughs.shape, True, dtype=bool)
                troughinds[d] = False
                troughs = troughs[troughinds]
            else:
                pass
            
            N_events = np.min((len(peaks), len(troughs)))
            d = d+1
    return(peaks, troughs)

def peaktroughcomp(peaks, troughs, Q, mindiff):
    # This function removes false troughs and false peaks so that peaks and 
    # troughs line up.   
    
    # For consecutive peaks or troughs, eliminate all peaks that aren't the 
    # maximum Q or all troughs that aren't the minimum Q.
    (peaks, troughs) = filter_multiples(peaks, troughs, Q)
    
    # Remove peaks that are less than 1.05*flow at the previous trough.
    (peaks, troughs) = filter_blips(peaks, troughs, Q, mindiff)
    
    # Go back and filter consecutives
    (peaks, troughs) = filter_multiples(peaks, troughs, Q)
                  
    return(peaks, troughs)

def state_det(Q, date, flood_thresh):
    
    L_days = len(Q)
    if isinstance(Q, np.ndarray): 
        pass
    else:
        Q = pd.Series.as_matrix(Q)
        Q = np.asarray(Q)

    troughs = findtroughs(Q)
    peaks = findpeaks(Q)
    troughs = np.squeeze(troughs)
    (peaks, troughs) = peaktroughcomp(peaks, troughs, Q, mindiff)

    decs = list()
    ascs = list()
    floods = list()
    
    asc_month = list()
    flood_month = list()
    decs_month = list()
    
    asc_year = list() # NEW
    flood_year = list() # NEW
    decs_year = list() # NEW
    
#    decs_drought = list()
#    decs_month_drought = list()
#    decs_year_droguht = list()
    
    state = np.empty((L_days,))
    
    # First, set up 0 behavior
    if troughs[0] > peaks[0]:
        # Starts with ascent to a peak, then decrease to first trough.
        if Q[peaks[0]] > flood_thresh:
            flood_month.append(date[0].month)
            flood_year.append(date[0].year) # NEW
            # Ascend from Q[0] to peaks[0]
            floods.append(Q[0:peaks[0]+1])
            state[0:peaks[0]+1] = 2   # Flooding
        else:
            asc_month.append(date[0].month)
            asc_year.append(date[0].year) # NEW
            ascs.append(Q[0:peaks[0]+1])
            state[0:peaks[0]+1] = 1   # Ascending
        # Decrease to first trough
        decs_month.append(date[0].month)
        decs_year.append(date[0].year) # NEW
        decs.append(Q[peaks[0]:troughs[0]+1])
        state[(peaks[0]):troughs[0]+1] = 0  # decreasing
        
        # Now continue with each trough coming after each peak
        for d in np.arange(1, np.min((len(peaks), len(troughs))), 1):
            
            # Ascent from trough[d-1] to peak[d]
            if Q[peaks[d]] > flood_thresh:
                flood_month.append(date[troughs[d]].month)
                flood_year.append(date[troughs[d]].year) # NEW
                floods.append(Q[troughs[d-1]:peaks[d]+1])
                state[(troughs[d-1]):peaks[d]+1] = 2   # Flooding
            else:
                asc_month.append(date[troughs[d]].month)
                asc_year.append(date[troughs[d]].year) # NEW
                ascs.append(Q[troughs[d-1]:peaks[d]+1])
                state[(troughs[d-1]):peaks[d]+1] = 1   # Normal ascent
            
            # descend from peak[d] to trough[d]
            decs_month.append(date[peaks[d]].month)
            decs_year.append(date[troughs[d]].year) # NEW
            decs.append(Q[(peaks[d]):troughs[d]+1])
            state[(peaks[d]):troughs[d]+1] = 0
            
    else:   # peak[0] > trough[0] 
        # Begins with descent from Q[0] to a trough at Q[troughs[0]]
        decs_month.append(date[0].month)
        decs_year.append(date[0].year) # NEW
        decs.append(Q[0:troughs[0]+1])
        state[0:troughs[0]+1] = 0   # Descending
        
        # Now ascent from troughs[0] to peaks[0]
        if Q[peaks[0]] > flood_thresh:
            flood_month.append(date[troughs[0]].month)
            flood_year.append(date[troughs[0]].year) # NEW
            floods.append(Q[(troughs[0]):peaks[0]+1])
            state[(troughs[0]):peaks[0]+1] = 2
        else:
            asc_month.append(date[troughs[0]].month)
            asc_year.append(date[troughs[0]].year) # NEW
            ascs.append(Q[(troughs[0]):peaks[0]])
            state[(troughs[0]):peaks[0]] = 1
                
        # Now continue with descent from peak[d-1] to trough[d]
        for d in np.arange(1, np.min((len(peaks), len(troughs))), 1):
            
            # Add descent from peaks[d-1] to troughs[d]
            decs_month.append(date[peaks[d-1]].month)
            decs_year.append(date[peaks[d-1]].year) # NEW
            decs.append(Q[(peaks[d-1]):troughs[d]+1])
            state[(peaks[d-1]+1):troughs[d]+1] = 0
            
            # And following ascent from troughs[d] to peaks[d]
            if Q[peaks[d]] > flood_thresh:
                flood_month.append(date[troughs[d]].month)
                flood_year.append(date[troughs[d]].year) # NEW
                floods.append(Q[(troughs[d]):peaks[d]+1])
                state[(troughs[d]+1):peaks[d]+1] = 2
            else:
                asc_month.append(date[troughs[d]].month)
                asc_year.append(date[troughs[d]].year) # NEW
                ascs.append(Q[(troughs[d]):peaks[d]+1])
                state[(troughs[d]+1):peaks[d]+1] = 1
                
    returns = dict()
    returns['state'] = state
    returns['ascs'] = ascs
    returns['decs'] = decs
    returns['floods'] = floods
    returns['ascs_month'] = asc_month
    returns['decs_month'] = decs_month
    returns['floods_month'] = flood_month
    returns['ascs_year'] = asc_year # NEW
    returns['decs_year'] = decs_year # NEW
    returns['floods_year'] = flood_year # NEW
                
    return(returns)

def count_transitions(statevect):
    L_days = len(statevect)  
    DDcount=0   # Number of dry-dry transitions
    DWcount=0   # Number of dry-wet transitions
    DFcount=0   # Number of dry-flood transitions
    WDcount=0   # Number of wet-dry transitions
    WWcount=0   # Number of wet-wet transitions
    WFcount=0   # Number of wet-flood transitions
    FDcount=0   # Number of flood-dry transitions
    FWcount=0   # Number of flood-wet transitions (is this technically possible?)
    FFcount=0   # Number of flood-flood transitions
    
    for k in range(L_days-1):
        if np.logical_and((statevect[k] == 0), statevect[k+1] == 0):
            DDcount = DDcount + 1
        elif np.logical_and((statevect[k] == 0), statevect[k+1] == 1):
            DWcount = DWcount + 1
        elif np.logical_and((statevect[k] == 0), statevect[k+1] == 2):
            DFcount = DFcount + 1
        elif np.logical_and((statevect[k] == 1), statevect[k+1] == 0):
            WDcount = WDcount + 1
        elif np.logical_and((statevect[k] == 1), statevect[k+1] == 1):
            WWcount = WWcount + 1
        elif np.logical_and((statevect[k] == 1), statevect[k+1] == 2):
            WFcount = WFcount + 1
        elif np.logical_and((statevect[k] == 2), statevect[k+1] == 0):
            FDcount = FDcount + 1
        elif np.logical_and((statevect[k] == 2), statevect[k+1] == 1):
            FWcount = FWcount + 1
        elif np.logical_and((statevect[k] == 2), statevect[k+1] == 2):
            FFcount = FFcount + 1
    
    return(DDcount, DWcount, DFcount, WDcount, WWcount, WFcount, FDcount, FWcount, FFcount)
            

def fit_trans_probs(statevect, date, start=1, stop=365, flag=1): # WINTER SEASON
    # Flag signifies whether Q is a tiem sereis of flow (0) or a list of flow
    # periods, each from a certain time of year (1)
    # UPDATED June 21 to accomodate separate, stochastically-sampled flood 
    # durations rather than relying on Markov chain transitions.
    if flag == 0:
        (DDcount, DWcount, DFcount, WDcount, WWcount, WFcount, FDcount, FWcount, FFcount) = count_transitions(statevect)
    else:
        state_doy = season_separate(statevect, date, start=start, stop=stop)
        N_pds = len(state_doy)
        DDcount=0   # Number of dry-dry transitions
        DWcount=0   # Number of dry-wet transitions
        DFcount=0   # Number of dry-flood transitions
        WDcount=0   # Number of wet-dry transitions
        WWcount=0   # Number of wet-wet transitions
        WFcount=0   # Number of wet-flood transitions
        FDcount=0   # Number of flood-dry transitions
        FWcount=0   # Number of flood-wet transitions (is this technically possible?)
        FFcount=0   # Number of flood-flood transitions
        for k in range(N_pds):
            state_pd = state_doy[k]
            (DDcount_new, DWcount_new, DFcount_new, WDcount_new, WWcount_new, WFcount_new, FDcount_new, FWcount_new, FFcount_new) = count_transitions(state_pd)
            DDcount = DDcount + DDcount_new
            DWcount = DWcount + DWcount_new
            DFcount = DFcount + DFcount_new
            WDcount = WDcount + WDcount_new
            WWcount = WWcount + WWcount_new
            WFcount = WFcount + WFcount_new
            FDcount = FDcount + FDcount_new
            FWcount = FWcount + FWcount_new
            FFcount = FFcount + FFcount_new
            
    FWcount = 0
    WFcount = 0
    if (DDcount + DWcount + DFcount) == 0:
        P_DD = float(0)
    else:
        P_DD = DDcount/(DDcount + DWcount + DFcount)
    if (DDcount + DWcount + DFcount) == 0:
        P_DW = float(0)
    else:
        P_DW = DWcount/(DDcount + DWcount + DFcount)
    if (DDcount + DWcount + DFcount) == 0:
        P_DF = float(0)
    else:
        P_DF = DFcount/(DDcount + DWcount + DFcount)
    if (WDcount + WWcount + WFcount) == 0:
        P_WD = float(0)
    else:
        P_WD = WDcount/(WDcount + WWcount + WFcount)
    if (WDcount + WWcount + WFcount) == 0:
        P_WW = float(0)
    else:
        P_WW = WWcount/(WDcount + WWcount + WFcount)
    if (WDcount + WWcount + WFcount) == 0:
        P_WF = float(0)
    else:
        P_WF = WFcount/(WDcount + WWcount + WFcount)
    if (FDcount + FWcount + FFcount) == 0:
        P_FD = float(0)
    else:
        P_FD = FDcount/(FDcount + FWcount + FFcount)
    if (FDcount + FWcount + FFcount) == 0:
        P_FW = float(0)
    else:
        P_FW = FWcount/(FDcount + FWcount + FFcount)
    if (FDcount + FWcount + FFcount) == 0:
        P_FF = float(0)
    else:
        P_FF = FFcount/(FDcount + FWcount + FFcount)
    
    # Update to accomodating selecting flood duration separately
    # P_FD = 1
    # P_FW = 0
    # P_FF = 0
    transprobs = np.empty(9)
    transprobs[0] = P_DD
    transprobs[1] = P_DW
    transprobs[2] = P_DF
    transprobs[3] = P_WD
    transprobs[4] = P_WW
    transprobs[5] = P_WF
    transprobs[6] = P_FD
    transprobs[7] = P_FW
    transprobs[8] = P_FF
    return (transprobs)


def find_incs(Q, date, flood_thresh):
    returns = state_det(Q, date, flood_thresh)
    ascs = returns['ascs']
    floods = returns['floods']
    Qincs = list()
    Fincs = list()
    
    for k in range(len(ascs)):
        asc_temp = ascs[k]
        asc1 = asc_temp[1:]
        asc2 = asc_temp[:-1]
        Qincs.extend(np.subtract(asc1, asc2))
        
    for j in range(len(floods)):
        flood_temp = floods[j]
        flood1 = flood_temp[1:]
        flood2 = flood_temp[:-1]
        Fincs.extend(np.subtract(flood1,flood2))
        
    return(Qincs, Fincs)

def fit_incs_pd(ascs, ascs_month, months, flag=0):
    # this period accepts a list of increasing periods and a month desired
    # and returns the parameter of the distribution of increments for that month
    
    ascs_select = list()
    for m in range(len(ascs)):
        #if ascs_month[m] == months:
        if np.in1d(ascs_month[m], months):
            ascs_select.append(ascs[m])
            
    L = len(ascs_select)
    incs = list()
    
    for k in range(L):
        asc_temp = ascs_select[k]
        if hasattr(asc_temp, "__len__"):
            asc1 = asc_temp[1:]
            asc2 = asc_temp[:-1]
            incs.extend(np.subtract(asc1, asc2))
        else:
            pass
    
    incs = list(filter(lambda a: a > 0, incs))
    if flag == 0:
        optparms = fitweibull(incs)
    elif flag == 1:
        optparms = fitlognorm(incs)
    elif flag == 2:
        optparms = gumbel_r.fit(incs)
    elif flag == 3:
        optparms = powerlaw.fit(incs)
    elif flag == 4:
        optparms = genextreme.it(incs)
    return(optparms)
    

def decs_sort(decs, Q_UL, Q_LD):
    
    b_storm = list()
    b_normal = list()
    b_drought = list()

    t_storm = int()
    t_normal = int()
    t_drought = int()
    

    numdecs = len(decs)

    for k in range(numdecs):
        dec_temp = decs[k]
        L_temp = len(dec_temp)
        # Cycle through decreases in dry spell
        for w in range(L_temp):
            if (dec_temp[w] > Q_UL):        # Storm flow
                if (w == 0):
                    Qpeak = dec_temp[w]
                    t_storm = 1
                else:
                    t_storm = t_storm + 1
                    b_storm.append((np.log(Qpeak) - np.log(dec_temp[w]))/(t_storm))           
            elif (dec_temp[w] > Q_LD):      # Normal recession
                if (w == 0):
                    Q_0 = dec_temp[w]
                    t_normal = 1
                elif (dec_temp[w-1] >= Q_UL):  #Transition has occurred
                    Q_0 = dec_temp[w]
                    t_normal = 1
                else:
                    t_normal = t_normal + 1
                    b_normal.append((np.log(Q_0) - np.log(dec_temp[w]))/(t_normal))           
            else:                           #extreme drought
                if (w == 0):
                    Q_star = dec_temp[w]
                    t_drought = 1
                elif (dec_temp[w-1] >= Q_LD):
                    t_drought = 1
                    Q_star = dec_temp[w]
                else:
                    t_drought = t_drought + 1
                    b_drought.append((np.log(Q_star) - np.log(dec_temp[w]))/(t_drought))
   
    b_storm = np.asarray(b_storm)
    b_normal = np.asarray(b_normal)
    b_drought = np.asarray(b_drought)
    
    return(b_storm, b_normal, b_drought)

def date_to_nth_day(date):
    datedt = pd.DatetimeIndex(date)
    year = datedt.year
    month = datedt.month
    day = datedt.day
    startyr = np.min(year)
    endyr = np.max(year)
    nyears = endyr - startyr
    DOY = list()
    yearlist = np.arange(startyr, endyr + 1, 1)
    for k in range(nyears + 1):
        if k == 0:
            if month[k] == 1:
                day1 = day[k]
            elif month[k] == 2:
                day1 = 31 + day[k]
            elif month[k] == 3: 
                day1 = 31 + 28 + day[k]
            elif month[k] == 4:
                day1 = 31 + 28 + 31 + day[k]
            elif month[k] == 5:
                day1 = 31 + 28 + 31 + 30 + day[k]
            elif month[k] == 6:
                day1 = 31 + 28 + 31 + 30 + 31 + day[k]
            elif month[k] == 7:
                day1 = 31 + 28 + 31 + 30 + 31 + 30 + day[k]
            elif month[k] == 8:
                day1 = 31 + 28 + 31 + 30 + 31 + 30 + 31 + day[k]
            elif month[k] == 9:
                day1 = 31 + 28 + 31 + 30 + 31 + 30 + 31 + 31 + day[k]
            elif month[k] == 10:
                day1 = 31 + 28 + 31 + 30 + 31 + 30 + 31 + 31 + 30 + day[k]
            elif month[k] == 11:
                day1 = 31 + 28 + 31 + 30 + 31 + 30 + 31 + 31 + 30 + 31 + day[k]
            elif month[k] == 12:
                day1 = 31 + 28 + 31 + 30 + 31 + 30 + 31 + 31 + 30 + 31 + 30 + day[k]
            DOY.extend(np.arange(day1, 366,1))
        elif k == nyears:
            diff = len(year) - len(DOY)
            DOY.extend(np.arange(1, diff + 1,1))
        elif sum(year==yearlist[k]) == 365:
            DOY.extend(np.arange(1,366,1))
        else:
            DOY.extend(np.arange(1,367,1))
    DOY = np.asarray(DOY)
    return(DOY)

def doy_check(datedt):
    # receives a single timestamp
    month = datedt.month
    day = datedt.day

    if month == 1:
        day1 = day
    elif month == 2:
        day1 = 31 + day
    elif month == 3: 
        day1 = 31 + 28 + day
    elif month == 4:
        day1 = 31 + 28 + 31 + day
    elif month == 5:
        day1 = 31 + 28 + 31 + 30 + day
    elif month == 6:
        day1 = 31 + 28 + 31 + 30 + 31 + day
    elif month == 7:
        day1 = 31 + 28 + 31 + 30 + 31 + 30 + day
    elif month == 8:
        day1 = 31 + 28 + 31 + 30 + 31 + 30 + 31 + day
    elif month == 9:
        day1 = 31 + 28 + 31 + 30 + 31 + 30 + 31 + 31 + day
    elif month == 10:
        day1 = 31 + 28 + 31 + 30 + 31 + 30 + 31 + 31 + 30 + day
    elif month == 11:
        day1 = 31 + 28 + 31 + 30 + 31 + 30 + 31 + 31 + 30 + 31 + day
    elif month == 12:
        day1 = 31 + 28 + 31 + 30 + 31 + 30 + 31 + 31 + 30 + 31 + 30 + day
    DOY = day1

    return(DOY)

def DOY_to_monthday(doy):
    if doy <= 31:
        month = 1
        day = doy
    elif doy <= 59:
        month = 2
        day = doy - 31
    elif doy <= 90:
        month = 3
        day = doy - 59
    elif doy <= 120:
        month = 4
        day = doy - 90
    elif doy <= 151:
        month = 5
        day = doy - 120
    elif doy <= 181:
        month = 6
        day = doy - 151
    elif doy <= 212:
        month = 7
        day = doy - 181
    elif doy <= 243:
        month = 8
        day = doy -212
    elif doy <= 273:
        month = 9
        day = doy - 243
    elif doy <= 304:
        month = 10
        day = doy - 273
    elif doy <= 334:
        month = 11
        day = doy - 304
    elif doy <= 365:
        month = 12
        day = doy - 334
        
    return(month, day)
        

def season_separate(Q, date, start=1, stop=365):
    date = pd.DatetimeIndex(date)
    year = date.year
    year = np.asarray(year)
    start_month, start_day = DOY_to_monthday(start)
    stop_month, stop_day = DOY_to_monthday(stop)
    start_year = np.min(year)
    stop_year = np.max(year)
    years = np.arange(start_year, stop_year + 1, 1)
    Q_periods = list()
    for k in range(stop_year - start_year):
        Q_yeartemp = Q[date.year==years[k]]
        Date_yeartemp = date[date.year==years[k]]
        if start > stop:
            Q_nextyear = Q[date.year==years[k+1]]
            Date_nextyear = date[date.year==years[k+1]]
            Start_date_temp = datetime.date(year=years[k], day=start_day, month=start_month)
            yr_end_date = datetime.date(year=years[k], day=31, month=12)
            Stop_date_temp = datetime.date(year=years[k+1], day=stop_day, month=stop_month)
            yr_start_date = datetime.date(year=years[k+1], day=1, month=1)
            dates_1 = pd.date_range(Start_date_temp, yr_end_date)
            dates_2 = pd.date_range(yr_start_date, Stop_date_temp)
            B = Q_yeartemp[np.logical_and(Date_yeartemp >= dates_1[0], Date_yeartemp <= dates_1[len(dates_1)-1])]
            if isinstance(Q, np.ndarray):
                period_k_1 = B
            else:
                period_k_1 = B.values
            C = Q_nextyear[np.logical_and(Date_nextyear >= dates_2[0], Date_nextyear <= dates_2[len(dates_2)-1])]
            if isinstance(Q, np.ndarray):
                period_k_2 = C
            else:
                period_k_2 = C.values
            period_k = np.append(period_k_1, period_k_2)
            Q_periods.append(period_k)
        else:
            Start_date_temp = datetime.date(year=years[k], day=start_day, month=start_month)
            Stop_date_temp = datetime.date(year=years[k], day=stop_day, month=stop_month)
            dates = pd.date_range(Start_date_temp, Stop_date_temp)
            B = Q_yeartemp[np.logical_and(Date_yeartemp >= dates[0], Date_yeartemp <= dates[len(dates)-1])]
            if isinstance(Q, np.ndarray):
                period_k = B
            else:
                period_k = B.values
            Q_periods.append(period_k)
            
    return(Q_periods)

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-    

# Contributions to climate mode optimization of CHESSIE. Daily streamflow dynamics
# flood event rate, magnitude, and chanes to transition probabilites of the three-
# state Markov Chain used to generate streamflow. 
    
# Aspects of the scripts can be changed to include three different climate
# mode forms - neutral/nonneutral, negative/neutral/positive, or continuous
# climate mode. Changes to model structures including or excluding certain sets
# of paramters. Or, you can change what set of climate mode predictors is used 
# to draw connections.

# @author: Matthew Koszuta
    
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

# For the following functions. Choose which model form is used by moving the comments (#) and adjust the 
# best 'guess' below in the post-processing.
def independent_peaks(Q, peak, flood_thresh, interval):
    peaks_index = np.full(Q.shape, False, dtype=bool)
    # A new column 'Peaks' is created filled with False
    for n in np.arange(len(peak)): # Only looks through pre-defined peaks
        if np.logical_and(Q[peak[n]] == np.max(Q[peak[n]-interval:peak[n]+interval]), Q[peak[n]] >= flood_thresh): # Checks to see if the current iteration is both the maximum in a series +/- 14 days and greater than the flood threshold
            peaks_index[peak[n]] = True
        else:
            pass
    return(peaks_index)


def beta_nll(params):
    l = len(dd_probs) # for lagged-NAO change l to l = len(dd_probs)-1
    alpha0 = params[0] # shape parameter
#    alpha1 = params[1]
#    alpha2 = params[2]
    beta0 = params[3] # shape parameter
#    beta1 = params[4]
#    beta2 = params[5]
    
    term1 = np.empty(l)
    term2 = np.empty(l)
    term3 = np.empty(l)
    
    loglik_beta = np.empty(l)
     
    for k in range(l):
        alpha = alpha0
#        alpha = alpha0 + alpha1*NAO_cat[k,1] 
#        alpha = alpha0 + alpha1*NAO_cat[k,0] + alpha2*NAO_cat[k,2]
#        alpha = alpha0 + alpha1*annual_nao.annual_NAO_index[k]
        beta = beta0
#        beta = beta0 + beta1*NAO_cat[k,1]
#        beta = beta0 + beta1*NAO_cat[k,0] + beta2*NAO_cat[k,2]
#        beta = beta0 + beta1*annual_nao.annual_NAO_index[k]
    
        # add +1 to the transition probability indexing term to lineup with appropriate NAO       
        term1[k] = gammaln(alpha+beta) - gammaln(alpha) - gammaln(beta)
        term2[k] = (alpha-1)*np.log(dd_probs[k]) # change to appropriate transition probabilities
        term3[k] = (beta-1)*np.log((1-dd_probs[k])) # change to appropriate transition probabilities
    loglik_beta = -(np.sum(term1) + np.sum(term2) + np.sum(term3))
    return(loglik_beta)

def gpd_poisson_nll(coeff, eta, exceedance, NAO_cat):
    Ts = 365.25 # Time scaling constants
    l = len(NAO_cat)
    nu0 = coeff[0] # event rate
#    nu1 = coeff[1]
#    nu2 = coeff[2]
    sigma0 = coeff[3] # scale parameter
#    sigma1 = coeff[4]
#    sigma2 = coeff[5]
    xi0 = coeff[6] # shape parameter
#    xi1 = coeff[7]
#    xi2 = coeff[8]

    loglik = np.empty(l)
    
    for k in np.arange(l):          
        nu = nu0
#        nu = nu0 + nu1*NAO_cat[k,1]
#        nu = nu0 + nu1*NAO_cat[k,0] + nu2*NAO_cat[k,2]
#        nu = nu0 + nu1*daily_nao.annual_NAO_index[k]
        
        sigma = sigma0
#        sigma = sigma0 + sigma1*NAO_cat[k,1]
#        sigma = sigma0 + sigma1*NAO_cat[k,0] + sigma2*NAO_cat[k,2]
#        sigma = sigma0 + sigma1*daily_nao.lagged_one_DJFM_NAO_index[k]

        xi = xi0
#        xi = xi0 + xi1*NAO_cat[k,1]
#        xi = xi0 + xi1*NAO_cat[k,0] + xi2**NAO_cat[k,2]
#        xi = xi0 + xi1*daily_nao.lagged_one_DJFM_NAO_index[k]
        
        if eta[k]==0:
            loglik[k] = -(nu / Ts)
        else:
            loglik[k] = (eta[k]*np.log(nu)) - (nu / Ts)
            - (eta[k]*np.log(sigma)) - (eta[k]*(1 + (1 / xi))*np.log(1 + 
            xi*(exceedance[k] / sigma)))
    return(-(np.sum(loglik)))
    

 
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    
#  Begin processing streamflow for calibration. USGS streamflow data can be 
# switched out for another dataset. Be mindful of whatelse may need to be 
# changed.
    
# @author: Dr. Caitlin Spence and Matthew Koszuta
    
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-



marietta_usgs = pd.read_excel("Marietta Daily Discharge October 1931 March 2017.xlsx",
                        header=0,
                        parse_cols="C:E")
marietta_usgs = marietta_usgs.join(pd.DataFrame(np.arange(1, 31213, 1), index=None, columns=['Day']))
Q = marietta_usgs["CFS"]
date = marietta_usgs["Date"]

flood_thresh = 156000 # 156000 CFS, stage == 44 feet, "action" stage. Flood stage is 49 ft
mindiff = 0.05
interval = 14 # Days of interval before and after a flood peak to remove dependent flood events from the maximum peak

exceedance = Q - flood_thresh

(peak, trough) = peaktroughcomp(findpeaks(Q), findtroughs(Q), Q, mindiff) # Determines which index of Q are either a peak or a trough
eta = independent_peaks(Q, peak, flood_thresh, interval) # Determines whether it exceeds flood threshold and is a peak, True if so, False if not.
eta = eta.astype(int)
marietta_usgs['Peaks'] = eta


Nyears = 10
Ndays = len(Q)
Startmonth = 10
Startdate=1

Q_drought = 3800
Q_normal_inc = 10000
Q_normal = Q_drought + Q_normal_inc

# load Rhodium output for each season's decay constants/flow thresholds
months = (10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9)

start_monthly = ([274,305,335,1,32,60,91,121,152,182,213,244])
stop_monthly = ([304,334,365,31,59,90,120,151,181,212,243,273])

start_3monthsliding = ([244,274,305,335,1,32,60,91,121, 152,182,213])
stop_3monthsliding = ([334,365,31,59,90,120,151,181,212,243,273,304])

months_3monthsliding = np.array(([9,10,11], 
                        [10,11,12], 
                        [11,12,1], 
                        [12,1,2], 
                        [1,2,3], 
                        [2,3,4], 
                        [3,4,5], 
                        [4,5,6], 
                        [5,6,7],
                        [6,7,8],
                        [7,8,9],
                        [8,9,10]))

b_storm = np.empty((12,1))
b_normal = np.empty((12,1))
b_drought = np.empty((12,1))

transprobs_monthly = np.empty((12,9))

state_returns = state_det(Q, date, flood_thresh)
statevect = state_returns['state']
ascs = state_returns['ascs']
floods = state_returns['floods']
decs = state_returns['decs']
ascs_month = state_returns['ascs_month']
floods_month = state_returns['floods_month']
decs_month = state_returns['decs_month'] 
ascs_year = state_returns['ascs_year']  # NEW
floods_year = state_returns['floods_year'] # NEW
decs_year = state_returns['decs_year'] # NEW
# State_returns is a dictionary containing ascending, descending, and flood ascents
# and the month each occurred in!

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

# Begin post-processing which connects large-scale climate patterns to the
# daily streamflow dynamics produced by the CHESSIE streamflow generator. The
# current climate mode being used the North Atlantic Oscillation (NAO), but it
# can be changed to another dataset if wanted, the NOAA Climate Prediction
# Center is a good place to start your search for data. 

# @author: Matthew Koszuta

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


# retieve annual PC-based NAO data from UCAR (will need to update link if changed);
# start data retrieval in 1931 to match available streamflow dataset (change with
# whichever river is being analyzed and the length of the available data)
df = pd.read_table('https://climatedataguide.ucar.edu/sites/default/files/nao_pc_annual.txt',
                   delim_whitespace=True,
                   skiprows=33,
                   names=['Year', 'annual_NAO_index'])
dates = pd.date_range('1931-10-01', '2016-12-31')
year = pd.DataFrame({'Year':dates.year, 'Date':dates}, index=dates)
merged_nao = year.merge(df, how='left', on='Year')

# retrieve winter (DJFM) station-based NAO data from  UCAR (will need to update
# link if changed); DJFM NAO is an important index since those are the months 
# where the pressure difference between the Icelandic low and Bermuda high is
# greatest causing more extreme weather patterns and NAO index
d = pd.read_table('https://climatedataguide.ucar.edu/sites/default/files/nao_station_djfm.txt', 
                 delim_whitespace=True,
                 skiprows=68,
                 names=['Year', 'DJFM_NAO_index'])

annual_nao = df.merge(d, how='left', on='Year') # merge DJFM NAO index to annual NAO index values
temp = pd.DataFrame(d.DJFM_NAO_index.shift(periods=1))
temp = temp.DJFM_NAO_index.rename('lagged_DJFM_NAO_index')
annual_nao = annual_nao.merge(pd.DataFrame(temp), how='left', left_index=True, right_index=True) # merge one-year lagged NAO index
annual_nao = annual_nao.shift(periods=-1) # exclude 1931 because of its limitations on transition probabilities since ww = 1.000
annual_nao = annual_nao.reindex(np.arange(0, 85, 1))

'''NOTE: Creating a DJFM NAO Index column with a one-year legged index and adding it 
to the DataFrame and reindexing because the original year has no corresponding
'Lagged-one_DJFM_NAO_index'.'''

djfm_nao_shift = d.DJFM_NAO_index.shift(periods=1)
djfm_nao_shift = djfm_nao_shift.rename(index='lagged_one_DJFM_NAO_index')
d = d.join(djfm_nao_shift)
daily_nao = merged_nao.merge(d, how='left', on='Year')
daily_nao = daily_nao.set_index('Date')
daily_nao = daily_nao[92:len(daily_nao)]

# retrieve daily NAO index data (interpolated data) from the Climate Prediction
# Center (CPC) and index by the date range of the dataset
d = pd.read_table('ftp://ftp.cpc.ncep.noaa.gov/cwlinks/norm.daily.nao.index.b500101.current.ascii',
                  delim_whitespace=True,
                  names=['Year', 'Month', 'Day', 'Daily NAO Index'])
interpolated_daily_nao = d.set_index(pd.to_datetime(dict(year=d.Year, month=d.Month, day=d.Day), errors='coerce', format='%Y-%m-%d', infer_datetime_format=True))
#interpolated_daily_nao = d.drop(['Year', 'Month', 'Day'], axis=1)

eta = eta[92:len(dates)]
exceedance = exceedance[92:len(dates)]

# Converting a list of years where flood events occurred (with repeated years for
# each flood that occurred) to a DataFrame of flood counts per year indexed by year.
dictionary = {'Year': floods_year}
pdf = pd.DataFrame(dictionary)
gb = pdf.groupby('Year')['Year'].count()
gb = gb.reindex(np.arange(1931, 2017), fill_value=0)
gb.name='flood_count'
g = pd.DataFrame(gb)


# determing the transition probabilities for historical data to determine what
# relationships they have with climate variability to better simulate streamflow
datedt = pd.DatetimeIndex(date)
year = datedt.year
year = np.asarray(year)
    
transition_probs = list()
for k in np.arange(1931, 2017+1, 1):
    state_yeartemp = statevect[year==k]
    date_yeartemp = date[year==k]
        
    transition_temp = fit_trans_probs(state_yeartemp, date_yeartemp, flag=0)
    transition_probs.append(transition_temp)
    
probs_DD = np.empty((len(transition_probs)))
probs_DW = np.empty((len(transition_probs)))
probs_DF = np.empty((len(transition_probs)))
probs_WD = np.empty((len(transition_probs)))
probs_WW = np.empty((len(transition_probs)))
probs_WF = np.empty((len(transition_probs)))
probs_FD = np.empty((len(transition_probs)))
probs_FW = np.empty((len(transition_probs)))
probs_FF = np.empty((len(transition_probs)))

for k in range(len(transition_probs)):
    probs_DD[k] = transition_probs[k][0]
    probs_DW[k] = transition_probs[k][1]
    probs_DF[k] = transition_probs[k][2]
    probs_WD[k] = transition_probs[k][3]
    probs_WW[k] = transition_probs[k][4]
    probs_WF[k] = transition_probs[k][5]
    probs_FD[k] = transition_probs[k][6]
    probs_FW[k] = transition_probs[k][7]
    probs_FF[k] = transition_probs[k][8]
        
dd_probs = probs_DD[1:-1]
dw_probs = probs_DW[1:-1]
df_probs = probs_DF[1:-1]
wd_probs = probs_WD[1:-1]
ww_probs = probs_WW[1:-1]
wf_probs = probs_WF[1:-1]
fd_probs = probs_FD[1:-1]
fw_probs = probs_FW[1:-1]
ff_probs = probs_FF[1:-1]

# Determining whether each state of the NAO falls into either the NAO-, NAOneutral, or NAO+ category
nao = pd.Series.as_matrix(annual_nao.annual_NAO_index) # sets which NAO dataset is used
NAO_cat = np.empty([len(nao), 3], dtype=int)
NAO_cat[:,0] = nao <= -0.5 # Set interval to what is preferred goruping of NAO idnex
NAO_cat[:,1] = np.logical_and(nao > -0.5, nao < 0.5)
NAO_cat[:,2] = nao >= 0.5

lagged_nao = pd.Series.as_matrix(annual_nao.lagged_DJFM_NAO_index[1:len(annual_nao)]) # sets which NAO dataset is used
NAO_cat = np.empty([len(lagged_nao), 3], dtype=int)
NAO_cat[:,0] = lagged_nao <= -0.5 # Set interval to what is preferred goruping of NAO idnex
NAO_cat[:,1] = np.logical_and(lagged_nao > -0.5, lagged_nao < 0.5)
NAO_cat[:,2] = lagged_nao >= 0.5

freq_results_beta = list()
guess = (173, 34, 20, 5, 45, 5)
freq_results_beta.append(fmin(beta_nll, guess))
freq_results_beta.append([beta_nll(freq_results_beta[0]), len(guess)])

nao = pd.Series.as_matrix(daily_nao.annual_NAO_index) # sets which NAO dataset is used
NAO_cat = np.empty([len(nao), 3], dtype=int)
NAO_cat[:,0] = nao <= -0.5 # Set interval to what is preferred goruping of NAO idnex
NAO_cat[:,1] = np.logical_and(nao > -0.5, nao < 0.5)
NAO_cat[:,2] = nao >= 0.5


lagged_nao = pd.Series.as_matrix(daily_nao.lagged_one_DJFM_NAO_index) # sets which NAO dataset is used
NAO_cat = np.empty([len(lagged_nao), 3], dtype=int)
NAO_cat[:,0] = lagged_nao <= -1 # Set interval to what is preferred goruping of NAO idnex
NAO_cat[:,1] = np.logical_and(lagged_nao > -1, lagged_nao < 1)
NAO_cat[:,2] = lagged_nao >= 1


freq_results_gpd_poisson = list()
guess = (2.19, 46650, 0.1, 0.1)
freq_results_gpd_poisson.append(fmin(gpd_poisson_nll, guess, args=(eta, exceedance, NAO_cat,)))
freq_results_gpd_poisson.append([gpd_poisson_nll(freq_results_gpd_poisson[0], eta, exceedance, NAO_cat), len(guess)])
freq_results_gpd_poisson.append(2*freq_results_gpd_poisson[1][0] + 2*freq_results_gpd_poisson[1][1])