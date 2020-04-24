
import numpy as np

def get_performances(target, prediction):
    mask = np.logical_or(np.isnan(target),
                         np.isnan(prediction),
                         )
    assert mask.sum().sum() == 0
    assert target.ndim > 1
    target_mean = target.mean(axis = 0)
    
    dikt = {}
    # Mean squared errors
    dikt['mses'] = ((target - prediction)**2).mean(axis = 0)
    # Mean absolute errors
    dikt['maes'] = (np.abs(target - prediction)).mean(axis = 0)

    dikt['scores'] = 1-((target-prediction)**2).sum(axis = 0)/((target-target_mean)**2).sum(axis = 0)
    dikt['mapes']  = 100*(np.abs(((target-prediction)/target))).mean(axis = 0)
    dikt['nmses']  = (((target - prediction)**2)/(target_mean**2)).mean(axis = 0)
    dikt['rmnmse'] = np.array([100*np.sqrt(dikt['nmses'].mean())])
    
    assert dikt['scores'].ndim == 1 
    assert dikt['scores'].shape[0] == target.shape[1]
    assert dikt['scores'].shape == dikt['mapes'].shape
    assert dikt['scores'].shape == dikt['nmses'].shape    
    assert dikt['rmnmse'].ndim == 1
    assert dikt['rmnmse'].shape[0] == 1   
    
    for key, value in dikt.items():
        dikt[key] = np.round(value, 8)

    return dikt


format_nb = '7.4f'

def print_performances(dikt_perf):
    scores = dikt_perf['scores']
    mapes  = dikt_perf['mapes']
    nmses  = dikt_perf['nmses']
    rmnmse = dikt_perf['rmnmse']
    assert rmnmse.shape == (1,)
    print('Scores - mean : ', '{0:{fmt}}; med : {1:{fmt}}; max : {2:{fmt}}; min : {3:{fmt}}'.format(scores.mean(), np.median(scores), scores.max(), scores.min(), fmt = format_nb))
    print('MAPEs  - mean : ', '{0:{fmt}}; med : {1:{fmt}}; max : {2:{fmt}}; min : {3:{fmt}}'.format(mapes .mean(), np.median(mapes ), mapes .max(), mapes .min(), fmt = format_nb))
    print('NMSEs  - mean : ', '{0:{fmt}}; med : {1:{fmt}}; max : {2:{fmt}}; min : {3:{fmt}}'.format(nmses .mean(), np.median(nmses ), nmses .max(), nmses .min(), fmt = format_nb))
    print('RMNMSE = {0:{fmt}}'.format(rmnmse[0],  fmt = format_nb))
    
    
def print_quantiles(perf):
    print('10  - ', '{0:{1}}'.format(np.percentile(perf, 10), format_nb))
    print('25  - ', '{0:{1}}'.format(np.percentile(perf, 25), format_nb))
    print('50  - ', '{0:{1}}'.format(np.percentile(perf, 50), format_nb))
    print('75  - ', '{0:{1}}'.format(np.percentile(perf, 75), format_nb))
    print('90  - ', '{0:{1}}'.format(np.percentile(perf, 90), format_nb))
    
 

