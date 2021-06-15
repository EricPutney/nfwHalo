from scipy import optimize
import numpy as np
from numpy.random import default_rng
import nfwFunctions as NFW
from scipy.interpolate import interp1d

def inverseTransformSampling(CDF, bracket, x0=None, x1=None):
    """
    Simple implementation of inverseTransformSampling. Currently a little slow since I need to loop though each sample instead of doing it in one step that can be parallelized. You win some, you lose some. Anyways, this takes any cumulative distribution function (CDF), subtracts some random offset from it between 0 and 1, and then finds the root of that offset CDF. When you repeat this process and record the roots, you actually recover the original distribution function and can draw from it truly at random.

    The CDF(bracket[1]) is multiplied onto the random number to accomodate CDFs that aren't properly normalized. The size of the random number can now never be larger than the max value of the CDF.
    """
    cdfmax = CDF(bracket[1])
    randNum = cdfmax*np.random.default_rng().uniform(0,1,1)[0]
    CDFsubRandomNumber = lambda x: CDF(x) - randNum
    return optimize.root_scalar(CDFsubRandomNumber , bracket = bracket, method='brentq', x0=x0, x1=x1).root

def inverseTransformSamplingRadius(nTracers, haloAttributes, bracket, x0=None, x1=None):
    """
    Specially built ITS method for the cumulative distribution function for picking a radius.
    """
    cdfmax = NFW.radialCDF(bracket[1], haloAttributes)
    resultList = []
    for i in range(nTracers):
        randNum = cdfmax*np.random.default_rng().uniform(0,1,1)[0]
        CDFsubRandomNumber = lambda x: NFW.radialCDF(x, haloAttributes) - randNum
        resultList.append(optimize.root_scalar(CDFsubRandomNumber , bracket = bracket, method='brentq', x0=x0, x1=x1).root)
    return resultList

def inverseTransformSamplingVelocity(nTracers, haloAttributes, radiiNorm, tabNormRadii, speedCubicSplines):
    """
    Specially built ITS method for the cumulative distribution function for picking a speed.
    """
    resultList = []
    for i in range(nTracers):
        bracket=(0,NFW.maxSpeed(radiiNorm[i], haloAttributes))
        cdfmax = NFW.speedCDF(radiiNorm[i],bracket[1],haloAttributes,tabNormRadii, speedCubicSplines)
        randNum = cdfmax*np.random.default_rng().uniform(0,1,1)[0]
        CDFsubRandomNumber = lambda x: NFW.speedCDF(radiiNorm[i],x,haloAttributes,tabNormRadii,speedCubicSplines) - randNum
        resultList.append(optimize.root_scalar(CDFsubRandomNumber , bracket = bracket, method='brentq').root)
    return resultList

#def inverseTransformSamplingVelocity(nTracers, haloAttributes, radiiNorm):
#    """
#    Specially built ITS method for the cumulative distribution function for picking a speed.
#    """
#    resultList = []
#    for i in range(nTracers):
#        disp = NFW.radialVelocityDispersion(radiiNorm[i], haloAttributes)
#        bracket=(0,NFW.maxSpeed(radiiNorm[i], haloAttributes))
#        cdfmax = NFW.speedCDFcallByDisp(bracket[1], disp)
#        randNum = cdfmax*np.random.default_rng().uniform(0,1,1)[0]
#        CDFsubRandomNumber = lambda x: NFW.speedCDFcallByDisp(x, disp) - randNum
#        resultList.append(optimize.root_scalar(CDFsubRandomNumber , bracket = bracket, method='brentq', x0=disp).root)
#    return resultList
    
    