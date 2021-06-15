from scipy import optimize
import numpy as np
from numpy.random import default_rng
import nfwFunctions as NFW
from scipy.interpolate import interp1d

def inverseTransformSampling(CDF, bracket, x0=None, x1=None):
    """
    Simple implementation of inverseTransformSampling. This takes any cumulative distribution function (CDF), subtracts some random offset between 0 and max(CDF) from it, and then finds the root of that offset CDF. When you repeat this process and record the roots, you recover the original distribution function and are sampling it directly.
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
    