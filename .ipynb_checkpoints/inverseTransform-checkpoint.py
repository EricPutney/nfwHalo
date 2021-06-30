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

def inverseTransformSamplingRadius(nTracers, haloAttributes, bracket, obsCutoff, obsRadius, earthRadius):
    """
    Picks a radius from the NFW radial CDF using inverse transform sampling. Also generates angular coordinates.
    """
    if obsCutoff == True:
        # The obsCutoff flag reduces the generation region to just a spherical region around the earth.
        cdfmax = NFW.radialCDF(bracket[1], haloAttributes)
        cdfmin = NFW.radialCDF(bracket[0], haloAttributes)
        # cutoffAngle defines the exact angular bounds required to encapsulate the observation sphere around the earth.
        cutoffAngle = 2*np.arctan(earthRadius/obsRadius - np.sqrt((earthRadius/obsRadius)**2-1))
        minCosTheta = np.cos(np.pi/2 - cutoffAngle)
        maxCosTheta = np.cos(np.pi/2 + cutoffAngle)
        minPhi = -cutoffAngle
        maxPhi = cutoffAngle
        earthVec = np.array((earthRadius/haloAttributes[6],0,0))
    else:
        # If obsCutoff is false, we generate the entire halo. So we allow all possibile values for the CDF and all angles.
        cdfmax = NFW.radialCDF(bracket[1], haloAttributes)
        cdfmin = 0.
        minCosTheta = -1
        maxCosTheta = 1
        minPhi = 0
        maxPhi = 2*np.pi

    radiiList = np.zeros((nTracers))
    angleList = np.zeros((2,nTracers))
    for i in range(nTracers):
        # This loop generates the coordinates (radius and angles) of each tracer.
        randNum = np.random.default_rng().uniform(cdfmin,cdfmax,1)[0]
        CDFsubRandomNumber = lambda x: NFW.radialCDF(x, haloAttributes) - randNum
        radiiList[i] = optimize.root_scalar(CDFsubRandomNumber , bracket = bracket, method='brentq').root
        angleList[:,[i]] = np.array([np.arccos(default_rng().uniform(minCosTheta,maxCosTheta,1)),default_rng().uniform(minPhi,maxPhi,1)]) # first row is theta, second row is phi
        if obsCutoff == True:
            # We need to check if this sample is within the observation region. This adds some time, but coordinate generation is hardly the rate limiting step.
            rejectionSample__FLAG = 0 # Zero means it hasn't been checked or it has been rejected
            while rejectionSample__FLAG == 0:
                tracerVec = np.array((radiiList[i]*np.sin(angleList[0,i])*np.cos(angleList[1,i]),radiiList[i]*np.sin(angleList[0,i])*np.sin(angleList[1,i]),radiiList[i]*np.cos(angleList[0,i])))
                if np.linalg.norm(tracerVec-earthVec) > obsRadius/haloAttributes[6]:
                    # It's outside the observation radius! Remake the sample and check again.
                    randNum = np.random.default_rng().uniform(cdfmin,cdfmax,1)[0]
                    CDFsubRandomNumber = lambda x: NFW.radialCDF(x, haloAttributes) - randNum
                    radiiList[i] = optimize.root_scalar(CDFsubRandomNumber , bracket = bracket, method='brentq').root
                    angleList[:,[i]] = np.array([np.arccos(default_rng().uniform(minCosTheta,maxCosTheta,1)),default_rng().uniform(minPhi,maxPhi,1)]) # first row is theta, second row is phi
                else:
                    rejectionSample__FLAG = 1 # It's within the observation radius! Move along.
            
    return (radiiList, angleList)

def inverseTransformSamplingVelocity(nTracers, haloAttributes, radiiNorm, tabNormRadii, speedCubicSplines):
    """
    Picks a speed from the NFW speed CDF using inverse transform sampling.
    """
    speedList = np.zeros((nTracers))
    angleList = np.zeros((2,nTracers))
    for i in range(nTracers):
        bracket=(0,NFW.maxSpeed(radiiNorm[i], haloAttributes))
        cdfmax = NFW.speedCDF(radiiNorm[i],bracket[1],haloAttributes,tabNormRadii, speedCubicSplines)
        randNum = cdfmax*np.random.default_rng().uniform(0,1,1)[0]
        CDFsubRandomNumber = lambda x: NFW.speedCDF(radiiNorm[i],x,haloAttributes,tabNormRadii,speedCubicSplines) - randNum
        speedList[i] = optimize.root_scalar(CDFsubRandomNumber , bracket = bracket, method='brentq').root
        angleList[:,[i]] = np.array([np.arccos(default_rng().uniform(-1,1,1)),default_rng().uniform(0,2*np.pi,1)]) # first row is theta, second row is phi

    return (speedList, angleList)
    