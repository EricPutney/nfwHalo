import numpy as np
from scipy import integrate
from scipy import special
import sys

def gFcn(x):
    """
    Helper function since things like this turn up a lot in NFW profiles.
    """
    return np.log(1+x)-(x/(1+x))

def nfwPotential(radiusNorm, tracerAttributes):
    """
    The potential (integrated GM/r^2 from +inf to r) at a given normalized radius.
    """
    rScaleNFW = tracerAttributes[0]
    rho0 = tracerAttributes[1]
    G = 4.3009*10**-3 # Newton's constant, (pc/solar mass) * (km/second)^2
    c = tracerAttributes[2]/tracerAttributes[0] # rDelta/rScaleNFW is the concentration
    return -(4*np.pi*rho0*G*rScaleNFW**2)*(np.log(1+c*radiusNorm)/(c*radiusNorm))
    
def massDensity(radiusNorm, tracerAttributes):
    rho0 = tracerAttributes[1]
    c = tracerAttributes[2]/tracerAttributes[0] # rDelta/rScaleNFW is the concentration
    return rho0/(c*radiusNorm*(1+c*radiusNorm)**2)

def relativeEnergy(radiusNorm, speed, tracerAttributes):
    rScaleNFW = tracerAttributes[0]
    rho0 = tracerAttributes[1]
    G = 4.3009*10**-3 # Newton's constant, (pc/solar mass) * (km/second)^2
    c = tracerAttributes[2]/tracerAttributes[0] # rDelta/rScaleNFW is the concentration
    energyNorm = 4.*np.pi*rho0*G*rScaleNFW**2
    Phi = -energyNorm*np.log(1+c*radiusNorm)/(c*radiusNorm)
    return -Phi - (1/2)*speed**2

def maxSpeed(radiusNorm, tracerAttributes):
    """
    Gives the max speed (the kinematic limit) at a particular radius. This is when a particle has enough kinetic energy to climb out of its potential (when KE=PE).
    """
    if 0<=np.max(radiusNorm)<=1:
        return np.sqrt(np.abs(2*nfwPotential(radiusNorm, tracerAttributes)))
    else:
        print("Out of acceptable radial bounds")

def radialPDF(radiusNorm, tracerAttributes):
    """
    This is the probability distribution function for the radius of tracer stars.
    """
    c = tracerAttributes[2]/tracerAttributes[0] # rDelta/rScaleNFW is the concentration

    if 0<=np.max(radiusNorm)<=1:
        return c**2*radiusNorm/(gFcn(c)*(1+c*radiusNorm)**2)
    else:
        print("Out of acceptable radial bounds")        

def radialCDF(radiusNorm, tracerAttributes):
    """
    This is the radial cumulative distribution function for an NFW halo. Takes an un-normalized radius and spits out the probability for
    r to be between 0 and the input radius. When paired with inverse transform sampling, you can rebuild the radial distribution function
    with a small set of tracers.
    """
    c = tracerAttributes[2]/tracerAttributes[0] # rDelta/rScaleNFW is the concentration
    if 0<=np.max(radiusNorm)<=1:
        return gFcn(c*radiusNorm)/gFcn(c)
    else:
        print("Out of acceptable radial bounds")
        
def A(PsiNorm):
    return np.real(special.lambertw(-PsiNorm*np.exp(-PsiNorm),k=-1))
    
def phaseSpaceDistribution(radiusNorm, speed, tracerAttributes):
    rScaleNFW = tracerAttributes[0]
    rho0 = tracerAttributes[1]
    G = 4.3009*10**-3 # Newton's constant, (pc/solar mass) * (km/second)^2
    c = tracerAttributes[2]/tracerAttributes[0] # rDelta/rScaleNFW is the concentration
    energyNorm = 4.*np.pi*rho0*G*rScaleNFW**2
    
    EpsNorm = relativeEnergy(radiusNorm, speed, tracerAttributes)/energyNorm
    if (EpsNorm<0 or np.abs(EpsNorm - 1) < 1e-3 or np.isnan(EpsNorm)==True):
        # This is a fairly desperate attempt on my part to control issues when EpsNorm is close to 1. This will not always work since special functions are a curse.
        if EpsNorm>0:
            sys.stderr.write('PSD was set to zero to control error at radiusNorm = '+str(radiusNorm)+ ' and speed = '+str(speed)+'. ')
        return 0.
    else:
        prefactor = 1./(np.sqrt(128)*np.pi**3*rScaleNFW**3*energyNorm**(3./2.)*gFcn(c))
        integrand = lambda PsiNorm: -(PsiNorm)**3*(1+(A(PsiNorm)/(PsiNorm)))*(2+(1+(A(PsiNorm)/(1+A(PsiNorm))))*((3*A(PsiNorm)/(PsiNorm))+2))/(np.sqrt(np.abs(PsiNorm-EpsNorm))*A(PsiNorm)**2*(1+A(PsiNorm))**2*(PsiNorm+A(PsiNorm)))
        return prefactor*integrate.quad(integrand,0,EpsNorm)[0]

def evalPsdLoop(tabulatedNormRadii, numTabulatedSpeeds, tabulatedSpeeds, tracerAttributes):
    psTable = np.zeros((len(tabulatedNormRadii), numTabulatedSpeeds))
    for i in range(len(tabulatedNormRadii)):
        for j in range(numTabulatedSpeeds):
            psTable[i,j] = phaseSpaceDistribution(tabulatedNormRadii[i], tabulatedSpeeds[j], tracerAttributes)
    return psTable
    
def speedCDF(radiusNorm, speed, tracerAttributes, tabulatedNormRadii, psdBivariateSpline):
    rDelta = tracerAttributes[2]
    radius = radiusNorm*rDelta
    return integrate.quad(lambda speedVar: (4*np.pi*radius*speedVar)**2*psdBivariateSpline(radiusNorm, speedVar),0,speed)[0]