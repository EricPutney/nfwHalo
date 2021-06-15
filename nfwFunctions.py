import numpy as np
from scipy import integrate
from scipy import special
from scipy.interpolate import interp1d

def gFcn(x):
    """
    Helper function since things like this turn up a lot in NFW profiles.
    """
    return np.log(1+x)-(x/(1+x))

def nfwPotential(radiusNorm, haloAttrib):
    """
    The potential (integrated GM/r^2 from +inf to r) at a given normalized radius.
    """
    rho0 = haloAttrib[2]
    G = haloAttrib[7]
    c = haloAttrib[5]
    rScale = haloAttrib[1]
    return -(4*np.pi*rho0*G*rScale**2)*(np.log(1+c*radiusNorm)/(c*radiusNorm))
    
def massDensity(radiusNorm, haloAttrib):
    rho0 = haloAttrib[2]
    c = haloAttrib[5]
    return rho0/(c*radiusNorm*(1+c*radiusNorm)**2)

def relativeEnergy(radiusNorm,speed,haloAttrib):
    rho0 = haloAttrib[2]
    G = haloAttrib[7]
    c = haloAttrib[5]
    rScale = haloAttrib[1]
    energyNorm = 4.*np.pi*rho0*G*rScale**2
    Phi = -energyNorm*np.log(1+c*radiusNorm)/(c*radiusNorm)
    return -Phi - (1/2)*speed**2

def maxSpeed(radiusNorm, haloAttrib):
    """
    Gives the max speed (the kinematic limit) at a particular radius. This is when a particle has enough kinetic energy to climb out of its potential (when KE=PE).
    """
    if 0<=np.max(radiusNorm)<=1:
        return np.sqrt(np.abs(2*nfwPotential(radiusNorm, haloAttrib)))
    else:
        print("Out of acceptable radial bounds")

def radialPDF(radiusNorm, haloAttrib):
    """
    This is the probability distribution function for the radius of tracer stars.
    """
    c = haloAttrib[5]

    if 0<=np.max(radiusNorm)<=1:
        return c**2*radiusNorm/(gFcn(c)*(1+c*radiusNorm)**2)
    else:
        print("Out of acceptable radial bounds")        

def radialCDF(radiusNorm, haloAttrib):
    """
    This is the radial cumulative distribution function for an NFW halo. Takes an un-normalized radius and spits out the probability for
    r to be between 0 and the input radius. When paired with inverse transform sampling, you can rebuild the radial distribution function
    with a small set of tracers.
    """
    c = haloAttrib[5]
    if 0<=np.max(radiusNorm)<=1:
        return gFcn(c*radiusNorm)/gFcn(c)
    else:
        print("Out of acceptable radial bounds")
        
def A(PsiNorm):
    return np.real(special.lambertw(-PsiNorm*np.exp(-PsiNorm),k=-1))
    
def phaseSpaceDistribution(radiusNorm, speed, haloAttrib):
    rho0 = haloAttrib[2]
    G = haloAttrib[7]
    c = haloAttrib[5]
    rScale = haloAttrib[1]
    energyNorm = 4.*np.pi*rho0*G*rScale**2
    
    EpsNorm = relativeEnergy(radiusNorm,speed,haloAttrib)/energyNorm
    if (EpsNorm<0 or np.abs(EpsNorm - 1) < 1e-3 or np.isnan(EpsNorm)==True):
        # This is a fairly desperate attempt on my part to control issues when EpsNorm is close to 1. This will not always work since special functions are a curse.
        return 0.
    else:
        prefactor = 1./(np.sqrt(128)*np.pi**3*rScale**3*energyNorm**(3./2.)*gFcn(c))
        integrand = lambda PsiNorm: -(PsiNorm)**3*(1+(A(PsiNorm)/(PsiNorm)))*(2+(1+(A(PsiNorm)/(1+A(PsiNorm))))*((3*A(PsiNorm)/(PsiNorm))+2))/(np.sqrt(np.abs(PsiNorm-EpsNorm))*A(PsiNorm)**2*(1+A(PsiNorm))**2*(PsiNorm+A(PsiNorm)))
        return prefactor*integrate.quad(integrand,0,EpsNorm)[0]
    
def phaseSpaceDistributionFromTable(radiusNorm, speed, tabulatedNormRadii, speedSplines, haloAttrib):
    rNearIndices = np.argpartition(np.abs(tabulatedNormRadii - radiusNorm),3)[:2]
    rMinIndex = rNearIndices[np.argmin(tabulatedNormRadii[rNearIndices])]
    rMin = tabulatedNormRadii[rMinIndex]
    rMaxIndex = rNearIndices[np.argmax(tabulatedNormRadii[rNearIndices])]
    rMax = tabulatedNormRadii[rMaxIndex]
    
    weight1 = (rMax-radiusNorm)/(rMax-rMin)
    weight2 = 1 - weight1
    return weight1*speedSplines[rMinIndex](speed) + weight2*speedSplines[rMaxIndex](speed)
    
    
def speedCDF(radiusNorm, speed, haloAttrib, psTable, tabulatedNormRadii, speedSplines):
    rDelta = haloAttrib[6]
    radius = radiusNorm*rDelta
    return integrate.quad(lambda speedVar: (4*np.pi*radius*speedVar)**2*phaseSpaceDistributionFromTable(radiusNorm, speedVar,tabulatedNormRadii,speedSplines,haloAttrib),0,speed)[0]

#def speedPDF(speed, radiusNorm, haloAttrib):
#    """
#    The probability distribution function for the speed of a tracer at a given radius. Supply this with a speed below the the virial speed or the kinematic limit
#    and a radius below the virial radius, and this will return the probability to have that speed at that radius.
#    """
#    if 0<=np.max(radiusNorm)<=1:
#        radDispSquared = radialVelocityDispersion(radiusNorm, haloAttrib)**2
#        return (1/(2*np.pi*radDispSquared))**(3/2)*(4*np.pi*speed**2)*np.exp(-speed**2/(2*radDispSquared))
#    else:
#        print("Out of acceptable radial bounds")
#
#def speedCDF(speed, radiusNorm, haloAttrib):
#    """
#    Integrates the speed PDF up from 0 to the given speed, at a particular (normalized) radius.
#    """
#    if 0<=np.max(radiusNorm)<=1:
#        radDispSquared = radialVelocityDispersion(radiusNorm, haloAttrib)**2
#        return special.erf(speed/np.sqrt(2*radDispSquared)) - np.sqrt(2/(np.pi*radDispSquared))*speed*np.exp(-speed**2/(2*radDispSquared))
#    else:
#        print("Out of acceptable radial bounds")
#
#def speedCDFcallByDisp(speed, dispersion):
#    """
#    Integrates the speed PDF up from 0 to the given speed, at a particular (normalized) radius.
#    """
#    radDispSquared = dispersion**2
#    return special.erf(speed/np.sqrt(2*radDispSquared)) - np.sqrt(2/(np.pi*radDispSquared))*speed*np.exp(-speed**2/(2*radDispSquared))
#
#def radialVelocityDispersion(radiusNorm, haloAttrib):
#    """
#    We are approximating the velocity distribution of the tracer stars with a Maxwell-Boltzmann distribution. The first moment of this
#    distribution of radial velocities should be zero, so we're actually extracting the 2nd moment and calling it the dispersion. Then,
#    I'm plugging this dispersion into a distribution of speeds (hopefully translating all of the factors correctly?) to get a speed distribution
#    to generate random velocities from.
#    """
#    c = haloAttrib[5]
#    virialSpeed = haloAttrib[9]
#    if 0<=np.max(radiusNorm)<=1:
#        return np.sqrt(virialSpeed**2*\
#                        ((c**2*radiusNorm*(1+c*radiusNorm)**2)/gFcn(c))*\
#                        integrate.quad(lambda y: gFcn(y)/(y**3*(1+y)**2), c*radiusNorm, np.inf)[0]
#                        )
#    else:
#        print("Out of acceptable radial bounds")
#        
#def phaseSpaceDistributionFunction(radiusNorm, speed, haloAttrib):
#    """
#    Analytic form for the 6D phase space distribution. This is properly normalized to 1 when integrated over all space and velocities, and is close to one when
#    integrated to the virial speed.
#    """
#    rho0 = haloAttrib[2]
#    c = haloAttrib[5]
#    rScale = haloAttrib[1]
#    if 0<=np.max(radiusNorm)<=1:
#        disp = radialVelocityDispersion(radiusNorm, haloAttrib)
#        return massDensity(radiusNorm, haloAttrib) * np.exp(-speed**2/(2*disp**2))/( np.sqrt(128*np.pi**5) * rScale**3 * disp**3 * gFcn(c) * rho0)
#    else:
#        print("Out of acceptable radial bounds")