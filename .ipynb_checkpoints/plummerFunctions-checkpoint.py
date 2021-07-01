import numpy as np

def plummerMassDensity(radius, rScalePlummer, plummerMass):
    return (3*plummerMass/(4*np.pi*rScalePlummer**3))*(1+(radius/rScalePlummer)**2)**(-5/2)

def plummerPotential(radius, rScalePlummer, plummerMass):
    G = 4.3009*10**-3 # Newton's constant, (pc/solar mass) * (km/second)^2
    return -G*plummerMass/np.sqrt(radius**2+rScalePlummer**2)

def radialPDF(radius, rScalePlummer, plummerMass):
    return 4*np.pi*radius**2*plummerMassDensity/plummerMass
    
def radialCDF(radiusNorm, rScalePlummer, rCutoff):
    # rCutoff should be the NFW rDelta, since that's how far we're saying the NFW halo reaches out to.
    # radiusNorm should be the radius normalized by whatever rCutoff is.
    return radiusNorm**3/((radiusNorm**2+(rScalePlummer/rCutoff)**2)**(3/2))