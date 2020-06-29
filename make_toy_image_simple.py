"""
This script, written by Daniel Palumb to be a thin method wrapper
around code from an iPython notebook by Ramesh Narayan,
creates eht-imaging image objects with I, Q and U maps corresponding
to Ramesh's toy model. An example call to generate an image with an
80x80 uas field of view across 160x160 pixels for an emission
region that is 1 M thick, inclined at a 20 degree angle to the observer,
emitting at a radius of 6 M, with a clockwise (negative) material rotation
of beta=0.41, and purely vertical field would be given by

im, coeffs, flipped_dividend = make_pol_toy(fov,npix,thickness_M,inc_dec,radius_M,beta,br,bphi,bz)
"""


import ehtim as eh
import numpy as np
import matplotlib.pyplot as plt

M87_ra = 12.513728717168174
M87_dec = 12.39112323919932
M87_dist = 16.9 * 3.086e22
M87_mass = 6.2e9 * 2e30
M_to_meter = 6.67e-11 / (3e8)**2



POLFLUX = True
INTENSITYISOTROPIC = True
SPECTRALINDEX = 0  # fnu propto nu^(-alpha)

def make_pol_toy(fov, npix, thickness_M,  inc_dec, radius_M, beta, br, bphi, bz, ra=M87_ra, dec=M87_dec, rf=230e9,mjd=57854, total_flux = 0.6, m=0.1, PWPs=[2]):
    """
    Returns the toy model image as well as PWP coefficients as specified by @PWPs. Also produces the flipped polarized dividend for investigation of symmetries.
    """

    inc_rad = inc_dec * np.pi/180
    radius_rad = radius_M * M87_mass*M_to_meter / M87_dist

    #make an empty image
    im = eh.image.make_empty(npix,fov, ra=ra, dec=dec, rf= rf, mjd = mjd, source='M87')

    pxi = (np.arange(npix)-0.01)/npix-0.5
    pxj = np.arange(npix)/npix-0.5
    # get angles measured East of North
    PXI,PXJ = np.meshgrid(pxi,pxj)
    varphi = np.arctan2(-PXJ,PXI)# - np.pi/2
    # varphi[varphi<0.] += 2.*np.pi

    #get grid of angular radii
    mui = pxi*fov
    muj = pxj*fov
    MUI,MUJ = np.meshgrid(mui,muj)
    MUDISTS = np.sqrt(np.power(MUI,2.)+np.power(MUJ,2.))

    #convert mudists to gravitational units
    rho = M87_dist / (M87_mass*M_to_meter) * MUDISTS


    #get emission coordinates
    def emission_coordinates(rho, varphi):
        phi = np.arctan2(np.sin(varphi),np.cos(varphi)*np.cos(inc_rad))
        sinprod = np.sin(inc_rad)*np.sin(phi)
        numerator = 1+rho**2 - (-3+rho**2)*sinprod+3*sinprod**2 + sinprod**3 
        denomenator = (-1+sinprod)**2 * (1+sinprod)
        sqq = np.sqrt(numerator/denomenator)
        r = (1-sqq + sinprod*(1+sqq))/(sinprod-1)
        return r, phi


    r, phi = emission_coordinates(rho, varphi)

    #begin Ramesh formalism - for details, see his notes
    bmag = np.sqrt(bx**2 + by**2 + bz**2)
    gfac = np.sqrt(1. - 2./r)
    gfacinv = 1. / gfac
    gamma = 1. / np.sqrt(1. - beta**2)

    sintheta = np.sin(inc_rad)
    costheta = np.cos(inc_rad)
    sinphi = np.sin(phi)
    cosphi = np.cos(phi)

    cospsi = -sintheta * sinphi
    sinpsi = np.sqrt(1. - cospsi**2)

    cosalpha = 1. - (1. - cospsi) * (1. - 2./r)
    sinalpha = np.sqrt(1. - cosalpha**2)

    
    sinxi = sintheta * cosphi / sinpsi
    cosxi = costheta / sinpsi
    
    kPthat = gfacinv
    kPxhat = cosalpha * gfacinv
    kPyhat = -sinxi * sinalpha * gfacinv
    kPzhat = cosxi * sinalpha * gfacinv
    
    kFthat = gamma * (kPthat - beta * kPyhat)
    kFxhat = kPxhat
    kFyhat = gamma * (-beta * kPthat + kPyhat)
    kFzhat = kPzhat

    delta = 1. / kFthat
    
    kcrossbx = kFyhat * bz - kFzhat * by
    kcrossby = kFzhat * bx - kFxhat * bz
    kcrossbz = kFxhat * by - kFyhat * bx
    polfac = np.sqrt(kcrossbx**2 + kcrossby**2 + kcrossbz**2) / (kFthat * bmag)
    
    profile = np.exp(-4*np.log(2)*((r-radius_M)/thickness_M)**2)

    polarizedintensity = polfac * delta**(3. + SPECTRALINDEX) * profile
    
    if INTENSITYISOTROPIC:
        intensity = delta**(3. + SPECTRALINDEX)
    else:
        intensity = polarizedintensity
    
    if POLFLUX:
        mag = polarizedintensity
    else:
        mag = 1
        
    fFthat = 0
    fFxhat = kcrossbx / (kFthat * bmag)
    fFyhat = kcrossby / (kFthat * bmag)
    fFzhat = kcrossbz / (kFthat * bmag)
    
    fPthat = gamma * (fFthat + beta * fFyhat)
    fPxhat = fFxhat
    fPyhat = gamma * (beta *fFthat + fFyhat)
    fPzhat = fFzhat
    
    kPrhat = kPxhat
    kPthhat = -kPzhat
    kPphat = kPyhat
    fPrhat = fPxhat
    fPthhat = -fPzhat
    fPphat = fPyhat
       
    k1 = r * (kPthat * fPrhat - kPrhat * fPthat)
    k2 = -r * (kPphat * fPthhat - kPthhat * fPphat)
        
    kOlp = r * kPphat
    radicand = kPthhat**2 - kPphat**2 * costheta**2 / sintheta**2

    #due to machine precision, some small negative values are present. We clip these here.
    radicand[radicand<0] = 0
    radical = np.sqrt(radicand)
    kOlth = r * radical * np.sign(sinphi)

    xalpha = -kOlp / sintheta
    ybeta = kOlth
    nu = -xalpha
    den = np.sqrt((k1**2 + k2**2) * (ybeta**2 + nu**2))

    ealpha = (ybeta * k2 - nu * k1) / den
    ebeta = (ybeta * k1 + nu * k2) / den

    q = -mag*(ealpha**2 - ebeta**2)
    u = -mag*(2*ealpha*ebeta)

    #flag out remaining nans
    q[np.isnan(q)]=0
    u[np.isnan(u)]=0
    #enforce pixel-wise fractional polarization m
    im.qvec = q.flatten()*m
    im.uvec = u.flatten()*m
    im.ivec = np.sqrt(im.qvec**2+im.uvec**2)/m
    #normalize to specified total_flux
    conversion = total_flux/im.total_flux()
    im.ivec *= conversion
    im.qvec *= conversion
    im.uvec *= conversion


    #define other utilities
    def compute_simple_PWPs(im, ms):
        #this computes the PWP coefficients across the entire image
        image_angle = varphi - np.pi/2.
        iarr = im.ivec.reshape(npix, npix)
        qarr = im.qvec.reshape(npix, npix)
        uarr = im.uvec.reshape(npix, npix)
        parr = qarr + 1j*uarr
        betas = []

        for m in ms:
            qbasis = np.cos(-image_angle*m)
            ubasis = np.sin(-image_angle*m)
            pbasis = qbasis + 1.j*ubasis
            prod = parr * pbasis
            coeff = prod.sum()/im.total_flux()
            betas.append(coeff)
        return betas

    def compute_flipped_dividend(im):
        parr = (im.qvec+1j*im.uvec).reshape(npix,npix)
        return parr / np.flip(parr, axis=0)

    coeffs = compute_simple_PWPs(im, PWPs)
    fd = compute_flipped_dividend(im)

    return [im, coeffs, fd]
