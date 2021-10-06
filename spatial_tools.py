import math
import numpy as np
from scipy.special import legendre

def pol2cart(phi, rho):
    if phi < 0:
        phi*= -1
        phi = 180 + (180-phi)
    phi = math.radians(phi+90)
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return[x, y]

def cart2sphere(xyz):
    '''
    takes an N x 3 array of catesian coordinates and converts them to sphereical coordinates

    output: N x 3 array - [rad,polar_angle,azimuth_angle]
    '''
    sph_out = np.zeros(xyz.shape)
    xy = xyz[:,0]**2 + xyz[:,1]**2
    sph_out[:,0] = np.sqrt(xy+xyz[:,2]**2)
    sph_out[:,1] = np.arctan2(np.sqrt(xy), xyz[:,2])
    sph_out[:,2] = np.arctan2(xyz[:,1], xyz[:,0])+math.pi/2
    # for i,azi in enumerate(sph_out[:,2]):
    #     if azi < 0:
    #         temp = azi * -1
    #         sph_out[:,2][i] = math.pi + (math.pi-temp)
    # sph_out[:,0] = np.sqrt(xyz[:,0]**2 + xyz[:,1]**2 + xyz[:,2]**2)
    # sph_out[:,1] = np.arctan(xyz[:,1]/xyz[:,2])
    # sph_out[:,2] = np.arccos(xyz[:,2]/np.sqrt(xyz[:,0]**2 + xyz[:,1]**2 + xyz[:,2]**2))

    return sph_out

def proj_sph2pln(pol, azi, c_lat=0, c_lon=0):
    '''
    lon = azi
    lat = pol
    '''
    cos_c = math.sin(c_lon)*math.sin(azi)+math.cos(c_lon)*math.cos(azi)*math.cos(pol-c_lat)
    x = (math.cos(azi)*math.sin(pol-c_lat))/cos_c
    y = (math.cos(c_lon)*math.sin(azi)-math.sin(c_lon)*math.cos(azi)*math.cos(pol-c_lat))/cos_c
    
    return [x,y]

def get_legendre(x, n):
    '''
    get the solution for x on the nth degree Legendre Polynomial
    '''
    leg = legendre(n)
    Px_n = leg(x)
    return Px_n

def cosdist(i,j):
    '''
    get the cosine distance between the points in cartesian space 
    defined by i and j

    --Parameters--
    i,j : 1 x 3 array of cartesian coordinates
    '''
    return 1 - sum(np.square(i-j)) * 0.5

def get_G(chanlocs, order=10, smooth=4):
    '''
    returns the G weight matrix for the surface laplacian

    --Parameters--

    chan_locs: an N x 3 array of catesian coordinates of channel locations
                NB: must be normalised onto unit sphere! 
    '''
    n_elecs = len(chanlocs)
    G = np.zeros((n_elecs,n_elecs))
    
    for i in range(0, n_elecs):
        for j in range(i,n_elecs):

            for k in range(1, order+1):
                G[i,j] += (((2*k+1)/((k*(k+1))**smooth)) 
                * get_legendre(cosdist(chanlocs[i],chanlocs[j]), k))

            G[i,j] /= (4*math.pi)

    G = G + G.T 
    G = G - (np.identity(n_elecs) * (G[0,0]/2))
    return G

def get_H(chanlocs, order=10, smooth=4):
    '''
    returns the H weight matrix for the surface laplacian

    --Parameters--

    chan_locs: an N x 3 array of catesian coordinates of channel locations
                NB: must be normalised onto unit sphere! 
    '''
    n_elecs = len(chanlocs)
    H = np.zeros((n_elecs,n_elecs))
    
    for i in range(0, n_elecs):
        for j in range(i,n_elecs):

            for k in range(1, order+1):
                H[i,j] -= ((2*(k+1)/((k*(k+1))**smooth-1)) 
                * get_legendre(cosdist(chanlocs[i],chanlocs[j]), k))

            H[i,j] /= -1*(4*math.pi)

    H = H + H.T 
    H = H - (np.identity(n_elecs) * (H[0,0]/2))
    return H

def surface_laplacian(data, Gs_matrix, Gsinv_matrix, H_matrix):
    '''
    Returns data filtered spatially by the surface laplacian
    as calculated from spherical splines 

    NB: electrode order of data must match order of chan locs used to calculate G and H matrices
    '''
    d_matrix = np.linalg.solve(Gs_matrix, data.T)
    d_sum = np.sum(d_matrix, axis=0)
    Gsinv_sum = np.sum(Gsinv_matrix, axis=0)
    C = d_matrix - (d_sum/np.sum(Gsinv_sum)) * Gsinv_sum
    return C @ H_matrix
