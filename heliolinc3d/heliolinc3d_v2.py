import numpy as np
import numba
import scipy.spatial as sc
import logging
import hdbscan

from numpy import sqrt
from numba import njit
from spiceypy import prop2b
from sklearn.cluster import DBSCAN, HDBSCAN
# from hdbscan import hdbscan
from time import time

from .transforms import radec2icrfu, frameChange
from .vector import sphereLineIntercept, sphere_line_intercept
from . import constants as cn

logger = logging.getLogger( __name__ )

def makeTracklets( ra, dec, mjd, max_vel=1.5, max_time=2/24, min_time=0.00119, deg=True,
    leafsize=16,
    balanced_tree=True,
    eps=0.0,
    ):
    """ generate a list of tracklets 

    parameters

    ra, dec     ... angular sky measurements
    t           ... mjd of observation
    maxvel      ... maximum angular velocity to consider (ang/day)

    """
    # # convert search radius (angle) to cartesian distance
    # use small angle approximation for now
    cr = max_vel*max_time
    if deg:
        cr *= np.pi / 180.0
    logger.info( f' clustering radius for tracklets: {cr} radians' )
    # convert to cartesian on the unit sphere (to handle periodic boundaries in ra dec)
    # normalize time, so that max_time is on the 4-sphere of radius 2*cr
    xyzt = np.zeros( (len(ra), 4) )
    xyzt[:, 0:3] = radec2icrfu( ra, dec, deg=deg ).T
    xyzt[:, 3] = mjd * max_vel

    # tree on position and time
    t0 = time()
    kdtree = sc.KDTree(
        xyzt, 
        leafsize=leafsize, 
        compact_nodes=True,
        copy_data=False, 
        balanced_tree=balanced_tree, 
        boxsize=None,
        )
    tf = time()
    logger.info( f' KD-tree construction time: {tf-t0} s' )
    
    # query in radius
    # we search with radius 2*cr to add a time cutoff
    # this results in a larger number of tracklets, but simplifies aplication over a large period of time
    cr2 = np.sqrt(2)*cr
    logger.info( f' scaled clustering radius for tracklets in time: {cr2} radians' )
    t0 = time()
    pairs = kdtree.query_pairs(cr2, p=2., eps=eps, output_type='ndarray')
    tf=time()
    logger.info( f' KD-tree pairs query time: {tf-t0} s' )
    logger.info( f' found {len(pairs)} candidate pairs' )
    # filter out pairs at same time 
    # used as a proxy for same image, hopefully simultaneous observations do not appear
    time_sep = np.abs( mjd[pairs[:,0]] -mjd[pairs[:,1]]  )
    mask = np.where( (min_time < time_sep) & (time_sep <= max_time) )
    pairs = pairs[mask]
    logger.info( f' found {len(pairs)} valid pairs' )

    return pairs

def make_heliocentric_arrows( ra, dec, mjd, obs_xyz, pairs, r, drdt, mjd_ref, max_vel=0.02, deg=True, GM=cn.GM, lttc=False ):
    """ convert tracklets to cartesian 3d arrows for r, rdot hypothesis
    """
    xyz = radec2icrfu( ra, dec, deg=deg )
    los = frameChange( xyz.T, 'icrf', 'ecliptic' )
    # logger.info( f'    LOS array shape: {los.shape}' )
    # los_nans = np.any( np.isnan(los) )
    # logger.info( f'    {los_nans} nans in LOS array' )
    dt = mjd - mjd_ref

    # compute the radius of the sphere to intersect
    # use first and second order expansions of heliocentric radius in time
    # second order coefficient computed from angular momentum
    # h = r*( sqrt(GM/r) - drdt )
    ddrdt = 0.0 # ( -GM/r**2 + h**2/r**3 )
    # assume angular momentum of a circular orbit for the given radius
    dr = drdt*dt + (ddrdt/2) * dt**2
    # logger.info( f'    obs array shape: {obs_xyz.shape}' )
    posu = sphere_line_intercept( los, obs_xyz, r + dr ) # nans where no intercept exists
    # pos_nans = np.any( np.isnan(posu) )
    # logger.info( f'    {pos_nans} nans in projected LOS array' )

    # good_pairs = pairs[ ~np.isnan( np.sum(posu, axis=1) ) ]
    # let the nans propagate to the final step

    # compute velocities (forward differencing)
    # also a little gravity to help
    # x1 = posu[ pairs[:, 0], : ]
    # x2 = posu[ pairs[:, 1], : ]
    # dt1 = dt[ pairs[:,0] ]
    # dt2 = dt[ pairs[:,1] ]
    # a1 = -GM / (r * dr)**3 * x1

    # v1 = (( x1 - x2 ).T / ( dt1 - dt2 )).T - 0.5*a1*( dt1 - dt2 )
    # t = mjd[ pairs[:,0] ]

    x1, v1, dt1 = arrows_from_pairs( posu, dt, pairs )

    mask = np.where( np.sum(v1*v1, axis=1) <= max_vel**2 )
    good_pairs = pairs[ mask ]
    logger.info( f' found {len(good_pairs)} valid pairs after velocity cut' )

    return x1[mask], v1[mask], dt1[mask], good_pairs

def make_heliocentric_arrows_2( ra, dec, mjd, obs_xyz, r, drdt, mjd_ref, max_vel=0.02, max_trk_time=1.5/24 ,deg=True, GM=cn.GM, lttc=False, leafsize=16, balanced_tree=True, eps=0.0, n_iter=1 ):
    """
    max_vel         ... AU/day
    max_trk_time    ... days
    """
    cr = np.sqrt(2)*max_vel*max_trk_time
    logger.info( f' tracklet search radius: {cr} AU' )
    if deg:
        cr = cr * np.pi / 180.

    dt = mjd - mjd_ref

    xyzt = np.zeros( (4, len(ra)) )
    xyzt[3] = dt * max_vel
    xyzt[0:3] = estimate_position( ra, dec, dt, obs_xyz, r, drdt )

    logger.debug( f' xyzt \n {xyzt[:, 0:10]}' )
    logger.debug( f' {np.abs( dt*max_vel ).min()} AU' )
    if np.any( np.isnan( xyzt ) ):
        logger.debug( f' NANS!!!!' )
        # return

    pairs = find_arrows( xyzt, cr, balanced_tree=balanced_tree, leafsize=leafsize )
    logger.info( f' found {pairs.shape[1]} potential pairs' )

    # cull arrows for reasonable behavior
    time_sep = np.abs( dt[pairs[0]] - dt[pairs[1]] )
    mask = np.where((0.0 < time_sep) & (time_sep<= max_trk_time) )
    logger.debug( f' pairs time sep \n { time_sep } ')
    logger.debug( f' time sep mask \n {mask}' )

    pairs = pairs.T[mask].T
    logger.debug( f' pairs shape {pairs.shape}' )
    logger.info( f' found {pairs.shape[1]} valid pairs' )

    # return pairs
    logger.debug( f' constructing arrows from pairs ...' )
    x1, v1, dt1 = arrows_from_pairs( xyzt[0:3], dt, pairs )

    # recompute
    for _ in range(n_iter): # generate propgressiviely better estimates on the angular momentum (tied closely to the d^2 r / dt^2 term)
        L = np.cross( x1, v1, axis=0 )
        x1, v1 = recompute_xv( L, pairs, ra, dec, mjd-mjd_ref, obs_xyz, r, drdt, GM=GM )

    # this should also cut the nans
    logger.debug( f' filtering tracklets on velocity...' )
    mask = np.where( np.sum(v1*v1, axis=0) <= max_vel**2 )[0]
    good_pairs = pairs[:, mask ]
    logger.info( f' found {good_pairs.shape[1]} valid pairs after velocity cut' )
    logger.debug( f' filtered tracklets on velocity...' )

    return x1[:, mask], v1[:, mask], dt1[mask], good_pairs


def heliolinc3d( ra, dec, mjd, obs_xyz, r, drdt, max_vel=0.02, max_trk_time=2/12, cr=0.001, deg=True, min_pts=3, GM=cn.GM, n_iter=1 ):
    """ generates tracklets from projected points ( since projection is cheap enough ) and searches for pairs within 
    a spatial limit given by a timespan and max velocity.

    clusters in angular momentum and eccentricty vector space

    paramters
    ---------
    ra, dec, mjd         ...observations
    obs_xyz              ...heliocentric position of opbserver at time of observation
    pairs                ...tracklet pairs (indices in observations)

    """
    mjd_ref = 0.5*(np.max(mjd) + np.min( mjd ))
    # compute 
    logger.info( ' calculating arrows...' )
    # returns AU and AU/day
    # get first guess
    x, v, dt, pairs = make_heliocentric_arrows_2( ra, dec, mjd, obs_xyz, r, drdt, mjd_ref, max_vel=max_vel, max_trk_time=max_trk_time, deg=deg )
    # recompute r dub dot term
    logger.debug( f' x, v shapes: {x.shape}, {v.shape}' )
    
    logger.debug( f' restimating rddot term' )
    L = np.cross( x, v, axis=0 )
    logger.debug( f' L shape: {L.shape}' )

    logger.debug( f' v shape: {v.shape}')
    logger.debug( f' l shape: {L.shape}')
    A = np.cross( v, L, axis=0 ) - GM*x/r 
    # scale eccentricity to angular momentum
    # or scale angular momentum to eccentricity?
    A /= np.linalg.norm( A, axis=0 )
    A *= np.linalg.norm( L, axis=0 )
    
    logger.info( ' clustering angular momentum and eccentricity...' )
    # not_nan_mask = ~np.isnan( np.sum(states, axis=1) ) # filter the nans
    # good_pairs = pairs[not_nan_mask]
    L_A = np.vstack([L, A]) # convert to row for each point
    # db = DBSCAN( eps=cr, min_samples=min_pts ).fit( L_A.T )

    # try hdbscan
    db = hdbscan.HDBSCAN( min_samples=3, cluster_selection_epsilon=cr )
    db.fit( L_A.T )    
    # db = HDBSCAN( min_samples=min_pts ).fit( L_E )

    # logger.info( 'filtering on variance of cluster mean states' )


    return db, L_A, pairs

# def reestimate_xv(x, v, t1, t2, r, rdot, GM):
#     """ use estimates of x, v to compute angular momentum -> transverse velocity at reference epoch
#     to get a better rddot estimate, and recompute x, v

#     t1 and t2 are time from reference epoch
#     """
#     L = np.cross( x, v, axis=1 ).T
#     v_transverse = np.linalg.norm( L, axis=0 ) / r
#     r_ddot = -GM/r**2 + (v_transverse**2)/r

#     r2_1 = r + rdot*t1 + 0.5*r_ddot*t1**2
#     r2_2 = r + rdot*t2 + 0.5*r_ddot*t2**2

#     x2 = project_to_hypothesis()
#     v2 = 

def recompute_xv( l, pairs, ra, dec, dt, obs_xyz, r0, r0dot, GM ):
    # L and pairs should have same length
    # pairs indexes ra, dec, dt, obs_xyz
    logger.debug( f' angular momentum shape: {l.shape}' )
    v0_t = l / r0
    logger.debug( f' v0_t shape: {v0_t.shape}' )
    r0ddot = -GM/r0**2 + (v0_t**2)/r0

    logger.debug( f' obs_xyz shape: {obs_xyz.shape}' )
    logger.debug( f' r0ddot shape {r0ddot.shape}' )
    x1 = estimate_position( ra[pairs[0]], dec[pairs[0]], dt[pairs[0]], obs_xyz[:, pairs[0]], r0, r0dot, r0ddot )
    x2 = estimate_position( ra[pairs[1]], dec[pairs[1]], dt[pairs[1]], obs_xyz[:, pairs[1]], r0, r0dot, r0ddot )

    v1 = (x2 - x1) / (dt[pairs[1]] - dt[pairs[0]])
    return x1, v1


def estimate_position( ra, dec, dt, obs_xyz, r0, r0dot, r0ddot=0.0, deg=True ):
    dr = r0dot*dt + 0.5*r0ddot*dt**2
    xyz = project_to_hypothesis( ra, dec, obs_xyz, dt, r0+dr, deg=deg )
    return xyz

def find_arrows( xyzt, cr, balanced_tree=True, leafsize=16 ):
    t0 = time()
    kdtree = sc.KDTree(
        xyzt.T, 
        leafsize=leafsize, 
        compact_nodes=True,
        copy_data=False, 
        balanced_tree=balanced_tree, 
        boxsize=None,
        )
    tf = time()
    logger.info( f' KD-tree construction time: {tf-t0} s' )

    t0 = time()
    pairs = kdtree.query_pairs(cr, p=2., eps=0.0, output_type='ndarray')
    tf = time()
    logger.info( f' KD-tree construction time: {tf-t0} s' )

    return pairs.T
            
@njit
def calculate_mean_states_2( labels, components ):
    n = len(np.unique(labels)) - 1
    d = components.shape[1]

    counts = np.zeros( n )
    mean_states = np.zeros( (n, d), dtype=components.dtype, )

    for label, point in zip(labels, components):
        mean_states[ label ] += point
        counts[ label ] += 1
    
    # mean_states /= counts
    return (mean_states.T / counts).T
# def filter_mean_states( mean_states ):

def project_to_hypothesis( ra, dec, obs_xyz, dt, r, deg=True ):
    """ 
    r may be single value or vector
    """
    xyz = radec2icrfu( ra, dec, deg=deg ).T
    logger.debug( f' xyz shape: {xyz.shape}' )
    los = frameChange( xyz, 'icrf', 'ecliptic' ).T

    return sphere_line_intercept( los, obs_xyz, r )

def arrows_from_pairs( xyz, dt, pairs, GM=cn.GM ):
    """
    dt is time from reference epoch (may be negative)
    """
    x1 = xyz[ :, pairs[0] ]
    x2 = xyz[ :, pairs[1] ]
    dt1 = dt[ pairs[0] ]
    dt2 = dt[ pairs[1] ]
    dt12 = dt1 - dt2

    r1 = np.linalg.norm(x1, axis=0)
    a1 = -GM/r1**3 * x1
    # a1 = 0.0
    v1 = ( x1 - x2 ) / dt12 - 0.5*a1*dt12

    return x1, v1, dt1