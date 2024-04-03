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
from .vector import sphere_line_intercept
from . import constants as cn

logger = logging.getLogger( __name__ )

#==============================================================================

def heliolinc3d( xyz_los, dt_ref, xyz_obs, r0, dr0, max_vel=0.02, max_trk_time=2/24, n_iter=1, GM=cn.GM ):
    """ HelioLinC small body linking routine

    parameters
    ----------
    xyz_los     ... (3, n) array with observation lone of sight unit vectors
    dt_ref      ... n array with time seperation of observations from reference epoch (units consistent with GM)
    image_ids   ... n array of integers indexing into xyz_obs
    xyz_obs     ... (3, n) array with observer position at time of observation (units consistent with GM)
    r0          ... hypothesis distance from central body (units consistent with GM)
    dr0         ... hypothesis radial velocity (units consistent with GM)

    """
    # find tracklets
    pairs = find_tracklets( xyz_los, dt_ref, xyz_obs, r0, dr0, max_vel=max_vel, max_trk_time=max_trk_time )

    # build arrows using iterations on 2nd order term
    xyz, vel = make_arrows_from_pairs( pairs, xyz_los, xyz_obs, dt_ref, r0, dr0, n_iter=n_iter )

    # cluster tracklets

def find_tracklets( xyz_los, dt_ref, xyz_obs, r0, dr0, max_vel=0.02, max_trk_time=2/24 ):
    n = xyz_los.shape[1]
    # project to hypothesis
    xyzt = np.zeros( (n, 4) )
    xyzt[:, 0:3] = sphere_line_intercept( xyz_los, xyz_obs, r0+dr0*dt_ref ).T
    xyzt[:, 3] = dt_ref*max_vel

    # kd trees don't like nans
    nan_mask = ~np.any(np.isnan( xyzt ), axis=1)
    ids = np.arange( n )[ nan_mask ]
    logger.info( f' {len(ids)} points projected to hypothesis' )

    # find pairs
    kd_r = max_vel*max_trk_time
    logger.info( ' building tracklet search tree ...' )
    t0 = time()
    kd_tree = sc.cKDTree( xyzt[nan_mask], leafsize=16)
    tf = time()

    logger.info( f' tracklet search tree construction time {tf-t0} s')
    logger.info( ' data bounding box dimensions:' )
    logger.info( f'    x: {xyzt[nan_mask, 0].max() - xyzt[nan_mask, 0].min()}')
    logger.info( f'    y: {xyzt[nan_mask, 1].max() - xyzt[nan_mask, 1].min()}')
    logger.info( f'    z: {xyzt[nan_mask, 2].max() - xyzt[nan_mask, 2].min()}')
    logger.info( f' querying for neighbors with search radius {kd_r}' )
    t0 = time()
    pairs = kd_tree.query_pairs( kd_r, p=np.inf, output_type='ndarray' )
    tf = time()

    logger.info( f' query time {tf-t0} s' )
    logger.info( f' found {len(pairs)} candidate pairs' )

    return pairs

def make_arrows_from_pairs( pairs, xyz_los, xyz_obs, dt, r0, dr0, n_iter=1, GM=cn.GM ):
    # tracklet masks
    los_1 = xyz_los[pairs[:, 0]]
    los_2 = xyz_los[pairs[:, 1]]
    
    obs_1 = xyz_obs[pairs[:, 0]]
    obs_2 = xyz_obs[pairs[:, 1]]

    dt1 = dt[pairs[:, 0]]
    dt2 = dt[pairs[:, 1]]

    xyz = sphere_line_intercept( los_1, obs_1, r0+dr0*dt1 )
    vel = (xyz - sphere_line_intercept( los_2, obs_2, r0+dr0*dt2 )) / ( dt1 - dt2 ) # I think this avoids unnecessery allocations?
    
    for _ in range(n_iter): 
        # |l| = r*v_t = r0*v0_t
        # ddr = -GM/r^2 + v_t^2/r
        # v_t^2/r = |l|^2 / r^3
        # then
        # v0_t^2/r0 = |l|^2 / r0^3
        ddr0 = -GM/r0**2 + np.sum( np.cross( xyz, vel, axis=0 )**2, axis=0 ) / r0**3
        xyz = sphere_line_intercept( los_1, obs_1, r0 + dr0*dt1 + 0.5*ddr0*dt1**2 )
        vel = (xyz - sphere_line_intercept( los_2, obs_2, r0 + dr0*dt2 + 0.5*ddr0*dt2**2 )) / ( dt1 - dt2 )
            
    return xyz, vel

# @njit
# def make_arrow_from_pair( xyz_los_1, xyz_los_2, xyz_obs_1, xyz_obs_2, dt1, dt2, r0, dr0, max_vel=0.02, max_trk_time=2/24, n_iter=1, GM=cn.GM ):
#     xyz_1 = sphere_line_intercept(  )

#     for 

# @njit
# def make_arrow_from_pair(
#     xlos1, ylos1, zlos1,
#     xlos2, ylos2, zlos2,
#     xobs1, yobs1, zobs1,
#     xobs2, yobs2, zobs2,
#     dt1, dt2,
#     r0, dr0,
#     n_iter=1, GM=cn.GM,
#     ):
#     x1, y1, z1 = sphere_line_intercept( xlos1, ylos1, zlos1, xobs1, yobs1, zobs1, r0+dr0*dt1 )
#     x2, y2, z2 = sphere_line_intercept( xlos2, ylos2, zlos2, xobs2, yobs2, zobs2, r0+dr0*dt2 )

#     vx = (x1 - x2) / (dt1 - dt2)
#     vy = (y1 - y2) / (dt1 - dt2)
#     vz = (z1 - z2) / (dt1 - dt2)

#     for _ in np.arange( n_iter ):
#         l2 = 

# @njit
# def sphere_line_intercept( 
#     xlos, ylos, zlos,
#     xobs, yobs, zobs,
#     r
#     ):
#     rlos = np.sqrt( xlos**2 + ylos**2 + zlos**2 )
#     xln = xlos / rlos
#     yln = ylos / rlos
#     zln = zlos / rlos

#     ol = xobs*xln + yobs*yln + zobs*zln
#     o2 = xobs*xobs + yobs*yobs + zobs*zobs

#     discrim = np.sqrt( ol**2 - o2 + r**2 )
#     k = -ol + discrim

#     xout = xobs + xln*k
#     yout = yobs + yln*k
#     zout = zobs + zln*k

#     return xout, yout, zout

# @njit
# def sphere_line_intercept( los, obs, r ):
#     ln = los / np.linalg.norm( los ) # normalize los
#     ol = np.sum( obs*ln )
#     o2 = np.sum( obs**2 )
#     # discrim may be nan
#     discrim = np.sqrt( ol**2 - o2 + r**2 )
#     k = -ol + discrim
#     return obs + ln * k

# def make_arrows( xyz_los_1, xyz_los_2, xyz_obs_1, xyz_obs_2, dt1, dt2, r0, dr0, max_vel=0.02, max_trk_time=2/24, n_iter=1, GM=cn.GM ): 
#     """ 
#     xyz_obs
#     """

#     xyz = sphere_line_intercept( xyz_los_1, xyz_obs_1, r0+dr0*dt1 )
#     vel = (xyz - sphere_line_intercept( xyz_los_2, xyz_obs_2, r0+dr0*dt2 )) / (dt1-dt2)

#     for _ in range(n_iter):

#=======

def makeTracklets_radec( ra, dec, mjd, max_vel=1.5, max_time=2/24, min_time=0.00119, deg=True,
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

def make_heliocentric_arrows_radec( ra, dec, dt, obs_xyz, r, drdt, max_vel=0.02, max_trk_time=2/24, n_iter=1, GM=cn.GM ):
    """
    max_vel         ... AU/day
    max_trk_time    ... days
    """
    logger.info( ' initial projection to hypothesis' )
    xyzt = np.zeros( (4, len(ra)) ).T
    xyzt[:, 0:3] = estimate_position( ra, dec, dt, obs_xyz, r, drdt ).T
    xyzt[:, 3] = dt*max_vel
    kd_r = max_vel*max_trk_time

    nan_mask = ~np.any(np.isnan( xyzt ), axis=1)
    non_nan = np.arange( len(ra) )[nan_mask] # indices in the observations table
    # xyzt = xyzt[ nan_mask ]

    logger.info( ' identifying pairs' )
    pairs = find_pairs( xyzt[nan_mask], kd_r )
    pairs = non_nan[ pairs ]

    logger.info( ' filtering pairs on time seperation' )
    time_sep = dt[ pairs[1] ] - dt[ pairs[0] ]
    mask = np.where((0.0 < time_sep) & (time_sep<= max_trk_time) )[0]
    pairs = pairs[:, mask]
    
    logger.info( ' iterating on tracklet positions and velocities' )
    x1, v1, dt1 = arrows_from_pairs( xyzt[nan_mask].T, dt, pairs )
    # for _ in range(n_iter): # generate propgressiviely better estimates on the angular momentum (tied closely to the d^2 r / dt^2 term)
    #     L = np.cross( x1, v1, axis=0 )
    #     x1, v1 = recompute_xv( L, pairs, ra, dec, dt, obs_xyz, r, drdt, GM=GM )

    # logger.info( ' filtering arrows on velocity' )
    # mask = np.where( np.sum(v1**2, axis=0) <= max_vel**2 )[0]
    # return x1, v1, dt1, pairs
    return xyzt, pairs
    # return x1[:, mask], v1[:, mask], dt1[mask], pairs[:, mask]


def heliolinc3d_radec( ra, dec, mjd, mjd_ref, r, drdt, obs_xyz, max_vel=0.02, max_trk_time=2/12, db_eps=0.00001, deg=True, min_pts=3, GM=cn.GM, n_iter=1, ):
    """ generates tracklets from projected points ( since projection is cheap enough ) and searches for pairs within 
    a spatial limit given by a timespan and max velocity.

    clusters in angular momentum and eccentricty vector space

    paramters
    ---------
    ra, dec, mjd         ...observations
    obs_xyz              ...heliocentric position of opbserver at time of observation
    pairs                ...tracklet pairs (indices in observations)

    """
    N = len(ra)
    assert N == len(dec) == len(mjd) == obs_xyz.shape[1]

    logger.info( ' generating tracklets' )
    dt = mjd - mjd_ref
    x, v, dt, pairs = make_heliocentric_arrows( ra, dec, dt, obs_xyz, r, drdt, max_vel=max_vel, max_trk_time=max_trk_time, deg=deg, GM=GM )

    logger.info( ' clustering in angular momentum and eccentricty vectors' )
    ang_mom = np.cross( x, v, axis=0 )
    ecc_vec = np.cross( v, ang_mom, axis=0 ) - GM*x/r 
    # scale eccentricity to angular momentum
    # or scale angular momentum to eccentricity?
    # A /= np.linalg.norm( A, axis=0 )
    # A *= np.linalg.norm( L, axis=0 )

    db = DBSCAN( min_samples=3, eps=db_eps )
    # db = hdbscan.HDBSCAN( min_samples=3, cluster_selection_epsilon=db_eps )
    db.fit( np.vstack( [ang_mom, ecc_vec] ).T )    

    # logger.info( 'filtering on variance of cluster mean states' )

    return db, pairs

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
    v0_t = l / r0
    r0ddot = -GM/r0**2 + np.sum(v0_t**2, axis=0)/r0

    x1 = estimate_position( ra[pairs[0]], dec[pairs[0]], dt[pairs[0]], obs_xyz[:, pairs[0]], r0, r0dot, r0ddot )
    x2 = estimate_position( ra[pairs[1]], dec[pairs[1]], dt[pairs[1]], obs_xyz[:, pairs[1]], r0, r0dot, r0ddot )

    v1 = (x2 - x1) / (dt[pairs[1]] - dt[pairs[0]])
    return x1, v1


def estimate_position( ra, dec, dt, obs_xyz, r0, r0dot, r0ddot=0.0, deg=True ):
    logger.debug( f' ra array shape : {ra.shape}' )
    logger.debug( f' obs array shape: {obs_xyz.shape}' )
    dr = r0dot*dt + 0.5*r0ddot*dt**2
    xyz = project_to_hypothesis( ra, dec, obs_xyz, dt, r0+dr, deg=deg )
    return xyz

def find_pairs( points, kd_r, balanced_tree=True, leafsize=16 ):
    logger.info( f' number of points in kd-tree: {points.shape}' )
    t0 = time()
    kdtree = sc.KDTree(
        points, 
        leafsize=leafsize, 
        compact_nodes=True,
        copy_data=False, 
        balanced_tree=balanced_tree, 
        boxsize=None,
        )
    tf = time()
    logger.info( f' KD-tree construction time: {tf-t0} s' )

    n_pairs = kdtree.count_neighbors( kdtree, kd_r, p=np.inf ) - len(points)
    logger.info( f' number of pairs: {n_pairs} ' )

    logger.info( f' KD tree search radius: {kd_r} AU' )

    t0 = time()
    pairs = kdtree.query_pairs(kd_r, p=np.inf, eps=0.0, output_type='ndarray')
    tf = time()
    logger.info( f' KD-tree query time: {tf-t0} s' )

    return pairs
            
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