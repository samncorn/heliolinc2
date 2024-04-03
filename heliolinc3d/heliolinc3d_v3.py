from platform import mac_ver
import numpy as np
import numba
import scipy.spatial as sc
import logging

from numba import njit
from sklearn.cluster import DBSCAN
from time import time

from .transforms import radec2icrfu
from .vector import sphere_line_intercept
from . import constants as cn

logger = logging.getLogger( __name__ )

def link_hypothesis( xyz_los, dt_ref, xyz_obs, r0, dr0, max_vel=0.02, max_trk_time=2/24, n_iter=1, GM=cn.GM ):
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
    pairs = find_tracklets_on_sky( xyz_los, dt_ref, xyz_obs, r0, dr0, max_vel=max_vel, max_trk_time=max_trk_time )

    # build arrows using iterations on 2nd order term
    xyz, vel, dt_trk = make_arrows_from_pairs( pairs, xyz_los, xyz_obs, dt_ref, r0, dr0, n_iter=n_iter )

    # cluster tracklets

def find_tracklets_on_sky( xyz_los, xyz_obs, t, max_ang_vel=2.0, max_trk_time=2.0/24 ):
    """

    max_ang_vel is DEG PER DAY
    """

    max_ang_vel_rad = max_ang_vel*np.pi/180.0

    n = xyz_los.shape[1]
    xyzt = np.zeros( (n, 4) )
    xyzt[:, 0:3] = xyz_los.T
    xyzt[:, 3] = t*max_ang_vel_rad

    kd_r = max_ang_vel_rad*max_trk_time
    logger.info( ' building tracklet search tree ...' )
    t0 = time()
    kd_tree = sc.cKDTree( xyzt, leafsize=16)
    tf = time()

    logger.info( f' tracklet search tree construction time {tf-t0} s')
    logger.info( f' querying for neighbors with search radius {kd_r}' )
    t0 = time()
    pairs = kd_tree.query_pairs( kd_r, p=np.inf, output_type='ndarray' )
    tf = time()

    logger.info( f' query time {tf-t0} s' )
    logger.info( f' removing same time pairs' )
    pairs = pairs[ np.where( t[pairs[:, 0]] != t[pairs[:, 1]] ) ]

    return pairs

def find_tracklets_projected( xyz_los, dt_ref, xyz_obs, r0, dr0, max_vel=0.02, max_trk_time=2/24 ):
    n = xyz_los.shape[1]
    # project to hypothesis
    xyzt = np.zeros( (n, 4) )
    xyzt[:, 0:3] = sphere_line_intercept( xyz_los, xyz_obs, r0+dr0*dt_ref ).T
    xyzt[:, 3] = dt_ref*max_vel

    # kd trees don't like nans
    nan_mask = ~np.any(np.isnan( xyzt ), axis=1)
    # xyzt = xyzt[nan_mask]
    ids = np.arange( n )[ nan_mask ]
    logger.info( f' {len(ids)} points projected to hypothesis' )

    # find pairs
    kd_r = max_vel*max_trk_time
    logger.info( ' building tracklet search tree ...' )
    t0 = time()
    kd_tree = sc.cKDTree( xyzt[nan_mask], leafsize=16)
    tf = time()

    logger.info( f' tracklet search tree construction time {tf-t0} s')
    # logger.info( ' data bounding box dimensions:' )
    # logger.info( f'    x: {xyzt[nan_mask, 0].max() - xyzt[nan_mask, 0].min()}')
    # logger.info( f'    y: {xyzt[nan_mask, 1].max() - xyzt[nan_mask, 1].min()}')
    # logger.info( f'    z: {xyzt[nan_mask, 2].max() - xyzt[nan_mask, 2].min()}')
    logger.info( f' querying for neighbors with search radius {kd_r}' )
    t0 = time()
    pairs = kd_tree.query_pairs( kd_r, p=np.inf, output_type='ndarray' )
    pairs = ids[pairs] # make sure pairs index the input arrays, not the nanless projections
    tf = time()

    logger.info( f' query time {tf-t0} s' )

    logger.info( f' removing same time pairs' )
    pairs = pairs[ np.where( dt_ref[pairs[:, 0]] != dt_ref[pairs[:, 1]] ) ]

    logger.info( f' removing tracklets above maximum velocity' )
    pairs = pairs[ np.where( 
        np.sum( (xyzt[pairs[:, 0], 0:3]-xyzt[pairs[:, 1], 0:3])**2, axis=1) / (dt_ref[pairs[:, 0]]-dt_ref[pairs[:, 1]]) <= max_vel**2
        ) ]

    logger.info( f' found {len(pairs)} candidate pairs' )

    return pairs

@njit
def make_arrows_from_pairs( pairs, xyz_los, xyz_obs, dt, r0, dr0, max_vel=0.02, n_iter=1, GM=cn.GM ):
    """ 

    xyz array needs to be shape (n, 3)

    TODO
    > test for convergence rather than n_iter
    """
    output = np.empty(( len(pairs), 6 )) # pos, vel array
    for i, (i1, i2) in enumerate(pairs):
        x1 = _sphere_line_intercept( xyz_los[i1], xyz_obs[i1], r0+dr0*dt[i1] )
        x2 = _sphere_line_intercept( xyz_los[i2], xyz_obs[i2], r0+dr0*dt[i2] )

        time_sep = dt[i1] - dt[i2]
        a1 = GM*x1/( np.sqrt( np.sum(x1**2) )**3 )
        v = (x1 - x2) / time_sep

        if np.sum( v**2 ) > max_vel**2:
            output[i, 0:3] = np.nan
            output[i, 3:6] = np.nan

        else:
            for _ in range(n_iter):
                l2 = np.sum( np.cross( x1, v )**2 )
                ddr0 = -GM/r0**2 + l2/r0**3

                x1 = _sphere_line_intercept( xyz_los[i1], xyz_obs[i1], r0+dr0*dt[i1]+ddr0*dt[i1]**2 )
                x2 = _sphere_line_intercept( xyz_los[i2], xyz_obs[i2], r0+dr0*dt[i2]+ddr0*dt[i2]**2 )

                a1 = GM*x1/( np.sqrt( np.sum(x1**2) )**3 )
                v = (x1 - x2) / time_sep + 0.5*a1*time_sep

            output[i, 0:3] = x1
            output[i, 3:6] = v

    return output
        

@njit
def _sphere_line_intercept( xyz_los, xyz_obs, r ):
    # ln = l / np.linalg.norm( l, axis=0 ) # normalize los
    ln = xyz_los / np.sqrt( np.sum( xyz_los**2 ))
    ol = np.sum( xyz_obs*ln,)
    o2 = np.sum( xyz_obs**2,)
    # discrim may be nan
    discrim = np.sqrt( ol**2 - o2 + r**2 )
    k = -ol + discrim
    return xyz_obs + ln * k 