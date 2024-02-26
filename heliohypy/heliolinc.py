import os
import tomllib
import logging
import argparse
import numpy as np
import pandas as pd
import sqlite3 as sql
import h5py

import dask
from dask.distributed import Client
from dask.diagnostics import ProgressBar

from heliohypy import heliohypy as hl
from heliohypy import utils

# def heliolinc():

# @dask.delayed
def heliolinc():
    DIR=os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser(
        prog='Heliolinc2',
        description='Python command line program to run helilinc C++ routines'
    )

    parser.add_argument( '--infile',  help='path to h5 file containing preprocessed inputs.' )
    parser.add_argument( '--hypfile', help='file containing hypotheses' )
    # parser.add_argument( '--hyp_line_start' help='lines in hyp file to use', default=1 )
    # parser.add_argument( '--hyp_lines' help='lines in hyp file to use', default=-1 )
    # parser.add_argument( '--out', help='directory + stem to write output to' )
    
    # parser.add_argument( '--mjd-start',     help='mjd of start of observing interval to use' )
    # parser.add_argument( '--mjd-end',       help='mjd of end of observation window to use' )
    parser.add_argument( '--night',  help='key identifying final night to include', type=str )
    parser.add_argument( '--window', help='timespan of previous nights data to include, days', default=14, type=float )

    # parser.add_argument( '--rmin', help='minimum heliocentric radius to check' )
    # parser.add_argument( '--rmax', help='minimum heliocentric radius to check' )

    # parser.add_argument( '--rdotmax', help='minimum heliocentric radial velocity to check' )
    # parser.add_argument( '--rdotmax', help='minimum heliocentric radial velocity to check' )

    parser.add_argument( '--clustrad',      help='clustering radius', default=200000, type=float)
    parser.add_argument( '--npt', help='number of points for a linkage', default=3, type=int )
    parser.add_argument( '--minobsnights', help='min number of distinct nights for a linkage', default=3, type=int )
    parser.add_argument( '--mintimespan', default=1.0, type=float )
    parser.add_argument( '--mingeodist', default=0.1, type=float )
    parser.add_argument( '--maxgeodist', default=100.0, type=float )
    parser.add_argument( '--geologstep', default=1.5, type=float )

    parser.add_argument( '--nproc', help='number of processes', default=1, type=int )

    args = parser.parse_args()

    # client = Client( n_workers=args.nproc )

    # load detections, images, earthpos, etc
    # inputs = load_tracklets( args.infile, args.night, args.window )
    inputs = dask.delayed( load_tracklets )( args.infile, args.night, args.window )
    # inputs = inputs.compute()
    detvec, trks, trk2det, images, earthpos, mjd_ref = inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5]
    # detvec, trks, trk2det, images, earthpos, mjd_ref = inputs
    # set up heliolinc config
    # hl_config = setup_config( args, mjd_ref )
    hl_config = dask.delayed( setup_config )( args, mjd_ref )
    # hl_config = hl_config.compute()
    
    # load hypothesis
    radhyps_table = pd.read_csv( args.hypfile )
    # radhyps = utils.format_hypothesis( radhyps['r(AU)'], radhyps['rdot(AU/day)'], radhyps['mean_accel'] )
    radhyps = np.empty((len(radhyps_table),), utils.hlradhyp )
    radhyps[ 'HelioRad' ] = radhyps_table[ 'r(AU)' ]
    radhyps[ 'R_dot' ] = 0.0 # radhyps_table[ 'rdot(AU/day)' ]
    radhyps[ 'R_dubdot' ] = 0.0 # currently defaulting to 0. will later add 2nd order gravity term
    #radhyps_table[ 'mean_accel' ]
    
    # set up dask jobs
    # inputs = []
    clusters = []
    # radhyp = radhyps
    # clusters = dask.delayed(hl.Heliolinc())( hl_config, images, detvec, trks, trk2det, radhyp, earthpos )
    # clusters = clusters.compute()
    # clusters = hl.heliolinc( hl_config, images, detvec, trks, trk2det, radhyps, earthpos )
    # print(clusters)
    for radhyp in radhyps:
        # inputs.append( load_tracklets( args.infile, args.night, args.window ) )

        # detvec, trks, trk2det, images, earthpos, mjd_ref = inputs[-1][0], inputs[-1][1], inputs[-1][2], inputs[-1][3], inputs[-1][4], inputs[-1][5]
        # hl_config = setup_config( args, mjd_ref )

        hyp_clust = dask.delayed(hl.Heliolinc())( hl_config, images, detvec, trks, trk2det, np.array([radhyp]), earthpos )
        # hyp_clust = client.submit(hl.Heliolinc, hl_config, images, detvec, trks, trk2det, radhyp, earthpos )
        clusters.append( hyp_clust )
    # inputs.compute()
    # print( inputs  )
    out = dask.compute(clusters)

    # for cluster in clusters:
    #     print( len(cluster) )

    # run the dask jobs, get clusters
    # with ProgressBar():
    #     dask.compute( clusters, scheduler='processes', num_workers=args.nproc )

    # deduplicate, check for good clusters
        
    # write to output

# @dask.delayed
# def heliolinc_hyp( detvec, tracklets, trk2det, hyp, images, earthpos, hl_config ):
#     """ run heliolinc on a hypothesis, delayed with dask
#     """
#     # r = hyp['HelioRad']
#     # rdot = hyp['R_dot']
#     # print( f'testing hypothesis heliocentric radius: {r}, radial velocity: {rdot}' )

#     clusters, clust2det = hl.heliolinc(
#         hl_config,
#         images,
#         detvec,
#         tracklets,
#         trk2det,
#         hyp,
#         earthpos,
#     )

#     return clusters, clust2det
        
# @dask.delayed
def setup_config( args, mjd_ref ):
    hl_config = hl.HeliolincConfig()
    hl_config.clustrad = args.clustrad
    hl_config.dbscan_npt = args.npt
    hl_config.minobsnights = args.minobsnights
    hl_config.mintimespan = args.mintimespan
    hl_config.mingeodist = args.mingeodist
    hl_config.maxgeodist = args.maxgeodist
    hl_config.geologstep = args.geologstep
    hl_config.MJDref = mjd_ref

    return hl_config

# @dask.delayed
def load_tracklets( trk_file, night_key, window ):
    """ load detections, tracklets, tracklet-2-detection table, image info, and earthpos from hdf5 file

    have to format the trk2det indices properly, since trackletes are computed by night
    """
    file = h5py.File( trk_file, 'r' )
    nights = file['nights'].keys()

    tf = np.trunc( file['nights'][night_key].attrs['start_mjd'] )

    detvec      = []
    tracklets   = []
    trk2det     = []

    for night in nights:
        
        t0 = np.trunc(file['nights'][ night ].attrs['start_mjd'])
        if 0 <= tf-t0 <= window:
        # if night == night_key:
            # print( night )
            detvec.append( np.array(file['nights'][ night ][ 'detections' ]) )
            tracklets.append( np.array(file['nights'][ night ][ 'tracklets' ]) )
            trk2det.append( np.array(file['nights'][ night ][ 'trk2det' ]) )
            # detvec.append( file['nights'][ night ][ 'detections' ][()] )
            # tracklets.append( file['nights'][ night ][ 'tracklets' ][()] )
            # trk2det.append( file['nights'][ night ][ 'trk2det' ][()] )

    # update indices and ids
    n_trks=0
    n_dets=0
    for i, tracklet_chunk in enumerate(tracklets):
        detvec[i]['index'] += n_dets
        trk2det[i]['i2'] += n_dets

        tracklet_chunk['trk_ID'] += n_trks
        trk2det[i]['i1'] += n_trks

        n_dets += len( detvec )
        n_trks += len( tracklet_chunk )

    detvec      = np.concatenate( detvec )
    tracklets   = np.concatenate( tracklets )
    trk2det     = np.concatenate( trk2det )

    # image[-1]ndex = file[ 'images' ][ 'index' ]
    # images_start[-1]ndex = tracklets['Img1'].min()
    # images_stop[-1]ndex = tracklets['Img2'].max()

    images = np.array(file[ 'images' ])#[()]
    earthpos = np.array(file[ 'earth_ephem' ])#[()]

    mjd_ref = 0.5*(detvec['MJD'].min() + detvec['MJD'].max())

    return detvec, tracklets, trk2det, images, earthpos, mjd_ref