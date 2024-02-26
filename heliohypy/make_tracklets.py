# heliolinc preprocessing
# makes tracklets, formats inputs, saves all to single hdf5 file
# multiple heliolinc realizations can be run out of the same prepocess output
    # for example, many hypotheis grid points
    # or a range of clustering radii
    # or different time intervals
# this allows for mor efficient storage of input data, without needing to store multiple copies

import os
import tomllib
import logging
import argparse
import numpy as np
import pandas as pd
import sqlite3 as sql
import h5py

# from heliolinc_utils import load_opsim
# import heliohypy as hl
from heliohypy import heliohypy as hl
from heliohypy import utils

logging.basicConfig( level=logging.INFO )

def make_tracklets():
    DIR=os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser(
        prog='make_tracklets',
        description='Preprocess heliolinc inputs'
        )
    # pathing args
    # parser.add_argument( '--config',    help='path to config file', default=os.path.join( os.path.realpath(__file__), 'default.toml') )
    parser.add_argument( '--dets',      help='path to input detections file' )
    parser.add_argument( '--imgs',      help='path to images database', )
    parser.add_argument( '--earthpos',   help='path to earth ephemeris file',    default=os.path.join(DIR, 'data/Earth_1day_2010_2040.csv') )
    parser.add_argument( '--obscodes',  help='path to observatory file',        default=os.path.join(DIR, 'data/ObsCodes.txt' ) )
    parser.add_argument( '--outfile',    help='output file stem', default=None )
    # value args
    parser.add_argument( '--observatory',   default='X05',  help='which observatory images are take from',)
    parser.add_argument( '--imrad',         default=2.2,   help='image field of view radius (degrees)',)
    parser.add_argument( '--maxtime',       default=1.5,    help='max time seperation between detections in a tracklet, hours',)
    parser.add_argument( '--mintime',       default=1/3600, help='min time seperation between detections in a tracklet, hours',)
    parser.add_argument( '--maxGCR',        default=0.5,    help='max residual for great circle fit in tracklet, arcsec',)
    parser.add_argument( '--mintrkpts',     default=2,      help='min number of points in tracklet',)
    parser.add_argument( '--minvel',        default=0.0,)
    parser.add_argument( '--maxvel',        default=1.5,)
    parser.add_argument( '--minarc',        default=0.0,)

    # parser.add_argument( '--tf', help='mjd to stop at', type=float )
    parser.add_argument( '--window', default=14.0 )

    # heliolinc args
    # parser.add_argument( '--clustrad',      help='clustering radius', default=200000. )
    # parser.add_argument( '--npt', help='number of points for a linkage', default=3 )
    # parser.add_argument( '--minobsnights', help='min number of distinct nights for a linkage', default=3 )
    # parser.add_argument( '--mintimespan', default=1.0 )
    # parser.add_argument( '--mingeodist', default=0.1 )
    # parser.add_argument( '--maxgeodist', default=100.0 )
    # parser.add_argument( '--geologstep', default=1.5 )

    # parser.add_argument( '--nproc', help='number of processes', default=1 )

    args = parser.parse_args()

    # setup tracklet config
    conf = hl.MakeTrackletsConfig()
    conf.mintrkpts  = args.mintrkpts
    # conf.imagetimetol # use default
    conf.maxvel     = args.maxvel
    conf.minvel     = args.minvel
    conf.minarc     = args.minarc
    conf.maxtime    = args.maxtime
    conf.mintime    = args.mintime
    conf.imagerad   = args.imrad
    conf.maxgcr     = args.maxGCR


    # conf.forecerun
    # conf.verbose    = False

    # config = tomllib.load( args['config'] )

    # keys mapping comman line args to equivalent config fields
    # arg_keys = {
    #     'dets' : 
    # }
    # move args to config
    # for arg in args:
    #     if arg is not None:
    #         # arg = config[ arg_keys[ arg ] ]
    #         config[ arg_keys[ arg ] ] = arg
    # read inputs
        # -detections
        # -observer positions
        # -obscodes
        # tracklet configurations

    # if arg not passed, check the config file
    logging.info( ' Loading Earth ephemeris file ...' )
    earthpos = utils.load_earth_ephemerides( args.earthpos )

    logging.info( ' Loading mpc observatory records ...' )
    obscodes = utils.load_obscodes( args.obscodes )

    logging.info( ' Loading LSST operations simulator database ...' )
    opsim = utils.load_opsim( args.imgs, 
                    columns=[
                        'observationId',
                        'fieldRA',
                        'fieldDec',
                        'filter',
                        'night',
                    ] ).reset_index(drop=True).reset_index()

    opsim.rename( columns = {
            'observationId' : 'FieldID',
            'observationStartMJD' : 'start_MJD',
            }, inplace=True )
    opsim[ 'obscode' ] = args.observatory

    logging.info( ' Loading detections file' )
    all_detections = pd.read_csv( 
        args.dets,
        usecols=[ 
            'ObjID', 
            'FieldID', 
            'AstRA(deg)', 
            'AstDec(deg)' 
            ],
        ).sort_values( 
            by='FieldID', 
            ignore_index=True 
            )
    fieldIds = all_detections[ 'FieldID' ].unique()
    opsim = opsim[ opsim['FieldID'].isin(fieldIds) ]

    # merge and reset index
    all_detections = all_detections.merge( opsim, how='left', on='FieldID' )

    images = utils.format_images( 
        opsim[ 'start_MJD' ],
        opsim[ 'fieldRA' ],
        opsim[ 'fieldDec' ],
        obscodes[ args.observatory ],
        earthpos
        )
    
        # store (everything) in hdf5 file
    # then heliolinc can operate on just the hdf5 file ( or subsets of it )
    if args.outfile is None:
        out_file_name = os.path.basename(args.dets).split( '.' )[0] + '_tracklets' + '.h5'
    else:
        out_file_name = args.outfile + '_tracklets' + '.h5'

    logging.info( ' converting inputs to structured arrays ...' )
    nights = opsim['night'].unique()

    with h5py.File( out_file_name, 'w' ) as file:
        logging.info( f' writing to {out_file_name}' )

        top_group = file.create_group( 'nights' )

        # write images and earthpos to file
        images_h5 = file.create_dataset(
            'images',
            (len(images),),
            dtype=images.dtype,
            )
        images_h5[()] = images

        earthpos_h5 = file.create_dataset(
            'earth_ephem',
            (len(earthpos),),
            dtype=earthpos.dtype,
            )
        earthpos_h5[()] = earthpos

        #======================================================================
        # detections = all_detections[ all_detections['start_MJD'] <= args.tf ]
        # detections = detections[ detections['start_MJD'] >= (args.tf - args.window) ]

        # hl_detections = utils.format_detections(
        #     detections['start_MJD'],
        #     detections['AstRA(deg)'],
        #     detections['AstDec(deg)'],
        #     0, # magnitude, optional
        #     detections['ObjID'],
        #     detections['index'],
        #     detections['filter'],
        #     detections['obscode']
        # )

        # paired_dets, tracklets, trk2det = hl.makeTracklets( conf, hl_detections, images )#images_night )
        # ref_mjd = 0.5*( paired_dets['MJD'].max() + paired_dets['MJD'].min() )
        # hl_config = setup_config( args, ref_mjd )

        # radhyp = np.empty( (2,), dtype=utils.hlradhyp )
        # radhyp[ 'HelioRad' ] = np.array( [1.1, 1.2] )
        # radhyp[ 'R_dot' ] = 0.0 # radhyps_table[ 'rdot(AU/day)' ]
        # radhyp[ 'R_dubdot' ] = 0.0

        # clusters = hl.heliolinc( hl_config, images, paired_dets, tracklets, trk2det, radhyp, earthpos )

        # night_group = file.create_group( 'detections' )

        # dets_h5 = night_group.create_dataset( 
        #     'detections',
        #     (len(paired_dets),),
        #     dtype=paired_dets.dtype,
        #     compression="gzip",
        #     )
        # dets_h5[()] = paired_dets

        # trks_h5 = night_group.create_dataset(
        #     'tracklets',
        #     (len(tracklets),),
        #     dtype=tracklets.dtype,
        #     compression="gzip",
        # )
        # trks_h5[()] = tracklets

        # trk2det_h5 = night_group.create_dataset(
        #     'trk2det',
        #     (len(trk2det),),
        #     dtype=trk2det.dtype,
        #     compression="gzip",
        # )
        # trk2det_h5[()] = trk2det

        #======================================================================
        # generate tracklets for each night
        for night in nights:
            detections = all_detections[ all_detections['night']==night ]

            # images_mask = opsim['night'] == night
            # images_night = images[ images_mask ]
            # images_start_ind = images_night.index[0]

            start_mjd = all_detections[ 'start_MJD' ].min()
            end_mjd    = all_detections[ 'start_MJD' ].max()

            night_group = top_group.create_group( str(night) )
            # attach metadata
            night_group.attrs.create( 'start_mjd', start_mjd )
            night_group.attrs.create( 'end_mjd', end_mjd )

            hl_detections = utils.format_detections(
                detections['start_MJD'],
                detections['AstRA(deg)'],
                detections['AstDec(deg)'],
                0, # magnitude, optional
                detections['ObjID'],
                detections['index'],
                detections['filter'],
                detections['obscode']
            )

            # what index is returned?
            # hl_detections['index'] = detections.index


            # generate tracklets, pairs
            # with hl.ostream_redirect(stdout=True, stderr=True): # necessary? needed it for notebooks, not sure here
            paired_dets, tracklets, trk2det = hl.makeTracklets( conf, hl_detections, images )#images_night )
            print ( paired_dets[:100] )
            # tracklets['Img1'] += images_start_ind
            # tracklets['Img2'] += images_start_ind

            dets_h5 = night_group.create_dataset( 
                'detections',
                (len(paired_dets),),
                dtype=paired_dets.dtype,
                compression="gzip",
                )
            dets_h5[()] = paired_dets

            trks_h5 = night_group.create_dataset(
                'tracklets',
                (len(tracklets),),
                dtype=tracklets.dtype,
                compression="gzip",
            )
            trks_h5[()] = tracklets

            trk2det_h5 = night_group.create_dataset(
                'trk2det',
                (len(trk2det),),
                dtype=trk2det.dtype,
                compression="gzip",
            )
            trk2det_h5[()] = trk2det

# def setup_config( args, mjd_ref ):
#     hl_config = hl.HeliolincConfig()
#     hl_config.clustrad = args.clustrad
#     hl_config.dbscan_npt = args.npt
#     hl_config.minobsnights = args.minobsnights
#     hl_config.mintimespan = args.mintimespan
#     hl_config.mingeodist = args.mingeodist
#     hl_config.maxgeodist = args.maxgeodist
#     hl_config.geologstep = args.geologstep
#     hl_config.MJDref = mjd_ref

#     return hl_config