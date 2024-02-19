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
from heliohypy import heliohypy as hl
from heliohypy import solarsyst_dyn_geo as hl_utils

logging.basicConfig( level=logging.INFO )

DIR=os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser(
    prog='make_tracklets',
    description='Preprocess heliolinc inputs'
    )

# pathing args
# parser.add_argument( '--config',    help='path to config file', default=os.path.join( os.path.realpath(__file__), 'default.toml') )
parser.add_argument( '--dets',      help='path to input detections file' )
parser.add_argument( '--imgs',      help='path to images database', )
parser.add_argument( '--earthpos',   help='path to earth ephemeris file',    default=os.path.join(DIR, '../data/Earth_1day_2010_2040.csv') )
parser.add_argument( '--obscodes',  help='path to observatory file',        default=os.path.join(DIR, '../data/ObsCodes.txt' ) )
parser.add_argument( '--outfile',    help='output file stem', default=None )

# value args
parser.add_argument( '--observatory',   default='X05',  help='which observatory images are take from',)
parser.add_argument( '--imrad',         default=1.75,   help='image field of view radius (degrees)',)
parser.add_argument( '--maxtime',       default=1.5,    help='max time seperation between detections in a tracklet, hours',)
parser.add_argument( '--mintime',       default=1/3600, help='min time seperation between detections in a tracklet, hours',)
parser.add_argument( '--maxGCR',        default=0.5,    help='max residual for great circle fit in tracklet, arcsec',)
parser.add_argument( '--mintrkpts',     default=2,      help='min number of points in tracklet',)
parser.add_argument( '--minvel',        default=0.0,)
parser.add_argument( '--maxvel',        default=1.5,)
parser.add_argument( '--minarc',        default=0.0,)

args = parser.parse_args()

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
earthpos = hl_utils.load_earth_ephemerides( args.earthpos )

logging.info( ' Loading mpc observatory records ...' )
obscodes = hl_utils.load_obscodes( args.obscodes )

logging.info( ' Loading LSST operations simulator database ...' )
opsim = hl_utils.load_opsim( args.imgs, 
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
detections = pd.read_csv( 
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
fieldIds = detections[ 'FieldID' ].unique()
opsim = opsim[ opsim['FieldID'].isin(fieldIds) ]

detections = detections.merge( opsim, how='left', on='FieldID' )

for night in 
logging.info( ' converting inputs to structured arrays ...' )
hl_detections = hl_utils.format_detections(
    detections['start_MJD'],
    detections['AstRA(deg)'],
    detections['AstDec(deg)'],
    0, # magnitude, optional
    detections['ObjID'],
    detections['index'],
    detections['filter'],
    detections['obscode']
)

images = hl_utils.format_images( 
    opsim[ 'start_MJD' ],
    opsim[ 'fieldRA' ],
    opsim[ 'fieldDec' ],
    obscodes[ args.observatory ],
    earthpos
    )

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

logging.info( ' generating tracklets ...' )
# generate tracklets, pairs
# with hl.ostream_redirect(stdout=True, stderr=True): # necessary? needed it for notebooks, not sure here
paired_dets, tracklets, trk2det = hl.makeTracklets( conf, hl_detections, images )

# store (everything) in hdf5 file
# then heliolinc can operate on just the hdf5 file ( or subsets of it )
if args.outfile is None:
    out_file_name = os.path.basename(args.dets).split( '.' )[0] + '.h5'
else:
    out_file_name = args.outfile + '.h5'

# print( f' ... writing to {out_file_name}' )
logging.info( f' writing to {out_file_name}' )
with h5py.File( out_file_name, 'w' ) as file:
    dets_h5 = file.create_dataset( 
        'detections',
        (len(paired_dets),),
        dtype=paired_dets.dtype,
        )
    dets_h5[()] = paired_dets

    trks_h5 = file.create_dataset(
        'tracklets',
        (len(tracklets),),
        dtype=tracklets.dtype,
    )
    trks_h5 = tracklets

    trk2det_h5 = file.create_dataset(
        'trk2det',
        (len(trk2det),),
        dtype=trk2det.dtype,
    )
    trk2det_h5 = trk2det

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