import numpy as np
import pandas as pd
import argparse
import logging
import tomllib
import os

from dask.distributed import LocalCluster

import heliolinc3d.heliolinc3d_v3 as hl

def main():
    # load args
    parser = argparse.ArgumentParser()
    # parser.add_argument( '-c', '--config', help='config file (toml)' )
    parser.add_argument( '--dets', help='detections file' )
    parser.add_argument( '--imgs', help='images file' )
    parser.add_argument( '--ephem', help='ephemeris file' )

    args=parser.parse_args()

    # load config file, either from args or default
    config = tomllib.load( args['config'] )

    # give args precedence over configs ( write args into configs )

    # load data files
        # coerce columns? 
    
    detections = load_file( args['dets'] )
    images = load_file( args['images'] )
    obs_staes = load_file( args['ephem'] ) # load from de440?
    
    # distribute over processes
        # get clusters
    
    # compare clusters from multiple hypothesis?

def load_file( path, **kwargs ):

    # handle kwargs

    suffix = path.split( '.' )[-1] # 
    match suffix:
        case ['csv']:
            return pd.read_csv( path )

        case ['h5']:
            return pd.read_hdf( path )
        
        case ['parquet']:
            return pd.read_parquet( path )
        
        
# def get_obs_states( ephem_path, times, obscodes ):
#     """ just earth observers atm
    
#     """
#     return None