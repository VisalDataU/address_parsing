# Import important libraries
# import os
# import glob
import pandas as pd
import geopandas as gpd

# from shapely.geometry import Polygon
# from shapely.geometry import MultiPolygon
from shapely import wkt

from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL
# import psycopg2
# import sys
# from os import environ

# create connection to the database
# config = dict(
#     drivername='postgresql+psycopg2',
#     username='postgres',
#     password='Z1@$2020',
#     host='192.168.22.131',
#     port='5432',
#     database='gis_db',
# )
# url = URL.create(**config)
# print(url)
# engine = create_engine(url, echo=True)

# read table with the sql query that extract geometry, which is converted into multipolygon points, along with 
# names and ID of location.
# gis_loc = pd.read_sql_query(

#     """SELECT c.id, ppvill.codephum as phum_id, c.com_id, c.dis_id, c.pro_id, ppvill.village as village_name, 
# 		c.com_name, d.dis_name, p.pro_name, COALESCE(ST_AsText(ppvill.geom), ST_AsText(c.geom)) as geometry
#         FROM commune_v4 c
#         INNER JOIN district_v4 d
#         ON c.dis_id = d.dis_id
#         INNER JOIN province_v4 p
#         ON c.pro_id = p.pro_id
#         LEFT JOIN pp_boundary_village ppvill
#         ON c.com_id = ppvill.codekhum""",

#     con=engine
# )

import sqlite3
database = "./gis_data_db.sqlite"
conn = sqlite3.connect(database)

gis_loc = pd.read_sql_query(
    """SELECT *
        FROM gis""",
    con=conn
)

# convert polygon points from string to geometry data type
gis_loc['geometry'] = gis_loc['geometry'].apply(wkt.loads)

# convert Pandas dataframe to GeoPandas dataframe. This dataframe will be used in get_residing_locs function.
polygon_area = gpd.GeoDataFrame(gis_loc, crs="EPSG:4326")

def pd_to_geopd(df: pd.DataFrame, longitude='longitude', latitude='latitude'):
    """

    Purpose:

    convert Pandas dataframe to GeoPandas dataframe and create a geometry column from latitude and longitude.
    
    Note:

    - Always input longitude parameter before latitude parameter. 

    - Latitude and longitude columns must be in float data type.
             
    ----------

    Parameter: 

    1. df: Pandas dataframe.
    2. longitude: Name of the column that contains longitude (default: "longitude").
    2. latitude: Name of the column that contains latitude (default: "latitude").

    ----------

    Returns:

    GeoPandas dataframe with a new column called "geometry".

    ----------

    Usage: 
    
    gpd_df = pd_to_geopd(df)

    """       
    point_cases = gpd.GeoDataFrame(
                    df,
                    geometry=gpd.points_from_xy(df[longitude], df[latitude]))

    return point_cases           


def get_residing_locs(point_cases: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """

    Purpose:

    Get villages, communes, districts, and provinces where the points (lat and long) from 
    the input dataframe lie within. 
             
    ----------

    Parameter: 

    1. point_cases: GeoPandas dataframe that contains geometry (lat and long).

    ----------

    Returns:

    GeoPandas dataframe with new columns of the following names:
        - index_right
        - id
        - phum_id 
        - com_id 
        - dis_id  
        - pro_id   
        - village_name  
        - com_name   
        - dis_name
        - pro_name

        Note: Those columns will show NaN or None if the input gemotry is not within the existing polygons
        in the database.  

    ----------

    Usage: 
    
    output = get_residing_locs(geo_df)

    """      
    # To get residing locations of the input geometry, sjoin (spatial join) function from GeoPandas module will be used 
    ## to join the input dataframe with polygon_area dataframe.
    df_out = gpd.sjoin(point_cases, polygon_area, how='left', op='within') 
    # df_out = df_out.drop_duplicates(subset="Property Code", keep="last")     
    return df_out

# sqlite connection     
database1 = "./gis_province_db.sqlite"
conn1 = sqlite3.connect(database1)

# get GIS province data for verifying whether a point (lat and long) is in Cambodia's provinces 
gis_province = pd.read_sql_query(
    "SELECT pro_id, pro_name, geometry FROM gis_province",
    con=conn1
)    

# convert polygon points from string to geometry data type
gis_province['geometry'] = gis_province['geometry'].apply(wkt.loads)

# convert Pandas dataframe to GeoPandas dataframe. This dataframe will be used in get_residing_locs function.
polygon_province = gpd.GeoDataFrame(gis_province, crs="EPSG:4326")

def verify_geom(point_cases: gpd.GeoDataFrame) -> gpd.GeoDataFrame:  
    """

    Purpose:

    Get villages, communes, districts, and provinces where the points (lat and long) from 
    the input dataframe lie within. 
             
    ----------

    Parameter: 

    1. point_cases: GeoPandas dataframe that contains geometry (lat and long).

    ----------

    Returns:

    GeoPandas dataframe with new columns of the following names:
        - index_right
        - id
        - phum_id 
        - com_id 
        - dis_id  
        - pro_id   
        - village_name  
        - com_name   
        - dis_name
        - pro_name

        Note: Those columns will show NaN or None if the input gemotry is not within the existing polygons
        in the database.  

    ----------

    Usage: 
    
    output = get_residing_locs(geo_df)

    """          
    # To get residing locations of the input geometry, sjoin (spatial join) function from GeoPandas module will be used 
    ## to join the input dataframe with polygon_province dataframe.
    df_out = gpd.sjoin(point_cases, polygon_province, how='left', op='within') 
    # df_out = df_out.drop_duplicates(subset="Property Code", keep="last")     
    return df_out

def view_non_legit_geom(df_pd: pd.DataFrame, longitude: str, latitude: str):
    test = pd_to_geopd(df_pd, longitude, latitude)
    out = verify_geom(test)
    non_legit = out[(out["index_right"].isna()) & 
                    (out[latitude].notnull() | out[longitude].notnull())]
    return non_legit    

def extract_address_ents(df_pd: pd.DataFrame, longitude: str, latitude: str):
    test = pd_to_geopd(df_pd, longitude, latitude)
    out = get_residing_locs(test)
    return out       