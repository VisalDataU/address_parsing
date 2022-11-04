# import sys
# sys.path.append('./')
from rapidfuzz import process, fuzz

import json
import tqdm
import pandas as pd
from pathlib import Path
import logging
logging.getLogger().setLevel(logging.ERROR)

from polyfuzz import PolyFuzz
from polyfuzz.models import RapidFuzz  
from pandas import DataFrame


def load_json(path: str) -> dict:
    """

    Purpose: 
    
    - Imports dictionary using json.load().

    ----------

    Parameter: 

    1. path: file path.
    
    ----------

    Returns: 
    
    A json file.

    ----------

    Usage: 
    
    word2idx = load_json(path+'word2idx.json')

    """    
    with open(path) as json_file:
        data = json.load(json_file)
    return data        


def import_vocab(path: str) -> dict:
    
    record_type_1 = load_json(path/'1_record_type.json')
    property_type_2 = load_json(path/'2_property_type.json')
    property_sector_3 = load_json(path/'3_property_sector.json')
    property_category_4 = load_json(path/'4_property_category.json')
    property_lookup = load_json(path/'property_lookup.json')

    # # convert keys into integer 
    # idx2word = {int(k):v for k,v in idx2word.items()}
    # idx2tag = {int(k):v for k,v in idx2tag.items()}

    return record_type_1, property_type_2, property_sector_3 , property_category_4, property_lookup

path = Path(__file__).parent.absolute()

json_path = path/'data_v2/'
# print("path: ", json_path)
# path = './data_v2/'
# record_type_1, property_type_2, property_sector_3 , property_category_4, property_lookup = import_vocab(json_path)

def df_to_list(df: pd.DataFrame, col: str) -> list:
    input = list(df[col].fillna('').str.lower())
    return input

def matching_rapid_fuzz(input: list):

    property_search_ls = []

    for x in tqdm.tqdm(input):
        property_search = process.extract(x, property_lookup, limit=1, scorer=fuzz.WRatio)
        # property_type = process.extract(x, property_category_4, limit=1)
        # print(f"input: {x}")
        property_search_ls.append(property_search)

    return property_search_ls
 
def mapping_property(property_search: list):        
    
    property_category_final = []
    property_sector_final = []
    property_type_final = []
    record_type_final = []

    for x in tqdm.tqdm(property_search):
        #property_search = process.extract(x, property_lookup, limit=1, scorer=fuzz.WRatio)
        # property_type = process.extract(x, property_category_4, limit=1)
        # print(f"input: {x}")
        # print(f"found: {property_type[0][0]}")
        # print(f"score: {property_type[0][1]}")
        # print("*" *30)
        property_category_ls = []
        property_sector_ls = []
        property_type_ls = []
        record_type_ls = []
        if x[0][1] < 80:
            property_category_final.append([])
            property_sector_final.append([])
            property_type_final.append([])
            record_type_final.append([])
        else:
            property_category = property_category_4[(x[0][2])]
            property_sector = property_sector_3[(x[0][2])[:6]]
            property_type = property_type_2[(x[0][2])[:4]]
            record_type = record_type_1[(x[0][2])[:2]]

            property_category_ls.append(property_category)
            property_sector_ls.append(property_sector)
            property_type_ls.append(property_type)
            record_type_ls.append(record_type)

            property_category_final.append(property_category_ls)
            property_sector_final.append(property_sector_ls)
            property_type_final.append(property_type_ls)
            record_type_final.append(record_type_ls)
            #print(record_type_ls)

    return property_category_final, property_sector_final, property_type_final, record_type_final           

def list_2_pd(property_category_final: list,
            property_sector_final: list,
            property_type_final: list,
            record_type_final: list):

    final_dict = {
                'property_category': [val for sublist in property_category_final for val in (sublist or [''])],
                'property_sector': [val for sublist in property_sector_final for val in (sublist or [''])],
                'property_type': [val for sublist in property_type_final for val in (sublist or [''])],
                'record_type': [val for sublist in record_type_final for val in (sublist or [''])]}  

    df = pd.DataFrame(final_dict)  
    df = df.apply(lambda x: x.astype(str).str.title())
    
    return df 


def get_property_type_pd(df: pd.DataFrame, col: str):
    print("dataframe to list")
    df_ls = df_to_list(df, col)

    print("macthing property")
    property_search = matching_rapid_fuzz(df_ls)

    print("get property type")
    property_category_final, property_sector_final, property_type_final, record_type_final = mapping_property(property_search)
    
    print("list to dataframe")
    output = list_2_pd(property_category_final, 
                        property_sector_final,
                        property_type_final,
                        record_type_final)
    output.replace({'': None}, inplace=True)

    print("merge input and output dataframe")
    final_df = df.merge(output, left_index=True, right_index=True)

    return final_df


### matching with Rapid Fuzz    

def initialize_matcher(n_jobs=1, score_cutoff = 0.8):

    matcher = RapidFuzz(n_jobs=n_jobs, score_cutoff= score_cutoff) # int:	Nr of parallel processes, use -1 to use all cores
    model = PolyFuzz(matcher)

    return model

def to_input(df: pd.DataFrame, col: str, property_lookup: dict):
    
    from_list = df[col].fillna('').str.lower().to_list() 
    to_list = list(property_lookup.values())

    return from_list, to_list


def matching(model: PolyFuzz, from_list: list, to_list: list):

    print('matching')
    model.match(from_list, to_list)   

    print('get mathces')
    lookup = model.get_matches()

    return lookup

def lookup_property(lookup: DataFrame):

    property_category_final = []
    property_sector_final = []
    property_type_final = []
    record_type_final = []

    for x in lookup['To']:
        property_category_ls = []
        property_sector_ls = []
        property_type_ls = []
        record_type_ls = []
        if x == None:
            property_category_final.append([])
            property_sector_final.append([])
            property_type_final.append([])
            record_type_final.append([])
        else: 
            property_lookup_key = ([k for k, v in property_lookup.items() if v == x])
            for key in property_lookup_key:
                property_category = property_category_4[(key)]
                property_sector = property_sector_3[(key)[:6]]
                property_type = property_type_2[(key)[:4]]
                record_type = record_type_1[(key)[:2]]

                property_category_ls.append(property_category)
                property_sector_ls.append(property_sector)
                property_type_ls.append(property_type)
                record_type_ls.append(record_type)           

                property_category_final.append(property_category_ls)
                property_sector_final.append(property_sector_ls)
                property_type_final.append(property_type_ls)
                record_type_final.append(record_type_ls) 

    return property_category_final, property_sector_final, property_type_final, record_type_final                   

def list_to_pd(property_category_final: list,
                property_sector_final: list,
                property_type_final: list,
                record_type_final: list):

    final_dict = {
                'property_category': [val for sublist in property_category_final for val in (sublist or [''])],
                'property_sector': [val for sublist in property_sector_final for val in (sublist or [''])],
                'property_type': [val for sublist in property_type_final for val in (sublist or [''])],
                'record_type': [val for sublist in record_type_final for val in (sublist or [''])]}

    df = pd.DataFrame(final_dict) 
    df = df.apply(lambda x: x.astype(str).str.title())    

    return df

def get_property_type_pd_rapid_fuzz(df: DataFrame,
                                    col: str,
                                    n_jobs=1, 
                                    score_cutoff=0.8
                                    ):
    
    print("initialize matching model")
    model = initialize_matcher(n_jobs, score_cutoff)
    
    print("prepare to lists")
    from_list, to_list = to_input(df, col, property_lookup)

    print("looking for similar string")
    lookup_result = matching(model, from_list, to_list)
    
    print("map properties")
    property_category_final, property_sector_final, property_type_final, record_type_final = lookup_property(lookup_result)


    print("list to dataframe")
    output = list_to_pd(property_category_final, 
                        property_sector_final,
                        property_type_final,
                        record_type_final)
    output.replace({'': None}, inplace=True)

    print("merge input and output dataframe")
    final_df = df.merge(output, left_index=True, right_index=True)

    return final_df    
