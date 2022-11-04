from keras.models import load_model
import numpy as np
import keras
import pandas as pd
import re
import tqdm


# import sys
# sys.path.append('./')
from .functions import import_vocab, get_house_no, get_street_no, get_borey, \
                        clean_borey, clean_house, clean_street, clean_address, \
                        process_txt, pred_to_label, decode_result, asseble_results     

from .fuzzy_verify_package.fuzzy_verify_preds import verify_prediction

word2idx, idx2tag, tag2idx , idx2word = import_vocab('./vocab/')

# load model
from keras_contrib.layers import CRF
from keras_contrib.losses import  crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy

custom_objects = {'CRF': CRF, 'crf_loss':crf_loss, 'crf_viterbi_accuracy':crf_viterbi_accuracy}
# ner_model = load_model('./model/model.h5', custom_objects)

# load NER model
def load_model_ner() -> keras.engine.training.Model:
    """

    Purpose: 

    A simple function to load model.
    
    ----------

    Returns:

    Model variable.

    ----------

    Usage: 
    
    model = load_model_ner()

    """  
    print("load_model_ner: begin", "\n")     
    ner_model = load_model('./model/model.h5', custom_objects)
    print("load_model_ner: done", "\n")
    return ner_model

def first_pipe(df: pd.DataFrame, address_col: str):
    """

    Purpose:

    This function extracts house number, street number, and borey name from address string. 
    It consists of three functions, including get_house_no, get_street_no, and get_borey. 
    After extracting the three components, it will remove the extracted string from the original 
    address and create a new Pandas column called "new_add_1". This new address column will be used 
    as input for NER prediction function, which is "predict_pipe". 
             
    ----------

    Parameter: 

    1. df: Pandas dataframe.
    2. address_col: column name that contains address.

    ----------

    Returns:

    New four columns => "new_add_1", "parsed_borey", "parsed_house_no", "parsed_street_no".

    ----------

    Usage: 
    
    df = (df, "full_address")

    """    
    # Regular expressions for cleaning outputs.    
    cleaning = {re.compile('(?i), St., |con, '): '',
            re.compile('\\(|\\)'): '',
            re.compile('(?i), St\..*?,'): '',
            re.compile('^,*'): '',
            re.compile('^\s*,*'): '',
            re.compile('(?i)st\..*?,'): '',
            re.compile('(?i)\s*home.*?,\s*,\s*'): '',
            re.compile('(?i)Rithy.*?,'): '',
            re.compile('(?i)^.*Blvd.*?,\s?,?'): '',
            re.compile('(?i)^.*N/a.*?,\s?,?'): '',
            re.compile('(?i)^.*road.*?,\s?,?'): '',
            re.compile('(?i)^.*Litterite.*?,\s?,?'): '',
            re.compile('(?i)^.*Concrete.*?,\s?,?|ផ្លូវលំ'): '',
            re.compile('(?i)^.*Land.*?,\s?,?'): '',
            re.compile('(?i)^.*E\d.*?,\s?,?'): '',
            re.compile('(?i)^.*#.*?,\s?,?'): '',
            re.compile('(?i)^.*betong.*?,\s?,?|Str,'): ''}        

    print("first_pipe: begin", "\n")

    # replace null address with "No address"
    print("replace null address with 'No address'")
    df["new_add"] = df[address_col].fillna("No address") 

    # remove whitespace
    print("remove whitespace")
    df["new_add"] = df["new_add"].replace(r'^\s+$', "No address", regex=True)


    # extract three components
    print("extract three components")
    df["house_no"] = get_house_no(df, "new_add", "house_no")
    df["street_no"] = get_street_no(df, "new_add", "street_no")
    df["borey"] = get_borey(df, "new_add", "borey")
    df.fillna("", inplace=True) # replace NaN will an empty string

    # remove the extracted string from address 
    print("remove the extracted string from address")
    df["new_add"] = df.apply(lambda row : row["new_add"].replace(str(row['house_no']), ''), axis=1)
    df["new_add"] = df.apply(lambda row : row['new_add'].replace(str(row['street_no']), ''), axis=1)
    df["new_add"] = df.apply(lambda row : row['new_add'].replace(str(row['borey']), ''), axis=1)

    # extract the three components again from "new_add" because, for example, an address might include redundant
    # house number (#No., E12E0, Street 21, Veal Vong, 7 Makara, Phnom Penh)
    print("extract the three components again from 'new_add'")
    df["house_no_1"] = get_house_no(df, 'new_add', "house_no_1")
    df["street_no_1"] = get_street_no(df, 'new_add', "street_no_1")
    df["borey_1"] = get_borey(df, "new_add", "borey_1")
    df.fillna("", inplace=True) # replace NaN will an empty string

    # remove the extracted string from "new_add" 
    print("remove the extracted string from 'new_add'")
    df["new_add_1"] = df.apply(lambda row : row['new_add'].replace(str(row['house_no_1']), ''), axis=1)
    df["new_add_1"] = df.apply(lambda row : row['new_add_1'].replace(str(row['street_no_1']), ''), axis=1)
    df["new_add_1"] = df.apply(lambda row : row['new_add_1'].replace(str(row['borey_1']), ''), axis=1)
    df["new_add_1"] = df["new_add_1"].replace(cleaning, regex=True)
    df.fillna('', inplace=True) # replace NaN will an empty string

    # coalesce the three components
    print("coalesce the three components")
    df["parsed_house_no"] = np.where(df["house_no"] == df["house_no_1"], df["house_no"], df["house_no"] + " " +  df["house_no_1"])
    df["parsed_street_no"] = np.where(df["street_no"] == df["street_no_1"], df["street_no"], df["street_no"] + " " +  df["street_no_1"])
    df["parsed_borey"] = np.where(df["borey"] == df["borey_1"], df["borey"], df["borey"] + " " +  df["borey_1"])
    df.fillna("", inplace=True) # replace NaN will an empty string

    # clean the extracted components
    print("clean the extracted components")
    df["parsed_borey"] = clean_borey(df, "parsed_borey")   
    df["parsed_house_no"] = clean_house(df, "parsed_house_no")
    df["parsed_street_no"] = clean_street(df, "parsed_street_no")

    # drop leftover columns
    col = ["house_no", "street_no", "borey", "new_add", "house_no_1", "street_no_1", "borey_1"]
    df.drop(col, axis=1, inplace=True)   

    print("first_pipe: done", "\n")

    return df    


# classifier
def predict_pipe(df: pd.DataFrame) -> pd.DataFrame:
    """

    Purpose:

    Parses address into village, commune, district, and province. The parsed tags will be 
    put into Pandas dataframe columns.
    
    Note:

    The parsed tags from this function are not condidered final output, since they need to be verified 
    by verify_prediction function.
             
    ----------

    Parameter: 

    1. df: Pandas dataframe from which address is being parsed.

    ----------

    Returns:

    New five columns => "village", "commune", "district", "province", "new_add_1".

    ----------

    Usage: 
    
    df = (df, "full_address")

    """    
    print("predict_pipe: begin", "\n")

    print("load model")
    ner_model = load_model('./model/model.h5', custom_objects)

    print("convert address to list")
    data_ls = [x for x in df['new_add_1'].values.tolist()]

    print("clean address")
    cleaned_add = clean_address(data_ls)
    
    print("process_txt")
    # padded = [process_txt(x, word2idx, idx2word) for x in tqdm.tqdm(cleaned_add)]
    padded = process_txt(cleaned_add, word2idx=word2idx, idx2word=idx2word)

    print("model predicting")
    preds = ner_model.predict(np.array(padded), verbose=1)

    print("prediction to labels")
    pred2label = pred_to_label(preds)

    print("decode predictions")
    result_decoded = decode_result(pred2label, padded, idx2word, idx2tag)

    print("asseble predictions to dataframe")
    predict_pipe_result = asseble_results(result_decoded, df)  
    
    print("predict_pipe: done", "\n")
    return predict_pipe_result

def address_parser(df: pd.DataFrame, address_col: str):
    """

    Purpose: 
    
    Parses address string into components. 
    It consists of three functions, including 
    first_pipe, predict_pipe, and verify_prediction.
    
    ----------

    Parameter: 

    1. df: Pandas dataframe.
    2. address_col: column name that contains address.
    
    ----------

    Usage: 
    
    df = (df, "full_address")

    """     
    first_pipe_result = first_pipe(df, address_col)
    predict_pipe_result = predict_pipe(first_pipe_result)
    verify_prediction_result = verify_prediction(predict_pipe_result)
    col = ["village", "commune", "district", "province", "new_add_1", "parsed_gazetteer"]
    verify_prediction_result = verify_prediction_result.drop(col, axis=1)  

    return verify_prediction_result    

