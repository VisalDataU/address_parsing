# from fuzzywuzzy import fuzz
# from fuzzywuzzy import process
from rapidfuzz import fuzz, process
import re
import nltk
nltk.data.path.append("./nltk_data")
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
import tqdm


import json
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
    """

    Purpose: 
    
    - Loads word2idx, idx2tag, tag2idx , idx2word using load_json function.

    ----------

    Parameter: 

    1. path: file path.
    
    ----------

    Returns: 
    
    word2idx, idx2tag, tag2idx , idx2word

    ----------

    Usage: 
    
    word2idx, idx2tag, tag2idx , idx2word = import_vocab('./vocab/')

    """     
    word2idx = load_json(path+'word2idx.json')
    idx2tag = load_json(path+'idx2tag.json')
    tag2idx = load_json(path+'tag2idx.json')
    idx2word = load_json(path+'idx2word.json')

    # convert keys into integer 
    idx2word = {int(k):v for k,v in idx2word.items()}
    idx2tag = {int(k):v for k,v in idx2tag.items()}

    return word2idx, idx2tag, tag2idx , idx2word

# word2idx, idx2tag, tag2idx , idx2word = import_vocab(path)
# path = './vocab/'
# print(path+'word2idx.json')

def clean_address(address: list):
    """

    Purpose: 
    
    - Cleans string of address.

    ----------

    Parameter: 

    1. address: a list/list of lists of address.
    
    ----------

    Returns: 
    
    A list/list of lists of cleaned address.

    ----------

    Usage: 
    
    address_a = ['Prey Kuy, Trapeang Kong, Samraong Tong, Kampong Speu Province']
    clean_address_a = clean_address(address_a)

    or

    address_b = [['Prey Kuy, Trapeang Kong, Samraong Tong, Kampong Speu Province'],
    ['#16, st01E, Phum Thma Da, Sangkat Kantaok, Khan Kamboul, Phnom Penh Capital City']] 
    clean_address_b = clean_address(address_b)

    """
    # Dictionary for common typos patterns          
    typo = {"Meanchey":"Mean Chey",
            "Kamdal":"Kandal",
            "Sihanoukville": "Sihanouk",
            "Chamkarmon": "Chamkar Mon",
            "Village of": "Phum",
            "Village": "",
            "Commune": "",
            "District": "",
            "Province": "",
            "Beoung": "Boeung",
            "Sangkat": "", 
            "Tonle Bassac": "Tonle Basak",
            "Por Sen Chey": "Por Senchey",
            "Ponhea Leu": "Ponhea Lueu",
            "Oulampik": "Olympic",
            "7 Makara": "Prampir Meakkakra",
            "Cambodia": "",
            "Kingdom of": "",
            "Capital": "",
            "Capital City": "",
            "City": "",
            "Ruessei Kaev": "Russey Keo",
            "Kam Bol": "Kamboul"} 

    # Dictionary for common abbreviations
    abbr = {"BKK" : "Boeng Keng Kang",
            "\bTK\b": "Toul Kork",
            " Str ": " Street ",
            " st ":" Street ",}                    

    # Regular expressions for cleaning
    cleaning = {re.compile('\\-+|\\:+|\\$+'): '',
                re.compile('\\b(\\d+)(\\s+\\1\\b)+'): ', ',
                re.compile('\\(|\\)'): '',
                re.compile(',+\s+,+'): ',',
                re.compile('\s+,\s+'): ', ',
                re.compile('^,|,$'): '',
                re.compile('^\.|\.$'): '',
                re.compile('^&|&$'): '',
                re.compile('\s+'): ' ',
                re.compile('^\s+|\s+$'): '',
                re.compile('`'): '\'',
                re.compile('\u200b'): ''}    

    # "|" works as an "Or" condition in regular expression. It will enable Regular expression to iterate over each pattern.
    pattern = '|'.join(sorted(re.escape(k) for k in typo)) # iterator for typo patterns   
    pattern_1 = '|'.join(sorted(re.escape(k) for k in abbr)) # iterator for abbr patterns

    # Function for handling multiple replacements 
    def multiple_replace(myDict, text):
        for rx,repl in myDict.items():
            text = rx.sub(repl, text)
        return text

    # clean typo
    address1 = [re.sub(pattern, lambda m: typo.get(m.group(0)), x, flags=re.IGNORECASE) for x in address]
    
    # clean abbr 
    address2 = [re.sub(pattern_1, lambda m: abbr.get(m.group(0)), x, flags=re.IGNORECASE) for x in address1]
    
    # clean leftover space and comma
    address3 = [multiple_replace(cleaning, x) for x in tqdm.tqdm(address2)]
    return address3


def process_txt(doc: list, word2idx: dict, idx2word: dict):    
    """

    Purpose: 
    
    - This function encodes address string into unique ID of each word in the current vocabulary file.
        Because NER model only accepts an address string that has 28 words, this function also pads the 
        whole string with "PADword" so that the address string will be lengthened to 28 words.
        Currently, ID of "PADword" is represented by number 1091. 
    
    Example:   
        input string: 
            [[' Ampil Rung Village, Tumnob Thum Commune, Ponhea Leu District, Kamdal Province']]

        encoded string: 
            [[[2588, 2408, 2904, 3742, 2772, 2904, 1717, 640, 2904, 3212, 
            1091, 1091, 1091, 1091, 1091, 1091, 1091, 1091, 1091, 1091, 1091, 1091, 1091, 1091, 1091, 1091, 1091, 1091]]]   
    ----------

    Parameter: 

    1. doc: a list/list of lists of address.
    2. word2idx: a dictionary used to encode words into number.
    3. idx2word: a dictionary used to decode number into words. 
    
    ----------

    Returns: 
    
    A list/list of lists of encoded address string.

    ----------

    Usage: 
    
    raw_address = [[' Ampil Rung Village, Tumnob Thum Commune, Ponhea Leu District, Kamdal Province'],
                [' Chey Otdam Village, Phsar Daek Commune, Ponhea Leu District, Kamdal Province']]

    encoded_address = process_txt(raw_address)       

    """    
    doc = [x.lower() for x in doc]
    # doc = map(lambda x: x.lower(), doc)
    token1 = [word_tokenize(x) for x in doc] 
    
    pad_tokens1 = []

    # print('input: ', token1, '\n') 
    for token_i in tqdm.tqdm(token1):
        out_i = []
        for t in token_i:
        # print(token_i)
            # print(t)
            try:
                out_i.append(word2idx[t])           
            except KeyError:
                # find misspelled words                    
                pro = process.extract(t, idx2word, limit=1, scorer=fuzz.ratio)
                # pro = process.extract(t, idx2word, limit=1)
                if pro[0][1] < 80:
                    pass
                else:
                    out_i.append(int(pro[0][2]))
                    # view misspelled words
                    # print('Typo: '+t, ', correction: '+pro[0][0], ', score: '+str(pro[0][1]), ', id: '+pro[0][2])
        # print(out_i)
            global decoded
            decoded = [] # for storing address that has leftover commas after their removed precedented string
            for o in out_i:
                decoded.append(idx2word[o])
        # print(decoded)           
        # # decoded_final = []                
        # # decoded_final.append(decoded)  
        # # print('decoded_final: ', out_i)      
        # # print('input: '+str(doc1))
        # # print(out_i)

        # clean leftover commas
        # print('decoded: ', decoded)
        decoded_str = ' '.join(decoded)
        cleaned_str = re.sub(',+\s+,+', ',', decoded_str)
        cleaned_str = re.sub('^,|,$', '', cleaned_str)
        cleaned_str = cleaned_str.strip()
        
        # resemble address after cleaning
        clean_str_final = []
        clean_str_final.append(cleaned_str)
        # print(clean_str_final)
        

        # tokenize cleaned address
        clean_str_final_tk = [word_tokenize(x) for x in clean_str_final]
        # print(clean_str_final_tk)

        # add PADword
        pad_clean_str_final_tk = []
        for seq in clean_str_final_tk:
            new_seq = []
            for i in range(28):
                try:
                    new_seq.append(seq[i])
                except:
                    new_seq.append("PADword")
            pad_clean_str_final_tk.append(new_seq) 
        # print(pad_clean_str_final_tk)            

        # encode cleaned address        
        for cl in  pad_clean_str_final_tk:
            w_cl = []
            for w_i in cl:
                try:
                    w_cl.append(word2idx[w_i])
                except KeyError:
                    w_cl.append(word2idx[','])
        pad_tokens1.append(w_cl)
    #print(pad_tokens1)
        #print("*"*20)
    return pad_tokens1        


def pred_to_label(preds: list):
    """

    Purpose: 
    
    - It performs the function of np.argmax() and put the result into a list.
        It is mainly used to transform immediate prediction results.
    
    Example:   
        input: 
            [array([[[1., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 1., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 1., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 1., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 1., 0.],
                    [0., 0., 0., 0., 0., 1., 0., 0., 0.],
                    [0., 0., 1., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 1., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 1., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0., 1.],
                    [0., 0., 0., 0., 0., 1., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 1., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 1., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 1., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 1., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 1., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 1., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 1., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 1., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 1., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 1., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 1., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 1., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 1., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 1., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 1., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 1., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 1., 0., 0., 0.]]]

        output: 
                    [array([[0, 1, 5, 6, 7, 5, 2, 4, 5, 8, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]])]
    ----------

    Parameter: 

    1. pred2label: a list/list of lists of predicted tags. It is the 
                    result from pred_to_label function.
    2. padded: a list/list of lists of encoded address string, which is 
                the results from process_txt function.
    3. idx2word: a dictionary used to decode number into words. 
    4. idx2tag: a dictionary used to decode number into tags.
    
    ----------

    Returns: 
    
    A list/list of lists of dedcoded tags.

    ----------

    Usage: 
    
    predictions = [array([[0, 1, 5, 6, 7, 5, 2, 4, 5, 8, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                            5, 5, 5, 5, 5, 5]])]

    result_decoded = decode_result(predictions, padded, idx2word, idx2tag)      

    """      
    pred2label = [np.argmax(x, axis=-1) for x in preds]
    return pred2label

def decode_result(pred2label: list, padded: list, idx2word: dict, idx2tag: dict):     
    """

    Purpose: 
    
    - It decodes predicted tags, which is represented in unique ID of each tag, 
        into string. 
    
    Example:   
        input: 
            [array([[0, 1, 5, 6, 7, 5, 2, 4, 5, 8, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]])]

        decoded tags: 
            [[['ampil', 'B-village_phum'],
            ['rung', 'I-village_phum'],
            [',', 'O'],
            ['tumnob', 'B-commune_khum'],
            ['thum', 'I-commune_khum'],
            [',', 'O'],
            ['ponhea', 'B-district_srok'],
            ['lueu', 'I-district_srok'],
            [',', 'O'],
            ['kandal', 'B-city_province']]]

    ----------

    Parameter: 

    1. pred2label: a list/list of lists of predicted tags. It is the 
                    result from pred_to_label function.
    2. padded: a list/list of lists of encoded address string, which is 
                the results from process_txt function.
    3. idx2word: a dictionary used to decode number into words. 
    4. idx2tag: a dictionary used to decode number into tags.
    
    ----------

    Returns: 
    
    A list/list of lists of dedcoded tags.

    ----------

    Usage: 
    
    predictions = [array([[0, 1, 5, 6, 7, 5, 2, 4, 5, 8, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                            5, 5, 5, 5, 5, 5]])]

    result_decoded = decode_result(predictions, padded, idx2word, idx2tag)      

    """       
    results = []
    for label,word in zip(pred2label, padded): # loop through the first outer list
        #print(label, word)
        tempo_result = []
        for label_i, word_i in zip(label, word): # loop through the second outer list
            
             tk_result = []
             if idx2word[word_i] == "PADword": # skip PADword
                pass
             else:
                tk_result.append(idx2word[word_i].capitalize())
                     # append words and tags (['ampil', 'B-village_phum'])
                tk_result.append(idx2tag[label_i])
                # print(tk_result)
                if tk_result == ['No', 'O']: # skip an empty list
                    pass
                else:                                
                    tempo_result.append(tk_result) # assemble outputs per address 
                
        # assemble tag lists of each address into a main list
        results.append(tempo_result)
    # print(results)
    return results       

def asseble_results(results: list, df: pd.DataFrame):   
    """

    Purpose: 
    
    - Assembles output from decode_result function into Pandas dataframe.

    ----------

    Parameter: 

    1. results: a list/list of lists of predicted tags. It is the 
                    result from decode_result function.
    2. df: a dataframe from which addresses are being parsed.                    
    
    ----------

    Returns: 
    
    The input dataframe with four new columns such as df['village'], df['commune'], df['district'], df['province']  

    ----------

    Usage: 

    predict_pipe_result = asseble_results(result_decoded, df)       

    """     
    B_vill = []
    B_comm = []
    B_dist = []
    B_city = []
    for x in range(len(df)):
        vill = []
        comm = []
        dist = []
        city = []
        
        # Concatenate the beginning and the inside of tags
        # Ex: Phnom (B-city_province), Penh (I-city_province) => Phnom Penh
        for token in results[x]:                
            if token[1] == 'B-village_phum':
                vill.append(token[0])    
            if token[1] == 'I-village_phum': 
                vill.append(token[0]) 
            if token[1] == 'B-commune_khum':
                comm.append(token[0])    
            if token[1] == 'I-commune_khum': 
                comm.append(token[0])             
            if token[1] == 'B-district_srok':
                dist.append(token[0])    
            if token[1] == 'I-district_srok': 
                dist.append(token[0])     
            if token[1] == 'B-city_province':
                city.append(token[0])    
            if token[1] == 'I-city_province': 
                city.append(token[0])

        # Assemble results of each tag                         
        B_vill.append([' '.join(map(str, vill))])                           
        B_comm.append([' '.join(map(str, comm))])
        B_dist.append([' '.join(map(str, dist))])
        B_city.append([' '.join(map(str, city))])

    # Put results of each entity into dataframe
    df['village'] = [val for sublist in B_vill for val in (sublist or [''])]
    df['commune'] = [val for sublist in B_comm for val in (sublist or [''])]
    df['district'] = [val for sublist in B_dist for val in (sublist or [''])]
    df['province'] = [val for sublist in B_city for val in (sublist or [''])]  

    return df


def view_pred(pad_tokens1, model, idx2word, idx2tag):
    # pad_tokens1 = [(i) for i in x for x in pad_tokens]
    i = np.random.randint(0, len(pad_tokens1))
    print("This is sentence:",i)
    p1 = model.predict(np.array(pad_tokens1))[i]
    p = np.argmax(p1, axis=-1)

    print("{:15}{:5}\t".format("Word", "Pred"))
    print("-" *30)
    for w, pred in zip(pad_tokens1[i], p):
        if idx2word[w] != 'PADword':
            print("{:15}{:15}\t".format(idx2word[w].capitalize(), idx2tag[pred]))


def get_house_no(df: pd.DataFrame, address_col: str, house_no="house_no"):
    """

    Purpose: 
    
    Extracts house number.

    ----------

    Parameter: 

    1. df = a dataframe from which addresses are being parsed. 
    2. address_col = name of the address column
    
    ----------

    Usage: 
    
    df['house_no'] = get_house_no(df, 'address_col') 

    """

    # patterns to search 
    house_regex = '((?i)^no.\d*(?:[-\s]?\w+)(?=,)|^#.*?(?=,)|(?i)house.*?(?=,)|(?i)E\d.*?(?=,))'
    
    # extract house number
    df[house_no] = df[address_col].str.extract(house_regex)

    return df[house_no]

def clean_house(df: pd.DataFrame, col: str):
    """

    Purpose: 
    
    Cleans house number which are extracted from get_house_no functions.

    ----------

    Parameter: 

    1. df = a dataframe from which addresses are being parsed. 
    2. col = name of the column that contains house number.
    
    ----------

    Usage: 
    
    df["parsed_house_no"] = clean_house(df, "parsed_house_no")

    """
    # Regular expressions for cleaning
    cleaning = {re.compile("^\s*House\s*$"): "",
                re.compile("^\s*#\s*$"): "",
                re.compile("(?i)^\s*#*NA#*\s*$"): "",
                re.compile("(?i)#*N/a"): ""}

    df[col] = df[col].replace(cleaning, regex=True)

    return df[col]


def get_street_no(df: pd.DataFrame, address_col: str, street_no="street_no"):
    """

    Purpose: 
    
    Extracts street number/name.
    
    ----------

    Parameter: 

    1. df = a dataframe from which addresses are being parsed.
    2. address_col = name of the address column.
    
    ----------

    Usage: 
    
    df = get_street_no(df, 'address_col') 

    """  
    # patterns to search 
    street_regex = '((?i)[^,]*blvd(?=,)|(?i)st\..*st\..*?(?=,)|(?i)st\s.*?(?=,)|^[1-9]\d*(?:[-\s]?[a-zA-Z0-9]+)|(?i)street.*?(?=,))'
    
    # extract street number
    df[street_no] = df[address_col].str.extract(street_regex)  

    return df[street_no] 

def clean_street(df: pd.DataFrame, col: str):
    """

    Purpose: 
    
    Cleans street number which are extracted from get_street_no functions.

    ----------

    Parameter: 

    1. df = a dataframe from which addresses are being parsed. 
    2. col = name of the column that contains street number.
    
    ----------

    Usage: 
    
    df["parsed_street_no"] = clean_street(df, "parsed_street_no")

    """    
    cleaning = {re.compile("^\s*Street\s*$"): ""}

    df[col] = df[col].replace(cleaning, regex=True)

    return df[col]

def get_borey(df: pd.DataFrame, address_col: str, borey="borey"):
    """

    Purpose: 
    
    If available, borey names will extracted from input address.

    ----------

    Parameter: 

    1. df = a dataframe from which addresses are being parsed. 
    2. address_col = name of the address column.
    
    ----------

    Usage: 
    
    df['borey'] = get_house_no(df, 'address_col') 

    """    
    patt = "((?i)borey.*?(?=,))"
    df[borey] = df[address_col].str.extract(patt)
    return df[borey]

def clean_borey(df: pd.DataFrame, col: str):
    """

    Purpose: 
    
    Cleans borey names which are extracted from get_house_no functions.

    ----------

    Parameter: 

    1. df = a dataframe from which addresses are being parsed. 
    2. col = name of the column that contains house number.
    
    ----------

    Usage: 
    
    df["parsed_borey"] = clean_house(df, "parsed_borey")

    """    
    cleaning = {re.compile('\)'): "",
                re.compile('(?i)borey\s*$'): ""}

    df[col] = df[col].replace(cleaning, regex=True)

    return df[col]    
