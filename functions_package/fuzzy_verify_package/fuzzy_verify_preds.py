import pandas as pd
import tqdm

import warnings
warnings.filterwarnings('ignore')
from polyfuzz import PolyFuzz
from polyfuzz.models import RapidFuzz    

# load model
matcher = RapidFuzz(n_jobs=1, score_cutoff= 0.60) # int:	Nr of parallel processes, use -1 to use all cores
model = PolyFuzz(matcher)
# functions
path_csv = "./functions_package/fuzzy_verify_package/Geo_Cambo_v2.csv"
def load_geo_db():
    # remove unneccessary colomn
    df = pd.read_csv(path_csv, engine= 'python')
    df = df[['pro_name', 'pro_id', 'dis_name', 'dis_id', 'com_com', 'com_id', 'village_name', 'village_id']]
    df.rename(columns = {'com_com':'com_name', 'village_name':'vil_name', 'village_id':'vil_id'}, inplace = True)

    # add leading zero
    df['pro_id'] = df['pro_id'].apply(lambda x: '{0:0>2}'.format(x))
    df['dis_id'] = df['dis_id'].apply(lambda x: '{0:0>4}'.format(x))
    df['com_id'] = df['com_id'].apply(lambda x: '{0:0>6}'.format(x))
    df['vil_id'] = df['vil_id'].apply(lambda x: '{0:0>8}'.format(x))
    df = df.apply(lambda x: x.astype(str).str.lower())
    return df

def remove_white(df2: pd.DataFrame):    
    # Remove white space
    df2['province'] = df2['province'].str.strip()
    df2['district'] = df2['district'].str.strip()
    df2['commune'] = df2['commune'].str.strip()
    df2['village'] = df2['village'].str.strip()

    df_input = df2
    return df_input

# matching address function
def match_addr(df_from: pd.DataFrame, df_to: pd.DataFrame):
    # create empty dictionary to retrive output result
    dict_matched = {'province' : [],
                    'district' : [],
                    'commune' : [],
                    'village' : [],
                    'gazetteer' : []}

    #for i in tqdm.tqdm(range(len(df_from))): 
    for i in tqdm.tqdm(range(len(df_from))): 

        # get each record of input data
        rows = df_from.iloc[i: i + 1]

        # id of input address
        #id = rows['id'].tolist()[0]

        # fuzzy province
        from_list = rows['province'].tolist()
        to_list = df_to['pro_name'].tolist()
        model.match(from_list, to_list)
        pro_matched = model.get_matches()

        # verify province name whether exist or not
        pro_name = ''
        if pro_matched['Similarity'][0] == 0.0 or (pro_matched['From'][0] == 'nan'):
            # dict_matched['id'].append(id)
            dict_matched['province'].append('Na')
            dict_matched['district'].append('Na')
            dict_matched['commune'].append('Na')
            dict_matched['village'].append('Na')
            dict_matched['gazetteer'].append('Na')

            continue
        else:
            pro_name = pro_matched['To'][0]

            # filter to get district in the province which is provided (pro_name)
            df_province = df_to[df_to['pro_name'] == pro_name]

            gazetteer = df_province['pro_id'].values[0]
            
            # fuzzy disctrict
            dis_name = ''
            from_list = rows['district'].tolist()
            to_list = df_province['dis_name'].tolist()
            model.match(from_list, to_list)
            dis_matched = model.get_matches()
                
            # verify disctrict whether exist or not
            if dis_matched['Similarity'][0] == 0.0 or (dis_matched['From'][0] == 'nan'):
                # dict_matched['id'].append(id)
                dict_matched['province'].append(pro_name)
                dict_matched['district'].append('Na')
                dict_matched['commune'].append('Na')
                dict_matched['village'].append('Na')
                dict_matched['gazetteer'].append(gazetteer)

            else:
                dis_name = dis_matched['To'][0]

                # filter to get district in the province which is provided (dis_name)
                df_district = df_province[df_province['dis_name'] == dis_name]

                gazetteer = df_district['dis_id'].values[0]

                # fuzzy commune
                from_list = rows['commune'].tolist()
                to_list = df_district['com_name'].tolist()
                model.match(from_list, to_list)
                com_matched = model.get_matches()

                # verify commune whether exist or not
                com_name = ''
                if com_matched['Similarity'][0] == 0.0 or (com_matched['From'][0] == 'nan'):
                    # dict_matched['id'].append(id)
                    dict_matched['province'].append(pro_name)
                    dict_matched['district'].append(dis_name)
                    dict_matched['commune'].append('Na')
                    dict_matched['village'].append('Na')
                    dict_matched['gazetteer'].append(gazetteer)
                else:
                    com_name = com_matched['To'][0]

                    # filter to get commune in the district which is provided (com_name)
                    df_commune = df_district[df_district['com_name'] == com_name]

                    gazetteer = df_commune['com_id'].values[0]

                    # fuzzy village
                    from_list = rows['village'].tolist()
                    to_list = df_commune['vil_name'].tolist()
                    model.match(from_list, to_list)
                    vil_matched = model.get_matches()

                    # verify village whether exist or not
                    vil_name = ''
                    if vil_matched['Similarity'][0] == 0.0 or (vil_matched['From'][0] == 'nan'):
                        # dict_matched['id'].append(id)
                        dict_matched['province'].append(pro_name)
                        dict_matched['district'].append(dis_name)
                        dict_matched['commune'].append(com_name)
                        dict_matched['village'].append('Na')
                        dict_matched['gazetteer'].append(gazetteer)
                    else:
                        vil_name = vil_matched['To'][0]

                        # filter to get villageID that corresponding to village name (vil_name)
                        df_village = df_commune[df_commune['vil_name'] == vil_name]

                        gazetteer = df_village['vil_id'].values[0]

                        # dict_matched['id'].append(id)
                        dict_matched['province'].append(pro_name)
                        dict_matched['district'].append(dis_name)
                        dict_matched['commune'].append(com_name)
                        dict_matched['village'].append(vil_name)
                        dict_matched['gazetteer'].append(gazetteer)
                        
    return dict_matched


def dict_to_pd(dict: dict):
    df = pd.DataFrame(dict) 
    return df

def capitalize_df(output: pd.DataFrame):
    output['province'] = output['province'].str.title()
    output['district'] = output['district'].str.title()
    output['commune'] = output['commune'].str.title()
    output['village'] = output['village'].str.title()   
    
    return output


def rename_col(output: pd.DataFrame):
    col = {'province':'parsed_province', 
        'district':'parsed_district', 
        'commune':'parsed_commune', 
        'village':'parsed_village',
        'gazetteer': 'parsed_gazetteer'}
    output.rename(columns=(col), inplace=True)
    return output

def get_all_id(output: pd.DataFrame):

    output['parsed_province_id'] = output["parsed_gazetteer"].str.extract("(..).*")
    output['parsed_district_id'] = output["parsed_gazetteer"].str.extract("(....).*")
    output['parsed_commune_id'] = output["parsed_gazetteer"].str.extract("(......).*")
    output['parsed_village_id'] = output["parsed_gazetteer"].str.extract("(........)")    

    return output 

def verify_prediction(preds: pd.DataFrame) -> pd.DataFrame:
    print("verify_prediction: begin")
    print("load_geo_db")
    df_subset = load_geo_db() 
    print("remove_white_space")
    df_input = remove_white(preds)
    print("match_addr")
    df_result = match_addr(df_input, df_subset)
    print("covnert dictionary to dataframe")
    df = dict_to_pd(df_result)
    print("capitalize output")
    capital_output = capitalize_df(df)
    print("rename columns")
    rename_result = rename_col(capital_output)
    print("get_id")
    print("replace Na with None")
    rename_result.replace({'Na': None}, inplace=True)
    get_all_id_result = get_all_id(rename_result)
    print("merge dataframe")
    preds = preds.merge(get_all_id_result, left_index=True, right_index=True)
    print("verify_prediction: done")
    return preds