![python](https://img.shields.io/badge/python-3.7.13-blue) ![keras](https://img.shields.io/badge/keras-2.2.4-red)
![keras-contrib](https://img.shields.io/badge/keras--contrib-2.0.8-important) ![tensorflow](https://img.shields.io/badge/tensorflow-1.13.1-yellow) <br> 
![cudatoolkit](https://img.shields.io/badge/cudatoolkit-10.0.13-brightgreen) ![cudnn](https://img.shields.io/badge/cudnn-7.6.5.32-success)
![fuzzywuzzy](https://img.shields.io/badge/fuzzywuzzy-0.18.0-ff69b4) <br> ![z1data](https://img.shields.io/badge/Z1 Data-R&D-informational)
![z1data](https://img.shields.io/badge/Z1 Data-Data Engineering-9cf) ![team](https://img.shields.io/badge/contributors-Visal-ff69b4)
![team](https://img.shields.io/badge/contributors-Thet-blueviolet)

### Z1 Address Parser
The aim of this project is to build a model that is able to parse address into respective components without depending entirely on regular expression. 
Still, after training and testing with different data, building a perfect deep learning model that could handle a variety of address patterns is not feasible, 
since local addresses manually inputted by human do not follow consistent format. Therefore, some components or tags, such as house number, street number, and borey 
names, will be extracted from address string using regular expression.

--------------------------------------

### Model Pipeline
![pipeline](https://bitbucket.org/z1-data/address_parsing_ner_model/raw/847fd6cf919ae0ff5aa19215c8f594bbd68f6bfa/model/model_illu/model_flow.png)

- The pipeline of the model consists of three stages:
    1. **First pipe** <br> 
        This stage is responsible for extracting house number, street number, and borey name from address string. 
        It consists of three functions, including get_house_no, get_street_no, and get_borey. 
        After extracting the three components, it will remove the extracted string from the original 
        address and create a new Pandas column called "new_add_1". This new address column will be used 
        as input for NER prediction function, which is "predict_pipe".
    2. **Predict pipe** <br>
        When receiving "new_add_1" column from the First pipe, this stage will parse address string into "village", "commune", "district", "province"
        by performing the following sub-funtions: 
        - clean_address(): executes cleaning address cleaning tasks, such as correcting typos, replacing abbreviations with full words, and removing 
                            unwanted spaces or commas.
        - process_txt(): encodes address string into unique ID of each word in the current vocabulary file.
                        Because the current NER model only accepts an address string that has 28 words, this function also pads the 
                        whole string with "PADword" so that the address string will be lengthened to 28 words.
                        Currently, ID of "PADword" is represented by number 1091.            
        - ner_model.predict(): run prediction task. It is a method of Keras library.
        - pred_to_label(): performs the function of np.argmax() and put the result into a list. It is mainly used to transform immediate prediction results. 
        - decode_result(): decodes predicted tags, which is represented in unique ID of each tag, into string.  
        - asseble_results(): assembles output from decode_result function into Pandas dataframe.

        Note:
        The parsed tags from this function are not condidered final output, since they need to be verified 
        by verify_prediction function.
    
    3. **Verify_prediction pipe**<br>
        This pipeline will verify the classified tags, including "village", "commune", "district", "province", from the previous stage. It mainly uses fuzzy search 
        to find the most similar string in Cambodian administrative level database that contains the four components, as well as lookup for ID number of the 
        verified components. The output of this stage include: 
        - parsed_house_no
        - parsed_street_no
        - parsed_borey
        - parsed_village
        - parsed_commune
        - parsed_district
        - parsed_province
        - parsed_province_id
        - parsed_district_id
        - parsed_commune_id
        - parsed_village_id
        - parsed_gazetteer
        
--------------------------------------

### Model Achitecture
![Model Achitecture](https://bitbucket.org/z1-data/address_parsing_ner_model/raw/847fd6cf919ae0ff5aa19215c8f594bbd68f6bfa/model/model_illu/model_plot.png)

The diagram shows each layer and its parameters used to construct the NER model:

- Embedding: converts tokens to vectors.    
- Bi-directional Long-Short Term Memory (Bi-LSTM) : captures infor backwards and forwards.
- Time Distributed: assembles outputs of each predicted tokens. 
- Conditional Random Field (CRF): improves the prediction output from the previous layers.

--------------------------------------
### How to use
To run the project with NVIDIA GPU, you need to install CUDA and cuDNN drivers from Anaconda distribution channels. You can refer to [Anaconda documentation](https://docs.anaconda.com/anaconda/install/) 
to install Anaconda distribution software. If GPU is not avaiable in your device, CPU will be used automatically.

Follow instructions below to run your first example.

Clone the project:

```bash
$ git clone git@bitbucket.org:z1data/address_parsing_ner_model.git
```

Create a conda environment with Python==3.7.13 and install dependencies:

```bash
$ conda env create --name <env_name> python==3.7.13 --file environment.yml
```

Open [predict.ipynb](https://bitbucket.org/z1-data/address_parsing_ner_model/src/master/predict.ipynb) Jupyter notebook and run all cells to see the result. 
![example](https://bitbucket.org/z1-data/address_parsing_ner_model/raw/847fd6cf919ae0ff5aa19215c8f594bbd68f6bfa/model/model_illu/example.png)