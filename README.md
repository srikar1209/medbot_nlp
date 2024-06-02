
Site Url : [MedBot](https://srikar1209.github.io/medbot_nlp/)

Medical Symptom Analysis and Diagnosis:
This project aims to develop a system for analyzing medical symptoms and providing potential disease diagnoses based on a given set of symptoms. The project utilizes natural language processing techniques, machine learning, and data analysis to process medical datasets and perform symptom-disease mapping.


Features:
Preprocess and clean medical symptom data

Extract symptoms and diseases from the dataset

Perform syntactic and semantic similarity analysis between symptoms

Suggest synonyms for input symptoms

Calculate one-hot vector representations of symptoms

Identify possible diseases based on the provided set of symptoms




Requirements:
Python 3.x

Pandas

NumPy

NLTK (Natural Language Toolkit)

spaCy

WordNet



Installation:

1. Clone the repository:

git clone https://github.com/your-username/medical-symptom-analysis.git

2. Install the required dependencies:

pip install -r requirements.txt

3. Download the necessary NLTK data:

pythonCopy codeimport nltk
nltk.download('punkt')
nltk.download('omw-1.4')


Usage:

1.Ensure that the Medical_dataset directory containing the Training.csv and Testing.csv files is present in the project directory.

2.Run the Python script:

python main.py

3.Follow the prompts to input the symptoms or use the provided functions to perform various operations, such as:

Preprocessing and cleaning symptoms

Calculating syntactic and semantic similarity between symptoms

Suggesting synonyms for input symptoms

Generating one-hot vector representations of symptoms

Identifying possible diseases based on the provided set of symptoms



Contributing:
Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.
```

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import wordnet as wn 
import csv
import json

data={"users":[]}
with open('DATA.json', 'w') as outfile:
    json.dump(data, outfile)


import json
def write_json(new_data, filename='DATA.json'):
    try:
        # Open the file in read mode
        with open(filename, 'r') as file:
            # Load existing data into a dict
            file_data = json.load(file)
    except FileNotFoundError:
        # If the file doesn't exist, create an empty dictionary
        file_data = {"users": []}

    # Append new_data to the "users" key
    file_data["users"].append(new_data)

    # Open the file in write mode
    with open(filename, 'w') as file:
        # Write the updated data back to the file
        json.dump(file_data, file, indent=4)


df_tr=pd.read_csv('Medical_dataset/Training.csv')
df_tr.head()


df_tr.shape

df_tt=pd.read_csv('Medical_dataset/Testing.csv')
df_tt.head()


symp = []
disease = []

for i in range(len(df_tr)):
    symptoms = df_tr.columns[df_tr.iloc[i] == 1].tolist()
    symp.append(symptoms)
    disease.append(df_tr.iloc[i, -1])
symp[0]

disease[50]

```
### PREPROCESSING
```

all_symp_col=list(df_tr.columns[:-1])
def clean_symp(sym):
    return sym.replace('_',' ').replace('.1','').replace('(typhos)','').replace('yellowish','yellow').replace('yellowing','yellow') 
all_symp=[clean_symp(sym) for sym in (all_symp_col)]
import nltk
nltk.download('omw-1.4')

from nltk.corpus import wordnet as wn

first_syns = []
second_syns = []

for sym in all_symp:
    # Checking if the symptom has synsets
    if not wn.synsets(sym):
        first_syns.append(sym)  
    else:
        second_syns.append(sym)

len(first_syns)
len(second_syns)
from spacy.lang.en.stop_words import STOP_WORDS
import spacy
nlp = spacy.load('en_core_web_sm')
#lemmatization
def preprocess(doc):
    nlp_doc = nlp(doc)
    d = []
    for token in nlp_doc:
        # Check if the lowercase version of the token is not a stop word and consists of alphabetic characters only
        if not token.text.lower() in STOP_WORDS and token.text.isalpha():
            # Lemmatize the token to its base form and append the lowercase lemma to the list
            d.append(token.lemma_.lower())
    return ' '.join(d)
preprocess("skin peeling")
#preprocessing all the symptoms
all_symp_pr = []

for sym in all_symp:
    preprocessed_sym = preprocess_sym(sym)
    all_symp_pr.append(preprocessed_sym)
#creating a dictionary where keys are the preprocessed symptoms and values are the corresponding original symptoms
col_dict = dict(zip(all_symp_pr, all_symp_col))
# defining a function that calculates jaccard similarity coefficient between two sets
def jaccard_set(str1, str2):
    list1=str1.split(' ')
    list2=str2.split(' ')
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union
def syntactic_similarity(symp_t, corpus):
    most_sim = []
    poss_sym = []
    
    for symp in corpus:
        d = jaccard_set(symp_t, symp)
         # Storing the symptom and its similarity coefficient
        most_sim.append((symp, d)) 
    
    # Sort by similarity coefficient in descending order
    most_sim.sort(key=lambda x: x[1], reverse=True)
    
    similar_symptoms = []
    for symp, sim in most_sim:
        if DoesExist(symp_t):
            return 1, [symp]  
        if symp != symp_t and sim != 0 and symp not in poss_sym:
            poss_sym.append(symp)  
            similar_symptoms.append((symp, sim))
    
    if len(similar_symptoms):
        return 1, similar_symptoms  
    else:
        return 0, None
import itertools
#Returns all the subsets of the set
def powerset(seq):
    seq_list = list(seq)
    subsets = [[]]
    for item in seq_list:
        subsets.extend([subset + [item] for subset in subsets])
    return subsets
def sort(a):
    # Sorting the list based on the lengths of the strings
    sorted_list = sorted(a, key=len)
    return sorted_list
# find all permutations of a list
def permutations(s):
    permutations = list(itertools.permutations(s))
    return([' '.join(permutation) for permutation in permutations])
def DoesExist(txt):
    # Split the input text into individual words
    txt = txt.split(' ')
    # Generating all possible combinations of words using powerset function
    combinations = [x for x in powerset(txt)]
    
    # Iterate over each combination
    for comb in combinations:
        joined_comb = ' '.join(comb)
        # Check if the joined combination exists in all_symp_pr
        if joined_comb in all_symp_pr:
            # If a matching symptom string is found, return it
            return joined_comb
    
    # If no matching symptom string is found, return False
    return False

    DoesExist('worried')
preprocess('really worried')
syntactic_similarity(preprocess('nervous') ,all_symp_pr)
import re

def check_pattern(inp, dis_list):
    pred_list = []
    pattern = re.compile(inp)

    for item in dis_list:
        # Check if the pattern matches the item
        if pattern.search(item):
            # If a match is found, append the item to the predicted list
            pred_list.append(item)
    # If any predicted matches are found, return them
    if pred_list:
        return 1, pred_list
    else:
        # If no matches are found, return None
        return 0, None
check_pattern('eye',all_symp_pr)
from nltk.wsd import lesk
from nltk.tokenize import word_tokenize

def WSD(word, context):
    tokens = word_tokenize(context)
    sense = lesk(tokens, word)

    return sense
def semanticD(doc1,doc2):
    doc1_p=preprocess(doc1).split(' ')
    doc2_p=preprocess_sym(doc2).split(' ')
    score=0
    for tock1 in doc1_p:
        for tock2 in doc2_p:
            syn1 = WSD(tock1,doc1)
            syn2 = WSD(tock2,doc2)
            #syn1=wn.synset(t)
            if syn1 is not None and syn2 is not None :
                x=syn1.wup_similarity(syn2)
                if x is not None and x>0.1:
                    score+=x
    return score/(len(doc1_p)*len(doc2_p))
semanticD('anxiety','nervous')
syna=wn.synsets('anxiety')
syna[0].definition()
syn2=wn.synsets('fatigue')
syn2[0].definition()
syn2[0].wup_similarity(syna[0])
fatigue_synsets = wn.synsets("fatigue") 
nervous_synsets = wn.synsets("nervous") 
path=[]
wup=[]



for s1 in fatigue_synsets:
    for s2 in nervous_synsets:
        path.append(s1.path_similarity(s2))#path similarity
        wup.append(s1.wup_similarity(s2))#wu-palmer similarity
    
        

pd.DataFrame([path,wup],["path","wup"])
import nltk
nltk.download('punkt')

#Finding the most semantically similar symptom
def semantic_similarity(symp_t, corpus):
    max_sim = 0
    most_sim = None
    
    # Calculating semantic similarity for each symptom in the corpus
    for symp in corpus:
        sim_score = semanticD(symp_t, symp)
        
        # Updating max_sim and most_sim if current similarity score is higher
        if sim_score > max_sim:
            max_sim = sim_score
            most_sim = symp
    
    # Returning the most similar symptom and its similarity score
    return max_sim, most_sim
semantic_similarity('nervous',all_symp_pr)
all_symp_pr.sort()
all_symp_pr
from itertools import chain
from nltk.corpus import wordnet

def suggest_syn(symptom):
    suggested_synonyms = set()
    synonyms = wordnet.synsets(symptom)
    
    # Collecting lemma names from synsets
    lemma_names = set(chain.from_iterable([word.lemma_names() for word in synonyms]))
    # Calculate semantic similarity with symptoms in the corpus for each synonym
    for synonym in lemma_names:
        similarity_score, most_similar_symptom = semantic_similarity(synonym, all_symp_pr)
        # If similarity score is non-zero, add the most similar symptom as a suggested synonym
        if similarity_score != 0:
            suggested_synonyms.add(most_similar_symptom)
    
    return list(suggested_synonyms)
suggest_syn('pain')
#One Hot Encoding
import numpy as np
import pandas as pd

def OHV(client_symptoms, all_symptoms):
    # Initialize an array with zeros
    one_hot_vector = np.zeros(len(all_symptoms))
    
    # Set values to 1 for client symptoms
    for symptom in client_symptoms:
        if symptom in all_symptoms:
            index = all_symptoms.index(symptom)
            one_hot_vector[index] = 1
    
    # Create DataFrame from the one-hot encoded vector
    one_hot_df = pd.DataFrame([one_hot_vector], columns=all_symptoms)
    
    return one_hot_df
def contains(small, big):
    a=True
    for i in small:
        if i not in big:
            a=False
    return a
def possible_diseases(l):
    posible_dis=[]
    for dis in set(disease):
        if contains(l,symVONdisease(df_tr,dis)):
            possible_dis.append(dis)
    return possible_dis
set(disease)
def symVONdisease(df, disease):
    # Filter the DataFrame to select rows where 'prognosis' equals the given disease
    filtered_df = df[df['prognosis'] == disease]
    
    # Get symptom columns where the value is 1
    symptom_columns = filtered_df.columns[filtered_df.eq(1).any()].tolist()
    
    return symptom_columns
symVONdisease(df_tr,'Psoriasis')


```


License:
This project is licensed under the MIT License.


Acknowledgments:

The medical dataset used in this project is sourced from [Source_Name].

This project utilizes the following libraries: Pandas, NumPy, NLTK, spaCy, and WordNet.

