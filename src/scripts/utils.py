from scripts import constants
import constants
import pandas as pd
import openai
from openai import OpenAI
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from prefixspan import PrefixSpan
import Levenshtein  
import numpy as np
import csv
import re
import os


client = OpenAI(api_key=os.environ.get("PERSONAL_HEKA_KEY", "..."))

def get_patient_information(patient_idx):
    """
    Retrieve patient information from a CSV file and process it.

    Args:
        patient_idx (int): Index of the patient in the dataset.

    Returns:
        dict: A dictionary containing the patient's information with formatted values.
    """
    test_df = pd.read_csv(constants.TEST_SET_PATH)
    patient_dict_orig = test_df.iloc[patient_idx].to_dict()
    patient_dict_qt_final = {k: v for k, v in patient_dict_orig.items() if k not in ['label']}
    patient_dict_semi_final = {k: v for k, v in patient_dict_qt_final.items() if v != -1}
    patient_dict_final = {k: round(v, 2) for k, v in patient_dict_semi_final.items()}
    if patient_dict_final['gender'] == 0.0:
        patient_dict_final['gender'] = 'female'
    else:
        patient_dict_final['gender'] = 'male'
    
    for k, v in patient_dict_final.items():
        patient_dict_final[k] = f'{v} {constants.UNITS_DICT[k]}'
    
    for k, v in constants.ABBRS_DICT.items():
        try:
            patient_dict_final[v] = patient_dict_final.pop(k)
        except:
            pass
        
    return patient_dict_final

def get_feature_name(chatgpt_response):
    """
    Identify the feature name mentioned in a response.

    Args:
        chatgpt_response (str): The response text to search for feature names.

    Returns:
        str: The name of the feature if found, otherwise 'unknown feature'.
    """
    for feature in constants.APPROVED_FEATURES:
        if (feature in chatgpt_response) or (feature.capitalize() in chatgpt_response):
            return feature
    return 'unknown feature'

def get_feature_value(feature, patient_idx):
    """
    Retrieve the value of a specific feature for a given patient.

    Args:
        feature (str): The name of the feature.
        patient_idx (int): Index of the patient in the dataset.

    Returns:
        str: The value of the feature if available, otherwise 'results unavailable'.
    """
    patient_info = get_patient_information(patient_idx)
    if (feature in patient_info) | (feature.lower() in patient_info):
        return patient_info[feature]
    else:
        return 'results unavailable'
    
def encode_classes(df, col):
    """
    Encode class labels in a dataframe column using a predefined dictionary.

    Args:
        df (pd.DataFrame): The dataframe containing the data.
        col (str): The name of the column to encode.

    Returns:
        pd.DataFrame: The dataframe with encoded class labels.
    """
    df[col] = df[col].replace(constants.CLASS_DICT)
    return df

def multiclass(actual_class, pred_class, average='macro'):
    """
    Calculate the ROC AUC score for a multiclass classification problem.

    Args:
        actual_class (list): List of actual class labels.
        pred_class (list): List of predicted class labels.
        average (str, optional): The averaging method to use. Defaults to 'macro'.

    Returns:
        float: The average ROC AUC score across all classes.
    """
    unique_class = set(actual_class)
    roc_auc_dict = {}
    for per_class in unique_class:
        other_class = [x for x in unique_class if x != per_class]
        new_actual_class = [0 if x in other_class else 1 for x in actual_class]
        new_pred_class = [0 if x in other_class else 1 for x in pred_class]
        roc_auc = roc_auc_score(new_actual_class, new_pred_class, average=average)
        roc_auc_dict[per_class] = roc_auc
    avg = sum(roc_auc_dict.values()) / len(roc_auc_dict)
    return avg
    
def test(ytest, ypred):
    """
    Calculate accuracy, F1 score, and ROC AUC score for predictions.

    Args:
        ytest (list): List of true class labels.
        ypred (list): List of predicted class labels.

    Returns:
        tuple: Accuracy, F1 score, and ROC AUC score as percentages.
    """
    acc = accuracy_score(ytest, ypred)
    f1 = f1_score(ytest, ypred, average='macro', labels=np.unique(ytest))
    try:
        roc_auc = multiclass(ytest, ypred)
    except:
        roc_auc = None
    return acc*100, f1*100, roc_auc*100

def compute_mean_pathway_length(results_df):
    """
    Calculate the mean length of pathways in the results dataframe.

    Args:
        results_df (pd.DataFrame): Dataframe containing pathway information.

    Returns:
        float: The mean pathway length.
    """
    mean_pathway_length = results_df['pathway_length'].mean()
    return mean_pathway_length

def generate_patient_string(patient_dict):
    """
    Generate a string summarizing a patient's laboratory results and gender.

    Args:
        patient_dict (dict): Dictionary containing patient information.

    Returns:
        str: A formatted string with the patient's results and gender.
    """
    patient_string = 'The patient has the following laboratory results and gender;'
    
    for name, value in patient_dict.items():
        patient_string = f'{patient_string} {name}: {value}, '
    return patient_string[:-2]

def get_patient_llm_string(patient_idx):
    """
    Generate a string of patient information for input to a language model.

    Args:
        patient_idx (int): Index of the patient in the dataset.

    Returns:
        str: A string with the patient's information formatted for language model input.
    """
    pt_dict = get_patient_information(patient_idx)
    pt_string = generate_patient_string(pt_dict)
    return pt_string

def get_diagnosis(patient_idx, llm, prompt):
    """
    Generate a diagnosis for a patient using a language model.

    Args:
        patient_idx (int): Index of the patient in the dataset.
        llm (object): The language model to use.
        prompt (str): The prompt to provide to the language model.

    Returns:
        str: The diagnosis generated by the language model.
    """
    print("Processing patient: ", patient_idx)
    patient_string = get_patient_llm_string(patient_idx)
    prompt = f"{prompt} {'So now consider this particular patient and give me the pathway for the final diagnosis:'}{patient_string}".replace("\n", "")
    response = llm.invoke(prompt, temperature=0, seed=constants.SEED)
    
    # Extract the diagnosis from the LLAMA response
    response_parts = re.split(', |\n', response.lower())
    diagnosis = 'none'  # Default diagnosis if not found
    
    # Search for specific diagnosis terms in the response
    for part in response_parts:
        for term in ['no anemia', 'vitamin b12', 'unspecified anemia', 
                     'anemia of chronic disease', 'iron', 'hemolytic anemia', 
                     'aplastic anemia', 'inconclusive']:
            if term in part:
                 if term == 'iron':
                    diagnosis = 'iron deficiency anemia'
                 elif term == 'vitamin b12':
                    diagnosis = 'vitamin b12/folate deficiency anemia'
                 elif term == 'inconclusive':
                    diagnosis = 'inconclusive diagnosis'
                 else:
                    diagnosis = term  # Update diagnosis to the current term (this ensures the last occurrence is taken)
    
    return diagnosis

def get_diagnosis_cot(patient_idx, llm, prompt):
    """
    Generate a diagnosis for a patient using a language model with chain-of-thought reasoning.

    Args:
        patient_idx (int): Index of the patient in the dataset.
        llm (object): The language model to use.
        prompt (str): The prompt to provide to the language model.

    Returns:
        tuple: The response from the language model and the diagnosis.
    """
    print("Processing patient: ", patient_idx)
    patient_string = get_patient_llm_string(patient_idx)
    prompt = f"{prompt} {'So now consider this particular patient and give me the pathway for the final diagnosis:'}{patient_string}".replace("\n", "")
    response = llm.invoke(prompt, temperature=0, seed=constants.SEED)
    
    # Extract the diagnosis from the LLAMA response
    response_parts = re.split(', |\n', response.lower())
    diagnosis = 'none'  # Default diagnosis if not found
    
    # Search for specific diagnosis terms in the response
    for part in response_parts:
        for term in ['no anemia', 'vitamin b12', 'unspecified anemia', 
                     'anemia of chronic disease', 'iron', 'hemolytic anemia', 
                     'aplastic anemia', 'inconclusive']:
            if term in part:
                 if term == 'iron':
                    diagnosis = 'iron deficiency anemia'
                 elif term == 'vitamin b12':
                    diagnosis = 'vitamin b12/folate deficiency anemia'
                 elif term == 'inconclusive':
                    diagnosis = 'inconclusive diagnosis'
                 else:
                    diagnosis = term  # Update diagnosis to the current term (this ensures the last occurrence is taken)
    
    return response, diagnosis

def get_llm_diagnosis_msg_pass(patient_idx, llm, prompt):
    print("Processing patient: ", patient_idx)
    messages = [HumanMessage(content="Please name the first feature whose results you want.")]
    patient_trajectory = []
    patient_pathway = []
    steps = 0
    done = False
    prompt = ChatPromptTemplate.from_messages([("system", prompt), MessagesPlaceholder(variable_name="messages"),])
    chain = prompt | llm
    chain
    
    while not done:
        steps += 1
        llm_response = chain.invoke({"messages": messages})
        patient_trajectory.append(llm_response)
        patient_pathway.append(llm_response)
        ai_message = AIMessage(content=llm_response)
        messages.append(ai_message)

        if ('anemia' in llm_response.lower()) or ('diagnosis' in llm_response.lower()): # diagnosis action
            messages.append(HumanMessage(content="End of chat"))
            diagnosis = llm_response.lower()
            done = True
        elif len(patient_trajectory) >= 10: # Termination action
            messages.append(HumanMessage(content="End of chat"))
            diagnosis = "inconclusive diagnosis"
            patient_trajectory.append(diagnosis)
            patient_pathway.append(diagnosis)
            done = True
        else:
            feature = get_feature_name(llm_response)
            feature_value = get_feature_value(feature, patient_idx)
            patient_pathway.append(feature_value)
            human_message = HumanMessage(content=feature_value)
            messages.append(human_message)
    return diagnosis, steps, patient_trajectory, patient_pathway, messages


def get_llm_diagnosis_msg_pass_cot(patient_idx, llm, prompt):
    print("Processing patient: ", patient_idx)
    messages = [HumanMessage(content="Please name the first feature whose results you want.")]
    patient_trajectory = []
    patient_pathway = []
    steps = 0
    done = False
    prompt = ChatPromptTemplate.from_messages([("system", prompt), MessagesPlaceholder(variable_name="messages"),])
    chain = prompt | llm
    
    while not done:
        steps += 1
        llm_response = chain.invoke({"messages": messages, "temperature": 0})
        ai_message = AIMessage(content=llm_response)
        messages.append(ai_message)

        if ('diagnosis found' in llm_response.lower()): # diagnosis action
            response_parts = re.split(', |\n', llm_response.lower())
            diagnosis = 'none'  # Default diagnosis if not found
    
            # Search for specific diagnosis terms in the response
            for part in response_parts:
                for term in ['no anemia', 'vitamin b12', 'unspecified anemia', 
                     'anemia of chronic disease', 'iron', 'hemolytic anemia', 
                     'aplastic anemia', 'inconclusive']:
                    if term in part:
                        if term == 'iron':
                            diagnosis = 'iron deficiency anemia'
                        elif term == 'vitamin b12':
                            diagnosis = 'vitamin b12/folate deficiency anemia'
                        elif term == 'inconclusive':
                            diagnosis = 'inconclusive diagnosis'
                        else:
                            diagnosis = term  # Update diagnosis to the current term (this ensures the last occurrence is taken)
            patient_trajectory.append(diagnosis)
            patient_pathway.append(diagnosis)
            messages.append(HumanMessage(content="End of chat"))
            done = True
        elif len(patient_trajectory) >= 10: # Termination action
            diagnosis = "inconclusive diagnosis"
            patient_trajectory.append(diagnosis)
            patient_pathway.append(diagnosis)
            messages.append(HumanMessage(content="End of chat"))
            done = True
        else:
            response_parts = re.split(', |\n', llm_response.lower())
            feature_name = 'none'  # Default diagnosis if not found
            # Search for specific diagnosis terms in the response
            for part in response_parts:
                for term in constants.APPROVED_FEATURES:
                    if term in part:
                        if term == 'mcv':
                            feature_name = 'mean corpuscular volume'
                        elif term == 'tibc':
                            feature_name = 'total iron binding capacity'
                        elif term == 'female':
                            feature_name = 'gender'
                        else:
                            feature_name = term  # Update diagnosis to the current term (this ensures the last occurrence is taken)
            patient_trajectory.append(feature_name)
            patient_pathway.append(feature_name)
            feature = get_feature_name(feature_name)
            feature_value = get_feature_value(feature, patient_idx)
            patient_pathway.append(feature_value)
            human_message = HumanMessage(content=feature_value)
            messages.append(human_message)
    return diagnosis, steps, patient_trajectory, patient_pathway, messages


### For chatgpt

def get_chatgpt_diagnosis(patient_idx, gpt_model, init_prompt):
    print("Processing patient: ", patient_idx)
    test_df = pd.read_csv(constants.TEST_SET_PATH)
    patient_string = get_patient_llm_string(patient_idx)
    messages = [{"role":"system", "content":init_prompt},
                {"role": "user", "content":patient_string}]
    response = client.chat.completions.create(model=gpt_model, messages=messages, temperature=0, 
                                              seed=constants.SEED) 
    diagnosis = response.choices[0].message.content
    return diagnosis.lower()
    
def get_chatgpt_diagnosis_cot(patient_idx, gpt_model, init_prompt):
    print("Processing patient: ", patient_idx)
    test_df = pd.read_csv(constants.TEST_SET_PATH)
    patient_string = get_patient_llm_string(patient_idx)
    messages = [{"role":"system", "content":init_prompt},
                {"role": "user", "content":patient_string}]
    response = client.chat.completions.create(model=gpt_model, messages=messages, temperature=0, 
                                              seed=constants.SEED) 
    diagnosis_response = response.choices[0].message.content
    # Extract the diagnosis from the GPT response
    diagnosis = 'None'  # Default diagnosis if not found
    for term in [
        'No anemia', 'Vitamin B12/Folate deficiency anemia', 'Unspecified anemia', 
        'Anemia of chronic disease', 'Iron deficiency anemia', 'Hemolytic anemia', 
        'Aplastic anemia', 'Inconclusive diagnosis'
    ]:
        if term in diagnosis_response:
            diagnosis = term
            break  # Stop searching once a diagnosis is found
    
    return diagnosis.lower(), diagnosis_response

def get_chatgpt_diagnosis_sequential(patient_idx, gpt_model, init_prompt):
    print("Processing patient: ", patient_idx)
    message = {}
    conversation = [{"role": "system", "content": init_prompt}] 
    patient_trajectory = []
    patient_pathway = []
    steps = 0
    done = False

    while not done:
        steps += 1
        completion = client.chat.completions.create(model=gpt_model, messages=conversation, temperature=0, 
                                                    seed=constants.SEED) 
        chatgpt_response = completion.choices[0].message.content
        patient_trajectory.append(chatgpt_response)
        patient_pathway.append(chatgpt_response)
        message = {"role": "assistant", "content": chatgpt_response}
        conversation.append(message)

        if ("anemia" in chatgpt_response) or ("diagnosis" in chatgpt_response):
            diagnosis = chatgpt_response
            done = True
        elif len(patient_trajectory) >= 10:
            diagnosis = "Inconclusive diagnosis"
            patient_trajectory.append(diagnosis)
            patient_pathway.append(diagnosis)
            done = True
        else:
            feature = get_feature_name(chatgpt_response)
            feature_value = get_feature_value(feature, patient_idx)
            message = {"role": "user", "content": feature_value}
            conversation.append(message)
            patient_pathway.append(feature_value)

    return diagnosis.lower(), steps, patient_trajectory, patient_pathway

    

def get_chatgpt_diagnosis_sequential_cot(patient_idx, llm, prompt):
    print("Processing patient: ", patient_idx)
    conversation = [{"role": "system", "content": prompt}] 
    patient_trajectory = []
    patient_pathway = []
    steps = 0
    done = False
    
    while not done:
        steps += 1
        completion = client.chat.completions.create(model=llm, messages=conversation, temperature=0, seed=constants.SEED) 
        chatgpt_response = completion.choices[0].message.content
        patient_pathway.append(chatgpt_response)
        message = {"role": "assistant", "content": chatgpt_response}
        conversation.append(message)

        if 'diagnosis found' in chatgpt_response.lower():  # Diagnosis action
            response_parts = re.split(', |\n', chatgpt_response.lower())
            diagnosis = 'none'  # Default diagnosis if not found
    
            # Search for specific diagnosis terms in the response
            for part in response_parts:
                for term in ['no anemia', 'vitamin b12', 'unspecified anemia', 'anemia of chronic disease', 'iron', 'hemolytic anemia', 'aplastic anemia', 'inconclusive']:
                    if term in part:
                        if term == 'iron':
                            diagnosis = 'iron deficiency anemia'
                        elif term == 'vitamin b12':
                            diagnosis = 'vitamin b12/folate deficiency anemia'
                        elif term == 'inconclusive':
                            diagnosis = 'inconclusive diagnosis'
                        else:
                            diagnosis = term  # Update diagnosis to the current term (this ensures the last occurrence is taken)
            patient_trajectory.append(diagnosis)
            patient_pathway.append(diagnosis)
            message = {"role": "user", "content": diagnosis}
            done = True
        elif len(patient_trajectory) >= 10:  # Termination action
            diagnosis = "inconclusive diagnosis"
            patient_trajectory.append(diagnosis)
            patient_pathway.append(diagnosis)
            done = True
        else:
            response_parts = re.split(', |\n', chatgpt_response.lower())
            feature_name = 'none'  # Default feature if not found
            # Search for specific feature terms in the response
            for part in response_parts:
                for term in constants.APPROVED_FEATURES:
                    if term in part:
                        if term == 'mcv':
                            feature_name = 'mean corpuscular volume'
                        elif term == 'tibc':
                            feature_name = 'total iron binding capacity'
                        elif term == 'female':
                            feature_name = 'gender'
                        else:
                            feature_name = term  # Update feature to the current term (this ensures the last occurrence is taken)
            patient_trajectory.append(feature_name)
            patient_pathway.append(feature_name)
            feature = get_feature_name(feature_name)
            feature_value = get_feature_value(feature, patient_idx)
            patient_pathway.append(feature_value)
            message = {"role": "user", "content": feature_value}
            conversation.append(message)
    return diagnosis.lower(), steps, patient_trajectory, patient_pathway, message

def get_results_chatgpt(n, gpt_model, init_prompt, save=False, filename="chatgpt_base.csv"):
    test_df = pd.read_csv(constants.TEST_SET_PATH)
    results_df = pd.DataFrame(columns=["y_actual", "y_pred"])
    results_df["y_actual"] = test_df[:n]["label"]
    results_df["y_pred"] = results_df.apply(lambda row: 
                                            get_chatgpt_diagnosis(row.name, gpt_model, init_prompt), axis=1)
    results_df = results_df.replace({"y_actual": constants.CLASS_DICT})
    if save:
        results_df.to_csv(f"{filename}.csv")
    return results_df


def get_results_chatgpt_cot(n, gpt_model, init_prompt, save=False, filename="chatgpt_plain_cot.csv"):
    test_df = pd.read_csv(constants.TEST_SET_PATH)
    results_df = pd.DataFrame(columns=["y_actual", "y_pred"])
    results_complete_df = pd.DataFrame(columns=["y_actual", "y_pred", "pathway"])

    for index, row in test_df.iterrows(): 
        if index >= n:
            break
        diagnosis, diagnosis_response = get_chatgpt_diagnosis_cot(row.name, gpt_model, init_prompt)
        results_df.loc[index, "y_pred"] = diagnosis
        results_complete_df.loc[index, "y_pred"] = diagnosis
        results_df.loc[index, "y_actual"] = row["label"]
        results_complete_df.loc[index, "y_actual"] = row["label"]
        results_complete_df.loc[index, "pathway"] = diagnosis_response

    results_df = results_df.replace({"y_actual": constants.CLASS_DICT})  
    results_complete_df = results_complete_df.replace({"y_actual": constants.CLASS_DICT})   

    if save:
        results_complete_df.to_csv(filename, index=False)

    return results_df


def get_results_chatgpt_sequential(n, gpt_model, init_prompt, save=False, filename="chatgpt_seq.csv"):
    test_df = pd.read_csv(constants.TEST_SET_PATH)
    results_df = pd.DataFrame(columns=["y_actual", "y_pred"])
    results_complete_df = pd.DataFrame(columns=["y_actual", "y_pred", "pathway", "trajectory"])

    for index, row in test_df.iterrows(): 
        if index >= n:
            break
        diagnosis, steps, patient_trajectory, patient_pathway = get_chatgpt_diagnosis_sequential(row.name, gpt_model, init_prompt)
        results_df.loc[index, "y_pred"] = diagnosis
        results_complete_df.loc[index, "y_pred"] = diagnosis
        results_df.loc[index, "y_actual"] = row["label"]
        results_df.loc[index, "pathway_length"] = steps
        results_complete_df.loc[index, "y_actual"] = row["label"]
        results_complete_df.loc[index, "pathway"] = " | ".join(patient_pathway) 
        results_complete_df.loc[index, "trajectory"] = " | ".join(patient_trajectory)  

    results_df = results_df.replace({"y_actual": constants.CLASS_DICT}) 
    results_complete_df = results_complete_df.replace({"y_actual": constants.CLASS_DICT}) 
    if save:
        results_complete_df.to_csv(filename, index=False)

    return results_df
    
def get_results_chatgpt_sequential_cot(n, gpt_model, init_prompt, save=False, filename="chatgpt_seq_cot.csv"):
    test_df = pd.read_csv(constants.TEST_SET_PATH)
    results_df = pd.DataFrame(columns=["y_actual", "y_pred"])
    results_complete_df = pd.DataFrame(columns=["y_actual", "y_pred", "pathway", "message"])
    results_sequential_df = pd.DataFrame(columns=["patient_trajectory"])

    for index, row in test_df.iterrows():  
        if index >= n:
            break
        diagnosis, steps, patient_trajectory, patient_pathway, message = get_chatgpt_diagnosis_sequential_cot(row.name, gpt_model, init_prompt)
        results_df.loc[index, "y_pred"] = diagnosis
        results_complete_df.loc[index, "y_pred"] = diagnosis
        results_df.loc[index, "y_actual"] = row["label"]
        results_df.loc[index, "pathway_length"] = steps
        results_complete_df.loc[index, "y_actual"] = row["label"]
        results_complete_df.loc[index, "pathway"] = " | ".join(patient_pathway) 
        results_complete_df.loc[index, "message"] = message["content"]  

        results_sequential_df.loc[index, "patient_trajectory"] = patient_trajectory 

    results_df = results_df.replace({"y_actual": constants.CLASS_DICT}) 
    results_complete_df = results_complete_df.replace({"y_actual": constants.CLASS_DICT}) 
    if save:
        results_complete_df.to_csv("chatgpt_seq_cot.csv", index=False)
        results_sequential_df.to_csv(filename, index = False)

    return results_df

### For Pathway comparison

def clean_data(data):
    """
    Cleans the input DataFrame by removing specific characters from each cell.

    Args:
    - data (pd.DataFrame): The input DataFrame containing the data to be cleaned.

    Returns:
    - pd.DataFrame: The cleaned DataFrame with specified characters removed from each cell.
    """
    for col in data.columns:
        data[col] = data[col].astype(str).apply(lambda x: x.replace('[', '').replace(']', '').replace("'", '').replace('-', ''))
    return data

def extract_variables(input_vector):
    """
    Extracts and cleans string variables from a comma-separated input vector, converting them to lowercase.

    Args:
    - input_vector (str): A comma-separated string containing variables and numbers.

    Returns:
    - str: A cleaned, comma-separated string of variables in lowercase.
    """
    variables = []
    for item in input_vector.split(','):
        if not item.strip().replace('.', '').isdigit():
            variables.append(item.strip().lower())
    return ', '.join(variables)

def process_csv_file(file_path: str):
    """
    Processes a CSV file by cleaning its data and extracting variables from the 'pathway' column.

    Args:
    - file_path (str): The file path of the CSV file to be processed.

    Returns:
    - None: The function saves the cleaned data back to the original file and prints a confirmation message.
    """
    data = pd.read_csv(file_path)
    
    # Clean the data
    data = clean_data(data)
    
    if 'pathway' in data.columns:
        data['cleaned_pathway'] = data['pathway'].apply(extract_variables)
        
        # Drop the original 'pathway' column and rename 'cleaned_pathway' to 'pathway'
        data = data.drop(columns=['pathway'])
        data = data.rename(columns={'cleaned_pathway': 'pathway'})
    
    data.to_csv(file_path, index=False)
    
    print(f"Cleaned data has been saved to {file_path}")


def mine_frequent_patterns(pathways_df, min_support=10):
    """
    Mines frequent sequential patterns from a DataFrame of pathways.

    Args:
    - pathways_df (pd.DataFrame): DataFrame containing the cleaned pathways.
    - min_support (int): Minimum support count for the frequent patterns.

    Returns:
    - None: The function prints the frequent sequential patterns and their support counts.
    """
    sequences = [tuple(pathway.split(', ')) for pathway in pathways_df['pathway']]
    print(f"Number of sequences: {len(sequences)}")

    ps = PrefixSpan(sequences)
    
    # Mine frequent sequential patterns with minimum support count
    patterns = ps.frequent(min_support)

    print("Frequent Sequential Patterns:")
    for pattern, support in patterns:
        print("Pattern:", pattern, "Support:", support)


def mine_longest_patterns(pathways_df, min_support=5):
    """
    Mines the longest common subsequences from a DataFrame of pathways.

    Args:
    - pathways_df (pd.DataFrame): DataFrame containing the cleaned pathways.
    - min_support (int): Minimum support count for the frequent patterns.

    Returns:
    - None: The function prints the longest common subsequences and their support counts.
    """
    sequences = [tuple(pathway.split(', ')) for pathway in pathways_df['pathway']]
    print(f"Number of sequences: {len(sequences)}")

    ps = PrefixSpan(sequences)

    # Mine frequent sequential patterns with minimum support count
    patterns = ps.frequent(min_support)

    max_length = max(len(pattern[1]) for pattern in patterns)
    longest_patterns = [(pattern[0], pattern[1]) for pattern in patterns if len(pattern[1]) == max_length]

    print("Longest Common Subsequence", max_length, ":")
    for pattern in longest_patterns:
        print("Pattern:", pattern[1], "Support:", pattern[0])

def filter_sequences_by_class(sequences, target_class):
    """
    Filters the sequences that end with the target class.

    Args:
    - sequences (list of tuples): List of sequences.
    - target_class (str): Target class to filter by.

    Returns:
    - filtered_sequences (list of tuples): Sequences that end with the target class.
    """
    return [seq for seq in sequences if seq[-1] == target_class]

def mine_longest_patterns_by_class(pathways_df, min_support=5):
    """
    Mines the longest common subsequences from a DataFrame of pathways, grouped by the ending class.

    Args:
    - pathways_df (pd.DataFrame): DataFrame containing the cleaned pathways.
    - class_dict (dict): Dictionary of class labels.
    - min_support (int): Minimum support count for the frequent patterns.

    Returns:
    - None: The function prints the longest common subsequences and their support counts for each class.
    """
    class_dict = constants.CLASS_DICT
    sequences = [tuple(pathway.split(', ')) for pathway in pathways_df['pathway']]
    print(f"Number of sequences: {len(sequences)}")

    for class_label, class_name in class_dict.items():
        filtered_sequences = filter_sequences_by_class(sequences, class_name)
        print(f"\nClass '{class_name}':")
        print(f"Number of sequences: {len(filtered_sequences)}")

        if len(filtered_sequences) < min_support:
            print(f"Not enough sequences with the class '{class_name}' (minimum required: {min_support}).")
            continue

        ps = PrefixSpan(filtered_sequences)

        # Mine frequent sequential patterns with minimum support count
        patterns = ps.frequent(min_support)

        if not patterns:
            print("No frequent patterns found.")
            continue

        max_length = max(len(pattern[1]) for pattern in patterns)
        longest_patterns = [(pattern[0], pattern[1]) for pattern in patterns if len(pattern[1]) == max_length]

        print("Longest Common Subsequences of Length", max_length, ":")
        for support, pattern in longest_patterns:
            print("Pattern:", pattern, "Support:", support)

def convert_pathways_to_letters(pathways_df, letter_dict):
    def convert_path_to_letters(path, letter_dict):
        """
        Convert pathway strings to alphabetic letters using the provided dictionary.

        Args:
        - path (str): String representing the pathway.
        - letter_dict (dict): Dictionary mapping strings to alphabetic letters.

        Returns:
        - str: String representing the pathway with letters.
        """
        components = path.split(', ')
        converted_components = [letter_dict.get(component, component) for component in components]
        return ''.join(converted_components)

    pathways_df['pathway_letters'] = pathways_df['pathway'].apply(lambda path: convert_path_to_letters(path, letter_dict))
    return pathways_df


def compute_mean_distance_and_max_index_between_dataframes(df1, df2, column_name='pathway_letters'):
    """
    Compute the mean Levenshtein distance between corresponding rows in two dataframes and
    return the index of the row with the maximum distance.

    Args:
        df1 (pd.DataFrame): The first dataframe containing pathway letters.
        df2 (pd.DataFrame): The second dataframe containing pathway letters.
        column_name (str, optional): The column name in both dataframes to compute distances for. Defaults to 'pathway_letters'.

    Returns:
        tuple: A tuple containing:
            - float: The mean Levenshtein distance between corresponding rows of the specified column from the two dataframes.
            - int: The index of the row with the maximum Levenshtein distance.
    """
    # Ensure both dataframes have the same number of rows
    min_len = min(len(df1), len(df2))
    
    # Calculate distances between corresponding rows
    distances = [
        Levenshtein.distance(df1.iloc[i][column_name], df2.iloc[i][column_name])
        for i in range(min_len)
    ]
    
    # Compute mean distance
    mean_distance = sum(distances) / len(distances) if distances else 0
    
    # Find the index of the maximum distance
    max_distance = max(distances, default=0)
    max_index = distances.index(max_distance) if distances else -1
    
    return mean_distance, max_index

def ep_length(df):
    length = df['pathway'].apply(lambda x: len(x.split(',')))
    mean_length = length.mean()
    return mean_length
