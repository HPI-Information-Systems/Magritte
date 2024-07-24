"""
This script is creating the file_dicts.jsonl file which contains the following information for each csv file:
    - filename
    - header
    - content
    - target
    - label
    - dataset
    - key
    - table_NE
    - most_common
    - coltypes_string
    - related_table
    - distances
    - subtable
    - related_cols
    - sub_related_cols
Furthermore, it creates the train, val and test splits for the experiments.

"""
import random
import csv
import json
import os
import pandas as pd 
import numpy as np
import pdb
import argparse
from sklearn.model_selection import StratifiedKFold
import spacy
import os
import numpy as np
from tqdm import tqdm
from pqdm.processes import pqdm
import itertools
from Levenshtein import distance as edit_distance
import sys
sys.path.append(".")

# N_JOBS = 1
spacy.require_gpu()
nlp = spacy.load('en_core_web_trf')

SEED = 42
np.random.seed(SEED)
random.seed(SEED)



ALL_DATASETS = ["deex","cius","govuk","mendeley","saus", "troy"]

TYPE_VOCAB = {'CARDINAL':'A', 'DATE1':'B','DATE2':'C', 'DATE3':'D', 'DATE4':'E', 'DATE5':'F', 'EVENT':'G', 'FAC':'H', 'GPE':'I', 'LANGUAGE':'J', 'LAW':'K', 'LOC':'L', 'MONEY':'M', 'NORP':'N', 'ORDINAL':'O', 'ORG':'P', 'PERCENT':'Q', 'PERSON1':'R','PERSON2':'S', 'PRODUCT':'T', 'QUANTITY':'U', 'TIME':'V', 'WORK_OF_ART':'W', 'EMPTY':'X'}


def pre_process_single_file(filename, csv_lines, header_indices, target_col, label):
    content = []
    header_lines = []
    num_col = 0
    # Extracting content
    for i, line in enumerate(csv_lines):
        if i in header_indices:
            header_lines.append(line)
            num_col = max(num_col,len(line))
            for idx,_ in enumerate(line):
                if line[idx] == target_col:
                    line[idx] = ""
        else:
            content.append(line)

    header = ["".join([h[i] for h in header_lines]) for i in range(num_col)]

    return {
        "filename": filename,
        "header": header,
        "content": content,
        "target": target_col,
        "label": label,
        "dataset": dataset,
        "key": dataset+'@@@@@@@'+filename,
        # 'table_NE': col_ne,
        # 'most_common': most_common_coltypes

    }


def refine_entities(content, docs):
    col_ne = [[ent.label_ for ent in d.ents] for d in docs]
    most_common_coltypes = [max(set(c), key=c.count) if c else 'EMPTY' for c in col_ne]
    events_dict = {'Prix','Olympics','Championships', 'Open', 'Challenger', 'Trophy', 'Tournament'}

    for i, data_type in enumerate(most_common_coltypes):
        if data_type == 'EMPTY':
            continue
        candidate = next(s for s in content[:,i] if s) # the first nonempty string in the column
        coltype = data_type
        if any(word in events_dict for word in candidate.split()):
            coltype = 'EVENT'
        elif data_type == 'PERSON':
            coltype = 'PERSON1' if "." in candidate else 'PERSON2'
        elif data_type == 'QUANTITY':
            coltype = 'QUANTITY' if any(char.isdigit() for char in candidate) else 'WORK_OF_ART'
        elif data_type == 'DATE':
            if candidate.isalpha(): #Contains letters but no digits or symbols
                coltype = 'WORK_OF_ART'
            elif candidate.isdigit(): #YYYY
                coltype = 'DATE1'
            elif candidate.isalnum(): #Contains letters and digits but no symbols
                coltype = 'DATE2'                   
            else: #Contains letters, digits and symbols
                splitted = candidate.split('-')
                if len(splitted) == 3: #YYYY-MM-DD
                    coltype = 'DATE3'
                elif len(splitted) == 2: #MM-DD
                    coltype = 'DATE4'
                else:
                    coltype = 'DATE5'
        
        most_common_coltypes[i] = coltype

    typestring = ''.join([TYPE_VOCAB[col] for col in most_common_coltypes])
    return col_ne, most_common_coltypes, typestring

def extract_entities(file_dicts, n_jobs, batch_size):

    all_contents = [np.asarray(f['content']) for f in file_dicts.values()]
    colstrings = [[' ; '.join(content[:,i]) for i in range(content.shape[1])] for content in all_contents]

    indices = np.cumsum([len(c) for c in colstrings])
    args = np.concatenate(colstrings)
    docs = []
    for d in tqdm(nlp.pipe(map(str,args), batch_size = batch_size), total=len(args)):
        docs.append(d)
    docs = np.asarray(docs, dtype=object)

    print("Refining entities...")
    docsplit = np.split(docs, indices[:-1])
    args = [(all_contents[i],docsplit[i]) for i in range(len(docsplit))]
    results = [refine_entities(*arg) for arg in args]

    pattern_dict = {}
    for idx,f_id in enumerate(file_dicts):
        fdict = file_dicts[f_id]
        col_ne, most_common_coltypes, typestring = results[idx]
        fdict['table_NE'] = col_ne
        fdict['most_common'] = most_common_coltypes
        fdict['coltypes_string'] = typestring
        pattern_dict[typestring] = pattern_dict.get(typestring,[]) + [f_id]

    for key,file_dict in file_dicts.items():
        file_dicts[key]['related_table'] = pattern_dict[file_dict['coltypes_string']]

    return file_dicts

def jaccard(list1, list2):
    intersection = len(list(set(list1).intersection(set(list2))))
    union = (len(list1)+len(list2)) - intersection
    if union == 0:
        return 0
    return float(intersection)/union


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', default="clean" ,help='Specify the type argument')
    parser.add_argument('--njobs', default=1 ,help='Number of jobs to run in parallel')
    parser.add_argument('--dataset', default="all", help='The dataset to process, or all if all datasets should be processed.')
    parser.add_argument('--batch_size', default=512, help='Batch size for spacy')
    args = parser.parse_args()
    n_jobs = int(args.njobs)
    batch_size = int(args.batch_size)
    type_arg = args.type
    data_dir = f"data/"
    result_dir = f"results/columntype/"
    selected_dataset = args.dataset

    line_annotations_path = f"{data_dir}/row_classification/strudel_annotations.jsonl"
    line_annotations = [json.loads(f) for f in open(line_annotations_path, "r", encoding="utf-8").readlines()]

    if not "clean" in type_arg:
        header_indices = {f["filename"]:[i for i,ann in enumerate(f["line_annotations"]) if ann == "header"] for f in line_annotations}
    else:
        header_indices = {f["filename"]:[0] for f in line_annotations}
    

    file_dicts = {}
    file_dict_path = f"{result_dir}/{type_arg}/overall_dicts_{selected_dataset}.jsonl"
    if os.path.exists(file_dict_path):
        for line in open(file_dict_path, "r", encoding="utf-8").readlines():
            fdict = json.loads(line)
            file_dicts[fdict['key']] = fdict
    else:
        print("Generating dictionaries...")
        for dataset in ALL_DATASETS:
            if selected_dataset != "all" and dataset != selected_dataset:
                continue
            annotations_path = f"{data_dir}/columntype/"+dataset+"_gt.csv"
            annotations_df = pd.read_csv(annotations_path, 
                                        header=None, 
                                        names=["filename", "dataset","target_col","label"])
            filenames = list(annotations_df["filename"].values)
            csv_paths = [f"{data_dir}/columntype/{type_arg}/{dataset}/{filename}" for filename in filenames]
            csv_lines = [list(csv.reader(open(csv_path, "r", encoding="utf-8"))) for csv_path in tqdm(csv_paths)]

            args = [(row["filename"],
                    csv_lines[i], 
                    header_indices[row["filename"]],
                    row["target_col"],
                    row["label"]) for i,row in annotations_df.iterrows()]

            results = pqdm(args, pre_process_single_file, n_jobs=n_jobs, argument_type='args')
            result_dict = {f["dataset"]+'@@@@@@@'+f["filename"]: f for f in results}
            file_dicts.update(result_dict)

        print("Extracting entities...")
        file_dicts = extract_entities(file_dicts, n_jobs=n_jobs, batch_size=batch_size)

        print('Now computing edit distances...')
        edit_distances = {}
        combinations = list(itertools.combinations(file_dicts.keys(), 2))
        args = ((file_dicts[table_1]["coltypes_string"], 
                file_dicts[table_2]["coltypes_string"]) 
                for table_1, table_2 in combinations)
        results = list(pqdm(args, edit_distance, n_jobs=n_jobs, argument_type='args'))
        edit_distances = {t_1_2: results[idx] for idx,t_1_2 in enumerate(combinations)}

        print('Identifying related and subrelated tables...')
        for f2,f2_dict in tqdm(file_dicts.items()):
            distances = {}
            tstr_2 = f2_dict['coltypes_string']
            for f1, f1_dict in file_dicts.items():
                tstr_1 = f1_dict['coltypes_string']
                if f1 == f2:
                    distances['0'] = distances.get('0',[]) + [f1]
                    distances['0-type'] = distances.get('0-type',[]) + [tstr_1]
                else:
                    try:
                        distance = edit_distances[(f1, f2)]
                    except KeyError:
                        distance = edit_distances[(f2, f1)]
                    if distance > (len(tstr_2))**0.5:
                        continue
                    distances[str(distance)] = distances.get(str(distance),[]) + [f1]
                    distances[str(distance)+'-type'] = distances.get(str(distance)+'-type',[]) + [tstr_1]
            
            file_dicts[f2]['distances'] = distances
            file_dicts[f2]['subtable'] = []
            file_dicts[f2]['subtable-type'] = []
            for key in distances.keys():
                if len(key.split("-")) == 2:
                    if key == "0-type":
                        file_dicts[f2]['table-type'] = distances[key][0]
                    continue
                elif key == '0':
                    continue
                else:
                    for index in range(len(distances[key])):
                        file_dicts[f2]['subtable'] = file_dicts[f2]['subtable'] + [distances[key][index]]
                        file_dicts[f2]['subtable-type'] = file_dicts[f2]['subtable-type'] + [distances[key+'-type'][index]]
       
        for fk, f_dict in file_dicts.items():
            target_col = int(f_dict['target'])
            related_cols = []
            for rel_table in f_dict['related_table']:
                arr_file = np.array(file_dicts[rel_table]["content"])
                related_cols.append(list(arr_file[:, target_col]))

            sub_related_cols = []
            sub_tables = []
            for index, table_filename in enumerate(f_dict.get('subtable',[])):
                cur_subtype = f_dict['subtable-type'][index]
                width = len(cur_subtype)

                arr_file = np.array(file_dicts[table_filename]["content"])
                if target_col < width and f_dict['table-type'][target_col] == cur_subtype[target_col]:
                    sub_related_cols.append(list(arr_file[:, target_col]))
                    sub_tables.append(table_filename)

            file_dicts[fk]['related_cols'] = related_cols
            file_dicts[fk]['sub_related_cols'] = sub_related_cols
            file_dicts[fk]['subtable'] = sub_tables     


        print('Filtering related tables using Jaccard...')
        # combinations = list(itertools.combinations(filekeys, 2))

        jac_pairs = []
        for k,f in file_dicts.items():
            jac_pairs.extend([(k, f['related_table'][i]) for i in range(len(f['related_table']))])
            jac_pairs.extend([(k, f['subtable'][i]) for i in range(len(f['subtable']))])
        args = ((np.unique(file_dicts[table_1]["content"]), 
                np.unique(file_dicts[table_2]["content"])) 
                for table_1, table_2 in jac_pairs)

        results = list(pqdm(args, jaccard, n_jobs=n_jobs, argument_type='args', total=len(jac_pairs)))
        jaccards = {t_1_2: results[idx] for idx,t_1_2 in enumerate(jac_pairs)}

        for fk,fdict in file_dicts.items():
            rel_tables = []
            rel_cols = []
            for idx,fk2 in enumerate(fdict['related_table']):
                try:
                    dist = jaccards[(fk,fk2)]
                except KeyError:
                    dist = jaccards[(fk2,fk)]
                if dist > 0.1:
                    rel_tables.append(fk2)
                    rel_cols.append(fdict['related_cols'][idx])
            file_dicts[fk]['related_table'] = rel_tables
            file_dicts[fk]['related_cols'] = rel_cols

            sub_tables = []
            sub_cols = []
            for idx, fk2 in enumerate(fdict['subtable']):
                try:
                    dist = jaccards[(fk,fk2)]
                except KeyError:
                    dist = jaccards[(fk2,fk)]
                if dist > 0.1:
                    sub_tables.append(fk2)
                    sub_cols.append(fdict['sub_related_cols'][idx])
            file_dicts[fk]['subtable'] = sub_tables
            file_dicts[fk]['sub_related_cols'] = sub_cols

        os.makedirs(os.path.dirname(file_dict_path), exist_ok=True)
        with open(file_dict_path, "w") as outfile:
            for fdict in file_dicts.values():
                json.dump(fdict, outfile)
                outfile.write("\n")

    # # Split train and test with stratified sampling
    filekeys = np.array(list(file_dicts.keys()))
    labels = np.array([f_dict['label'] for f_dict in file_dicts.values()])
    sfolder_test = StratifiedKFold(n_splits=10, random_state = 42, shuffle=True)
    trainval_indices, test_indices = next(sfolder_test.split(filekeys, labels))
    
    test_files = filekeys[test_indices]
    test_dicts = [file_dicts[k] for k in test_files]
    test_dicts_path = f"{result_dir}/{type_arg}/folds/{selected_dataset}_test_dicts.jsonl"
    os.makedirs(os.path.dirname(test_dicts_path), exist_ok=True)

    with open(test_dicts_path, "w") as outfile:
        for fdict in test_dicts:
            json.dump(fdict, outfile)
            outfile.write("\n")

    trainval_files = filekeys[trainval_indices]
    trainval_labels = labels[trainval_indices]

    trainval_dicts = [file_dicts[k] for k in trainval_files]
    with open(f"{result_dir}/{type_arg}/folds/{selected_dataset}_train_dicts_full.jsonl", "w") as outfile:
        for fdict in trainval_dicts:
            json.dump(fdict, outfile)
            outfile.write("\n")
    
    if selected_dataset == "all":        
        sfolder_train_val = StratifiedKFold(n_splits=10, random_state = 42, shuffle=True)
        for idx,(train_indices, val_indices) in enumerate(sfolder_train_val.split(trainval_files, trainval_labels)):
            train_files = trainval_files[train_indices]
            val_files = trainval_files[val_indices]

            train_dicts = [file_dicts[k] for k in train_files]
            val_dicts = [file_dicts[k] for k in val_files]

            train_dicts_path = f"{result_dir}/{type_arg}/folds/{selected_dataset}_train_dicts_{idx}.jsonl"
            val_dicts_path = f"{result_dir}/{type_arg}/folds/{selected_dataset}_val_dicts_{idx}.jsonl"

            with open(train_dicts_path, "w") as outfile:
                for fdict in train_dicts:
                    json.dump(fdict, outfile)
                    outfile.write("\n") 
            
            with open(val_dicts_path, "w") as outfile:
                for fdict in val_dicts:
                    json.dump(fdict, outfile)
                    outfile.write("\n")





