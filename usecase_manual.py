"""
Data extraction from "semi-structured" sources.
Steps:
Need to parse the table from the file. I would like to have a csv file with the following columns:
```Date, Daily rainfall (in.), Sampling location  #1, ..., Sampling location #n```
"""
import os
import pandas as pd
from typing import List, Any, Iterable
import pdb

def parse_column(column: pd.Series):
    """
    Infer the type and header (if any) of the column. We are assuming the column has a non-descriptive dtype, i.e. "object"
    Heuristic algorithm:
    1. If the column is completely empty, return "empty".
    2. Delete all the NaN values from the column.
    3. If the column values only have one type according to python inference, return that type.
    3. If the values are string and something else, return the other type (assuming string is for metadata purposes).
    """
    nan_ratio = column.isna().sum() / len(column)
    if nan_ratio == 1.:
        return pd.Series([])
    ret_column = column.dropna()

    header = None

    types = [pd.api.types.infer_dtype([x]) for x in list(ret_column)]
    n_types = len(set(types))
    if n_types == 1:
        typ = types[0]
    elif n_types == 2 and "string" in types:
        typ = [x for x in types if x != "string"][0]
        header = [x for idx,x in enumerate(ret_column) if types[idx] == "string"][0]
    else:
        if types[0] == str:
            pd_guess = column[1:].infer_objects().dtype
            if pd_guess != object:
                header = column[0]
                typ = pd_guess
        else:
            typ = "object"

    # return pd.api.types.pandas_dtype(typ)
    ret_column = pd.Series(ret_column, name=header)
    if typ == ["datetime","date","time","timedelta"]:
        return pd.to_datetime(ret_column, errors="coerce")
    elif typ in ["floating","integer", "mixed-integer","mixed-integer-float","decimal","complex"]:
        return pd.to_numeric(ret_column, errors="coerce")
    else:
        return ret_column


def parse_coltype(column: pd.Series):
    nan_ratio = column.isna().sum() / len(column)
    if nan_ratio > 0.9:
        return "empty"
    
    types = [pd.api.types.infer_dtype([x]) for x in list(ret_column)]
    n_types = len(set(types))
    if n_types == 1:
        typ = types[0]
    elif n_types == 2 and "string" in types:
        typ = [x for x in types if x != "string"][0]
        header = [x for idx,x in enumerate(ret_column) if types[idx] == "string"][0]
    else:
        if types[0] == str:
            pd_guess = column[1:].infer_objects().dtype
            if pd_guess != object:
                header = column[0]
                typ = pd_guess
        else:
            typ = "object"


    ret_column = pd.Series(ret_column, name=header)
    if typ == ["datetime","date","time","timedelta"]:
        return pd.to_datetime(ret_column, errors="coerce")
    elif typ in ["floating","integer", "mixed-integer","mixed-integer-float","decimal","complex"]:
        return pd.to_numeric(ret_column, errors="coerce")
    else:
        return ret_column


records = []
csv_dir = "data/massbay/"
for file in os.listdir(csv_dir):
    print(file)
    df = pd.read_csv(os.path.join(csv_dir, file))
    
    # The first column is sometimes not empty but contains metadata only
    del df[df.columns[0]]
    #Sometimes the last column is not empty but it contains "ghost" data (i.e. it's colored like the backgrund)
    for c in df.columns:
        if df[c].isna().sum() / len(df[c]) > 0.75: 
            del(df[c])
    
    # find the row that contains the string "Rain" in the second column, which is the header
    name_row = df[df[df.columns[1]].str.contains("Rain", na=False)]
    names = name_row.iloc[0].to_list()
    names[0] = "Date"

    # name row index
    name_row_idx = name_row.index[0]
    df = df.iloc[name_row_idx+1:]

    # sometimes the last column is junk. In this case the name is not a string but an int
    df.columns = names
    if type(df.columns[-1]) != str:
        del df[df.columns[-1]]

    # delete empty rows
    empty_rows = []
    for idx, row in df.iterrows():
        if row.isna().sum() / len(row) == 1.:
            empty_rows.append(idx)
    df = df.drop(empty_rows)

    df.reset_index(drop=True, inplace=True)
    """
    Consolidate all individual tables into a single table. This table will have the following schema:

    ```Date, Daily Rainfall, Sampling location #1 (...)```
    3. Ensure that all data types for the columns are consistent.
    4. Ensure that all records have semantically consistent values. In our case:
    - Date is a valid date
    - Beach is unambiguous 
    - Numeric columns have the same units
    - Missing values are handled consistently (e.g. NaN, 0, -1, etc.)
    """

    # extract each record as a dictionary
    for idx, row in df.iterrows():
        rowdict = row.to_dict()
        date = rowdict["Date"]
        if not date: 
            continue
        raincol = [x for x in rowdict.keys() if "rain" in x.lower()][0]
        # rainfall is the second column, but sometimes has inconsistent names
        for key in rowdict.keys(): 
            if raincol in key or "date" in key.lower():
                continue
            sampling_location = key.strip()
            record = {"Date": date,  
                      "Daily Rainfall (in.)": rowdict[raincol], 
                      sampling_location: rowdict[key]}

            records.append(record)
    del df

# convert to dataframe
df = pd.DataFrame.from_records(records)
df.to_csv("results/massbay/manual_integrated.csv", index=False)