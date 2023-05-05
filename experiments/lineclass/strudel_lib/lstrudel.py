# this is the implementation of line classification version of strudel.
# Created by lan at 2020/3/2
# import logging
import pdb
import string
from functools import reduce
import joblib

import numpy as np
import pandas
import pandas as pd
import rolling
from nltk import word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ndcg_score
import sys
sys.path.append("related/lineclass/strudel_lib/")
from derived_detector import detect_derived_cells, is_numeric
from utility import detect_datatype, calculate_hist_diff

def is_empty_row(row):
    return all([True if (cell or "").strip() == '' else False for cell in row])



class LStrudel:
    algo_name = 'lstrudel'

    def __init__(self, n_jobs=1):
        self.n_jobs = n_jobs
        self.clf = RandomForestClassifier(n_estimators=100, n_jobs=n_jobs)

    def save(self, model_path):
        joblib.dump(self.clf, model_path)
    
    def load(self, model_path):
        self.clf = joblib.load(model_path)

    def fit(self, train_set: pandas.DataFrame):
        """ Input is the training set, which is a pandas dataframe created for every file
        with the function create_line_feature_vector."""
        profile_columns = ['filename', 'line_number']

        clean_train_set = train_set[train_set['label'] != 'empty'].reset_index(drop=True)
        clean_train_set = clean_train_set.drop(profile_columns, axis=1)

        #Create X_train as clean_train_set without the column called 'label'
        y_train = clean_train_set["label"]
        X_train = clean_train_set.drop("label", axis=1)
        self.clf.fit(X_train, np.ravel(y_train), self.n_jobs)

    def predict(self, test_set: pandas.DataFrame):
        profile_columns = ['filename', 'line_number']

        empty_line_removed_test_set = test_set[test_set['label'] != 'empty'].reset_index(drop=True)

        test_set_line_profile = empty_line_removed_test_set[profile_columns]
        clean_test_set = empty_line_removed_test_set.drop(profile_columns, axis=1)

        # X_test = clean_test_set.iloc[:, 0:len(clean_test_set.columns)-1]
        # y_test = clean_test_set.iloc[:, len(clean_test_set.columns) - 1: len(clean_test_set.columns)]
        y_test = clean_test_set["label"]
        X_test = clean_test_set.drop("label", axis=1)

        pred_prob = self.clf.predict_proba(X_test)
        pred_prob_df = pandas.DataFrame(data=pred_prob, columns=self.clf.classes_ + '_prob')
        results_df = pandas.concat([test_set_line_profile, pred_prob_df, y_test], axis=1)

        names =[x.split("_")[0] for x in results_df.columns if "prob" in x]
        prob_idx =[idx for idx,x in enumerate(results_df.columns) if "prob" in x]
        detected_line_idx  = np.argmax(results_df.iloc[:,prob_idx].values, axis=1)
        results_df["predicted"] = [names[x] for x in detected_line_idx]

        return results_df
    
    def create_line_feature_vector(self,filename, table_array, line_labels, max_rows = None, sample_data=False):

        empty_idx = [idx for idx, x in enumerate(line_labels) if x == "empty"]
        all_rows = [x for idx, x in enumerate(table_array) if idx not in empty_idx]
        all_classes = [x for idx, x in enumerate(line_labels) if idx not in empty_idx]

        rows = []
        row_classes = []
        if max_rows is not None:
            if len(all_rows) > max_rows:
                data_idx = [idx for idx, x in enumerate(all_classes) if x == "data"]
                nondata_idx = [idx for idx, x in enumerate(all_classes) if x != "data"]
                if len(nondata_idx) < max_rows-1:
                    sampled_idx = data_idx[:(max_rows-len(nondata_idx))]
                    for idx,r in enumerate(all_rows):
                        if idx in nondata_idx+list(sampled_idx):
                            rows.append(r)
                            row_classes.append(all_classes[idx])
                else: # this happens in 5 cases
                    rows = all_rows[:max_rows]
                    row_classes = all_classes[:max_rows]
                        
            else:
                rows = all_rows
                row_classes = all_classes
        else:
            rows = all_rows
            row_classes = all_classes

        max_size = len(max(rows, key=lambda x: len(x)))
        file_rows = [['']*max_size for _ in rows]
        for i,row in enumerate(rows):
            for j,cell in enumerate(row):
                file_rows[i][j] = (cell or "")
        file_array = np.array(file_rows)

        num_lookup_lines = 5

        def row_emptiness_pattern(row):
            pattern = []
            for cell in row:
                cell_tmp = cell or ""
                if len(cell_tmp.strip()):
                    pattern.append(1)
                else:
                    pattern.append(0)
            return np.array(pattern)

        def discounted_cumulative_gain(row):
            # We hope all "1" are at the left of a pattern to all "0". Translating it to a dcg calculation,
            # an element's relevance is always 1 more than the element to its right, and the right-most element has a relevance score of 1.
            # For example, given a pattern [1,1,0,1,0,0], the relevance of the sequence is [6,5,4,3,2,1]
            # For an ideal pattern, the relevance of all "1" is large than that of all "0".
            re_pattern = np.asarray([row_emptiness_pattern(row)])
            pattern_len = re_pattern.shape[1]
            relevance = np.asarray([[pattern_len - index for index in range(0, pattern_len)]])
            if len(re_pattern) == 0:
                return 0
            elif len(re_pattern) ==1:
                return 1
            return ndcg_score(re_pattern, relevance)

        f_dcg = np.apply_along_axis(func1d=discounted_cumulative_gain, axis=1, arr=file_array)
        num_lines = len(file_array)
        f_line_position = []
        for index, row in enumerate(file_array):
            f_line_position.append((index + 1) / num_lines)
        f_line_position = np.array(f_line_position)

        def empty_cell_ratio(row):
            re_pattern = row_emptiness_pattern(row)
            empty_cell_ratio = 1 - np.count_nonzero(re_pattern) / len(row)
            return empty_cell_ratio

        f_empty_cell_ratio = np.apply_along_axis(func1d=empty_cell_ratio, axis=1, arr=file_array)

        def numeric_cell_ratio(row):
            nc_ratio = len([cell for cell in row if is_numeric(cell)]) / len(row)
            return nc_ratio

        f_numeric_cell_ratio = np.apply_along_axis(func1d=numeric_cell_ratio, axis=1, arr=file_array)

        def string_cell_ratio(row):
            sc_ratio = len([cell for cell in row if detect_datatype(cell) == 4]) / len(row)
            return sc_ratio

        f_string_cell_ratio = np.apply_along_axis(func1d=string_cell_ratio, axis=1, arr=file_array)

        def average_word_count(row):
            cell_tokens = [list(filter(lambda token: token not in string.punctuation, word_tokenize(cell or ""))) for cell in row]
            len_cell_tokens = [len(tokens) for tokens in cell_tokens if len(tokens) > 0]
            if not len_cell_tokens:
                awc = 0.0
            else:
                awc = np.sum(len_cell_tokens) / len(len_cell_tokens)
            return awc

        f_average_word_count = np.apply_along_axis(func1d=average_word_count, axis=1, arr=file_array)

        def empty_line_ratio_after(window):
            row_window = window[0]
            rows_after = row_window[1:]
            empty_rows_after = [row_after for row_after in rows_after if is_empty_row(row_after)]
            if not rows_after:
                ratio = 0.0
            else:
                ratio = len(empty_rows_after) / len(rows_after)
            return ratio

        windows = list(rolling.Apply(file_array, num_lookup_lines + 1, operation=list, window_type='variable'))
        windows = np.array([windows[len(windows) - len(file_array):]], dtype=object).transpose()

        f_empty_line_ratio_after = np.apply_along_axis(func1d=empty_line_ratio_after, axis=1, arr=windows)

        def empty_line_ratio_before(window):
            row_window = window[0]
            rows_before = row_window[:-1]
            empty_row_before = [row_before for row_before in rows_before if is_empty_row(row_before)]
            if not rows_before:
                ratio = 0.0
            else:
                ratio = len(empty_row_before) / len(rows_before)
            return ratio

        windows = list(rolling.Apply(file_array, num_lookup_lines + 1, operation=list, window_type='variable'))
        windows = np.array([windows[:len(file_array)]], dtype=object).transpose()

        f_empty_line_ratio_before = np.apply_along_axis(func1d=empty_line_ratio_before, axis=1, arr=windows)

        def hist_diff_after(window):
            row_window = window[0]
            row = row_window[0]
            row_value_length = [len(cell.strip()) for cell in row]
            rows_after = row_window[1:]
            hist_diff = 0.0
            for row_after in rows_after:
                if is_empty_row(row_after):
                    continue
                row_after_value_length = [len((cell or "").strip()) for cell in row_after]
                hist_diff = calculate_hist_diff(row_value_length, row_after_value_length)
                break
            return hist_diff

        windows = list(rolling.Apply(file_array, num_lookup_lines + 1, operation=list, window_type='variable'))
        windows = np.array([windows[-len(file_array):]], dtype=object).transpose()

        f_hist_diff_after = np.apply_along_axis(func1d=hist_diff_after, axis=1, arr=windows)

        def hist_diff_before(window):
            row_window = window[0]
            row = row_window[-1]
            row_value_length = [len(cell.strip()) for cell in row]
            rows_before = row_window[:-1]
            hist_diff = 0.0
            for row_before in reversed(rows_before):
                if is_empty_row(row_before):
                    continue
                row_before_value_length = [len(cell.strip()) for cell in row_before]
                hist_diff = calculate_hist_diff(row_value_length, row_before_value_length)
                break
            return hist_diff

        windows = list(rolling.Apply(file_array, num_lookup_lines + 1, operation=list, window_type='variable'))
        windows = np.array([windows[:len(file_array)]], dtype=object).transpose()

        f_hist_diff_before = np.apply_along_axis(func1d=hist_diff_before, axis=1, arr=windows)

        def data_type_after(window):
            row_window = window[0]
            row = row_window[0]
            row_data_type = [detect_datatype(cell) for cell in row]
            rows_after = row_window[1:]
            data_type_same_ratio = 0.0
            for row_after in rows_after:
                if is_empty_row(row_after):
                    continue
                row_after_data_type = [detect_datatype(cell) for cell in row_after]
                cell_same_data_type = [(this_cell, that_cell) for this_cell, that_cell in zip(row_data_type, row_after_data_type) if this_cell == that_cell]
                data_type_same_ratio = len(cell_same_data_type) / len(row_data_type)
            return data_type_same_ratio

        windows = list(rolling.Apply(file_array, num_lookup_lines + 1, operation=list, window_type='variable'))
        windows = np.array([windows[-len(file_array):]], dtype=object).transpose()

        f_data_type_after = np.apply_along_axis(func1d=data_type_after, axis=1, arr=windows)

        def data_type_before(window):
            row_window = window[0]
            row = row_window[-1]
            row_data_type = [detect_datatype(cell) for cell in row]
            rows_before = row_window[:-1]
            data_type_same_ratio = 0.0
            for row_before in rows_before:
                if is_empty_row(row_before):
                    continue
                row_before_data_type = [detect_datatype(cell) for cell in row_before]
                cell_same_data_type = [(this_cell, that_cell) for this_cell, that_cell in zip(row_data_type, row_before_data_type) if this_cell == that_cell]
                data_type_same_ratio = len(cell_same_data_type) / len(row_data_type)
            return data_type_same_ratio

        windows = list(rolling.Apply(file_array, num_lookup_lines + 1, operation=list, window_type='variable'))
        windows = np.array([windows[:len(file_array)]], dtype=object).transpose()

        f_data_type_before = np.apply_along_axis(func1d=data_type_before, axis=1, arr=windows)

        is_derived_file_array, _ = detect_derived_cells(file_array)

        f_derived_cell_coverage = []
        for row, is_derived_row in zip(file_array, is_derived_file_array):
            num_is_derived = len([cell for cell in is_derived_row if cell == 'd'])
            num_numeric = len([cell for cell in row if is_numeric(cell)])
            coverage = num_is_derived / num_numeric if num_numeric != 0 else 0.0
            f_derived_cell_coverage.append(coverage)
        f_derived_cell_coverage = np.array(f_derived_cell_coverage)

        line_number = np.array(range(file_array.shape[0]))

        lstrudel_feature_vector = pandas.DataFrame({'filename': filename,
                                                    'line_number': line_number,
                                                    'line_position': f_line_position,
                                                    'discounted_cumulative_gain': f_dcg,
                                                    'line_empty_cell_ratio': f_empty_cell_ratio,
                                                    'line_numeric_cell_ratio': f_numeric_cell_ratio,
                                                    'line_string_cell_ratio': f_string_cell_ratio,
                                                    'line_average_word_count': f_average_word_count,
                                                    'empty_line_ratio_after': f_empty_line_ratio_after,
                                                    'empty_line_ratio_before': f_empty_line_ratio_before,
                                                    'hist_diff_after': f_hist_diff_after,
                                                    'hist_diff_before': f_hist_diff_before,
                                                    'data_type_after': f_data_type_after,
                                                    'data_type_before': f_data_type_before,
                                                    'derived_cell_coverage': f_derived_cell_coverage,
                                                    'label': row_classes})

        return lstrudel_feature_vector
