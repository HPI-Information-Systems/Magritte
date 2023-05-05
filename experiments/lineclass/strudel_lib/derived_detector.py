# Created by lan at 2021/11/9
import numpy as np
import pandas

def is_numeric(value):
    """
    check whether the given value is a numeric value. Numbers with decimal point or thousand separator can be properly determined.
    :param value: the given value to be verified
    :return: True if the given value is numeric
    """
    value_str = str(value).replace(',', '')
    try:
        float(value_str)
        return True
    except:
        return False


def to_numeric(value):
    if not is_numeric(value):
        raise Exception('Value is not numeric.')
    value_str = str(value).replace(',', '')
    try:
        return float(value_str)
    except:
        raise Exception('Conversion error.')



def get_top_neighbour(row_index, anchor_row_indices):
    top_indices = [index for index in anchor_row_indices if index < row_index]
    top_index = max(top_indices) if len(top_indices) > 0 else None
    return top_index


def get_bottom_neighbour(row_index, anchor_row_indices):
    bottom_indices = [index for index in anchor_row_indices if index > row_index]
    bottom_index = min(bottom_indices) if len(bottom_indices) > 0 else None
    return bottom_index


def get_left_neighbour(col_index, anchor_col_indices):
    left_indices = [index for index in anchor_col_indices if index < col_index]
    left_index = max(left_indices) if len(left_indices) > 0 else None
    return left_index


def get_right_neighbour(col_index, anchor_col_indices):
    right_indices = [index for index in anchor_col_indices if index > col_index]
    right_index = min(right_indices) if len(right_indices) > 0 else None
    return right_index


def detect_derived_cells(table, aggr_delta=1.0E-1, satisfied_ratio=0.5, floating_error=1.0E-10, keyword_filter_anchor=True) -> np.array:
    """
    this algorithm_package detects the position of derived cells in the given table.

    :param table: a list of lists that represents the two-dimensional table
    :param aggr_delta: the delta threshold for sum that gives how much error is allowed for an aggregation
    :param satisfied_ratio: Todo: find a good way to describe it
    :return: a list of lists that represents the annotations, it has the same shape as the given table parameter.
    The value of each element is either 'derived' or non-'derived'
    """
    table_array = np.array(table, dtype=str)
    annotation_array = np.full_like(table_array, fill_value='n', dtype='<U1')

    # generate anchor first, get the indices of all anchoring cells
    derived_keywords = ['total', 'all', 'totals', 'sum', 'average', 'avg']
    if keyword_filter_anchor:
        array_incl_kws = np.zeros(table_array.shape, dtype=bool)
        for keyword in derived_keywords:
            judge = np.core.defchararray.find(np.char.lower(table_array), keyword) != -1
            array_incl_kws = array_incl_kws | judge
        indices_anchor_cells = np.where(array_incl_kws)
    else:
        # all numeric cells are anchor.
        table_df = pandas.DataFrame(data=table_array)
        indices_anchor_cells = np.where(table_df.applymap(is_numeric).to_numpy())

    # if there is no anchoring cells, just return
    if len(indices_anchor_cells) == 0:
        return annotation_array, 0

    anchor_row_indices = indices_anchor_cells[0]
    anchor_col_indices = indices_anchor_cells[1]
    indices_anchor_cells = list(zip(anchor_row_indices, anchor_col_indices))
    # print('Number of anchor cells: {}'.format(str(len(indices_anchor_cells))))
    # print('Size of the file: {}, {}'.format(table_array.shape[0], table_array.shape[1]))
    # create neighbour dictionary for each anchor
    dict_anchor_neighbour = dict.fromkeys(indices_anchor_cells)
    for anchor_index in dict_anchor_neighbour:
        row_index = anchor_index[0]
        col_index = anchor_index[1]
        top_index = get_top_neighbour(row_index, anchor_row_indices)
        bottom_index = get_bottom_neighbour(row_index, anchor_row_indices)
        left_index = get_left_neighbour(col_index, anchor_col_indices)
        right_index = get_right_neighbour(col_index, anchor_col_indices)
        dict_anchor_neighbour[anchor_index] = {'top': top_index, 'bottom': bottom_index, 'left': left_index, 'right': right_index}

    # generate derived cell candidates, each candidate is a numeric cell in either the same row or the same column as an anchoring cell
    indices_derived_cells = []  # each element is a tuple of two elements representing the row and column indices
    for row_index, col_index in indices_anchor_cells:
        row = table_array[row_index, :]
        horizontal_candidate_indices = [index for index, value in enumerate(row) if is_numeric(value)]
        # upwards
        # top_index = get_top_neighbour(row_index, anchor_row_indices)
        # top_index = -1 if top_index is None else top_index
        top_index = -1
        upwards_sum = [0] * len(row)
        upwards_avg = [0] * len(row)
        for count, index in enumerate(reversed(range(top_index + 1, row_index)), start=1):
            # if there is no number cell in this line anymore, searching can be terminated.
            if len(horizontal_candidate_indices) == 0:
                break
            parsed_row = table_array[index, :]
            column_indices_numbers = [index for index in horizontal_candidate_indices if is_numeric(parsed_row[index])]
            for number_col_index in column_indices_numbers:
                upwards_sum[number_col_index] += to_numeric(parsed_row[number_col_index])
                upwards_avg[number_col_index] = upwards_sum[number_col_index] / count
            satisfied_indices_sum = [candidate_index for candidate_index in horizontal_candidate_indices
                                     if abs(to_numeric(row[candidate_index]) - upwards_sum[candidate_index]) - floating_error <= aggr_delta]
            satisfied_indices_avg = [candidate_index for candidate_index in horizontal_candidate_indices
                                     if abs(to_numeric(row[candidate_index]) - upwards_avg[candidate_index]) - floating_error <= aggr_delta]
            if count == 1:
                continue
            # is more than that many cells see such a summation, the candidates are indeed derived cells.
            if len(satisfied_indices_sum) / len(horizontal_candidate_indices) > satisfied_ratio \
                    or len(satisfied_indices_avg) / len(horizontal_candidate_indices) > satisfied_ratio:
                indices_derived_cells.extend([(row_index, candidate_col_index) for candidate_col_index in horizontal_candidate_indices])
                break

        # downwards
        # bottom_index = get_bottom_neighbour(row_index, anchor_row_indices)
        # bottom_index = table_array.shape[0] if bottom_index is None else top_index
        bottom_index = table_array.shape[0]
        downwards_sum = [0] * len(row)
        downwards_avg = [0] * len(row)
        for count, index in enumerate(range(row_index + 1, bottom_index), start=1):
            if len(horizontal_candidate_indices) == 0:
                break
            parsed_row = table_array[index, :]
            column_indices_numbers = [index for index in horizontal_candidate_indices if is_numeric(parsed_row[index])]
            for number_col_index in column_indices_numbers:
                downwards_sum[number_col_index] += to_numeric(parsed_row[number_col_index])
                downwards_avg[number_col_index] = downwards_sum[number_col_index] / count
            satisfied_indices_sum = [candidate_index for candidate_index in horizontal_candidate_indices
                                     if abs(to_numeric(row[candidate_index]) - downwards_sum[candidate_index]) - floating_error <= aggr_delta]
            satisfied_indices_avg = [candidate_index for candidate_index in horizontal_candidate_indices
                                     if abs(to_numeric(row[candidate_index]) - downwards_avg[candidate_index]) - floating_error <= aggr_delta]
            if count == 1:
                continue
            if len(satisfied_indices_sum) / len(horizontal_candidate_indices) > satisfied_ratio \
                    or len(satisfied_indices_avg) / len(horizontal_candidate_indices) > satisfied_ratio:
                indices_derived_cells.extend([(row_index, candidate_col_index) for candidate_col_index in horizontal_candidate_indices])
                break

        column = table_array[:, col_index]
        vertical_candidate_indices = [index for index, value in enumerate(column) if is_numeric(value)]
        # leftwards
        # left_index = get_left_neighbour(col_index, anchor_col_indices)
        # left_index = -1 if left_index is None else left_index
        left_index = -1
        leftwards_sum = [0] * len(column)
        leftwards_avg = [0] * len(column)
        for count, index in enumerate(reversed(range(left_index + 1, col_index)), start=1):
            if len(vertical_candidate_indices) == 0:
                break
            parsed_column = table_array[:, index]
            row_indices_numbers = [index for index in vertical_candidate_indices if is_numeric(parsed_column[index])]
            for number_row_index in row_indices_numbers:
                leftwards_sum[number_row_index] += to_numeric(parsed_column[number_row_index])
                leftwards_avg[number_row_index] = leftwards_sum[number_row_index] / count
            satisfied_indices_sum = [candidate_index for candidate_index in vertical_candidate_indices
                                 if abs(to_numeric(column[candidate_index]) - leftwards_sum[candidate_index]) - floating_error <= aggr_delta]
            satisfied_indices_avg = [candidate_index for candidate_index in vertical_candidate_indices
                                     if abs(to_numeric(column[candidate_index]) - leftwards_avg[candidate_index]) - floating_error <= aggr_delta]
            if count == 1:
                continue
            if len(satisfied_indices_sum) / len(vertical_candidate_indices) > satisfied_ratio \
                    or len(satisfied_indices_avg) / len(vertical_candidate_indices) > satisfied_ratio:
                indices_derived_cells.extend([(candidate_row_index, col_index) for candidate_row_index in vertical_candidate_indices])
                break

        # rightwards
        # right_index = get_right_neighbour(col_index, anchor_col_indices)
        # right_index = table_array.shape[1] if right_index is None else right_index
        right_index = table_array.shape[1]
        rightwards_sum = [0] * len(column)
        rightwards_avg = [0] * len(column)
        for count, index in enumerate(range(col_index + 1, right_index), start=1):
            if len(vertical_candidate_indices) == 0:
                break
            parsed_column = table_array[:, index]
            row_indices_numbers = [index for index in vertical_candidate_indices if is_numeric(parsed_column[index])]
            for number_row_index in row_indices_numbers:
                rightwards_sum[number_row_index] += to_numeric(parsed_column[number_row_index])
                rightwards_avg[number_row_index] = rightwards_sum[number_row_index] / count
            satisfied_indices_sum = [candidate_index for candidate_index in vertical_candidate_indices
                                     if abs(to_numeric(column[candidate_index]) - rightwards_sum[candidate_index]) - floating_error <= aggr_delta]
            satisfied_indices_avg = [candidate_index for candidate_index in vertical_candidate_indices
                                     if abs(to_numeric(column[candidate_index]) - rightwards_avg[candidate_index]) - floating_error <= aggr_delta]
            if count == 1:
                continue
            if len(satisfied_indices_sum) / len(vertical_candidate_indices) > satisfied_ratio \
                    or len(satisfied_indices_avg) / len(vertical_candidate_indices) > satisfied_ratio:
                indices_derived_cells.extend([(candidate_row_index, col_index) for candidate_row_index in vertical_candidate_indices])
                break

    indices_derived_cells = list(set(indices_derived_cells))
    for derived_cell_index in indices_derived_cells:
        annotation_array[derived_cell_index[0], derived_cell_index[1]] = 'd'
    return annotation_array, len(indices_anchor_cells)


def cal_is_derived(table_cells, aggr_delta=1.0E-1, satisfied_ratio=0.5):
    indices_is_derived = []
    pred_annotations, num_derived_candidates = detect_derived_cells(table_cells, aggr_delta, satisfied_ratio, keyword_filter_anchor=False)
    indices = np.where(pred_annotations == 'd')
    if len(indices[0]) == 0:
        return indices_is_derived, 0
    for row_index, column_index in zip(indices[0], indices[1]):
        indices_is_derived.append((row_index,column_index))
    return indices_is_derived, num_derived_candidates