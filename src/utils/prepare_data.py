# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

"""
Function for train-test split of the data
"""
from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.model_selection import train_test_split


def prepare_data(df0, input_cols_list, output_var, test_size=0.3):
    df1 = df0.copy()
    x_data = df1[input_cols_list]
    y_data = df1[str(output_var)]
    x_train_df, x_test_df, y_train_df, y_test_df = train_test_split(
        x_data, y_data, test_size=test_size, random_state=42)
    return (x_train_df, x_test_df, y_train_df, y_test_df)
