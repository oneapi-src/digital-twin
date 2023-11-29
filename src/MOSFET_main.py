# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

"""
Main executable code
"""
import argparse
import logging
import numpy as np
import pandas as pd
import os
import joblib, pathlib
import warnings
from utils.synthetic_datagen import data_gen, data_gen_aug
from utils.prepare_data import prepare_data
from utils.training import linreg, XGBHyper_train, XGBReg_train, XGB_predict, XGB_predict_daal4py, XGB_predict_aug

warnings.simplefilter(action="ignore", category=FutureWarning)

def main(FLAGS):

    if FLAGS.logfile == "":
        logging.basicConfig(level=logging.DEBUG)
    else:
        path = pathlib.Path(FLAGS.logfile)
        path.parent.mkdir(parents=True, exist_ok=True)    
        logging.basicConfig(filename=FLAGS.logfile, level=logging.DEBUG)

    
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    
    
    logging.getLogger('').addHandler(console)
    logger = logging.getLogger()

    if FLAGS.n_data_len:
        linspace_factor = int(FLAGS.n_data_len)
    else:
        linspace_factor = 1

    
    model_file = FLAGS.modelfile
    
    
    logger.info("===== Running benchmarks for oneAPI tech =====")
    


    logger.info("===== Generating Synthetic Data =====")
    if FLAGS.data_path:
        print("Reading customized data file in path  ",FLAGS.data_path)
        synth_data=pd.read_csv(FLAGS.data_path)
        logger.info("Customized data shape %s", ' '.join(
            map(str, list(synth_data.shape))))
    else: 
        data_gen(linspace_factor)
        ##read generated data
        data_dir=os.environ.get('DATA_DIR')
        synth_data=pd.read_csv(os.path.join(data_dir,'synthetic_data.csv'))
        logger.info("Synthetic data shape %s", ' '.join(
            map(str, list(synth_data.shape))))

    if FLAGS.x_cols:
        print("Reading customized dataset 'X' columns ", FLAGS.x_cols)
        x_columns=FLAGS.x_cols
    else:
        print("Synthetic dataset 'X' columns: ['w_l', 'vgs', 'vth', 'eta','temp', 'w_l_bins', 'vgs_bins', 'vth_bins'] ")
        x_columns= ['w_l', 'vgs', 'vth', 'eta','temp', 'w_l_bins', 'vgs_bins', 'vth_bins']
    
    if FLAGS.y_col:
        print("Reading customized dataset target 'Y' column: ", FLAGS.y_col)
        target=FLAGS.y_col
    else:
        print("Synthetic dataset 'Y' target column: 'log-leakage'")
        target='log-leakage'

    if FLAGS.model == 'lr':
        # Linear Regression for reference

        input_cols = x_columns
        synth_data[input_cols] = synth_data[input_cols].astype(float)
        x_train, x_test, y_train, y_test = prepare_data(
            synth_data, input_cols_list=input_cols, output_var=target)

        logger.info("===== Running Benchmarks for Linear Regression =====")
        train_time, pred_time, MSE = linreg(
            x_train, x_test, y_train, y_test)
        logger.info("Training time = %s", train_time)
        logger.info("Prediction time = %s", pred_time)
        logger.info('Mean SQ Error: %s', str(round(np.mean(MSE), 3)))

    if FLAGS.model == 'xgbh':
        # XGB Regression Hyperparameter training

        input_cols = x_columns
        synth_data[input_cols] = synth_data[input_cols].astype(float)
        x_train, x_test, y_train, y_test = prepare_data(
            synth_data, input_cols_list=input_cols, output_var=target)
        loop_ctr = 5
        parameters = {'nthread': [1],
                      'learning_rate': [0.02],  # so called `eta` value
                      'max_depth': [3, 5],
                      'min_child_weight': [6, 7],
                      'n_estimators': [750, 1000],
                      'tree_method': ['hist']}
        logger.info(
            "===== Running Benchmarks for XGB Hyperparameter Training =====")
        train_time, trained_model, model_params = XGBHyper_train(
            x_train, y_train, parameters)
        logger.info("Training time = %s", train_time)
        
        prediction, pred_time, MSE = XGB_predict(
            x_test, y_test, trained_model, loop_ctr)
        prediction, pred_time_daal4py, MSE_daal4py = XGB_predict_daal4py(
            x_test, y_test, trained_model, loop_ctr)
        logger.info("Prediction time = %s", pred_time)
        logger.info("daal4py Prediction time = %s", pred_time_daal4py)
        logger.info('Mean SQ Error: %s', str(round(np.mean(MSE), 3)))
        logger.info('daal4py Mean SQ Error: %s',
                    str(round(np.mean(MSE_daal4py), 3)))
    
        if model_file != "":
            joblib.dump(trained_model, model_file)
        else:
            joblib.dump(trained_model, "model.pkl")

    if FLAGS.model == 'xgb':
        # XGB Regression

        input_cols = x_columns
        synth_data[input_cols] = synth_data[input_cols].astype(float)
        x_train, x_test, y_train, y_test = prepare_data(
            synth_data, input_cols_list=input_cols, output_var=target)

        logger.info("===== Running Benchmarks for XGB Regression =====")
        loop_ctr = 5
        train_time, trained_model = XGBReg_train(
            x_train, y_train, loop_ctr)
        logger.info("Training time = %s", train_time)
        
        prediction, pred_time, MSE = XGB_predict(
            x_test, y_test, trained_model, loop_ctr)
        prediction, pred_time_daal4py, MSE_daal4py = XGB_predict_daal4py(
            x_test, y_test, trained_model, loop_ctr)
        logger.info("Prediction time = %s", pred_time)
        logger.info("daal4py Prediction time = %s", pred_time_daal4py)
        logger.info('Mean SQ Error: %s', str(round(np.mean(MSE), 3)))
        logger.info('daal4py Mean SQ Error: %s',
                    str(round(np.mean(MSE_daal4py), 3)))
    
        if model_file != "":
            joblib.dump(trained_model, model_file)
        else:
            joblib.dump(trained_model, "model.pkl")            

    if FLAGS.model == 'xgbfull':  # this will report results for daal4py converted model only
        # XGB Regression Hyperparameter training

        input_cols = x_columns
        synth_data[input_cols] = synth_data[input_cols].astype(float)
        x_train, x_test, y_train, y_test = prepare_data(
            synth_data, input_cols_list=input_cols, output_var=target)
        loop_ctr = 1
        parameters = {'nthread': [1],
                      'learning_rate': [0.02],  # so called `eta` value
                      'max_depth': [3, 5],
                      'min_child_weight': [6, 7],
                      'n_estimators': [750, 1000],
                      'tree_method': ['hist']}
        logger.info("\n")
        logger.info(
            "===== Running Benchmarks for full pipeline execution =====")
        train_time, trained_model, model_params = XGBHyper_train(
            x_train, y_train, parameters)
        
        prediction, pred_time, MSE = XGB_predict_daal4py(
            x_test, y_test, trained_model, loop_ctr)
    
        logger.info('Mean SQ Error for initial train: %s',
                    str(round(np.mean(MSE), 3)))
        
        synth_data_0 = synth_data
        
        synth_data_0 = synth_data.drop(columns='sub-vth')
        train_trend_df = pd.DataFrame(columns=['len_data', 'train_time'])
        train_time_vals = []
        len_data_vals = []
        for counter in range(10):
            # generating new synthetic data and making a prediction on it
            synth_data_aug = data_gen_aug(linspace_factor)
            synth_data_aug = synth_data_aug.astype(float)

            y_pred = XGB_predict_aug(synth_data_aug, trained_model)
            synth_data_aug[target] = y_pred

            # concatenating the original dataframe to
            synth_data_0 = pd.concat(
                [synth_data_0, synth_data_aug], axis=0, ignore_index=True)
            if len(synth_data_0) > 2500000:
                synth_data_0 = synth_data_0.tail(2500000)

            # semi supervised learning on the new "augmented data"
            train_time, trained_model = XGBReg_train(
                synth_data_0[input_cols], synth_data_0[target], loop_ctr, model_params)
            len_data_vals.append(len(synth_data_0))
            train_time_vals.append(train_time)
            print('augmented supervised learning round finished ' + str(counter))
        train_trend_df['len_data'] = len_data_vals
        train_trend_df['train_time'] = train_time_vals
        
        train_trend_df.to_csv('train_time_intel.csv', index=False)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-l',
                        '--logfile',
                        type=str,
                        default="",
                        help="log file to output benchmarking results to")
    parser.add_argument('-m',
                        '--model',
                        type=str,
                        default="",
                        required=True,
                        help="type of model lr:linreg, xgb:xgboost, xgbh: \
                        xgb with hyperparameter tuning, xgbfull:")
    parser.add_argument('-mf',
                        '--modelfile',
                        type=str,
                        default="",
                        required=False,
                        help="name for the built model please add extension if desired")
    parser.add_argument('-n',
                        '--n_data_len',
                        type=str,
                        default="1",
                        help="option for data length. Provide 1 2 or 3, default 1")
    parser.add_argument('-d',
                        '--data_path',
                        type=str,
                        default="",
                        required=False,
                        help="path to the customized csv dataset, optional")
    parser.add_argument('-x',
                        '--x_cols',
                        type=str,
                        nargs="*",
                        default=[],
                        required=False,
                        help="provide the independent columns of customized dataset space separated")
    parser.add_argument('-y',
                        '--y_col',
                        type=str,
                        default='',
                        required=False,
                        help="provide the dependent column of customized dataset")
    FLAGS = parser.parse_args()
    main(FLAGS)
