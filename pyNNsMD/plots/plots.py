#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 22:58:56 2022

@author: Chen Zhou
"""
import os
import sys
import numpy as np
import matplotlib as mpl
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser(description='Train a energy-gradient model from data, parameters given in a folder')

parser.add_argument("-i", "--index", required=True, help="Index of the NN to train")
parser.add_argument("-f", "--filepath", required=True, help="Filepath to weights, hyperparameter, data etc. ")
parser.add_argument("-g", "--gpus", default=-1, required=True, help="Index of gpu to use")
parser.add_argument("-l", "--load", required=True, help="Whether to load the model or remake it")
parser.add_argument("-m", "--mode", default="training", required=True, help="Which mode to use train or retrain")
args = vars(parser.parse_args())

fstdout = open(os.path.join(args['filepath'], "fitlog.txt"), 'a')
sys.stderr = fstdout
sys.stdout = fstdout

print("\n")
print("Input argpars for training/validation plots:", args)

from pyNNsMD.src.device import set_gpu

set_gpu([int(args['gpus'])])
print("Logic Devices:", tf.config.experimental.list_logical_devices('GPU'))

from pyNNsMD.models.mlp_eg import EnergyGradientModel
from pyNNsMD.scaler.energy import EnergyGradientStandardScaler, MaskedEnergyGradientStandardScaler
from pyNNsMD.utils.data import load_json_file, read_xyz_file, save_json_file
from pyNNsMD.plots.loss import plot_loss_curves, plot_learning_curve
from pyNNsMD.plots.pred import plot_scatter_prediction
from pyNNsMD.plots.error import plot_error_vec_mean, plot_error_vec_max

def plot_results(i=0, out_dir=None, mode='training', load_model: bool = False):
    """Plot the training/validation results. Uses precomputed feature and model representation.

    Args:
        i (int, optional): Model index. The default is 0.
        out_dir (str, optional): Directory for fit output. The default is None.
        mode (str, optional): Fit-mode to take from hyperparameters. The default is 'training'.
        load_model (bool): Whether to load model without remaking the model. Default is False.

    Raises:
        ValueError: Wrong input shape.

    Returns:
        error_val (list): Validation error for (energy,gradient).

    """
    i = int(i)
    # Load everything from folder
    training_config = load_json_file(os.path.join(out_dir, mode+"_config.json"))
    model_config = load_json_file(os.path.join(out_dir, "model_config.json"))
    i_train = np.load(os.path.join(out_dir, "train_index.npy"))
    i_val = np.load(os.path.join(out_dir, "test_index.npy"))
    scaler_config = load_json_file(os.path.join(out_dir, "scaler_config.json"))
    
    # Info from Config
    use_mask = training_config["mask"]
    energies_only = model_config["config"]['energy_only']
    dir_save = os.path.join(out_dir, "fit_stats")
    os.makedirs(dir_save, exist_ok=True)
    unit_label_energy = training_config['unit_energy']
    unit_label_grad = training_config['unit_gradient']
    batch_size = training_config['batch_size']
    
    # Load data.
    data_dir = os.path.dirname(out_dir)
    xyz = read_xyz_file(os.path.join(data_dir, "geometries.xyz"))
    x = np.array([x[1] for x in xyz])
    y1 = np.array(load_json_file(os.path.join(data_dir, "energies.json")))
    y2 = np.array(load_json_file(os.path.join(data_dir, "forces.json")))
    print("INFO: Shape of x", x.shape)
    print("INFO: Shape of y", y1.shape, y2.shape)
    y = [y1, y2]
    
    # Scale x,y
    if use_mask:
        scaler = MaskedEnergyGradientStandardScaler(**scaler_config["config"])
    else:
        scaler = EnergyGradientStandardScaler(**scaler_config["config"])
    scaler.load_weights(os.path.join(out_dir, "scaler_weights.npy"))
    x_rescale, _ = scaler.transform(x=x, y=None)
        
    # Load model
    if load_model:
        out_model = tf.keras.models.load_model(os.path.join(out_dir, "model_tf"), compile=False)
    else:
        out_model = EnergyGradientModel(**model_config["config"])
        out_model.load_weights(os.path.join(out_dir, "model_weights.h5"))
    out_model.precomputed_features = True
    out_model.output_as_dict = True
    out_model.energy_only = energies_only
    
    # Model + Model precompute layer +feat
    feat_x, feat_grad = out_model.precompute_feature_in_chunks(x_rescale, batch_size=batch_size)
    
    # Plot and Save
    xtrain = [feat_x[i_train], feat_grad[i_train]]
    xval = [feat_x[i_val], feat_grad[i_val]]
    yval_plot = [y[0][i_val], y[1][i_val]]
    ytrain_plot = [y[0][i_train], y[1][i_train]]
    # Convert back scaler
    pval = out_model.predict(xval)
    ptrain = out_model.predict(xtrain)
    _, pval = scaler.inverse_transform(y=[pval['energy'], pval['force']])
    _, ptrain = scaler.inverse_transform(y=[ptrain['energy'], ptrain['force']])

    print("Info: Predicted Energy shape:", ptrain[0].shape)
    print("Info: Predicted Gradient shape:", ptrain[1].shape)
    print("Info: Plot fit stats...")

    # Plot
    plot_scatter_prediction(pval[0], yval_plot[0], save_plot_to_file=True, dir_save=dir_save,
                            filename='fit' + str(i) + "_energy",
                            filetypeout='.png', unit_actual=unit_label_energy, unit_predicted=unit_label_energy,
                            plot_title="Prediction Energy")

    plot_scatter_prediction(pval[1], yval_plot[1], save_plot_to_file=True, dir_save=dir_save,
                            filename='fit' + str(i) + "_grad",
                            filetypeout='.png', unit_actual=unit_label_grad, unit_predicted=unit_label_grad,
                            plot_title="Prediction Gradient")

    plot_error_vec_mean([pval[1], ptrain[1]], [yval_plot[1], ytrain_plot[1]],
                        label_curves=["Validation gradients", "Training Gradients"], unit_predicted=unit_label_grad,
                        filename='fit' + str(i) + "_grad", dir_save=dir_save, save_plot_to_file=True,
                        filetypeout='.png', x_label='Gradients xyz * #atoms * #states ',
                        plot_title="Gradient mean error")

    plot_error_vec_max([pval[1], ptrain[1]], [yval_plot[1], ytrain_plot[1]],
                       label_curves=["Validation", "Training"],
                       unit_predicted=unit_label_grad, filename='fit' + str(i) + "_grad",
                       dir_save=dir_save, save_plot_to_file=True, filetypeout='.png',
                       x_label='Gradients xyz * #atoms * #states ', plot_title="Gradient max error")

    pval = out_model.predict(xval)
    ptrain = out_model.predict(xtrain)
    _, pval = scaler.inverse_transform(y=[pval['energy'], pval['force']])
    _, ptrain = scaler.inverse_transform(y=[ptrain['energy'], ptrain['force']])
    out_model.precomputed_features = False
    out_model.output_as_dict = False
    ptrain2 = out_model.predict(x_rescale[i_train])
    _, ptrain2 = scaler.inverse_transform(y=[ptrain2[0], ptrain2[1]])
    print("Info: Max error precomputed and full gradient computation:")
    print("Energy", np.max(np.abs(ptrain[0] - ptrain2[0])))
    print("Gradient", np.max(np.abs(ptrain[1] - ptrain2[1])))
    error_val = [np.mean(np.abs(pval[0] - y[0][i_val])), np.mean(np.abs(pval[1] - y[1][i_val]))]
    error_train = [np.mean(np.abs(ptrain[0] - y[0][i_train])), np.mean(np.abs(ptrain[1] - y[1][i_train]))]
    print("error_val:", error_val)
    print("error_train:", error_train)
    error_dict = {"train": [error_train[0].tolist(), error_train[1].tolist()],
                  "valid": [error_val[0].tolist(), error_val[1].tolist()]}
    save_json_file(error_dict, os.path.join(out_dir, "fit_error.json"))
    
if __name__ == "__main__":
    plot_results(args['index'], args['filepath'], args['mode'], args['load'])

fstdout.close()