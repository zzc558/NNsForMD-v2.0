import matplotlib as mpl
import numpy as np
import tensorflow as tf
mpl.use('Agg')
import os
import json
import pickle
import sys
import argparse

parser = argparse.ArgumentParser(description='Train a energy-gradient model from data, parameters given in a folder')

parser.add_argument("-i", "--index", required=True, help="Index of the NN to train")
parser.add_argument("-f", "--filepath", required=True, help="Filepath to weights, hyperparameter, data etc. ")
parser.add_argument("-g", "--gpus", default=-1, required=True, help="Index of gpu to use")
parser.add_argument("-m", "--mode", default="training", required=True, help="Which mode to use train or retrain")
args = vars(parser.parse_args())

fstdout = open(os.path.join(args['filepath'], "fitlog.txt"), 'w')
sys.stderr = fstdout
sys.stdout = fstdout

print("Input argpars:", args)

from pyNNsMD.src.device import set_gpu

set_gpu([int(args['gpus'])])
print("Logic Devices:", tf.config.experimental.list_logical_devices('GPU'))

import pyNNsMD.utils.callbacks
import pyNNsMD.utils.activ
from pyNNsMD.models.schnet_kgcnn import SchnetEnergy
from pyNNsMD.scaler.energy import EnergyGradientStandardScaler
from pyNNsMD.utils.loss import get_lr_metric, ScaledMeanAbsoluteError, r2_metric, ZeroEmptyLoss
from pyNNsMD.utils.data import load_json_file, read_xyz_file, save_json_file
from pyNNsMD.plots.loss import plot_loss_curves, plot_learning_curve
from pyNNsMD.plots.pred import plot_scatter_prediction
from pyNNsMD.plots.error import plot_error_vec_mean, plot_error_vec_max
from kgcnn.utils.adj import define_adjacency_from_distance, coordinates_to_distancematrix
# from kgcnn.utils.data import ragged_tensor_from_nested_numpy
from kgcnn.mol.methods import global_proton_dict


def ragged_tensor_from_nested_numpy(numpy_list: list, dtype="int64"):
    return tf.RaggedTensor.from_row_lengths(np.concatenate(numpy_list, axis=0),
                                            np.array([len(x) for x in numpy_list], dtype=dtype))


def train_model_energy_gradient(i=0, out_dir=None, mode='training'):
    """Train an energy plus gradient model. Uses precomputed feature and model representation.
    Args:
        i (int, optional): Model index. The default is 0.
        out_dir (str, optional): Directory for fit output. The default is None.
        mode (str, optional): Fit-mode to take from hyperparameters. The default is 'training'.
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
    energies_only = model_config["config"]['energy_only']
    range_dist = model_config["config"]["schnet_kwargs"]["gauss_args"]["distance"]
    unit_label_energy = training_config['unit_energy']
    unit_label_grad = training_config['unit_gradient']
    epo = training_config['epo']
    batch_size = training_config['batch_size']
    epostep = training_config['epostep']
    initialize_weights = training_config['initialize_weights']
    learning_rate = training_config['learning_rate']
    loss_weights = training_config['loss_weights']
    use_callbacks = list(training_config["callbacks"])

    # Load data.
    data_dir = os.path.dirname(out_dir)
    xyz = read_xyz_file(os.path.join(data_dir, "geometries.xyz"))
    x = np.array([x[1] for x in xyz])
    coords = [np.array(x[1]) for x in xyz]
    atoms = [np.array([global_proton_dict[at] for at in x[0]]) for x in xyz]
    range_indices = [define_adjacency_from_distance(coordinates_to_distancematrix(x),
                                                    max_distance=range_dist)[1] for x in coords]

    y1 = np.array(load_json_file(os.path.join(data_dir, "energies.json")))
    y2 = np.array(load_json_file(os.path.join(data_dir, "forces.json")))
    print("INFO: Shape of y", y1.shape, y2.shape)
    y = [y1, y2]

    # Fit stats dir
    dir_save = os.path.join(out_dir, "fit_stats")
    os.makedirs(dir_save, exist_ok=True)

    # cbks, Learning rate schedule
    cbks = []
    for x in use_callbacks:
        if isinstance(x, dict):
            # tf.keras.utils.get_registered_object()
            cb = tf.keras.utils.deserialize_keras_object(x)
            cbks.append(cb)

    # Index train test split
    print("Info: Train-Test split at Train:", len(i_train), "Test", len(i_val), "Total", len(x))

    # Make all Model
    assert model_config["class_name"] == "SchnetEnergy", "Training script only for EnergyModel"
    out_model = SchnetEnergy(**model_config["config"])
    out_model.energy_only = energies_only
    out_model.output_as_dict = False

    # Look for loading weights
    npeps = np.finfo(float).eps
    if not initialize_weights:
        out_model.load_weights(os.path.join(out_dir, "model_weights.h5"))
        print("Info: Load old weights at:", os.path.join(out_dir, "model_weights.h5"))
        print("Info: Transferring weights...")
    else:
        print("Info: Making new initialized weights.")

    # Scale x,y
    scaler = EnergyGradientStandardScaler(**scaler_config["config"])
    scaler.fit(x[i_train], [y[0][i_train], y[1][i_train]])
    x_rescale, y_rescale = scaler.transform(x, y)
    y1, y2 = y_rescale

    # Train Test split
    # Train Test split
    xtrain = [
        ragged_tensor_from_nested_numpy([atoms[i] for i in i_train]),
        ragged_tensor_from_nested_numpy([coords[i] for i in i_train]),
        ragged_tensor_from_nested_numpy([range_indices[i] for i in i_train])
    ]
    xval = [
        ragged_tensor_from_nested_numpy([atoms[i] for i in i_val]),
        ragged_tensor_from_nested_numpy([coords[i] for i in i_val]),
        ragged_tensor_from_nested_numpy([range_indices[i] for i in i_val])
    ]
    ytrain = y1[i_train]
    yval = y1[i_val]

    # Compile model
    # This is only for metric to without std.
    scaled_metric = ScaledMeanAbsoluteError(scaling_shape=scaler.energy_std.shape)
    scaled_metric.set_scale(scaler.energy_std)
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    lr_metric = get_lr_metric(optimizer)
    out_model.compile(optimizer=optimizer,
                      loss='mean_squared_error',
                      metrics=[scaled_metric, lr_metric, r2_metric])

    scaler.print_params_info()

    out_model.summary()
    print("")
    print("Start fit.")
    hist = out_model.fit(x=xtrain, y=ytrain, epochs=epo, batch_size=batch_size, callbacks=cbks, validation_freq=epostep,
                         validation_data=(xval, yval), verbose=2)
    print("End fit.")
    print("")

    outname = os.path.join(dir_save, "history.json")
    outhist = {a: np.array(b, dtype=np.float64).tolist() for a, b in hist.history.items()}
    with open(outname, 'w') as f:
        json.dump(outhist, f)

    print("Info: Saving auto-scaler to file...")
    scaler.save_weights(os.path.join(out_dir, "scaler_weights.npy"))

    # Plot and Save
    yval_plot = [y[0][i_val], y[1][i_val]]
    ytrain_plot = [y[0][i_train], y[1][i_train]]

    # Convert back scaler and predict with new model
    out_model.save_weights(os.path.join(out_dir, "model_weights.h5"))
    out_model = SchnetEnergy(**model_config["config"])
    out_model.energy_only = False
    out_model.output_as_dict = True
    out_model.load_weights(os.path.join(out_dir, "model_weights.h5"))

    pval = out_model.predict_to_numpy_output(out_model.predict(xval))
    ptrain = out_model.predict_to_numpy_output(out_model.predict(xtrain))
    _, pval = scaler.inverse_transform(y=[pval['energy'], pval['force']])
    _, ptrain = scaler.inverse_transform(y=[ptrain['energy'], ptrain['force']])

    print("Info: Predicted Energy shape:", ptrain[0].shape)
    print("Info: Predicted Gradient shape:", ptrain[1].shape)
    print("Info: Plot fit stats...")

    # Plot
    plot_loss_curves(hist.history['mean_absolute_error'],
                     hist.history['val_mean_absolute_error'],
                     val_step=epostep, save_plot_to_file=True, dir_save=dir_save,
                     filename='fit' + str(i), filetypeout='.png', unit_loss=unit_label_energy, loss_name="MAE",
                     plot_title="Energy")

    plot_learning_curve(hist.history['lr'], filename='fit' + str(i), dir_save=dir_save)

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

    error_val = [np.mean(np.abs(pval[0] - y[0][i_val])), np.mean(np.abs(pval[1] - y[1][i_val]))]
    error_train = [np.mean(np.abs(ptrain[0] - y[0][i_train])), np.mean(np.abs(ptrain[1] - y[1][i_train]))]
    print("error_val:", error_val)
    print("error_train:", error_train)
    error_dict = {"train": [error_train[0].tolist(), error_train[1].tolist()],
                  "valid": [error_val[0].tolist(), error_val[1].tolist()]}
    save_json_file(error_dict, os.path.join(out_dir, "fit_error.json"))

    print("Info: Saving model to file...")
    out_model.save_weights(os.path.join(out_dir, "model_weights.h5"))
    out_model.save(os.path.join(out_dir, "model_tf"))

    return error_val


if __name__ == "__main__":
    print("Training Model: ", args['filepath'])
    print("Network instance: ", args['index'])
    out = train_model_energy_gradient(args['index'], args['filepath'], args['mode'])

fstdout.close()