"""
Tensorflow keras model definitions for energy and gradient.

There are two definitions: the subclassed EnergyModel and a precomputed model to 
train energies. The subclassed Model will also predict gradients.
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras as ks

from pyNNsMD.layers.features import FeatureGeometric
from pyNNsMD.layers.mlp import MLP
from pyNNsMD.layers.normalize import ConstLayerNormalization
from pyNNsMD.scaler.general import SegmentStandardScaler


class EnergyModel(ks.Model):
    """
    Subclassed tf.keras.model for energy/gradient which outputs both energy and gradient from coordinates.
    
    It can also
    """

    def __init__(self,
                 states=1,
                 atoms=2,
                 invd_index=None,
                 angle_index=None,
                 dihed_index=None,
                 nn_size=100,
                 depth=3,
                 activ='selu',
                 use_reg_activ=None,
                 use_reg_weight=None,
                 use_reg_bias=None,
                 use_dropout=False,
                 dropout=0.01,
                 **kwargs):
        """
        Initialize an EnergyModel with hyperparameters.

        Args:
            hyper (dict): Hyperparamters.
            **kwargs (dict): Additional keras.model parameters.

        Returns:
            tf.keras.model.
            
        """
        super(EnergyModel, self).__init__(**kwargs)
        out_dim = int(states)
        indim = int(atoms)
        self.out_dim = out_dim

        # Allow for all distances, backward compatible
        if isinstance(invd_index, bool):
            if invd_index:
                invd_index = [[i, j] for i in range(0, int(atoms)) for j in range(0, i)]

        use_invd_index = len(invd_index) > 0 if isinstance(invd_index, list) or isinstance(invd_index,
                                                                                           np.ndarray) else False
        use_angle_index = len(angle_index) > 0 if isinstance(angle_index, list) or isinstance(angle_index,
                                                                                              np.ndarray) else False
        use_dihed_index = len(dihed_index) > 0 if isinstance(dihed_index, list) or isinstance(dihed_index,
                                                                                              np.ndarray) else False

        invd_index = np.array(invd_index, dtype=np.int64) if use_invd_index else None
        angle_index = np.array(angle_index, dtype=np.int64) if use_angle_index else None
        dihed_index = np.array(dihed_index, dtype=np.int64) if use_dihed_index else None

        invd_shape = invd_index.shape if use_invd_index else None
        angle_shape = angle_index.shape if use_angle_index else None
        dihed_shape = dihed_index.shape if use_dihed_index else None

        self.feat_layer = FeatureGeometric(invd_shape=invd_shape,
                                           angle_shape=angle_shape,
                                           dihed_shape=dihed_shape,
                                           name="feat_geo"
                                           )
        self.feat_layer.set_mol_index(invd_index, angle_index, dihed_index)

        self.std_layer = ConstLayerNormalization(axis=-1, name='feat_std')
        self.mlp_layer = MLP(nn_size,
                             dense_depth=depth,
                             dense_bias=True,
                             dense_bias_last=True,
                             dense_activ=activ,
                             dense_activ_last=activ,
                             dense_activity_regularizer=use_reg_activ,
                             dense_kernel_regularizer=use_reg_weight,
                             dense_bias_regularizer=use_reg_bias,
                             dropout_use=use_dropout,
                             dropout_dropout=dropout,
                             name='mlp'
                             )
        self.energy_layer = ks.layers.Dense(out_dim, name='energy', use_bias=True, activation='linear')
        self.precomputed_features = False
        self.energy_only = True

        self.build((None, indim, 3))

    def call(self, data, training=False, **kwargs):
        """
        Call the model output, forward pass.

        Args:
            data (tf.tensor): Coordinates.
            training (bool, optional): Training Mode. Defaults to False.

        Returns:
            y_pred (list): List of tf.tensor for predicted [energy,gradient]

        """
        # Unpack the data
        x = data
        y_pred = None
        # Compute predictions
        if self.energy_only and not self.precomputed_features:
            feat_flat = self.feat_layer(x)
            feat_flat_std = self.std_layer(feat_flat)
            temp_hidden = self.mlp_layer(feat_flat_std, training=training)
            temp_e = self.energy_layer(temp_hidden)
            y_pred = temp_e
        elif not self.energy_only and not self.precomputed_features:
            with tf.GradientTape() as tape2:
                tape2.watch(x)
                feat_flat = self.feat_layer(x)
                feat_flat_std = self.std_layer(feat_flat)
                temp_hidden = self.mlp_layer(feat_flat_std, training=training)
                temp_e = self.energy_layer(temp_hidden)
            temp_g = tape2.batch_jacobian(temp_e, x)
            y_pred = [temp_e, temp_g]
        elif self.precomputed_features:
            feat_flat_std = self.std_layer(x)
            temp_hidden = self.mlp_layer(feat_flat_std, training=training)
            temp_e = self.energy_layer(temp_hidden)
            y_pred = temp_e

        return y_pred

    @tf.function
    def predict_chunk_feature(self, tf_x):
        with tf.GradientTape() as tape2:
            tape2.watch(tf_x)
            feat_pred = self.feat_layer(tf_x, training=False)  # Forward pass
        grad = tape2.batch_jacobian(feat_pred, tf_x)
        return feat_pred, grad

    def precompute_feature_in_chunks(self, x, batch_size, normalization_mode=1):
        np_x = []
        np_grad = []
        for j in range(int(np.ceil(len(x) / batch_size))):
            a = int(batch_size * j)
            b = int(batch_size * j + batch_size)
            tf_x = tf.convert_to_tensor(x[a:b], dtype=tf.float32)
            feat_pred, grad = self.predict_chunk_feature(tf_x)
            np_x.append(np.array(feat_pred.numpy()))
            np_grad.append(np.array(grad.numpy()))

        np_x = np.concatenate(np_x, axis=0)
        np_grad = np.concatenate(np_grad, axis=0)

        self.set_const_normalization_from_features(np_x, normalization_mode=normalization_mode)
        return np_x, np_grad

    def set_const_normalization_from_features(self, feat_x, normalization_mode=1):

        feat_x_mean, feat_x_std = self.get_layer('feat_std').get_weights()
        if normalization_mode == 1:
            feat_x_mean = np.mean(feat_x, axis=0, keepdims=True)
            feat_x_std = np.std(feat_x, axis=0, keepdims=True)
        elif normalization_mode == 2:
            seg_scaler = SegmentStandardScaler(self.get_layer('feat_geo').get_feature_type_segmentation())
            seg_scaler.fit(y=feat_x)
            feat_x_mean, feat_x_std = np.array(seg_scaler.get_params()["feat_mean"]), np.array(
                seg_scaler.get_params()["feat_std"])

        self.get_layer('feat_std').set_weights([feat_x_mean, feat_x_std])
