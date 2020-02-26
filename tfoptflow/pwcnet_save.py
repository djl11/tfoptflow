# global
import tensorflow as tf
from copy import deepcopy
import argparse

# local
from tfoptflow.model_pwcnet import ModelPWCNet, _DEFAULT_PWCNET_VAL_OPTIONS

tf.enable_resource_variables()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-cp', '--checkpoint_path', type=str, required=True,
                        help='path of pwcnet checkpoint file')
    parser.add_argument('-smd', '--saved_model_dir', type=str, required=True,
                        help='directory of pwcnet tf saved model')
    ckpt_path = parser.parse_args().checkpoint_path
    saved_model_dir = parser.parse_args().saved_model_dir

    gpu_devices = ['/device:GPU:0']  # We're doing the evaluation on a single GPU
    controller = '/device:GPU:0'
    mode = 'val'  # We're doing the evaluation on the validation split of the dataset

    # Configure the model for evaluation, starting with the default evaluation options
    nn_opts = deepcopy(_DEFAULT_PWCNET_VAL_OPTIONS)
    nn_opts['verbose'] = True
    nn_opts['ckpt_path'] = ckpt_path
    nn_opts['batch_size'] = 1  # Setting this to 1 leads to more accurate evaluations of the processing time
    nn_opts['use_tf_data'] = False  # Don't use tf.data reader for this simple task
    nn_opts['gpu_devices'] = gpu_devices
    nn_opts['controller'] = controller  # Evaluate on CPU or GPU?

    # We're evaluating the PWC-Net-large model in quarter-resolution mode
    # That is, with a 6 level pyramid, and up-sampling of level 2 by 4 in each dimension as the final flow prediction
    nn_opts['use_dense_cx'] = True
    nn_opts['use_res_cx'] = True
    nn_opts['pyr_lvls'] = 6
    nn_opts['flow_pred_lvl'] = 2

    # Instantiate the model in evaluation mode and display the model configuration
    nn = ModelPWCNet(mode=mode, options=nn_opts)
    nn.save_model(saved_model_dir)


if __name__ == '__main__':
    main()
