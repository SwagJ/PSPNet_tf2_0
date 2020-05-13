import numpy as np
import tensorflow as tf

def main():
    ckpt = tf.train.get_checkpoint_state('./checkpoint')
    checkpoint_path = ckpt.model_checkpoint_path
    reader = tf.compat.v1.train.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()

    parameters = {}

    for key in var_to_shape_map:

        str_name = key

        if str_name.find('/') > -1:
            names = str_name.split('/')
            layer_name = names[0]
            if names[1] == 'weights':
                layer_info = 'kernel'
            else:
                layer_info = names[1]
        else:
            layer_name = str_name
            layer_info = None

        key_dict = layer_name + '/' + layer_info

        if key_dict not in parameters.keys():
            parameters[key_dict] = reader.get_tensor(key)
        else:
            raise Exception("Same key in the same network!")

    np.save('checkpoint.npy', parameters)

if __name__ == '__main__':
    main()