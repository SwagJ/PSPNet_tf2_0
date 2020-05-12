from __future__ import print_function

import argparse
import imageio
from model import PSPNet50
from tools import *
from pathlib import Path

ADE20k_param = {'crop_size': [473, 473],
                'num_classes': 150,
                'model': PSPNet50}

CHECKPOINTS_DIR = './model'

def get_arguments():
    parser = argparse.ArgumentParser(description="Reproduced PSPNet")
    parser.add_argument("--data_path", type=str, default='',
                        help="Path to the datasets' directory.")
    parser.add_argument("--flipped_eval", action="store_true",
                        help="whether to evaluate with flipped img.")
    parser.add_argument("--dataset", type=str, default='',
                        choices=['megadepth', 'coco'],
                        required=True)

    return parser.parse_args()


def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    saver.save(sess, checkpoint_path, global_step=step)
    print('The checkpoint has been created.')


def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))


def main():
    args = get_arguments()

    # disable eager execution
    tf.compat.v1.disable_eager_execution()

    # load parameters
    param = ADE20k_param

    crop_size = param['crop_size']
    num_classes = param['num_classes']
    PSPNet = param['model']

    # load dataset images' paths
    if args.dataset == 'coco':
        base_path = Path(args.data_path, 'COCO/train2014/')
        image_paths = list(base_path.iterdir())
    elif args.dataset == 'megadepth':
        image_paths = []
        base_path = Path(args.data_path, 'megadepth/phoenix/S6/zl548/MegaDepth_v1/')
        for sub_dir in list(base_path.iterdir()):
            num_dir = base_path / sub_dir
            for sub_dir2 in list(num_dir.iterdir()):
                dense_dir = num_dir / sub_dir2
                imgs_path = dense_dir / 'imgs'
                if imgs_path.exists():
                    for p in list(imgs_path.iterdir()):
                        image_paths.append(p)

    for image_path in image_paths:
        print(image_path)
        tf.compat.v1.reset_default_graph()  # reset graph

        # preprocess images
        img, filename = load_img(os.fspath(image_path))
        img_shape = tf.shape(img)
        h, w = (tf.maximum(crop_size[0], img_shape[0]), tf.maximum(crop_size[1], img_shape[1]))
        img = preprocess(img, h, w)

        # Create network.
        net = PSPNet({'data': img}, is_training=False, num_classes=num_classes)
        with tf.compat.v1.variable_scope('', reuse=True):
            flipped_img = tf.image.flip_left_right(tf.squeeze(img))
            flipped_img = tf.expand_dims(flipped_img, 0)
            net2 = PSPNet({'data': flipped_img}, is_training=False, num_classes=num_classes)

        raw_output = net.layers['conv6']

        # Do flipped eval or not
        if args.flipped_eval:
            flipped_output = tf.image.flip_left_right(tf.squeeze(net2.layers['conv6']))
            flipped_output = tf.expand_dims(flipped_output, dim=0)
            raw_output = tf.add_n([raw_output, flipped_output])

        # Predictions.
        raw_output_up = tf.compat.v1.image.resize_bilinear(raw_output, size=[h, w], align_corners=True)
        raw_output_up = tf.image.crop_to_bounding_box(raw_output_up, 0, 0, img_shape[0], img_shape[1])
        raw_output_up = tf.argmax(raw_output_up, axis=3)
        pred = decode_labels(raw_output_up, img_shape, num_classes)

        # Init tf Session
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.compat.v1.Session(config=config)
        init = tf.compat.v1.global_variables_initializer()

        sess.run(init)

        restore_var = tf.compat.v1.global_variables()

        ckpt = tf.train.get_checkpoint_state(CHECKPOINTS_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            loader = tf.compat.v1.train.Saver(var_list=restore_var)
            load(loader, sess, ckpt.model_checkpoint_path)
        else:
            print('No checkpoint file found.')

        preds = sess.run(pred)

        save_dir = image_path.parent.parent / 'labels'
        if not os.path.exists(os.fspath(save_dir)):
            os.makedirs(os.fspath(save_dir))
        print(os.fspath(save_dir))
        imageio.imwrite(os.fspath(save_dir) + '/' + filename, preds[0])


if __name__ == '__main__':
    main()
