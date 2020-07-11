from __future__ import print_function

import argparse
import imageio
import tqdm
from model_eager import PSPNet50
from tools import *
from pathlib import Path

ADE20k_param = {'crop_size': [473, 473],
                'num_classes': 150,
                'model': PSPNet50}

CHECKPOINTS_DIR = 'C:/Users/CHIOATH/Downloads/checkpoint.npy'

def get_arguments():
    parser = argparse.ArgumentParser(description="Reproduced PSPNet")
    parser.add_argument("--data_path", type=str, default='',
                        help="Path to the datasets' directory.",
                        required=True)
    parser.add_argument("--checkpoints", type=str, default=CHECKPOINTS_DIR,
                        help="Path to restore weights.")
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


def _read_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    return tf.cast(image, tf.float32)

def _preprocess(image, resizing=[473, 473]):
    # image = tf.image.rgb_to_grayscale(image)
    image = tf.expand_dims(image, 0)
    target_size = resizing
    # if resizing:
        # image = tf.image.resize(image, target_size,
        #                         method=tf.image.ResizeMethod.GAUSSIAN,
        #                         preserve_aspect_ratio=True)
    image = tf.image.resize_with_crop_or_pad(image, target_size[0], target_size[1])
    img_r, img_g, img_b = tf.split(axis=3, num_or_size_splits=3, value=image)
    
    image = tf.cast(tf.concat(axis=3, values=[img_b, img_g, img_r]), dtype=tf.float32)
    # Extract mean.
    image -= IMG_MEAN

    return image

class InferenceFilter(tf.keras.Model):
    def __init__(self, img_shape, num_classes):
        super(InferenceFilter, self).__init__(name='InferenceFilter')
        self.img_shape = img_shape
        self.num_classes = num_classes
        crop_size = ADE20k_param['crop_size']
        self.h, self.w = (tf.maximum(crop_size[0], img_shape[0]), tf.maximum(crop_size[1], img_shape[1]))
    
    def call(self, raw_output):
        raw_output_up = tf.compat.v1.image.resize_bilinear(raw_output, size=[self.h, self.w], align_corners=True)
        raw_output_up = tf.image.crop_to_bounding_box(raw_output_up, 0, 0, self.img_shape[0], self.img_shape[1])
        raw_output_up_print = tf.argmax(raw_output_up, axis=3)
        pred = tf.image.convert_image_dtype(decode_labels(raw_output_up_print, self.img_shape, self.num_classes), dtype=tf.uint8)
        return raw_output_up_print, pred

def main():
    args = get_arguments()

    # disable eager execution
    # tf.compat.v1.disable_eager_execution()

    # load parameters
    param = ADE20k_param

    crop_size = param['crop_size']
    num_classes = param['num_classes']
    PSPNet = param['model']

    # load dataset images' paths
    if args.dataset == 'coco':
        base_path = Path(args.data_path, 'COCO/train2014/')
        image_paths = list(base_path.iterdir())
        image_paths = [str(p) for p in image_paths]
    elif args.dataset == 'megadepth':
        image_paths = []
        base_path = Path(args.data_path, 'megadepth/phoenix/S6/zl548/MegaDepth_v1/')
        for sub_dir in list(base_path.iterdir()):
            num_dir = base_path / sub_dir
            for sub_dir2 in list(num_dir.iterdir()):
                dense_dir = num_dir / sub_dir2
                imgs_path = dense_dir / 'imgs'
                if imgs_path.exists():
                    image_paths.extend([x for x in imgs_path.iterdir() if not os.fspath(x).startswith(".")])

        image_paths = [str(p) for p in image_paths]
    
    save_dir = os.path.join(args.data_path, args.dataset, 'semantics')
    if(not os.path.exists(save_dir)):
        os.makedirs(save_dir)
    paths = tf.data.Dataset.from_tensor_slices(image_paths)

    num_images = len(image_paths)
    names = tf.data.Dataset.from_tensor_slices([os.path.join(save_dir, os.path.basename(str(p))) for p in image_paths])
    
    images = paths.map(_read_image)
    images = images.map(_preprocess)
    # images = images.map(lambda im: preprocess_eager(im, ADE20k_param['crop_size'][0], ADE20k_param['crop_size'][1]))
    
    dataset = tf.data.Dataset.zip((images, names))
    # # for image_path in image_paths:
        # tf.compat.v1.reset_default_graph()  # reset graph

        # # preprocess images
        # img, filename = load_img(os.fspath(image_path))
        # img_shape = tf.shape(img)
        # h, w = (tf.maximum(crop_size[0], img_shape[0]), tf.maximum(crop_size[1], img_shape[1]))
        # img = preprocess(img, h, w)

    # Create network.

    net = PSPNet50(num_classes=num_classes, checkpoint_npy_path=args.checkpoints)
    
    # tf.keras.Sequential([
    #     ,
    #     InferenceFilter(ADE20k_param['crop_size'], num_classes)
    # ])
    net.trainable = False
    output_filter = InferenceFilter(ADE20k_param['crop_size'], num_classes)
    # if args.flipped_eval:
    #     net2 = PSPNet50(num_classes=num_classes,checkpoint_npy_path=args.checkpoints)

    # save_dir = image_path.parent.parent / 'semantic_labels'
    # if not os.path.exists(os.fspath(save_dir)):
    #     os.makedirs(os.fspath(save_dir))

    # preds = net.predict(images)
    bar_index = tqdm.tqdm(range(num_images))
    for image, name in dataset:
        raw_output = net.predict_on_batch(image)
        bar_index.update(1)
        label_matrix, pred = output_filter(raw_output)
        
        imageio.imwrite(name.numpy().decode('utf-8'), pred[0])
        # 
        # print(label_matrix)
        # np.save(os.fspath(save_dir) + '/' + str(img_name) + '_labels.npy', label_matrix)
    # print(
    #     net.predict(images))


        # # Init tf Session
        # config = tf.compat.v1.ConfigProto()
        # config.gpu_options.allow_growth = True
        # sess = tf.compat.v1.Session(config=config)
        # init = tf.compat.v1.global_variables_initializer()

        # sess.run(init)

        # restore_var = tf.compat.v1.global_variables()



        # # label_matrix, preds = sess.run([raw_output_up_print, pred])
        # label_matrix = sess.run(raw_output_up_print)


        # print(os.fspath(save_dir))
        # # 
        # # Save labels of pixels
        # img_name = filename.split('.')[0]

        # print(os.fspath(save_dir) + '/' + str(img_name) + '_labels.npy')
        # 


if __name__ == '__main__':
    main()
