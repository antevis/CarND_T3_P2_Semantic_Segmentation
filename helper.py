import re
import random
import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
import time
import tensorflow as tf
from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm
import pickle
from moviepy.editor import ImageSequenceClip, VideoFileClip


class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def maybe_download_pretrained_vgg(data_dir):
    """
    Download and extract pretrained vgg model if it doesn't exist
    :param data_dir: Directory to download the model to
    """
    vgg_filename = 'vgg.zip'
    vgg_path = os.path.join(data_dir, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        # Download vgg
        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                os.path.join(vgg_path, vgg_filename),
                pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(vgg_path, vgg_filename))


def gen_batch_function(data_folder, image_shape):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """

    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
        label_paths = {
            re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
            for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}
        background_color = np.array([255, 0, 0])

        random.shuffle(image_paths)
        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            gt_images = []
            for image_file in image_paths[batch_i:batch_i + batch_size]:
                gt_image_file = label_paths[os.path.basename(image_file)]

                image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
                gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)

                gt_bg = np.all(gt_image == background_color, axis=2)
                gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
                gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)

                images.append(image)
                gt_images.append(gt_image)

            yield np.array(images), np.array(gt_images)

    return get_batches_fn


def gen_test_output(sess, logits, keep_prob, image_pl, data_folder, image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    for image_file in glob(os.path.join(data_folder, 'image_2', '*.png')):
        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)

        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, image_pl: [image]})
        im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
        segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)

        yield os.path.basename(image_file), np.array(street_im)


def save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(
        sess, logits, keep_prob, input_image, os.path.join(data_dir, 'data_road/testing'), image_shape)
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)


"""
Extended functionality
"""
def saveVideo(sess, image_shape, logits, keep_prob, image_pl):
    """
    Embeds predicted drivable area into a video stream
    :param sess: TF session
    :param image_shape: input image shape
    :param logits: TF tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :return: void
    """
    fileName = input("Please enter file name: ")
    sourceVideoFile = "./data/video/{}".format(fileName)
    clip = VideoFileClip(sourceVideoFile)

    resultFrames = []

    for frame in tqdm(clip.iter_frames()):

        frame = frame[180:540,:,:]
        smallFrame = scipy.misc.imresize(frame, image_shape)
        dst = scipy.misc.toimage(frame)

        prediction = sess.run([tf.nn.softmax(logits)], {keep_prob: 1.0, image_pl: [smallFrame]})

        prediction = prediction[0][:, 1].reshape(image_shape[0], image_shape[1])
        segmentation = (prediction > 0.5).reshape(image_shape[0], image_shape[1], 1)
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))

        # Up-sampling mask to the frame's dimensions
        mask = scipy.misc.imresize(mask, (360, 1280))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        dst.paste(mask, box=None, mask=mask)

        resultFrames.append(np.array(dst))

    resultClip = ImageSequenceClip(resultFrames, fps=25, with_mask=False)

    resultFileName = "./data/video/{}_out.mp4".format(fileName.split('.')[0])
    resultClip.write_videofile(resultFileName, progress_bar=True)


def convTranspose(inputs, filters, kernel_size, strides, pad='same', k_init_std=1e-2, k_reg_scale=1e-3, name=None):
    """
    Wrapper for tf.layers.conv2d_transpose with some reasonable defaults
    :param inputs: Input tensor.
    :param filters: Integer, the dimensionality of the output space
    :param kernel_size:  A tuple or list of 2 positive integers specifying the spatial dimensions of the filters.
    Can be a single integer to specify the same value for all spatial dimensions.
    :param strides: A tuple or list of 2 positive integers specifying the strides of the convolution.
    Can be a single integer to specify the same value for all spatial dimensions.
    :param pad: one of "valid" or "same" (case-insensitive).
    :param k_init_std: standard deviation for kernel_initializer
    :param k_reg_scale: scale for kernel_regualarizer
    :return: Output tensor
    """
    return tf.layers.conv2d_transpose(inputs, filters, kernel_size, strides, pad,
                                      kernel_initializer=tf.truncated_normal_initializer(stddev=k_init_std),
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=k_reg_scale),
                                      name=name)


def conv1x1(inputs, filters, pad='same', k_init_std=1e-2, k_reg_scale=1e-3, name=None):
    """
    1x1 2D Convolution version for tf.layers.conv2d
    :param inputs: Input tensor
    :param filters: Integer, the dimensionality of the output space
    :param pad: one of "valid" or "same" (case-insensitive).
    :param k_init_std: standard deviation for kernel_initializer
    :param k_reg_scale: scale for kernel_regualarizer
    :return: Output tensor
    """
    return tf.layers.conv2d(inputs, filters, 1, padding=pad,
                            kernel_initializer=tf.truncated_normal_initializer(stddev=k_init_std),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=k_reg_scale), name=name)

# Three methods below used for loss tracking and preserving between sessions
def saveToFile(data, filename):
    pickle.dump(data, open(filename, 'wb'))


def fileExists(fullPath):
    return os.path.exists(fullPath)


def loadFromFile(fileName):
    with open(fileName, mode='rb') as f:
        data = pickle.load(f)
    return data

# Doesn't seem to help a lot
def augment_brightness(image):

    random_brightness = np.random.uniform(0.75, 1.25)

    def clamp(a):
        return min(255, a * random_brightness)

    vfunc = np.vectorize(clamp)
    image = vfunc(image)

    return image

def promptForInputCategorical(message, options):
    """
    Prompts for user input with limited number of options
    :param message: Message displayed to the user
    :param options: limited number of options.
    Prompt will repeat until user input matches one of the provided options.
    :return: user response
    """
    response = ''

    options_list = ', '.join(options)

    while response not in options:
        response = input('{} ({}): '.format(message, options_list))

    return response



