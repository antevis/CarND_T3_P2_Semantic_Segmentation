import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
from tqdm import tqdm



# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

"""
# PERSISTANCE
lossStorage = "./model/loss.p"
"""


# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    tf.saved_model.loader.load(sess, ['vgg16'], vgg_path)
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    graph = tf.get_default_graph()
    input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    l3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    l4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    l7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return input, keep_prob, l3_out, l4_out, l7_out

tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    l7_1x1 = helper.conv1x1(inputs=vgg_layer7_out, filters=num_classes)
    l4_1x1 = helper.conv1x1(inputs=vgg_layer4_out, filters=num_classes)
    l3_1x1 = helper.conv1x1(inputs=vgg_layer3_out, filters=num_classes)

    output = helper.convTranspose(inputs=l7_1x1, filters=num_classes, kernel_size=4, strides=2) # 1st upsampling
    output = tf.add(output, l4_1x1)                                                             # 1st skip connection
    output = helper.convTranspose(inputs=output, filters=num_classes, kernel_size=4, strides=2) # 2nd upsampling
    output = tf.add(output, l3_1x1)                                                             # 2nd skip connection

    # Final upsampling. Produces the dimensionality of the input image
    output = helper.convTranspose(inputs=output, filters=num_classes, kernel_size=16, strides=8)

    return output


tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, loss)
    """
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label = tf.reshape(correct_label, (-1, num_classes))

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    return logits, optimizer, loss

tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """

    """
    # PERSISTANCE
    saver = tf.train.Saver()

    if helper.fileExists(lossStorage):
        currentLoss = helper.loadFromFile(lossStorage)
    else:
        currentLoss = 1e2  # knowingly big number
    """


    for epoch in tqdm(range(epochs)):
        for batch, (image, label) in enumerate(get_batches_fn(batch_size)):

            feed_dict = {input_image: image, correct_label: label, keep_prob: 0.5, learning_rate: 1e-4}
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict=feed_dict)

            """
            # PERSISTANCE
            if loss < currentLoss:
                _ = saver.save(sess, "./model/vggfcn")
                helper.saveToFile(loss, lossStorage)
                currentLoss = loss
            """

            print('Epoch: {}, batch: {}, loss: {}'.format(epoch, batch, loss))


tests.test_train_nn(train_nn)


def run():

    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    # Path to vgg model
    vgg_path = os.path.join(data_dir, 'vgg')
    learning_rate = tf.placeholder(tf.float32)
    correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes])

    with tf.Session() as sess:

        input, keep_prob, l3, l4, l7 = load_vgg(sess, vgg_path)
        output = layers(l3, l4, l7, num_classes)
        logits, optimizer, loss = optimize(output, correct_label, learning_rate, num_classes)

        """
        # PERSISTANCE
        saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state("model")
        
        # Borrowed from Aaron Brown
        if checkpoint and checkpoint.model_checkpoint_path:

            options = ["y", "n"]
            loadResponse = helper.promptForInputCategorical("Trained model found. Should it be loaded?", options)

            if loadResponse == "y":

                saver.restore(sess, checkpoint.model_checkpoint_path)
                trainResponse = helper.promptForInputCategorical("Continue training loaded model?", options)

                if trainResponse == "y":
                    train(correct_label, data_dir, image_shape, input, keep_prob, learning_rate, loss, optimizer, sess,
                          resume=True)
            else:
                train(correct_label, data_dir, image_shape, input, keep_prob, learning_rate, loss, optimizer, sess)
        else:
        """
        train(correct_label, data_dir, image_shape, input, keep_prob, learning_rate, loss, optimizer, sess)

        # OPTIONAL: Apply the trained model to a video
        outputResponse = helper.promptForInputCategorical("Save (s)amples or process (v)ideo?", ["s", "v"])
        if outputResponse == "s":
            helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input)
        else:
            helper.saveVideo(sess, image_shape, logits, keep_prob, input)


def train(correct_label, data_dir, image_shape, input, keep_prob, learning_rate, loss, optimizer, sess, resume=False):
    # Create function to get batches
    get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)
    # OPTIONAL: Augment Images for better results
    #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network
    epochs = 30
    batch_size = 8

    if not resume:
        sess.run(tf.global_variables_initializer())
    train_nn(sess, epochs, batch_size, get_batches_fn, optimizer, loss, input,
             correct_label, keep_prob, learning_rate)

    print("Training complete.")


if __name__ == '__main__':
    run()
