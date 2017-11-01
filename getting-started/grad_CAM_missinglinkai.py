from tensorflow.contrib.slim.nets import vgg

import tensorflow as tf
import missinglink


input_placeholder = tf.placeholder(tf.float32, [None, 224, 224, 3])

logits, end_points = vgg.vgg_16(inputs=input_placeholder)
probs = tf.nn.softmax(logits)

session = tf.Session()
# Download pre-trained checkpoint from http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
tf.train.Saver().restore(session, 'vgg_16.ckpt')

OWNER_ID = 'replace me with owner id'
PROJECT_TOKEN = 'replace me with project token'

missinglink_project = missinglink.TensorFlowProject(OWNER_ID, PROJECT_TOKEN)

path = 'http://cmeimg-a.akamaihd.net/640/photos.demandstudios.com' + \
    '/getty/article/103/49/516464087.jpg'

with missinglink_project.create_experiment() as experiment:
    experiment.set_properties(
        input_placeholder=input_placeholder.op.name,
        output_layer=probs.op.name,
        last_conv_layer=end_points['vgg_16/conv5/conv5_3'].op.name
    )

    experiment.generate_grad_cam(path, session)
