from tensorflow.contrib.slim.nets import vgg

import tensorflow as tf
import missinglink


input_placeholder = tf.placeholder(tf.float32, [None, 224, 224, 3])

logits, end_points = vgg.vgg_16(inputs=input_placeholder)

session = tf.Session()
# Download pre-trained checkpoint from http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
tf.train.Saver().restore(session, 'vgg_16.ckpt')

OWNER_ID = 'replace me with owner id'
PROJECT_TOKEN = 'replace me with project token'

missinglink_project = missinglink.TensorFlowProject(OWNER_ID, PROJECT_TOKEN)

path = 'http://l7.alamy.com/zooms/b76d255dd51e493e8c0fd5d5aa85f96f/lumbermill-cp93p7.jpg'

with missinglink_project.create_experiment() as experiment:
    experiment.set_properties(
        input_placeholder=input_placeholder.op.name,
        output_layer=logits.op.name
    )

    experiment.visual_back_prop(path, session)
