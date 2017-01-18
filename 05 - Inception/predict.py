# -*- coding: utf-8 -*-

import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys


tf.app.flags.DEFINE_string("output_graph", "./workspace/flowers_graph.pb", "Trained graph path")
tf.app.flags.DEFINE_string("output_labels", "./workspace/flowers_labels.txt", "Trained graph's label path")
tf.app.flags.DEFINE_boolean("show_image", True, "Show image after predict.")

FLAGS = tf.app.flags.FLAGS


class Inception:

    def __init__(self, graph_path, label_path):
        self.labels = [line.rstrip() for line in tf.gfile.GFile(label_path)]

        with tf.gfile.FastGFile(graph_path, 'rb') as fp:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(fp.read())
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.Session()
        self.logits = self.sess.graph.get_tensor_by_name('final_result:0')

    def predict(self, image_path, top=5):
        image = tf.gfile.FastGFile(image_path, 'rb').read()

        prediction = self.sess.run(self.logits, {'DecodeJpeg/contents:0': image})

        top_indices = prediction[0].argsort()[::-1][:top]

        for i in top_indices:
            name = self.labels[i]
            score = prediction[0][i]
            print '%s (%.2f%%)' % (name, score * 100)


def main(_):
    if len(sys.argv) < 2:
        print 'Usage: predict.py image_path'

    else:
        inception = Inception(FLAGS.output_graph, FLAGS.output_labels)

        inception.predict(sys.argv[1])

        if FLAGS.show_image:
            img = mpimg.imread(sys.argv[1])
            plt.imshow(img)
            plt.show()


if __name__ == "__main__":
    tf.app.run()
