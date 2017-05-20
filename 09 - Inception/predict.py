import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys


tf.app.flags.DEFINE_string("output_graph", "./workspace/flowers_graph.pb", "학습된 신경망이 저장된 위치")
tf.app.flags.DEFINE_string("output_labels", "./workspace/flowers_labels.txt", "학습할 레이블 데이터 파일")
tf.app.flags.DEFINE_boolean("show_image", True, "이미지 추론 후 이미지를 보여줍니다.")

FLAGS = tf.app.flags.FLAGS


def main(_):
    labels = [line.rstrip() for line in tf.gfile.GFile(FLAGS.output_labels)]

    with tf.gfile.FastGFile(FLAGS.output_graph, 'rb') as fp:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(fp.read())
        tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        logits = sess.graph.get_tensor_by_name('final_result:0')
        image = tf.gfile.FastGFile(sys.argv[1], 'rb').read()
        prediction = sess.run(logits, {'DecodeJpeg/contents:0': image})
        top_results = prediction[0].argsort()[::-1][:5]

    for i in top_results:
        name = labels[i]
        score = prediction[0][i]
        print('%s (%.2f%%)' % (name, score * 100))

    if FLAGS.show_image:
        img = mpimg.imread(sys.argv[1])
        plt.imshow(img)
        plt.show()


if __name__ == "__main__":
    tf.app.run()
