# -*- coding: utf-8 -*-

import tensorflow as tf


tf.app.flags.DEFINE_string("train_dir", "./model", "Training directory")
tf.app.flags.DEFINE_string("log_dir", "./logs", "Log directory")
tf.app.flags.DEFINE_string("ckpt_name", "conversation.ckpt", "Checkpoint name")

tf.app.flags.DEFINE_boolean("train", False, "Start training")
tf.app.flags.DEFINE_boolean("test", True, "Run a self test")
tf.app.flags.DEFINE_boolean("data_loop", True, "For experiment with small sets")
tf.app.flags.DEFINE_integer("batch_size", 100, "batch size")
tf.app.flags.DEFINE_integer("epoch", 1000, "max epoch for training")

tf.app.flags.DEFINE_string("data_path", "./data/chat.log", "Conversation data path")
tf.app.flags.DEFINE_string("voc_path", "./data/chat.voc", "Vocabulary ids path")
tf.app.flags.DEFINE_boolean("voc_test", False, "Vocabulary test with conversation data")
tf.app.flags.DEFINE_boolean("voc_build", False, "Vocabulary build from conversation data")

tf.app.flags.DEFINE_integer("max_decode_len", 20, "Max decode/reply length.")


FLAGS = tf.app.flags.FLAGS
