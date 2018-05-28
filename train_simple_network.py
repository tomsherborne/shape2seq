"""
SEQ2SEQ IMAGE CAPTIONING
Tom Sherborne 8/5/18
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os, time

import numpy as np
import tensorflow as tf
from tqdm import trange

seq2seq = tf.contrib.seq2seq

from shapeworld import Dataset, tf_util

from src.model import CaptioningModel
from src.glove_loader import GloveLoader
from src.batch_parser import SimpleBatchParser
from src.config import Config

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("data_dir", "", "Location of ShapeWorld data")
tf.flags.DEFINE_string("log_dir", "./models/final/short", "Directory location for logging")
tf.flags.DEFINE_string("cnn_ckpt", "", "Directory to load CNN checkpoint")
tf.flags.DEFINE_string("dtype", "agreement", "Shapeworld Data Type")
tf.flags.DEFINE_string("name", "simple", "Shapeworld Data Name")
tf.flags.DEFINE_string("parse_type", "", "shape, color or shape_color for input data formatting")
tf.flags.DEFINE_string("glove_dir", "", "Directory of GloVe embeddings to load")
tf.flags.DEFINE_integer("glove_dim", 50, "Dimensionality of GloVe embeddings")
tf.flags.DEFINE_integer("batch_size", 128, "Training batch size")
tf.flags.DEFINE_string("exp_tag", "", "Sub-folder labelling under log_dir for this experiment")

tf.logging.set_verbosity(tf.logging.INFO)


def main(_):
    # FILESYSTEM SETUP ------------------------------------------------------------
    assert FLAGS.data_dir, "Must specify data location!"
    assert FLAGS.log_dir, "Must specify experiment to log to!"
    assert FLAGS.exp_tag, "Must specify experiment tag subfolder to log_dir %s" % FLAGS.log_dir
    assert FLAGS.cnn_ckpt, "Must specify where to load CNN checkpoint from!"
    assert FLAGS.parse_type

    # Build saving folders
    save_root = FLAGS.log_dir + os.sep + FLAGS.exp_tag
    train_path = save_root + os.sep + "train"
    eval_path = save_root + os.sep + "eval"
    test_path = save_root + os.sep + "test"
    
    if not tf.gfile.IsDirectory(train_path):
        tf.gfile.MakeDirs(train_path)
        tf.gfile.MakeDirs(eval_path)
        tf.gfile.MakeDirs(test_path)
        
        tf.logging.info("Creating training directory: %s", train_path)
        tf.logging.info("Creating eval directory: %s", eval_path)
        tf.logging.info("Creating eval directory: %s", test_path)
    else:
        tf.logging.info("Using training directory: %s", train_path)
        tf.logging.info("Using eval directory: %s", eval_path)
    
    # Sanity check
    tf.reset_default_graph()
    tf.logging.info("Clean graph reset...")
    
    try:
        dataset = Dataset.create(dtype=FLAGS.dtype, name=FLAGS.name, config=FLAGS.data_dir)
        dataset.pixel_noise_stddev = 0.1
    except Exception:
        raise ValueError("config=%s did not point to a valid Shapeworld dataset" % FLAGS.data_dir)
    
    # Get parsing and parameter feats
    params = Config(mode="train", sw_specification=dataset.specification())
    params.cnn_checkpoint = FLAGS.cnn_ckpt
    params.batch_size = FLAGS.batch_size
    
    # MODEL SETUP ------------------------------------------------------------
    g = tf.Graph()
    with g.as_default():
        parser = SimpleBatchParser(src_vocab=dataset.vocabularies['language'], batch_type=FLAGS.parse_type)
        params.vocab_size = len(parser.tgt_vocab)
        
        batch = tf_util.batch_records(dataset, mode="train", batch_size=params.batch_size)
        
        model = CaptioningModel(config=params, batch_parser=parser)
        
        if FLAGS.glove_dir:
            tf.logging.info("Loading GloVe Embeddings...")
            gl = GloveLoader(vocab=parser.tgt_vocab, glove_dir=FLAGS.glove_dir, dims=FLAGS.glove_dim, load_new=False)
            glove_initials = gl.get_embeddings_matrix()
            tf.logging.info("Building model with GloVe initialisation...")
            model.build_model(batch, embedding_init=glove_initials)
        else:
            tf.logging.info("Building model without GloVe initialisation...")
            model.build_model(batch)
        tf.logging.info("Network built...")
        
        # TRAINING OPERATION SETUP ------------------------------------------------------------
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = tf.contrib.layers.optimize_loss(
                loss=model.batch_loss,
                global_step=model.global_step,
                learning_rate=params.initial_learning_rate,
                optimizer=params.optimizer,
                clip_gradients=params.clip_gradients,
            )
        
        logging_saver = tf.train.Saver(max_to_keep=params.max_checkpoints_to_keep)
        summary_op = tf.summary.merge_all()
    
    train_writer = tf.summary.FileWriter(logdir=train_path, graph=g)
    
    tf.logging.info('###' * 20)
    tf.logging.info("Begin shape2seq network training for %d steps" % params.num_total_steps)
    
    with tf.Session(graph=g) as sess:
        
        tf.logging.info("### Trainable Variables")
        for var in tf.trainable_variables():
            print("-> %s" % var.op.name)
        
        coordinator = tf.train.Coordinator()
        queue_threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)
        
        # Initialise everything
        sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
        
        tf.logging.info("Restoring CNN...")
        model.init_fn(sess)
        
        start_train_time = time.time()
        
        # Loss accumulator and logging interval generator at [25%, 50%, 75%, 100%] * epoch
        logging_loss = []
        logging_points = np.linspace(0, params.num_steps_per_epoch, 4, endpoint=False, dtype=np.int32)
        logging_points = np.fliplr([params.num_steps_per_epoch - logging_points])[0]
        
        for c_epoch in range(0, params.num_epochs):
            tf.logging.info("Running epoch %d" % c_epoch)
            for c_step in trange(0, params.num_steps_per_epoch):
                
                if c_step in logging_points:
                    _, loss_, summaries = sess.run(fetches=[train_op, model.batch_loss, summary_op])
                    
                    loss_ = logging_loss + [loss_]
                    logging_loss = []
                    
                    avg_loss = np.mean(loss_).squeeze()
                    new_summ = tf.Summary()
                    new_summ.value.add(tag="train/avg_loss", simple_value=avg_loss)
                    train_writer.add_summary(new_summ, tf.train.global_step(sess, model.global_step))
                    train_writer.add_summary(summaries, tf.train.global_step(sess, model.global_step))
                    train_writer.flush()
                    
                    tf.logging.info(" -> Average loss step %d, for last %d steps: %.5f" % (c_step,
                                                                                           len(loss_),
                                                                                           avg_loss))
                # Run without summaries
                else:
                    _, loss_, = sess.run(fetches=[train_op, model.batch_loss])
                    logging_loss.append(loss_)
            
            logging_saver.save(sess=sess,
                               save_path=train_path + os.sep + "model",
                               global_step=tf.train.global_step(sess, model.global_step))
        
        coordinator.request_stop()
        coordinator.join(threads=queue_threads)
        
        end_time = time.time() - start_train_time
        tf.logging.info('Training complete in %.2f-secs/%.2f-mins/%.2f-hours', end_time, end_time / 60,
                        end_time / (60 * 60))


if __name__ == "__main__":
    tf.app.run()
