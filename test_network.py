"""
SEQ2SEQ IMAGE CAPTIONING
Tom Sherborne 8/5/18
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os, time, sys

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
tf.flags.DEFINE_string("log_dir", "./models/exp7", "Directory location for logging")
tf.flags.DEFINE_string("dtype", "agreement", "Shapeworld Data Type")
tf.flags.DEFINE_string("name", "simple", "Shapeworld Data Name")
tf.flags.DEFINE_string("data_partition", "validation", "Which part of the dataset to test using")
tf.flags.DEFINE_string("parse_type", "", "shape, color or shape_color for input data formatting")
tf.flags.DEFINE_string("exp_tag", "", "Subfolder labelling under log_dir for this experiment")
tf.flags.DEFINE_integer("num_imgs", 1000, "How many images to test with")

tf.logging.set_verbosity(tf.logging.INFO)

def main(_):
    # FILESYSTEM SETUP ------------------------------------------------------------
    
    assert FLAGS.data_dir, "Must specify data location!"
    assert FLAGS.log_dir, "Must specify experiment to log to!"
    assert FLAGS.exp_tag, "Must specify experiment tag subfolder to log_dir %s" % FLAGS.log_dir
    assert FLAGS.parse_type in ["shape", "color", "shape_color"], "Must specify a valid batch parser type"

    # Build saving folders
    save_root = FLAGS.log_dir + os.sep + FLAGS.exp_tag
    test_path = save_root + os.sep + "test"

    # Sanity check
    tf.reset_default_graph()

    tf.logging.info("Clean graph reset...")

    try:
        dataset = Dataset.create(dtype=FLAGS.dtype, name=FLAGS.name, config=FLAGS.data_dir)
        dataset.pixel_noise_stddev = 0.1
    except Exception:
        raise ValueError("config=%s did not point to a valid Shapeworld dataset" % FLAGS.data_dir)

    # Get parsing and parameter feats
    params = Config(mode="test", sw_specification=dataset.specification())
    
    # MODEL SETUP ------------------------------------------------------------
    
    g = tf.Graph()
    with g.as_default():
        parser = SimpleBatchParser(batch_type=FLAGS.parse_type)
        vocab, rev_vocab = parser.get_vocab()
        params.vocab_size = len(parser.tgt_vocab)
        
        batch = tf_util.batch_records(dataset, mode=FLAGS.data_partition, batch_size=params.batch_size)
        
        model = CaptioningModel(config=params, batch_parser=parser)
        model.build_model(batch)
        
        restore_model = tf.train.Saver()

        tf.logging.info("Network built...")
        
    # TESTING SETUP ------------------------------------------------------------

    if FLAGS.num_imgs < 1:
        num_imgs = params.instances_per_shard * params.num_shards
    else:
        num_imgs = FLAGS.num_imgs
    
    tf.logging.info("Running test for %d images", num_imgs)

    test_writer = tf.summary.FileWriter(logdir=test_path, graph=g)

    #  Find model checkpoint
    train_path = FLAGS.log_dir + os.sep + FLAGS.exp_tag + os.sep + "train"
    model_ckpt = tf.train.latest_checkpoint(train_path)

    assert model_ckpt, "Checkpoints could not be loaded, check that train_path %s exists" % train_path

    tf.logging.info("Loading checkpoint %s", model_ckpt)

    with tf.Session(graph=g, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        coordinator = tf.train.Coordinator()
        queue_threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

        restore_model.restore(sess, model_ckpt)
        tf.logging.info("Model restored!")

        # Initialise everything
        sess.run([tf.tables_initializer()])

        #  Freeze graph
        sess.graph.finalize()
    
        # Get global step
        global_step = tf.train.global_step(sess, model.global_step)
        tf.logging.info("Successfully loaded %s at global step = %d.",
                        os.path.basename(model_ckpt), global_step)

        start_train_time = time.time()

        correct_accumulator = []
        loss_accumulator = []
        
        for b_idx in trange(num_imgs):
            reference_caps, inf_decoder_outputs, loss_ = sess.run(fetches=[model.reference_captions,
                                                                           model.inf_decoder_output,
                                                                           model.batch_loss],
                                                                  feed_dict={model.phase : 0})
            ref_cap = reference_caps.squeeze()
            inf_cap = inf_decoder_outputs.sample_id.squeeze()
            print(b_idx)
            print("REF -> %s | INF -> %s" %
                  (" ".join(rev_vocab[r] for r in ref_cap), " ".join(rev_vocab[r] for r in inf_cap)))
            
            correct_cap = inf_cap[1:-1] == ref_cap[1:-1]
            correct_accumulator.append(int(correct_cap))
            loss_accumulator += loss_
            import pdb;pdb.set_trace()

        avg_loss = np.mean(loss_accumulator).squeeze()
        std_loss = np.std(loss_accumulator).squeeze()
        avg_acc = np.mean(correct_accumulator).squeeze()
        std_acc = np.std(correct_accumulator).squeeze()
        
        new_summ = tf.Summary()
        new_summ.value.add(tag="test/avg_loss", simple_value=avg_loss)
        new_summ.value.add(tag="test/std_loss", simple_value=std_loss)
        new_summ.value.add(tag="test/avg_loss_%s" % FLAGS.parse_type, simple_value=avg_acc)
        new_summ.value.add(tag="test/std_loss_%s" % FLAGS.parse_type, simple_value=std_acc)
        test_writer.add_summary(new_summ, tf.train.global_step(sess, model.global_step))
        test_writer.flush()
        
        coordinator.request_stop()
        coordinator.join(threads=queue_threads)

        end_time = time.time()-start_train_time
        tf.logging.info('Training complete in %.2f-secs/%.2f-mins/%.2f-hours', end_time, end_time/60, end_time/(60*60))


if __name__=="__main__":
    tf.app.run()