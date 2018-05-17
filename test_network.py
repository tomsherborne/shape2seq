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
seq2seq = tf.contrib.seq2seq

from shapeworld import Dataset, tf_util
from src.model import CaptioningModel
from src.batch_parser import OneshapeBatchParser
from src.config import Config

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("data_dir", "", "Location of ShapeWorld data")
tf.flags.DEFINE_string("log_dir", "./models/exp8", "Directory location for logging")
tf.flags.DEFINE_string("dtype", "agreement", "Shapeworld Data Type")
tf.flags.DEFINE_string("name", "oneshape", "Shapeworld Data Name")
tf.flags.DEFINE_string("data_partition", "validation", "Which part of the dataset to test using")
tf.flags.DEFINE_string("exp_tag", "", "Subfolder labelling under log_dir for this experiment")
tf.flags.DEFINE_integer("num_imgs", 1000, "How many images to test with")
tf.flags.DEFINE_boolean("greedy", False, "Greedy decoding [TRUE] or softmax sampling [FALSE]")
tf.logging.set_verbosity(tf.logging.INFO)

def main(_):
    # FILESYSTEM SETUP ------------------------------------------------------------
    assert FLAGS.data_dir, "Must specify data location!"
    assert FLAGS.log_dir, "Must specify experiment to log to!"
    assert FLAGS.exp_tag, "Must specify experiment tag subfolder to log_dir %s" % FLAGS.log_dir

    # Folder setup for saving summaries and loading checkpoints
    save_root = FLAGS.log_dir + os.sep + FLAGS.exp_tag
    test_path = save_root + os.sep + "test"
    train_path = FLAGS.log_dir + os.sep + FLAGS.exp_tag + os.sep + "train"
    
    model_ckpt = tf.train.latest_checkpoint(train_path)     # Get checkpoint to load
    tf.logging.info("Loading checkpoint %s", model_ckpt)
    assert model_ckpt, "Checkpoints could not be loaded, check that train_path %s exists" % train_path

    # Sanity check graph reset
    tf.reset_default_graph()
    tf.logging.info("Clean graph reset...")

    try:
        dataset = Dataset.create(dtype=FLAGS.dtype, name=FLAGS.name, config=FLAGS.data_dir)
        dataset.pixel_noise_stddev = 0.1
        dataset.random_sampling = False
    except Exception:
        raise ValueError("config=%s did not point to a valid Shapeworld dataset" % FLAGS.data_dir)

    # Get parsing and parameter feats
    params = Config(mode="test", sw_specification=dataset.specification())
    
    # Greedy decoding
    if FLAGS.greedy:
        params.inference_greedy = True
        params.inference_sample = False
        
    # MODEL SETUP ------------------------------------------------------------
    
    g = tf.Graph()
    with g.as_default():
        parser = OneshapeBatchParser(src_vocab=dataset.vocabularies['language'])
        vocab, rev_vocab = parser.get_vocab()
        params.vocab_size = len(parser.tgt_vocab)
        
        caption_pl = tf.placeholder(dtype=tf.int32, shape=(params.batch_size, 9))
        caption_len_pl = tf.placeholder(dtype=tf.int32, shape=(params.batch_size,))
        world_pl = tf.placeholder(dtype=tf.float32, shape=(params.batch_size, 64, 64, 3))
        batch = {"caption": caption_pl, "caption_length": caption_len_pl, "world": world_pl}
        
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
    
    with tf.Session(graph=g, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # Model restoration
        restore_model.restore(sess, model_ckpt)
        tf.logging.info("Model restored!")

        # Trained model does not need initialisation. Init the vocab conversation tables
        sess.run([tf.tables_initializer()])

        #  Freeze graph
        sess.graph.finalize()
    
        # Get global step
        global_step = tf.train.global_step(sess, model.global_step)
        tf.logging.info("Successfully loaded %s at global step = %d.",
                        os.path.basename(model_ckpt), global_step)

        start_test_time = time.time()
        misses = []
        cap_scores = []
        for b_idx in range(num_imgs):
            idx_batch = dataset.generate(n=params.batch_size, mode=FLAGS.data_partition, include_model=True)
            
            reference_caps, inf_decoder_outputs = sess.run(fetches=[model.reference_captions,
                                                                    model.inf_decoder_output],
                                                           feed_dict={model.phase: 0,
                                                                      world_pl: idx_batch['world'],
                                                                      caption_pl: idx_batch['caption'],
                                                                      caption_len_pl: idx_batch['caption_length']
                                                            })
            
            ref_cap = reference_caps.squeeze()
            inf_cap = inf_decoder_outputs.sample_id.squeeze()
            
            if inf_cap.ndim > 0 and inf_cap.ndim > 0:
                cap_scores.append(parser.score_cap_against_world(idx_batch['world_model'][0], ref_cap, inf_cap))
                print("%d REF -> %s | INF -> %s | Spec-Correct: %d |  Underspec-Correct %d | Inc %d" %
                      (b_idx,
                       " ".join(rev_vocab[r] for r in ref_cap),
                       " ".join(rev_vocab[r] for r in inf_cap),
                       cap_scores[-1].specific_correct,
                       cap_scores[-1].underspecify_correct,
                       cap_scores[-1].incorrect)
                      )
            else:
                print("Skipping %d as inf_cap %s is malformed" % (b_idx, inf_cap))
                misses.append(1)
        
        num_specific_correct = sum([s.specific_correct for s in cap_scores]) / len(cap_scores)
        num_underspecify_correct = sum([s.underspecify_correct for s in cap_scores]) / len(cap_scores)
        num_incorrect = sum([s.incorrect for s in cap_scores]) / len(cap_scores)
        
        print("SPECIFIC CORRECT -> %.3f\nUNDERSPECIFY CORRECT -> %.3f\nINCORRECT -> %.3f" % (num_specific_correct,
                                                                                             num_underspecify_correct,
                                                                                             num_incorrect))
        new_summ = tf.Summary()
        new_summ.value.add(tag="test/specific_correct_%s" % (FLAGS.data_partition),
                           simple_value=num_specific_correct)

        new_summ.value.add(tag="test/underspecify_correct_%s" % (FLAGS.data_partition),
                           simple_value=num_underspecify_correct)
        
        new_summ.value.add(tag="test/incorrect_%s" % (FLAGS.data_partition),
                           simple_value=num_incorrect)
        
        test_writer.add_summary(new_summ, tf.train.global_step(sess, model.global_step))
        test_writer.flush()
        
        end_time = time.time()-start_test_time
        tf.logging.info('Testing complete in %.2f-secs/%.2f-mins/%.2f-hours', end_time, end_time/60, end_time/(60*60))

if __name__=="__main__":
    tf.app.run()

