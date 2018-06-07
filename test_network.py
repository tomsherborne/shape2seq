"""
SEQ2SEQ IMAGE CAPTIONING
Tom Sherborne 8/5/18
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os, time, csv

import numpy as np
import tensorflow as tf
seq2seq = tf.contrib.seq2seq

from shapeworld import Dataset
from src.model import CaptioningModel
from src.batch_parser import FullSequenceBatchParser
from src.config import Config

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("data_dir", "/home/trs46/data", "Location of ShapeWorld data")
tf.flags.DEFINE_string("log_dir", "./models/final/sequence", "Directory location for logging")
tf.flags.DEFINE_string("dtype", "agreement", "Shapeworld Data Type")
tf.flags.DEFINE_string("name", "existential", "Shapeworld Data Name")
tf.flags.DEFINE_string("variant", "", "Shapeworld dataset variant [required]")
tf.flags.DEFINE_string("data_partition", "validation", "Which part of the dataset to test using")
tf.flags.DEFINE_string("exp_tag", "", "Subfolder labelling under log_dir for this experiment")
tf.flags.DEFINE_integer("num_imgs", 1000, "How many images to test with")
tf.flags.DEFINE_string("decode_type", "greedy", "Greedy decoding [TRUE] or softmax sampling [FALSE]")
tf.flags.DEFINE_integer('pholder_sz', 8, "size of placeholder values in feed_dict")
tf.logging.set_verbosity(tf.logging.INFO)

def main(_):
    # FILESYSTEM SETUP ------------------------------------------------------------
    assert FLAGS.data_dir, "Must specify data location!"
    assert FLAGS.log_dir, "Must specify experiment to log to!"
    assert FLAGS.exp_tag, "Must specify experiment tag subfolder to log_dir %s" % FLAGS.log_dir
    assert FLAGS.variant, "Must specific shapeworld variant"

    # Folder setup for saving summaries and loading checkpoints
    save_root = FLAGS.log_dir + os.sep + FLAGS.exp_tag
    test_path = save_root + os.sep + "test_2"
    if not tf.gfile.IsDirectory(test_path):
        tf.gfile.MakeDirs(test_path)
        
    train_path = FLAGS.log_dir + os.sep + FLAGS.exp_tag + os.sep + "train"
    
    model_ckpt = tf.train.latest_checkpoint(train_path)     # Get checkpoint to load
    tf.logging.info("Loading checkpoint %s", model_ckpt)
    assert model_ckpt, "Checkpoints could not be loaded, check that train_path %s exists" % train_path

    # Sanity check graph reset
    tf.reset_default_graph()
    tf.logging.info("Clean graph reset...")

    try:
        dataset = Dataset.create(dtype=FLAGS.dtype, name=FLAGS.name, variant=FLAGS.variant, config=FLAGS.data_dir)
        dataset.pixel_noise_stddev = 0.1
        dataset.random_sampling = False
    except Exception:
        raise ValueError("variant=%s did not point to a valid Shapeworld dataset" % FLAGS.variant)

    # Get parsing and parameter feats
    params = Config(mode="test", sw_specification=dataset.specification())
    
    # Parse decoding arg from CLI
    params.decode_type = FLAGS.decode_type
    assert params.decode_type in ['greedy', 'sample', 'beam']
    
    # MODEL SETUP ------------------------------------------------------------
    g = tf.Graph()
    with g.as_default():
        parser = FullSequenceBatchParser(src_vocab=dataset.vocabularies['language'])
        vocab, rev_vocab = parser.get_vocab()
        params.vocab_size = len(parser.tgt_vocab)
        
        caption_pl = tf.placeholder(dtype=tf.int32, shape=(params.batch_size, FLAGS.pholder_sz))
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
        perplexities = []

        sem_parser = parser.build_semparser()

        idx_batch = dataset.generate(n=params.batch_size, mode=FLAGS.data_partition, include_model=True)
        
        # Dict of lists -> list of dicts
        idx_batch = [{k: v[idx] for k, v in idx_batch.items()}
                     for idx in range(0, params.batch_size)]
        
        for b_idx, batch in enumerate(idx_batch):
            reference_caps, inf_decoder_outputs, batch_perplexity = sess.run(fetches=[model.reference_captions,
                                                                                      model.inf_decoder_output,
                                                                                      model.batch_perplexity],
                                                                             feed_dict={model.phase: 0,
                                                                                        world_pl: [batch['world']],
                                                                                        caption_pl: [batch['caption']],
                                                                                        caption_len_pl: [batch['caption_length']]
                                                                              })
            
            perplexities.append(batch_perplexity)
            ref_cap = reference_caps.squeeze()
            inf_cap = inf_decoder_outputs.sample_id.squeeze()
            if inf_cap.ndim > 0 and inf_cap.ndim > 0:
                
                ref_cap = " ".join(rev_vocab[r] for r in ref_cap if r!=parser.pad_token_id)

                inf_cap = " ".join([rev_vocab[w] for w in
                                        filter(lambda y: y != parser.pad_token_id and y != parser.eos_token_id, inf_cap)
                                    ])

                try:
                    cap_scores.append(sem_parser(batch['world_model'], inf_cap))
                except Exception as exc:
                    print("Uncaught failure")
                    print(exc)
                    continue
                    
                print("-------------------------------------------")
                print("%d | REF -> %s | INF -> %s" % (b_idx, ref_cap, inf_cap))
                print(cap_scores[-1])
                
            else:
                print("Skipping %d as inf_cap %s is malformed" % (b_idx, inf_cap))
                misses.append(inf_cap)
        
        avg_perplexity = np.mean(perplexities).squeeze()
        std_perplexity = np.std(perplexities).squeeze()
        agree_rate = np.mean([sc.agreement for sc in cap_scores])
        false_rate = np.mean([sc.false for sc in cap_scores])
        ooscope_rate = np.mean([sc.out_of_scope for sc in cap_scores])
        ungramm_rate = np.mean([sc.ungrammatical for sc in cap_scores])
        
        print("--------------------------")
        print("PERPLEXITY -> %.5f +- %.5f" % (avg_perplexity, std_perplexity))
        print("AGREEMENT RATE -> %.2f" % agree_rate)
        print("FALSE RATE -> %.2f" % false_rate)
        print("OOSCOPE RATE -> %.2f" % ooscope_rate)
        print("UNGRAMMATICAL RATE -> %.2f" % ungramm_rate)
        print("misses -> %d" % sum(misses))
        
        new_summ = tf.Summary()
        new_summ.value.add(tag="%s/perplexity_avg_%s_%s" % (FLAGS.data_partition, FLAGS.decode_type, FLAGS.name),
                           simple_value=avg_perplexity)
        new_summ.value.add(tag="%s/perplexity_std_%s_%s" % (FLAGS.data_partition, FLAGS.decode_type, FLAGS.name),
                           simple_value=std_perplexity)
        new_summ.value.add(tag="%s/agree_rate_%s_%s" % (FLAGS.data_partition, FLAGS.decode_type, FLAGS.name),
                            simple_value=agree_rate)
        new_summ.value.add(tag="%s/false_rate_%s_%s" % (FLAGS.data_partition, FLAGS.decode_type, FLAGS.name),
                            simple_value=false_rate)
        new_summ.value.add(tag="%s/ooscope_rate_%s_%s" % (FLAGS.data_partition, FLAGS.decode_type, FLAGS.name),
                            simple_value=ooscope_rate)
        new_summ.value.add(tag="%s/ungramm_rate_%s_%s" % (FLAGS.data_partition, FLAGS.decode_type, FLAGS.name),
                            simple_value=ungramm_rate)
        
        test_writer.add_summary(new_summ, tf.train.global_step(sess, model.global_step))
        test_writer.flush()
        
        test_outputs_fname = test_path + os.sep + "caps_%d_%s_%s.csv" % (tf.train.global_step(sess, model.global_step),
                                                                   FLAGS.data_partition, FLAGS.decode_type)
        
        with open(test_outputs_fname, 'w', newline='\n') as fh:
            writer = csv.writer(fh, delimiter=',')
            writer.writerow(cap_scores[0]._fields)
            writer.writerows(list(c) for c in cap_scores)
        
        end_time = time.time()-start_test_time
        tf.logging.info('Testing complete in %.2f-secs/%.2f-mins/%.2f-hours', end_time, end_time/60, end_time/(60*60))

if __name__=="__main__":
    tf.app.run()

