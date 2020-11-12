# coding=utf-8
# Copyright 2018 The Google AI Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Lint as: python2, python3
# coding=utf-8
"""Create masked_lm TF examples for ALBERT-seq.
   No next-sentence labels are used.
   Input is FASTA file containing sequences for a protein family."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import random
from albert import tokenization
import numpy as np
import six
from six.moves import range
from six.moves import zip
import tensorflow.compat.v1 as tf

"""
Amino acid frequency in all Pfam sequences.
    It is used for masked_lm_weights. (inversely weighted; frequent AA -> low weight)
    This feature for triplets is not available yet.
    Some amino acids are too rare to be considered, so manually set to a small value
    Manual values (small): set to large values so their weights are low 
        B=0.1500
        J=0.8000
        X=0.8000
        Z=0.1500
    Manual values (large): set to small values so their weights are high
        O=0.0150
        U=0.0150
"""
PAD_TOKEN = "[PAD]"
CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"
UNK_TOKEN = "[UNK]"
MASK_TOKEN = "[MASK]"
AA_FREQ = {
    "A": 0.0900, "B": 0.1500, "C": 0.0142, "D": 0.0548, 
    "E": 0.0604, "F": 0.0419, "G": 0.0760, "H": 0.0230, 
    "I": 0.0604, "J": 0.8000, "K": 0.0490, "L": 0.1014, 
    "M": 0.0228, "N": 0.0372, "O": 0.0150, "P": 0.0439, 
    "Q": 0.0353, "R": 0.0557, "S": 0.0616, "T": 0.0541, 
    "U": 0.0150, "V": 0.0739, "W": 0.0132, "X": 0.8000, 
    "Y": 0.0313, "Z": 0.1500
}
AA_WEIGHT = {a:0.05/AA_FREQ[a] for a in AA_FREQ.keys()}
AA_WEIGHT[PAD_TOKEN] = 0.0
AA_WEIGHT[CLS_TOKEN] = 0.0
AA_WEIGHT[SEP_TOKEN] = 0.0
AA_WEIGHT[UNK_TOKEN] = 0.0

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", None,
                    "FASTA format sequences (or comma-separated list of FASTA files).")

flags.DEFINE_string(
    "output_file", None,
    "Output TF example file (or comma-separated list of files).")

flags.DEFINE_string(
    "vocab_file", None,
    "The vocabulary file that the ALBERT model was trained on.")

flags.DEFINE_integer("max_seq_length", 384, "Maximum sequence length.\
        For Pfam database, approx. 4.63% sequences are longer than 384. Average token density is 0.3863 (148.34/384)")

flags.DEFINE_bool("triplet", False,
                 "Set True for amino acid triplet sentences. Default is singlet.")

flags.DEFINE_integer("ngram", 2, "Maximum number of ngrams to mask.")

flags.DEFINE_integer("max_predictions_per_seq", 20,
                     "Maximum number of masked LM predictions per sequence.")

flags.DEFINE_integer("random_seed", 12345, "Random seed for data generation.")

flags.DEFINE_integer(
    "dupe_factor", 20,
    "Number of times to duplicate the input data (with different masks).")

flags.DEFINE_float("masked_lm_prob", 0.15, "Masked LM probability.")



class TrainingInstance(object):
    """A single training instance (a protein sequence)."""

    def __init__(self, tokens, masked_lm_positions, masked_lm_labels):
        self.tokens = tokens
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels

    def __str__(self):
        s = ""
        s += "tokens: %s\n" % (" ".join(
            [tokenization.printable_text(x) for x in self.tokens]))
        s += "masked_lm_positions: %s\n" % (" ".join(
            [str(x) for x in self.masked_lm_positions]))
        s += "masked_lm_labels: %s\n" % (" ".join(
            [tokenization.printable_text(x) for x in self.masked_lm_labels]))
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()


def write_instance_to_example_files(instances, tokenizer, max_seq_length,
                                    max_predictions_per_seq, output_files
                                   ):
    """Create TF example files from TrainingInstances."""
    writers = []
    for output_file in output_files:
        writers.append(tf.python_io.TFRecordWriter(output_file))

    writer_index = 0

    total_written = 0
    for (inst_index, instance) in enumerate(instances):
        input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
        input_mask = [1] * len(input_ids)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length

        masked_lm_positions = list(instance.masked_lm_positions)
        masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
        masked_lm_weights = []
        for tok in instance.masked_lm_labels:
            if tok.upper() in AA_WEIGHT: #for singlets, use frequency-based weights
                masked_lm_weights.append(AA_WEIGHT[tok.upper()])
            else: #for triplets, uniform weights
                masked_lm_weights.append(1.0)

        while len(masked_lm_positions) < max_predictions_per_seq:
            masked_lm_positions.append(0)
            masked_lm_ids.append(0)
            masked_lm_weights.append(0.0)

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(input_ids)
        features["input_mask"] = create_int_feature(input_mask)

        features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
        features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
        features["masked_lm_weights"] = create_float_feature(masked_lm_weights)

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))

        writers[writer_index].write(tf_example.SerializeToString())
        writer_index = (writer_index + 1) % len(writers)

        total_written += 1

        if inst_index < 20:
            tf.logging.info("*** Example ***")
            tf.logging.info("tokens: %s" % " ".join(
              [tokenization.printable_text(x) for x in instance.tokens]))

            for feature_name in features.keys():
                feature = features[feature_name]
                values = []
                if feature.int64_list.value:
                    values = feature.int64_list.value
                elif feature.float_list.value:
                    values = feature.float_list.value
                tf.logging.info(
                    "%s: %s" % (feature_name, " ".join([str(x) for x in values])))

    for writer in writers:
        writer.close()

    tf.logging.info("Wrote %d total instances", total_written)


def create_int_feature(values):
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature


def create_float_feature(values):
    feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return feature


def create_training_instances(input_files, tokenizer, max_seq_length,
                              dupe_factor, masked_lm_prob,
                              max_predictions_per_seq, rng):
    """Create `TrainingInstance`s from FASTA file(s)."""
    all_documents = []
    for input_file in input_files:
        with tf.gfile.GFile(input_file, 'r') as reader:
            sequence=''
            while True:
                line = reader.readline()
                if not line:
                    break
                if line.startswith('>'):
                    if len(sequence)>0:
                        residues = [r for r in sequence.strip()]
                        if FLAGS.triplet:
                            triplets = []
                            for i in range(1,len(residues)-1):
                                triplets.append(residues[i-1]+residues[i]+residues[i+1])
                            residues=triplets
                        
                        tokens = tokenizer.tokenize(' '.join(residues))
                        all_documents.append(tokens)

                    sequence=''
                else:
                    sequence+=line.strip()
                

  # Remove empty documents
    all_documents = [x for x in all_documents if x]
    rng.shuffle(all_documents)

    vocab_words = list(tokenizer.vocab.keys())
    instances = []
    for _ in range(dupe_factor):
        instances.extend(
                create_instances_from_document(
                    all_documents, max_seq_length,
                    masked_lm_prob, max_predictions_per_seq, vocab_words, rng))

    rng.shuffle(instances)
    return instances

def create_instances_from_document(
    all_documents, max_seq_length,
    masked_lm_prob, max_predictions_per_seq, vocab_words, rng):
    """Creates `TrainingInstance`s for a single document."""

    # Account for [CLS]
    max_num_tokens = max_seq_length - 1

    instances = []
    current_chunk = []
    current_length = 0
    i = 0
    while i < len(all_documents):
        segment = all_documents[i]
        if segment:
            tokens = [CLS_TOKEN]
            for j in range(min(len(segment),max_num_tokens)):
                tokens.extend(segment[j])

            assert len(tokens) >= 1

            while len(tokens) <= max_num_tokens:
                tokens.append(PAD_TOKEN)

            (tokens, masked_lm_positions,
                masked_lm_labels) = create_masked_lm_predictions(
                    tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng)
            instance = TrainingInstance(
                    tokens=tokens,
                    masked_lm_positions=masked_lm_positions,
                    masked_lm_labels=masked_lm_labels)
            instances.append(instance)

        i += 1
    return instances

MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])

def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, rng):
    """Creates the predictions for the masked LM objective."""
    cand_indexes = []

    for (i, token) in enumerate(tokens):
        if token == CLS_TOKEN or token == SEP_TOKEN:
            continue
        else:
            cand_indexes.append(i)

    num_to_predict = min(max_predictions_per_seq,
                       max(1, int(round(len(tokens) * masked_lm_prob))))

    ngrams = np.arange(1, FLAGS.ngram + 1, dtype=np.int64)
    pvals = 1. / np.arange(1, FLAGS.ngram + 1)
    pvals /= pvals.sum(keepdims=True)
    
    masked_lm_positions = []
    masked_lm_labels = []
    masked_lms = []
    output_tokens = list(tokens)
    covered_indices = set()

    if masked_lm_prob == 0:
        return (output_tokens, masked_lm_positions,
            masked_lm_labels)

    ngram_indexes = []
    for idx in range(len(cand_indexes)):
        ngram_index = []
        for n in ngrams:
            ngram_index.append(cand_indexes[idx:idx+n])
        ngram_indexes.append(ngram_index)

    rng.shuffle(ngram_indexes)
    for ngram_index in ngram_indexes:

        if len(covered_indices) >= num_to_predict:
            break
        if not ngram_index:
            continue

        n = np.random.choice(ngrams[:len(ngram_index)],
                             p=pvals[:len(ngram_index)]/pvals[:len(ngram_index)].sum(keepdims=True))

        covered_index = set() 
        for idx in ngram_index[n-1]:
            if idx in covered_indices:
                continue
            if (idx+1 in covered_indices) or (idx-1 in covered_indices):
                #prevent accidental consecutive masks
                continue
            else:
                covered_index.add(idx)
        covered_indices = covered_indices.union(covered_index)

    for idx in covered_indices:
        if len(masked_lms)>=num_to_predict:
            break
        if rng.random() <= 0.85:
            tok = MASK_TOKEN
        else:
            tok = tokens[idx]
        output_tokens[idx] = tok
        masked_lms.append(MaskedLmInstance(index=idx, label=tokens[idx]))

    masked_lms = sorted(masked_lms, key=lambda x: x.index)
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    assert len(masked_lms) <= num_to_predict

    return (output_tokens, masked_lm_positions, masked_lm_labels)

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=True,
        spm_model_file=None)

    input_files = []
    for input_pattern in FLAGS.input_file.split(","):
        input_files.extend(tf.gfile.Glob(input_pattern))

    tf.logging.info("*** Reading from input files ***")
    for input_file in input_files:
        tf.logging.info("  %s", input_file)

    rng = random.Random(FLAGS.random_seed)
    instances = create_training_instances(
        input_files, tokenizer, FLAGS.max_seq_length, FLAGS.dupe_factor,
        FLAGS.masked_lm_prob, FLAGS.max_predictions_per_seq,
        rng)

    tf.logging.info("number of instances: %i", len(instances))

    output_files = FLAGS.output_file.split(",")
    tf.logging.info("*** Writing to output files ***")
    for output_file in output_files:
        tf.logging.info("  %s", output_file)

    write_instance_to_example_files(instances, tokenizer, FLAGS.max_seq_length,
                                  FLAGS.max_predictions_per_seq, output_files)


if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("output_file")
    flags.mark_flag_as_required("vocab_file")
    tf.app.run()
