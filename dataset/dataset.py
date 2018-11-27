import tensorflow as tf
from light_kaldi_io import FeatureReader
import os
import random
import numpy as np


class KaldiDataReader():
    """Used to read data from a kaldi data directory."""

    def __init__(self, data_dir, num_parallel=1, num_speakers=None, num_segments=None, min_len=None, max_len=None):
        """ Create a data_reader from a given directory.

        Args:
            data_dir: The kaldi data directory.
            num_parallel: The number of threads to read features.
            num_speakers: The number of speakers per batch.
            num_segments: The number of semgents per speaker.
              batch_size = num_speakers * num_segments
              When num_segments = 1, the batch is randomly chosen from n speakers,
              which is used for softmax-like loss function. While we can sample multiple segments for each speaker,
              which is used for triplet-loss or GE2E loss.
            min_len: The minimum length of the sampled sequence.
            max_len: The maximum length of the sampled sequence.
        """
        self.data = data_dir
        self.num_speakers = num_speakers
        self.num_segments = num_segments
        self.min_len = min_len
        self.max_len = max_len

        # We process the data directory and fetch speaker information
        self.dim = FeatureReader(data_dir).get_dim()
        spk2index, spk2features = self.get_speaker_info(data_dir)
        self.num_total_speakers = len(spk2features)
        self.spk2features = spk2features
        self.num_parallel_datasets = num_parallel

        if self.num_parallel_datasets != 1:
            raise NotImplementedError("When num_parallel_datasets != 1, we got some strange problem with the dataset. Waiting for fix.")

    @staticmethod
    def get_speaker_info(data):
        """Get speaker information from the data directory

        Args:
            data: The kaldi data directory
        :return:
            spk2index: A dictionary map the speaker name to the index
            spk2features: A list. Each entry is a list containing features (in text) of this speaker
        """
        assert(os.path.isdir(data))
        spk2index = {}
        utt2spk = {}
        spk2features = []
        with open(os.path.join(data, "spk2utt"), "r") as f:
            for (index, line) in enumerate(f.readlines()):
                spk, utts = line.strip().split(" ", 1)
                spk2index[spk] = index
                for utt in utts.split(" "):
                    utt2spk[utt] = index
                spk2features.append([])

        with open(os.path.join(data, "feats.scp"), "r") as f:
            for line in f.readlines():
                (key, rxfile) = line.decode().split(' ')
                spk2features[utt2spk[key]].append(rxfile)

        return spk2index, spk2features

    def set_batch(self, num_speakers, num_segments):
        """Set the batch-related parameters

        Args:
            num_speakers: The number of speakers per batch.
            num_segments: The number of semgents per speaker.
        """
        self.num_speakers = num_speakers
        self.num_segments = num_segments

    def set_length(self, min_len, max_len):
        """Set the length of the sequence

        Args:
            min_len: The minimum length of the sampled sequence.
            max_len: The maximum length of the sampled sequence.
        """
        self.min_len = min_len
        self.max_len = max_len

    def random_batch(self, feature_reader):
        """Sample a batch randomly.

        Args:
            feature_reader: The reader used to load the features from Kaldi
        :return: A tuple which is (features, labels).
        """
        # The function never stop since it just randomly pick speakers to form a batch
        batch_speakers = random.sample(xrange(self.num_total_speakers), self.num_speakers)
        # The random length of this batch
        batch_length = random.randint(self.min_len, self.max_len)
        features = np.zeros((self.num_speakers * self.num_segments, batch_length, feature_reader.dim), dtype=np.float32)
        labels = np.zeros((self.num_speakers * self.num_segments), dtype=np.int32)
        for i, speaker in enumerate(batch_speakers):
            labels[i * self.num_segments:(i + 1) * self.num_segments] = speaker
            feature_list = self.spk2features[speaker]
            if len(feature_list) < self.num_segments:
                feature_list *= (int(self.num_segments / len(feature_list)) + 1)
            # Now the length of the list must be greater than the sample size.
            speaker_features = random.sample(feature_list, self.num_segments)
            for j, feat in enumerate(speaker_features):
                features[i * self.num_segments + j, :, :] = feature_reader.read(feat, batch_length, shuffle=True)
        return features, labels

    def random_batch_sequence(self):
        """Randomly load features to form a batch

        This function is used in the load routine to feed data to the dataset object
        It can also be used as a generator to get data directly.
        """
        feature_reader = FeatureReader(self.data)
        while True:
            features, labels = self.random_batch(feature_reader)
            yield (features, labels)

    def load_dataset(self):
        """ Load data from Kaldi features and return tf.dataset.
        The function is useful for training, since it randomly loads features and labels from N speakers,
          with K segments per speaker.
        The batch is sampled randomly, so there is no need to do shuffle again.

        :return: A nested tensor (features, labels)
        """
        batch_size = self.num_speakers * self.num_segments
        if self.num_parallel_datasets == 1:
            # Single thread loading
            dataset = tf.data.Dataset.from_generator(self.random_batch_sequence, (tf.float32, tf.int32),
                                                     (tf.TensorShape([batch_size, None, self.dim]),
                                                      tf.TensorShape([batch_size])))
        else:
            # Multiple threads loading
            # It is very stange that the following code doesn't work properly.
            # I guess the reason may be the py_func influence the performance of parallel_interleave.
            dataset = tf.data.Dataset.range(self.num_parallel_datasets).apply(
                tf.contrib.data.parallel_interleave(
                    lambda x: tf.data.Dataset.from_generator(self.random_batch_sequence, (tf.float32, tf.int32),
                                                             (tf.TensorShape([batch_size, None, self.dim]),
                                                              tf.TensorShape([batch_size]))),
                    cycle_length=self.num_parallel_datasets,
                    sloppy=False))
        dataset = dataset.prefetch(1)
        return dataset.make_one_shot_iterator().get_next()

    def simple_load(self):
        pass


class TFDataReader():
    """Used to read data from a directory containing TFRecords."""

    def __init__(self, data, params):
        """ Create a data_reader from a given directory.

        Args:
            data: The directory containing TFRecords (one record per speaker).
            params: The parameters derived from a JSON config file.
        """
        pass


class KaldiDataReaderFIFO():
    def __init__(self,
                 data_dir,
                 coord,
                 num_speakers=None,
                 num_segments=None,
                 min_len=None,
                 max_len=None,
                 max_queue_size=32,
                 wait_time=0.01,
                 num_parallel=1):
        # Try to use tensorflow queue to see whether it is faster
        # Most of the loading time is due to the uncompression process, so it should be useful to have multiple
        # processings to load the data at the same time.
        # Change the shape of the input data here with the parameter shapes.
        self.data = data_dir
        self.num_speakers = num_speakers
        self.num_segments = num_segments
        self.min_len = min_len
        self.max_len = max_len

        self.dim = FeatureReader(data_dir).get_dim()
        spk2index, spk2features = self.get_speaker_info(data_dir)
        self.num_total_speakers = len(spk2features)
        self.spk2features = spk2features
        self.num_parallel_datasets = num_parallel

        self.wait_time = wait_time
        self.max_queue_size = max_queue_size

        batch_size = self.num_speakers * self.num_segments
        self.queue = tf.PaddingFIFOQueue(self.max_queue_size, (tf.float32, tf.int32),
                                  shapes=(tf.TensorShape([batch_size, None, self.dim]), tf.TensorShape([batch_size])))
        self.queue_size = self.queue.size()
        self.threads = []
        self.coord = coord
        self.feature_placeholder = tf.placeholder(dtype=tf.float32, shape=[batch_size, None, self.dim])
        self.label_placeholder = tf.placeholder(dtype=tf.int32, shape=[batch_size])
        self.enqueue = self.queue.enqueue([self.feature_placeholder, self.label_placeholder])

    def dequeue(self):
        output = self.queue.dequeue()
        return output

    def random_batch(self, feature_reader):
        """Sample a batch randomly.

        Args:
            feature_reader: The reader used to load the features from Kaldi
        :return: A tuple which is (features, labels).
        """
        # The function never stop since it just randomly pick speakers to form a batch
        batch_speakers = random.sample(xrange(self.num_total_speakers), self.num_speakers)
        # The random length of this batch
        batch_length = random.randint(self.min_len, self.max_len)
        features = np.zeros((self.num_speakers * self.num_segments, batch_length, feature_reader.dim), dtype=np.float32)
        labels = np.zeros((self.num_speakers * self.num_segments), dtype=np.int32)
        for i, speaker in enumerate(batch_speakers):
            labels[i * self.num_segments:(i + 1) * self.num_segments] = speaker
            feature_list = self.spk2features[speaker]
            if len(feature_list) < self.num_segments:
                feature_list *= (int(self.num_segments / len(feature_list)) + 1)
            # Now the length of the list must be greater than the sample size.
            speaker_features = random.sample(feature_list, self.num_segments)
            for j, feat in enumerate(speaker_features):
                features[i * self.num_segments + j, :, :] = feature_reader.read(feat, batch_length, shuffle=True)
        return features, labels

    def random_batch_sequence(self):
        """Randomly load features to form a batch

        This function is used in the load routine to feed data to the dataset object
        It can also be used as a generator to get data directly.
        """
        feature_reader = FeatureReader(self.data)
        while True:
            features, labels = self.random_batch(feature_reader)
            yield (features, labels)

    def thread_main(self, sess):
        stop = False
        while not stop:
            iterator = self.random_batch_sequence()
            for features, labels in iterator:
                while self.queue_size.eval(session=sess) == self.max_queue_size:
                    if self.coord.should_stop():
                        break
                    time.sleep(self.wait_time)
                sess.run(self.enqueue, feed_dict={self.feature_placeholder: features,
                                                  self.label_placeholder: labels})
                if self.coord.should_stop():
                    stop = True
                    print("Enqueue thread receives stop request.")
                    break

    def start_threads(self, sess, n_threads=1):
        import threading
        for _ in range(n_threads):
            thread = threading.Thread(target=self.thread_main, args=(sess,))
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)
        return self.threads

    @staticmethod
    def get_speaker_info(data):
        """Get speaker information from the data directory

        Args:
            data: The kaldi data directory
        :return:
            spk2index: A dictionary map the speaker name to the index
            spk2features: A list. Each entry is a list containing features (in text) of this speaker
        """
        assert (os.path.isdir(data))
        spk2index = {}
        utt2spk = {}
        spk2features = []
        with open(os.path.join(data, "spk2utt"), "r") as f:
            for (index, line) in enumerate(f.readlines()):
                spk, utts = line.strip().split(" ", 1)
                spk2index[spk] = index
                for utt in utts.split(" "):
                    utt2spk[utt] = index
                spk2features.append([])

        with open(os.path.join(data, "feats.scp"), "r") as f:
            for line in f.readlines():
                (key, rxfile) = line.decode().split(' ')
                spk2features[utt2spk[key]].append(rxfile)

        return spk2index, spk2features


if __name__ == "__main__":
    data = "/home/dawna/mgb3/transcription/exp-yl695/Snst/xvector/cpdaic_1.0_50/data/voxceleb_train_combined_no_sil/"
    # data = "/scratch/yl695/voxceleb/data/voxceleb_train_combined_no_sil"
    data_loader = KaldiDataReader(data, num_parallel=1)
    data_loader.set_batch(64, 10)
    data_loader.set_length(200, 400)
    num_loads = 10
    import time

    # # A very simple network
    # features = tf.placeholder(tf.float32, shape=[None, None, None])
    # features = features + 1
    # labels = tf.placeholder(tf.int32, shape=[None,])
    #
    # counter = 0
    # ts = time.time()
    # time1 = 0
    # time2 = 0
    # time3 = 0
    # for (features_val, labels_val, t1, t2, t3) in data_loader.random_batch_sequence():
    #     counter += 1
    #     time1 += t1
    #     time2 += t2
    #     time3 += t3
    #     if counter == num_loads:
    #         break
    # te = time.time()
    # print("Time: %.4f s, time 1: %.4f s, time 2: %.4f s, time 3: %.4f s" % (te - ts, time1, time2, time3))

    with tf.Session() as sess:
        features, labels = data_loader.load_dataset()
        sess.run(tf.global_variables_initializer())
        print("start...")
        start_time = time.time()
        for _ in xrange(num_loads):
            features_val, labels_val = sess.run([features, labels])
        end_time = time.time()
        print("Time: %.4f s" % (end_time - start_time))
        #TODO: try to close the file


    # This result varies on different computers and disks, and may also be affected by the cache.
    # I load the data for a long time before testing to insure the cache is used.
    # Batch size = 15 * 10
    #   With one dataset reading, the performance is 3.00 s/batch (100 batches)
    #   With 2 datasets, the performance is 2.38 s/batch (100 batches)
    #   With 4 datasets, the performance is 2.4 s/batch (100 batches)
    #
    # When the dataset is really large, it is better to optimize the input pipeline.
    # I used to use TFRecords before, but if we use TFRecords, the files should be open when the graph is
    # constructed. To carefully control the number of segments per speaker per batch, we need at least one files for
    # each speaker. This is terrible if we have thousands of speakers, meaning that we need to open thousands of
    # files at the same time. The available file descriptors are limited and you may need to set `ulimit -n` very
    # large. If this is impossible (e.g. you are not the system administrator), you can only load parts of the
    # dataset, and reload the dataset after several training steps.
    #

