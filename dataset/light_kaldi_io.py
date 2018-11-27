#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2014-2016  Brno University of Technology (author: Karel Vesely)
# Licensed under the Apache License, Version 2.0 (the "License")

# Modified by Yi Liu

import os
import numpy as np
import random

class UnknownMatrixHeader(Exception):
    pass


class BadSampleSize(Exception):
    pass


class FeatureReader():
    """Read kaldi features"""
    def __init__(self, data):
        """This is a modified version of read_mat_scp in kaldi_io.

        I wrote the class because we don't want to open and close file frequently.
        The number of file descriptors is limited (= the num of arks) so we can keep all the files open.
        The performance bottleneck will be the seek operation during reading.
        Once the feature archive is opened, it just keeps the file descriptors until the class is closed.

        Args:
            data: The kaldi data directory.
        """
        self.fd = {}
        self.data = data
        self.dim = self.get_dim()

    def get_dim(self):
        with open(os.path.join(self.data, "feats.scp"), "r") as f:
            dim = self.read(f.readline().strip().split(" ")[1]).shape[1]
        return dim

    def close(self):
        for name in self.fd:
            self.fd[name].close()

    def read(self, file_or_fd, length=None, shuffle=False):
        """ [mat] = read_mat(file_or_fd)
         Reads single kaldi matrix, supports ascii and binary.
         file_or_fd : file, gzipped file, pipe or opened file descriptor.

         In our case, file_or_fd can only be a filename with offset. We will save the fd after opening it.

         Note:
             It is really painful to load data from compressed archives. To speed up training, the archives should be
             prepared as uncompressed data. Directly exit if loading data from compressed data. If you really like to
             use that, modify by yourself.
             Maybe other high-performance library can be used to accelerate the loading. No time to try here.
        """
        (filename, offset) = file_or_fd.rsplit(":", 1)
        if filename not in self.fd:
            fd = open(filename, 'rb')
            assert fd is not None
            self.fd[filename] = fd
        # Move to the target position
        self.fd[filename].seek(int(offset))
        try:
            binary = self.fd[filename].read(2).decode()
            if binary == '\0B':
                mat = self._read_mat_binary(self.fd[filename])
            else:
                pass
        except:
            raise IOError("Cannot read features from %s" % file_or_fd)

        if length is not None:
            num_features = mat.shape[0]
            length = num_features if length > num_features else length
            start = random.randint(0, num_features - length) if shuffle else 0
            mat = mat[start:start+length, :]
        return mat

    def _read_mat_binary(self, fd):
        # Data type
        header = fd.read(3).decode()
        # 'CM', 'CM2', 'CM3' are possible values,
        if header.startswith('CM'):
            # import sys
            # sys.exit("Using compressed archives ")
            return self._read_compressed_mat(fd, header)
        elif header == 'FM ':
            sample_size = 4  # floats
        elif header == 'DM ':
            sample_size = 8  # doubles
        else:
            raise UnknownMatrixHeader("The header contained '%s'" % header)
        assert (sample_size > 0)
        # Dimensions
        s1, rows, s2, cols = np.frombuffer(fd.read(10), dtype='int8,int32,int8,int32', count=1)[0]
        # Read whole matrix
        buf = fd.read(rows * cols * sample_size)
        if sample_size == 4:
            vec = np.frombuffer(buf, dtype='float32')
        elif sample_size == 8:
            vec = np.frombuffer(buf, dtype='float64')
        else:
            raise BadSampleSize
        mat = np.reshape(vec, (rows, cols))
        return mat

    def _read_compressed_mat(self, fd, format):
        """ Read a compressed matrix,
            see: https://github.com/kaldi-asr/kaldi/blob/master/src/matrix/compressed-matrix.h
            methods: CompressedMatrix::Read(...), CompressedMatrix::CopyToMat(...),
        """
        assert (format == 'CM ')  # The formats CM2, CM3 are not supported...

        # Format of header 'struct',
        global_header = np.dtype([('minvalue', 'float32'), ('range', 'float32'), ('num_rows', 'int32'),
                                  ('num_cols', 'int32')])  # member '.format' is not written,
        per_col_header = np.dtype([('percentile_0', 'uint16'), ('percentile_25', 'uint16'), ('percentile_75', 'uint16'),
                                   ('percentile_100', 'uint16')])

        # Mapping for percentiles in col-headers,
        def uint16_to_float(value, min, range):
            return np.float32(min + range * 1.52590218966964e-05 * value)

        # Mapping for matrix elements,
        def uint8_to_float_v2(vec, p0, p25, p75, p100):
            # Split the vector by masks,
            mask_0_64 = (vec <= 64)
            mask_65_192 = np.all([vec > 64, vec <= 192], axis=0)
            mask_193_255 = (vec > 192)
            # Sanity check (useful but slow...),
            # assert(len(vec) == np.sum(np.hstack([mask_0_64,mask_65_192,mask_193_255])))
            # assert(len(vec) == np.sum(np.any([mask_0_64,mask_65_192,mask_193_255], axis=0)))
            # Build the float vector,
            ans = np.empty(len(vec), dtype='float32')
            ans[mask_0_64] = p0 + (p25 - p0) / 64. * vec[mask_0_64]
            ans[mask_65_192] = p25 + (p75 - p25) / 128. * (vec[mask_65_192] - 64)
            ans[mask_193_255] = p75 + (p100 - p75) / 63. * (vec[mask_193_255] - 192)
            return ans

        # Read global header,
        globmin, globrange, rows, cols = np.frombuffer(fd.read(16), dtype=global_header, count=1)[0]

        # The data is structed as [Colheader, ... , Colheader, Data, Data , .... ]
        #                         {           cols           }{     size         }
        col_headers = np.frombuffer(fd.read(cols * 8), dtype=per_col_header, count=cols)
        data = np.reshape(np.frombuffer(fd.read(cols * rows), dtype='uint8', count=cols * rows),
                          newshape=(cols, rows))  # stored as col-major,

        mat = np.empty((cols, rows), dtype='float32')
        for i, col_header in enumerate(col_headers):
            col_header_flt = [uint16_to_float(percentile, globmin, globrange) for percentile in col_header]
            mat[i] = uint8_to_float_v2(data[i], *col_header_flt)

        return mat.T  # transpose! col-major -> row-major,


if __name__ == "__main__":
    def read(file_or_fd, length=None, shuffle=False):
        """ [mat] = read_mat(file_or_fd)
         Reads single kaldi matrix, supports ascii and binary.
         file_or_fd : file, gzipped file, pipe or opened file descriptor.

         In our case, file_or_fd can only be a filename with offset. We will save the fd after opening it.
        """
        (filename, offset) = file_or_fd.rsplit(":", 1)
        fd = open(filename, 'rb')
        fd.seek(int(offset))

        binary = fd.read(2).decode()
        if binary == '\0B':
            mat, time1, time2, time3 = read_mat_binary(fd)
        else:
            pass

        if length is not None:
            num_features = mat.shape[0]
            length = num_features if length > num_features else length
            start = random.randint(0, num_features - length) if shuffle else 0
            mat = mat[start:start+length, :]
        fd.close()
        return mat, time1, time2, time3

    def read_mat_binary(fd):
        # Data type
        import time
        ts = time.time()
        header = fd.read(3).decode()
        # 'CM', 'CM2', 'CM3' are possible values,
        if header.startswith('CM'):
            return read_compressed_mat(fd, header)
        elif header == 'FM ':
            sample_size = 4  # floats
        elif header == 'DM ':
            sample_size = 8  # doubles
        else:
            raise UnknownMatrixHeader("The header contained '%s'" % header)
        assert (sample_size > 0)
        # Dimensions
        s1, rows, s2, cols = np.frombuffer(fd.read(10), dtype='int8,int32,int8,int32', count=1)[0]
        t1 = time.time() - ts
        # Read whole matrix
        ts = time.time()
        buf = fd.read(rows * cols * sample_size)
        t2 = time.time() - ts
        ts = time.time()
        if sample_size == 4:
            vec = np.frombuffer(buf, dtype='float32')
        elif sample_size == 8:
            vec = np.frombuffer(buf, dtype='float64')
        else:
            raise BadSampleSize
        mat = np.reshape(vec, (rows, cols))
        t3 = time.time() - ts
        return mat, t1, t2, t3

    def read_compressed_mat(fd, format):
        """ Read a compressed matrix,
            see: https://github.com/kaldi-asr/kaldi/blob/master/src/matrix/compressed-matrix.h
            methods: CompressedMatrix::Read(...), CompressedMatrix::CopyToMat(...),
        """
        assert (format == 'CM ')  # The formats CM2, CM3 are not supported...

        # Format of header 'struct',
        global_header = np.dtype([('minvalue', 'float32'), ('range', 'float32'), ('num_rows', 'int32'),
                                  ('num_cols', 'int32')])  # member '.format' is not written,
        per_col_header = np.dtype([('percentile_0', 'uint16'), ('percentile_25', 'uint16'), ('percentile_75', 'uint16'),
                                   ('percentile_100', 'uint16')])

        # Mapping for percentiles in col-headers,
        def uint16_to_float(value, min, range):
            return np.float32(min + range * 1.52590218966964e-05 * value)

        # Mapping for matrix elements,
        def uint8_to_float_v2(vec, p0, p25, p75, p100):
            # Split the vector by masks,
            mask_0_64 = (vec <= 64)
            mask_65_192 = np.all([vec > 64, vec <= 192], axis=0)
            mask_193_255 = (vec > 192)
            # Sanity check (useful but slow...),
            # assert(len(vec) == np.sum(np.hstack([mask_0_64,mask_65_192,mask_193_255])))
            # assert(len(vec) == np.sum(np.any([mask_0_64,mask_65_192,mask_193_255], axis=0)))
            # Build the float vector,
            ans = np.empty(len(vec), dtype='float32')
            ans[mask_0_64] = p0 + (p25 - p0) / 64. * vec[mask_0_64]
            ans[mask_65_192] = p25 + (p75 - p25) / 128. * (vec[mask_65_192] - 64)
            ans[mask_193_255] = p75 + (p100 - p75) / 63. * (vec[mask_193_255] - 192)
            return ans

        import time
        ts = time.time()
        # Read global header,
        globmin, globrange, rows, cols = np.frombuffer(fd.read(16), dtype=global_header, count=1)[0]
        t1 = time.time() - ts
        # The data is structed as [Colheader, ... , Colheader, Data, Data , .... ]
        #                         {           cols           }{     size         }
        ts = time.time()
        col_headers = np.frombuffer(fd.read(cols * 8), dtype=per_col_header, count=cols)
        data = np.reshape(np.frombuffer(fd.read(cols * rows), dtype='uint8', count=cols * rows),
                          newshape=(cols, rows))  # stored as col-major,
        t2 = time.time() - ts

        ts = time.time()
        mat = np.empty((cols, rows), dtype='float32')
        for i, col_header in enumerate(col_headers):
            col_header_flt = [uint16_to_float(percentile, globmin, globrange) for percentile in col_header]
            mat[i] = uint8_to_float_v2(data[i], *col_header_flt)
        t3 = time.time() - ts

        return mat.T, t1, t2, t3  # transpose! col-major -> row-major,

    # data = "/home/dawna/mgb3/transcription/exp-yl695/Snst/xvector/cpdaic_1.0_50/data/voxceleb_train_combined_no_sil"
    data = "/scratch/yl695/voxceleb/data/voxceleb_train_combined_no_sil"
    feats_scp = []
    with open(os.path.join(data, "feats.scp"), "r") as f:
        for line in f.readlines():
            utt, scp = line.strip().split(" ")
            feats_scp.append(scp)
    import random
    import time
    ts = time.time()
    time1 = 0
    time2 = 0
    time3 = 0
    for _ in xrange(2):
        num_samples = 640
        batch_length = random.randint(200, 400)
        selected = random.sample(feats_scp, num_samples)
        for utt in selected:
            _, t1, t2, t3 = read(utt, batch_length, shuffle=True)
            time1 += t1
            time2 += t2
            time3 += t3
    te = time.time() - ts
    print("Total time: %f s, time 1: %f s, time 2: %f s, time 3: %f s" % (te, time1, time2, time3))
