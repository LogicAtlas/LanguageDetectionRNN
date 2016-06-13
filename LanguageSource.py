import AlphaBase as AlphaBase
import os
import numpy as np


class LanguageSource(object):
    def __init__(self, alpha_set: AlphaBase):
        self.language_file_id = {}
        self.language_name_to_index = {}
        self.language_index_to_name = {}
        self.alpha_set = alpha_set
        self.current_language_id = 0
        self.num_languages = 0
        pass

    def begin(self, data_dir):
        for ix, file in enumerate(os.listdir(data_dir)):
            lang_name = file.split('-')[1].split('.')[0]  # get the language's name (string representation)
            full_file_name = data_dir + '/' + file
            self.language_file_id[lang_name] = open(full_file_name, 'r')
            self.language_name_to_index[lang_name] = ix
            self.language_index_to_name[ix] = lang_name
        self.num_languages = len(self.language_file_id)

    @staticmethod
    def read_with_restart(fh, read_len):
        ft1 = fh.tell()
        r_data = fh.read(read_len)
        ft2 = fh.tell()
        if ft1 == ft2:
            print('End of file found on:', fh)
            fh.seek(0)
            r_data = fh.read(read_len)
        return r_data

    def get_next_batch(self, batch_size: int, seq_len) -> ([str], [str]):
        lang_str_list = []
        lang_id_list = []
        for bi in range(batch_size):
            # get the next language id and update it for the cycle
            lang_id = self.current_language_id
            self.current_language_id += 1
            if self.current_language_id >= self.num_languages:
                self.current_language_id = 0
            # Get the file handle for the language and read from it.
            lang_name = self.language_index_to_name[lang_id]
            fd = self.language_file_id[lang_name]
            lang_str = self.alpha_set.filter(self.read_with_restart(fd, seq_len))
            # Continue reading until a string of the correct length is created
            while len(lang_str) < seq_len:
                rem = seq_len - len(lang_str)
                lang_str += self.alpha_set.filter(self.read_with_restart(fd, rem))
            lang_str_list.append(lang_str)
            lang_id_list.append(lang_id)
        return lang_str_list, lang_id_list

    def get_next_batch_one_hot(self, batch_size: int, seq_len):
        batch_x, batch_y = self.get_next_batch(batch_size, seq_len)
        # one-hot encode the strings, batch_xs = [n_step x batch_size x n_input]
        batch_xs = self.get_ml_data_matrix(self.alpha_set.alpha_compressed_size, batch_x)
        # one-hot encode the languages ids of each string, batch_ys = [n_step x batch_size x n_classes]
        batch_ys = self.get_class_rep(batch_y, self.num_languages)
        return batch_xs, batch_ys

    def get_ml_data_matrix(self, n_char: int, lang_strings: [str]):
        """ Return a numpy matrix representing the list of strings in lang_str.
        Each character of the strings in lang_string is represented by a one-hot encoding vector
        Each string is then a concatenation of the one-hot vectors.
        :param n_char:
        :param lang_strings: a list of strings to find representations for
        :return: a numpy matrix with a row for each string in lang_string
        """
        n = len(lang_strings)
        m = len(lang_strings[0])  # each string is the same length
        # Create the empty matrix. Each row is a string. Each row has m one-hot vectors, each of length n_char.
        rep_mat = np.zeros((m, n, n_char), dtype=np.float32)
        for i, str_x in enumerate(lang_strings):
            for j, char in enumerate(str_x):
                rep = self.alpha_set.get_alpha_index(char)  # look up the characters integer representation
                rep_mat[j, i, rep] = 1  # set the one-hot bit in the vector for 'char' in string 'str_x'
        return rep_mat

    @staticmethod
    def get_class_rep(class_id_list, n_class):
        """ Get the one-hot representation matrix of a list of class indices.
        :param class_id_list:
        :param n_class: maximum number of classes
        :return: a matrix of one-hot vectors. Each row is a one-hot vector.
        """
        n_class_id = len(class_id_list)
        class_vec = np.zeros((n_class_id, n_class), dtype=np.float32)
        for i, class_id in enumerate(class_id_list):
            class_vec[i, class_id] = 1
        return class_vec

    @staticmethod
    def self_test():
        ab = AlphaBase.AlphaBase.load_object_from_file('alpha_dog.pk')
        ls = LanguageSource(ab)
        ls.begin('/Users/frank/data/LanguageDetectionModel/exp_data_test')
        # x, y = ls.get_next_batch(420, 32)
        # for x1, y1 in zip(x, y):
        #    print(y1, x1, len(x1))
        # bs = 64
        # for i in range(10000000):
        #    x, y = ls.get_next_batch(bs, 128)
        #    if i % 1000 == 0:
        #        print(i, (i+1)*bs)
        print('test complete.')
# LanguageSource.self_test()
