import os
import pickle
import re


class AlphaBase(object):
    def __init__(self):
        self.alpha_count = {}
        self.alpha_count_total = 0
        self.alpha_prob = {}
        self.alpha_index = {}
        self.alpha_compressed_size_value = 0
        self.num_languages_value = 0
        pass

    def start(self, data_dir, char_limit_per_language=0):
        for file in os.listdir(data_dir):
            full_file = os.path.join(data_dir, file)
            char_count = 0
            for line_num, line in enumerate(open(full_file, 'r')):
                for char in line:
                    self.alpha_count_total += 1
                    if char in self.alpha_count:
                        self.alpha_count[char] += 1
                    else:
                        self.alpha_count[char] = 1
                char_count += len(line)
                if char_limit_per_language > 0:
                    if char_count >= char_limit_per_language:
                        break
            print(file, self.alpha_count_total)
        for alpha, alpha_count in self.alpha_count.items():
            self.alpha_prob[alpha] = alpha_count / self.alpha_count_total
        self.num_languages_value = len(os.listdir(data_dir))

    def compress(self, compression_factor):
        alpha_list = [(k, v) for k, v in self.alpha_prob.items()]
        alpha_sorted = sorted(alpha_list, key=lambda x: x[1], reverse=True)
        acc_prob = 0
        ix = 1
        for k, v in alpha_sorted:
            acc_prob += v
            if acc_prob < compression_factor:
                self.alpha_index[k] = ix
                ix += 1
            else:
                self.alpha_index[k] = 0  # not found class
        self.alpha_compressed_size_value = ix

    @property
    def alpha_size(self):
        return len(self.alpha_prob)

    @property
    def alpha_compressed_size(self):
        return self.alpha_compressed_size_value

    @property
    def num_languages(self):
        return self.num_languages_value

    def get_alpha_index(self, alpha):
        if alpha in self.alpha_index:
            return self.alpha_index[alpha]
        else:
            return 0  # return 0 for those characters not found

    def get_alpha_string_index_list(self, alpha_string):
        return [self.get_alpha_index(alpha) for alpha in alpha_string]

    @staticmethod
    def filter(inx: str) -> str:
        outx = inx.replace('\n', ' ')
        outx = outx.replace('\t', ' ')
        outx = re.sub('[().,;*:!?]', '', outx)
        return outx

    def save_object_to_file(self, file_name: str):
        """ Save this object to a file
        """
        with open(file_name, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_object_from_file(file_name: str):
        """ Construct an object of type LangByWord from a file
        """
        with open(file_name, 'rb') as input_file:
            obj = pickle.load(input_file)
            return obj

    @staticmethod
    def self_test():
        base_dir = '/Users/frank/data/LanguageDetectionModel/exp_data_test'
        a = AlphaBase()
        a.start(base_dir, 1)
        a.compress(0.999)
        print(a.alpha_size, a.alpha_compressed_size)
        print(a.get_alpha_string_index_list('This is a test'))
        a.save_object_to_file('dog.pk')
        b = AlphaBase.load_object_from_file('dog.pk')
        print(b.alpha_size, b.alpha_compressed_size)
        print(b.get_alpha_string_index_list('This is a test'))

#AlphaBase.self_test()
