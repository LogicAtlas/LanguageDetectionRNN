"""
  A Class for reading Language Test Data.
"""


class LangTestData(object):
    def __init__(self):
        pass

    @staticmethod
    def read_data(test_file, seq_len, filter):
        """ Read a file and extract the data.
        :param test_file: A file name containing test data in a specific format.
        :param seq_len: the length of the sequence to extract.
        :param filter: a mapping from input string to output string
        :return: x - a list of sequence each of length seq_len, y - a list of classes (indexes) corresponding to x
        """
        x = []  # the test strings
        y = []  # the class indexes
        for line in open(test_file, 'r'):
            lang_id, lang_text = line.split('\t', 2)  # split into two parts using the tab
            lang_text = filter(lang_text)
            # Construct a string of the exact length 'seq_len'
            if len(lang_text) == seq_len:
                x.append(lang_text)  # exact length match
            elif len(lang_text) > seq_len:
                x.append(lang_text[0:seq_len])  # too long, so truncate
            else:  # too short, append circularly
                lang_text_2 = lang_text + ' '
                nlt = [lang_text_2[i % len(lang_text_2)] for i in range(seq_len)]
                nlt = ''.join(nlt)
                x.append(nlt)
            y.append(lang_id)
        return x, y

