import AlphaBase as AlphaBase


class LangTestData(object):
    def __init__(self):
        pass

    @staticmethod
    def read_data(test_file, seq_len):
        x = []
        y = []
        for line in open(test_file, 'r'):
            lang_id, lang_text = line.split('\t', 2)  # split into two parts using the tab
            lang_text = AlphaBase.AlphaBase.filter(lang_text)
            if len(lang_text) == seq_len:
                x.append(lang_text)
            elif len(lang_text) > seq_len:
                x.append(lang_text[0:seq_len])
            else:
                lang_text_2 = lang_text + ' '
                nlt = [lang_text_2[i % len(lang_text_2)] for i in range(seq_len)]
                nlt = ''.join(nlt)
                x.append(nlt)
            y.append(lang_id)
        return x, y

    @staticmethod
    def read_data_natural(test_file):
        x = []
        y = []
        for line in open(test_file, 'r'):
            line = LanguageSource.LanguageSource.filter(line)
            lang_id, lang_text = line.split('\t', 2)  # split into two parts using the tab
            x.append(lang_text)
            y.append(lang_id)
        return x, y
