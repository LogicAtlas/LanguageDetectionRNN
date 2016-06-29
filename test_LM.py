
"""
A Reccurent Neural Network (LSTM) implementation example using TensorFlow library
"""
import os
import AlphaBase
import LanguageSource as LanguageSource
import LangTestData as langTestData
import LMSystem as LMSystem
import ParmSet as ParmSet

# Set the location of training and testing files
base_dir = '/home/frank'
lang_data_dir = base_dir + '/data/LanguageDetectionModel/exp_data_test'
test_data_file_name = base_dir + '/data/LanguageDetectionModel/europarl.test'
alpha_file_name = 'alpha_dog.pk'

# Get the character set used as network input. If it exists then read it. If
# it does not exist then generate it. Compress the number of characters in the set.
if os.path.isfile(alpha_file_name):
    alpha_set = AlphaBase.AlphaBase.load_object_from_file(alpha_file_name)
else:
    alpha_set = AlphaBase.AlphaBase()
    alpha_set.start(lang_data_dir, 10000000)  # read data but limit the number of characters per language
    alpha_set.compress(0.999)  # use only this fraction of the characters cumulative distribution
    alpha_set.save_object_to_file(alpha_file_name)
print('alpha size:', alpha_set.alpha_size, 'alpha compressed size', alpha_set.alpha_compressed_size)

# Setup the source of training data.
train_lang_data = LanguageSource.LanguageSource(alpha_set)
train_lang_data.begin(lang_data_dir)

# Parameters for training are set here select for (very-small,small,medium,large,very-large)
parms = ParmSet.ParmSet('mega-large')
parms.print()

# Get test data
lang_db_test = langTestData.LangTestData()
x_test, y_test = lang_db_test.read_data(test_data_file_name, parms.n_steps, alpha_set.filter)
test_data = train_lang_data.get_ml_data_matrix(alpha_set.alpha_compressed_size, x_test)  # get one-hot version of data
y_test2 = [train_lang_data.language_name_to_index[y_l] for y_l in y_test]  # convert the language names to indexes
test_label = train_lang_data.get_class_rep(y_test2, parms.n_classes)  # convert the class indexes to one-hot vectors

# Train and then test a RNN language model for the data
my_lm = LMSystem.LMSystem(alpha_set.alpha_compressed_size, parms)
model_base_file_name = 'models/model_t'  # save the models periodically in this directory
my_lm.train(train_lang_data, model_base_file_name)
my_lm.test(test_data, test_label, model_base_file_name + '_final')
