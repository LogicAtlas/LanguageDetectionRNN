# LanguageDetectionRNN
This project uses Recurrent Neural Networks constructed from the TensorFlow toolkit to build models for language identification.
### Files
* AlphaBase.py - A class for representing the characters of a set of languages.
* LanguageSource.py - A class that acts as a source of training data.
* LangTestData.py - A class that represents the test data.
* LMSystem.py - A class for training and testing a RNN for the language identification task.
* ParmSet.py - A class for selecting the RNN model by picking a set of parameters from predefined values.
* test_LM.py - A class for running a training and testing session on actual data.

### Data
Data is taken from European Parliament Proceedings Parallel Corpus 1996-2011

* training data should be in the one file per language format as created in the LanguageDetectionModel project
* testing file should be europarl.test as in the LanguageDetectionModel project

### Instructions for use - change the file test_LM.py as follows:
* Change the location of the training data. 
* Change the name of the test file. 
* Select the model size (parameter of ParmSet).
* Then run it: 

python test_LM.py

### Results
Model | CPU training time | GPU training time | Accuracy
------|-------------------|-------------------|-----------
very-small | 20 secs | 10 secs | 0.0841904765749
small | 45 secs | 16 secs | 0.27419047337
medium | 11 mins 10 secs | 2 mins 35 secs | 0.937761902809
large | ? | 27 mins | 0.93928
very-large | ? | 24 hours | 0.99761912

CPU - MacBook Pro (Late 2013, 2.3 GHz Intel Core i7)

GPU - NVidia Titan X