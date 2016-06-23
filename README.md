# LanguageDetectionRNN
This project uses Recurrent Neural Networks constructed from the TensorFlow toolkit to build models for language identification.
### Files
* AlphaBase.py - A class for representing the characters of a set of languages.
* LanguageSource.py - A class that acts as a source of training data.
* LangTestData.py - A class that represents the test data.
* LMSystem.py - A class for training and testing a RNN for the language identification task.
* test_LM.py - A class for running a training and testing session on actual data.

### Data
Data is taken from ???
* training data should be in the one file per language format as created in the LanguageDetectionModel project
* testing file should be 

### Instructions for use
* Change the location of the training data. 
* Change the name of the test file. 
* Select the model size.
* Then run: 

python test_LM.py

### Results
Model | CPU training time | GPU training time | Accuracy
------|-------------------|-------------------|-----------
very-small | 20 secs | ? | 0.0841904765749
small | 45 secs | ? | 0.27419047337
medium | 11 mins 10 secs | ? | 0.937761902809
large | ? | ? | 0.99
very-large | ? | ? | 0.99

CPU - MacBook Pro (Late 2013, 2.3 GHz Intel Core i7)

GPU - NVidia Titan X