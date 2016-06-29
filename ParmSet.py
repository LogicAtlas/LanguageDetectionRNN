class ParmSet:
    """ A class for holding the training parameters of a Language Detection RNN model
    """
    learning_rate = 0.0005  # model learning rate
    training_cycles = 16000  # total number of training cycle to perform
    batch_size = 64  # mini-batch size
    display_step = 10  # display a line of status this often
    model_step = 100  # write models this often
    # Network Parameters
    # number of characters in the set of languages, also the size of the one-hot vector encoding characters
    n_steps = 64  # time steps
    n_hidden = 128  # hidden layer num of features
    n_classes = 21  # total number of class, (the number of languages in the database)

    def __init__(self, set_size):
        if set_size == 'very-small':
            self.learning_rate = 0.0005
            self.training_cycles = 2000
            self.batch_size = 32
            self.display_step = 10
            self.model_step = 100
            self.n_steps = 32
            self.n_hidden = 64
            self.n_classes = 21
        elif set_size == 'small':
            self.learning_rate = 0.0005
            self.training_cycles = 10000
            self.batch_size = 64
            self.display_step = 10
            self.model_step = 100
            self.n_steps = 32
            self.n_hidden = 64
            self.n_classes = 21
        elif set_size == 'medium':
            # time on cpu: 15 min, time on GPU: ? min, accuracy: 0.93576190869
            self.learning_rate = 0.0005
            self.training_cycles = 100000
            self.batch_size = 64
            self.display_step = 10
            self.model_step = 100
            self.n_steps = 64
            self.n_hidden = 128
            self.n_classes = 21
        elif set_size == 'large':
            # time on cpu: ? min, time on GPU: ? min, accuracy: ?
            self.learning_rate = 0.00001
            self.training_cycles = 1*1000*1000
            self.batch_size = 64
            self.display_step = 10
            self.model_step = 100
            self.n_steps = 64
            self.n_hidden = 256
            self.n_classes = 21
        elif set_size == 'very-large':
            # time on cpu: ? min, time on GPU: ? min, accuracy: ?
            self.learning_rate = 0.00001
            self.training_cycles = 50 * 1000 * 1000
            self.batch_size = 128
            self.display_step = 10
            self.model_step = 10000
            self.n_steps = 64
            self.n_hidden = 1024
            self.n_classes = 21
        elif set_size == 'mega-large':
            # time on cpu: ? min, time on GPU: ? min, accuracy: ?
            self.learning_rate = 0.00001
            self.training_cycles = 50 * 1000 * 1000
            self.batch_size = 128
            self.display_step = 10
            self.model_step = 50000
            self.n_steps = 64
            self.n_hidden = 2048
            self.n_classes = 21          
        else:
            print('error in ParmSet parameters size')

    def print(self):
        print('learning rate', self.learning_rate,
              'training cycles', self.training_cycles,
              'batch size', self.batch_size,
              'display_step', self.display_step,
              'model_step', self.model_step,
              'n_step', self.n_steps,
              'n_hidden', self.n_hidden,
              'n_classes', self.n_classes)

    @staticmethod
    def self_test():
        my_parms = ParmSet('very-small')
        print('learning rate', my_parms.learning_rate,
              'training cycles', my_parms.training_cycles,
              'batch size', my_parms.batch_size)
