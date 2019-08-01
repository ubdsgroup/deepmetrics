# Basic tools
import numpy as np
import functools
import time

# Keras imports
from keras import Input
from keras.layers import Dense, Embedding, LSTM, Bidirectional, concatenate
from keras.models import Sequential, load_model, Model
from keras.utils import to_categorical
from keras.metrics import top_k_categorical_accuracy

# Custom classes
from Attention import Attention

class DeepLog:
    def __init__(self, model_path, data_path, retrain=False):

        self.window_size = 15
        self.input_size = 1
        self.hidden_size = 64
        self.num_hidden_layers = 1
        self.num_candidates = 2
        self.num_classes = 29
        self.num_epochs = 50
        self.batch_size = 2048

        self.model_path = model_path
        self.data_path = data_path

        # Defines custom metric for comparing accuracy against the top k most probable predictions
        self.topk_acc = functools.partial(top_k_categorical_accuracy, k=self.num_candidates)
        self.topk_acc.__name__ = 'topk_acc'

        if retrain:
            print("Training new DeepLog model...")
            self.model = self.create_Keras_model("attention")
        else:
            if "Attention" in model_path:
                self.model = load_model(model_path, custom_objects={"Attention":Attention, "topk_acc":self.topk_acc})

    ###################################################################################################################
    """ Generates sequences of length self.window_size from the data.
    
    @:returns inputs: a list of inputs, of length self.window_size
    @:returns outputs: the next value after the input sequence
    """

    def generate_sequences(self, path):
        num_sessions = 0
        inputs = []
        outputs = []
        with open(self.data_path + "/" + path, "r") as datafile:
            for line in datafile.readlines():
                num_sessions += 1
                line = tuple(map(lambda n: n - 1, map(int, line.strip().split())))
                for i in range(len(line) - self.window_size):
                    inputs.append(list(line[i:i + self.window_size]))
                    outputs.append(line[i + self.window_size])
        print("Number of sessions: {}".format(num_sessions))
        print("Number of sequences: {}".format(len(inputs)))

        return np.array(inputs), to_categorical(outputs, self.num_classes)

    ###################################################################################################################

    def generate_anomalies(self, path):
        num_sessions = 0
        inputs = []
        outputs = []
        with open(self.data_path + "/" + path, "r") as datafile:
            for line in datafile.readlines():
                num_sessions += 1
                line = tuple(map(lambda n: n - 1, map(int, line.strip().split())))
                for i in range(len(line) - self.window_size):
                    if line[i + self.window_size] == 28:
                        inputs.append(list(line[i:i + self.window_size]))
                        outputs.append(line[i + self.window_size])
        print("Number of sessions: {}".format(num_sessions))
        print("Number of sequences: {}".format(len(inputs)))

        return np.array(inputs), to_categorical(outputs, self.num_classes)

    ###################################################################################################################

    """ Creates the Keras model to be used for prediction. 
    
    @:param type:   defines the type of model to be trained. 'basic' sets up a two layer LSTM as described in the
                    original DeepLog paper. 'attention' sets up a two layer bi-directional LSTM with an 
                    attention layer, and generally seems to give better results.
        
    @:returns model: the Keras model
    """


    def create_Keras_model(self, type):

        if type == "basic":
            model = Sequential()
            model.add(LSTM(self.window_size, return_sequences=True, input_shape=(self.window_size, self.input_size), activation="relu"))

            for i in range(self.num_hidden_layers):
                model.add(LSTM(self.hidden_size, return_sequences=False, activation="relu"))

            model.add(Dense(self.num_classes, activation="softmax"))
            model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy", self.topk_acc])

            return model

        if type == "attention":

            input = Input(shape=(self.window_size, self.input_size))
            lstm = Bidirectional(LSTM(self.hidden_size,
                                       return_sequences=True,
                                       return_state=True,
                                       dropout=0.1,
                                       activation="relu"))(input)

            lstm, forward_h, forward_c, backward_h, backward_c = Bidirectional(LSTM(self.hidden_size,
                                                                                    return_sequences=True,
                                                                                    return_state=True,
                                                                                    dropout=0.1,
                                                                                    activation="relu"))(lstm)

            state_h = concatenate([forward_h, backward_h])
            context_vector = Attention(self.window_size)([lstm, state_h])
            output = Dense(self.num_classes, activation="softmax", name="output_layer")(context_vector)

            model = Model(inputs=input, outputs=output)
            model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy", self.topk_acc])

            return model

    ###################################################################################################################
    """ Trains the Keras model, and saves it in model directory specified when creating the DeepLog object
     
    @:param train_data:    The name of the file containing the training data. Should be located inside of the data 
                            directory specified when creating the DeepLog object
    @:param validation_data:    Optional validation data to test performance during training
    """

    def train(self, train_data, validation_data=None):

        inputs, outputs = self.generate_sequences(train_data)
        inputs = np.reshape(inputs, (len(inputs), self.window_size, 1))
        if validation_data is not None:
            test_inputs, test_outputs = self.generate_sequences(validation_data)
            test_inputs = np.reshape(test_inputs, (len(test_inputs), self.window_size, 1))
            self.model.fit(inputs, outputs, batch_size=128, verbose=1, epochs=self.num_epochs, validation_data=(test_inputs, test_outputs))
        else:
            self.model.fit(inputs, outputs, batch_size=128, verbose=1, epochs=self.num_epochs)

        print("Trained DeepLog model")

        self.model.save(self.model_path)
        print("Saved model as: " + self.model_path)

    ###################################################################################################################
    """ Evaluates the DeepLog model on a testing set.
    
    @:param file:   The name of the file containing the test data. Should be located inside of the data
                    directory specified when creating the DeepLog object
                    
                    As of 07/30, model achieves 92.7% accuracy for top prediction, 98.45% for top 2.
    """

    def evaluate(self, file):

        if "abnormal" in file:
            inputs, outputs = self.generate_anomalies(file)
        else:
            inputs, outputs = self.generate_sequences(file)
        inputs = np.reshape(inputs, (len(inputs), self.window_size, 1))

        return self.model.evaluate(x=inputs, y=outputs, batch_size=128)

    ###################################################################################################################

if __name__ == '__main__':

    deeplog = DeepLog("DeepLog_Attention_model.h5", "data", retrain=False)
    #deeplog.train("hdfs_train")
    results = deeplog.evaluate("hdfs_test_abnormal")
    print(deeplog.model.metrics_names)
    print(results)