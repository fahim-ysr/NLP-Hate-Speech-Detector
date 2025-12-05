# Importing required modules
from entity.config_entity import ModelTrainerConfig

# For Machine Learning
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, SpatialDropout1D
from tensorflow.keras.optimizers import RMSprop
from constants import *

class ModelArchitecture:
    """
    Defines LSTM model architecture for hate speech detection
    """

    # Constructor function
    def __init__(self):
        pass

    def get_model(self):
        """
        Builds and returns the compiled LSTM model
        """

        model = Sequential()

        # Converts each word indices to a vectors of fixed size (model learns words relationships)
        model.add(Embedding(MAX_WORDS,100,input_length=MAX_LENGTH))

        # Helps prevent overfitting by ignoring 20% of the output during training
        model.add(SpatialDropout1D(0.2))

        # Long-short term memory (LSTM) is the heart of our model. It learns sequential patterns in the text
        model.add(LSTM(100,dropout=0.2,recurrent_dropout=0.2))

        # Binary output of 0 and 1. 0 means no hate and 1 means hate. Single neuron with sigmoid activation
        model.add(Dense(1,activation='sigmoid'))

        # Builds the model explicitly since we are using the new version of Keras
        model.build(input_shape=(None, MAX_LENGTH))

        # Prints the model summary
        model.summary()

        # Compiles with binary crossentropy since this is a binary classification
        model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])

        return model