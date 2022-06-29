import tensorflow
from tensorflow.keras.layers.experimental.preprocessing import Rescaling, Resizing
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, BatchNormalization, MaxPooling2D, Dropout, Dense, Flatten
from tensorflow.keras.layers.experimental.preprocessing import  RandomRotation, RandomZoom, RandomHeight, RandomWidth, RandomContrast, RandomTranslation
from tensorflow.keras.models import  Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L1, L2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping,  ReduceLROnPlateau
from tensorflow.keras.utils import plot_model

import matplotlib.pyplot as plt
import numpy as np
from validators import Max

import deepl_data_preproc as preproc


class DLModel:

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model_save_path = f'saved_models/{self.model_name}'

        self.training_history = None
        self.model = self.__create_model()
        self.compile_model()
        
        plot_model(self.model, show_shapes=True, to_file=f"{model_name}.png")


    def train(self, train_X, train_y, validation_X, validation_y, epochs=100, batch_size=32, stop_early=True):

        checkpointer_callback = ModelCheckpoint(
		filepath=self.model_save_path,
		verbose=1,
		save_weights_only=False,
		save_best_only=True
	    )
    
        earlystopper_callback = EarlyStopping( monitor="val_loss", min_delta=0.0001, patience=20, verbose=1)
        
        reduceLR = ReduceLROnPlateau(
                monitor='val_loss', factor=0.3, patience=8, min_lr=0.000000001, verbose=1)
        
        
        callbacks = [checkpointer_callback, reduceLR]
        if stop_early:
            callbacks.append(earlystopper_callback)

        self.training_history = self.model.fit(
            train_X,
            train_y,
            validation_data=(validation_X, validation_y),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            callbacks=callbacks
        )
    
    def train_full_data(self, train_X, train_y, epochs=100, batch_size=32):
    
        
        reduceLR = ReduceLROnPlateau(
                monitor='loss', factor=0.3, patience=8, min_lr=0.000000001, verbose=1)
        
        
        callbacks = [reduceLR]

        self.training_history = self.model.fit(
            train_X,
            train_y,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            callbacks=callbacks
        )

    def predict_test(self, input):
        probs = self.model.predict(input, batch_size=1)
        predictions = []
        for pred in probs:
            activity_index = np.argmax(pred)
            output = [ 1 if i == activity_index else 0 for i in range(len(pred))]
            predictions.append(output)
        return np.array(predictions)

    def predict(self, input_frame):
        preds = self.model(np.expand_dims(input_frame, axis=0))
        print(f"Preds: {preds}")
        activity_index = np.argmax(preds)
        return activity_index

    def predict_pretty(self, input_frame):
        activ = self.predict(input_frame)

        if activ == 0:
            return 'andar'
        elif activ == 1:
            return 'bicicleta'
        elif activ == 2:
            return 'correr'
        else:
            return 'repouso'

   

    def evaluate(self, X, y, batch_size=1):
        return self.model.evaluate(X, y, batch_size=batch_size)

    def save(self):
        self.model.save(f'{self.model_save_path}.h5', overwrite=True ,include_optimizer=True, save_format='h5')

    def load(self, filename):
        self.model = tensorflow.keras.models.load_model(filename, None)
        self.compile_model()



    def compile_model(self):
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
             metrics=['accuracy']
        )

    def plot_training_history(self):
        # summarize history for accuracy
        plt.plot(self.training_history.history['accuracy'])
        plt.plot(self.training_history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='lower right')
        plt.show()
        # summarize history for loss
        plt.plot(self.training_history.history['loss'])
        plt.plot(self.training_history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper right')
        plt.show()

    def plot_full_training_history(self):
        # summarize history for accuracy
        plt.plot(self.training_history.history['accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='lower right')
        plt.show()
        # summarize history for loss
        plt.plot(self.training_history.history['loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper right')
        plt.show()

    def __create_model(self):
        input_layer = Input(shape=(200, 6, 1), name="Input-Layer")

    

        
        hidden_layer = Flatten()(input_layer)
        
        hidden_layer = Dense(units=512, 
                        activation='relu',
                        bias_regularizer=L2(1e-4))(hidden_layer)
        hidden_layer = Dropout(rate=0.4)(hidden_layer)

        hidden_layer = Dense(units=512, 
                        activation='relu',
                        bias_regularizer=L2(1e-4))(hidden_layer)
        hidden_layer = Dropout(rate=0.3)(hidden_layer)

        hidden_layer = Dense(units=64, 
                        activation='relu',
                        bias_regularizer=L2(1e-4))(hidden_layer)
        hidden_layer = Dropout(rate=0.3)(hidden_layer)

        output_layer = Dense(units=4, activation='softmax', name="Output-Layer")(hidden_layer)

        model = Model(
            inputs=input_layer,
            outputs=output_layer
        )

        return model
