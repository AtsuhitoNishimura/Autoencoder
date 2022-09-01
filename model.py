import os
import cv2
import gc
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras.callbacks import TensorBoard, Callback
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Reshape, Dropout


class PlotLossGraph(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        plt.axis([0, self.epochs, 0, 0.25])
        plt.ion()
        
    def on_epoch_end(self, epoch, logs={}):
        self.graph_protter(logs)
        plt.clf()
        
    def on_train_end(self, logs={}):
        self.graph_protter(logs)
        plt.show()
    
    def graph_protter(self, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        plt.plot(range(len(self.losses)), self.losses, label='loss')
        plt.plot(range(len(self.val_losses)), self.val_losses, label='val_loss')
        plt.legend(loc='best')
        plt.draw()
        plt.pause(0.05)


class AutoEncorder():
    def __init__(self):
        self.build_model()
        
    def build_model(self):
        input_img = Input(shape=(224,224,3))
        x = Conv2D(64, (3, 3) , activation = 'relu', padding = 'same')(input_img)
        x = Conv2D(64, (3, 3) , activation = 'relu', padding = 'same')(x)
        x = MaxPooling2D((2, 2), strides=None)(x)
        x = Conv2D(128, (3, 3) , activation = 'relu', padding = 'same')(x)
        x = Conv2D(128, (3, 3) , activation = 'relu', padding = 'same')(x)
        x = MaxPooling2D((2, 2), strides=None)(x)
        x = Conv2D(256, (3, 3) , activation = 'relu', padding = 'same')(x)
        x = Conv2D(256, (3, 3) , activation = 'relu', padding = 'same')(x)
        x = MaxPooling2D((2, 2), strides=None)(x)
        x = Conv2D(512, (3, 3) , activation = 'relu', padding = 'same')(x)
        x = Conv2D(512, (3, 3) , activation = 'relu', padding = 'same')(x)
        x = MaxPooling2D((2, 2), strides=None)(x)
        x = Conv2D(512, (3, 3) , activation = 'relu', padding = 'same')(x)
        x = Conv2D(512, (3, 3) , activation = 'relu', padding = 'same')(x)
        x = MaxPooling2D((2, 2), strides=None)(x)
        x = Flatten()(x)
        x = Dropout(0.5)(x)
        x = Dense(4096, activation = 'relu')(x)
        x = Dense(1024, activation = 'relu')(x)
        x = Dense(150528, activation = 'sigmoid')(x)
        output_img = Reshape((224,224,3))(x)
        
        model = Model(inputs=input_img, outputs=output_img)
        model.compile(optimizer='adam', loss='binary_crossentropy')
        self.model = model
        
    def data_loder(self, train_path, test_path):
        list_sequences_train = os.listdir(os.path.join(train_path))
        list_sequences_test = os.listdir(os.path.join(test_path))
        train_img_list = []
        test_img_list = []
        test_img_shape_list = []
        
        for train_img_ID in list_sequences_train:
            train_img = cv2.imread(f"{train_path}/{train_img_ID}")
            train_img = cv2.resize(train_img, [224,224])
            train_img_list.append(train_img)
            
        train_img_list = np.array(train_img_list)
        train_img_list = train_img_list.astype('float32')
        train_img_list /= 255
        
        for test_img_ID in list_sequences_test:
            test_img = cv2.imread(f"{test_path}/{test_img_ID}")
            test_img_shape_list.append(test_img.shape)
            test_img = cv2.resize(test_img, [224,224])
            test_img_list.append(test_img)
            
        test_img_list = np.array(test_img_list)
        test_img_list = test_img_list.astype('float32')
        test_img_list /= 255
        
        gc.collect()
        self.train_img_list = train_img_list
        self.test_img_list = test_img_list
        self.test_img_shape_list = test_img_shape_list
    
    def display_img(self):
        decode_img = self.model.predict(self.test_img_list)
        n = 10
        plt.figure(figsize=(20, 4))
        
        for i in range(1, n + 1):
            ax = plt.subplot(2, n, i)
            plt.imshow(cv2.resize(self.test_img_list[i][..., ::-1].reshape(224, 224, 3), 
                                  [self.test_img_shape_list[i][1],self.test_img_shape_list[i][0]]))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax = plt.subplot(2, n, i + n)
            plt.imshow(cv2.resize(decode_img[i][..., ::-1].reshape(224, 224, 3), 
                                  [self.test_img_shape_list[i][1],self.test_img_shape_list[i][0]]))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            
        plt.show()
        keyboardClick=False
        while keyboardClick != True:
            keyboardClick=plt.waitforbuttonpress()       
    
    def train(self, train_path, test_path, epoch=1000, batch=8):
        self.data_loder(train_path, test_path)
        plot_loss_graph = PlotLossGraph()
        plot_loss_graph.epochs = epoch

        self.model.fit(self.train_img_list, self.train_img_list, 
                       epochs=epoch, 
                       verbose=1, 
                       batch_size=batch, 
                       shuffle=True, 
                       validation_data=(self.test_img_list, self.test_img_list), 
                       callbacks=[plot_loss_graph, TensorBoard(log_dir='/tmp/autoencoder')])
        
        self.display_img()