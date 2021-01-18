import numpy as np
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier


class Models:
    @staticmethod
    def compileANN(ann, trainFeatures, trainLabels, testFeatures):
        loss = 'categorical_crossentropy'  # loss function which is used with classes that are greater than 2
        # compile and fit ann
        ann.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
        # epochs is the number of iterations
        # batch_size is the number of samples per gradient update for training
        ann.fit(trainFeatures, to_categorical(trainLabels), epochs=50, batch_size=32)
        # test ann
        prediction = ann.predict(testFeatures)
        prediction = np.argmax(prediction, axis=1)
        return prediction

    def applyANN(self, trainFeatures, trainLabels, testFeatures):
        print('Wait for fitting ANN classifier and testing it...')
        # Initialize the constructor
        ann = Sequential()
        # Add an input layer
        ann.add(Dense(512, activation='relu', input_dim=392))
        # Add an hidden layer
        ann.add(Dense(256, activation='relu'))
        ann.add(Dense(128, activation='relu'))
        ann.add(Dense(64, activation='relu'))
        ann.add(Dense(32, activation='relu'))
        ann.add(Dense(16, activation='relu'))
        # softmax normalizes it into a probability distribution consisting of
        # K probabilities proportional to the exponentials of the input
        # Add an output layer
        ann.add(Dense(10, activation='softmax'))
        prediction = self.compileANN(ann, trainFeatures, trainLabels, testFeatures)
        return prediction

    @staticmethod
    def applyKNN(trainFeatures, trainLabels, testFeatures):
        print('Wait for fitting knn classifier and testing it...')
        # Apply KNN classifier with k = 10 , i find 10 best k
        knn = KNeighborsClassifier(10, metric='euclidean')
        knn.fit(trainFeatures, trainLabels)  # fit train data
        prediction = knn.predict(testFeatures)  # test data
        return prediction

    @staticmethod
    def applySVM(trainFeatures, trainLabels, testFeatures):
        print('Wait for fitting SVM classifier and testing it...')
        SVM = svm.SVC()
        SVM.fit(trainFeatures, trainLabels)
        prediction = SVM.predict(testFeatures)
        return prediction

    @staticmethod
    def compileCnn(cnn, trainFeatures, trainLabels, testFeatures, testLabels):
        # compile cnn
        cnn.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        cnn.fit(trainFeatures.reshape(60000, 28, 28, 1), to_categorical(trainLabels), epochs=15, batch_size=128,
                validation_data=(testFeatures.reshape(10000, 28, 28, 1), to_categorical(testLabels)))
        prediction = cnn.predict(testFeatures.reshape(10000, 28, 28, 1))
        score = cnn.evaluate(testFeatures.reshape(10000, 28, 28, 1), to_categorical(testLabels), verbose=0)
        print("accuracy CNN", score[1])
        prediction = np.argmax(prediction, axis=1)
        return prediction

    def applyCNN(self,trainFeatures, trainLabels, testFeatures, testLabels):
        cnn = Sequential()
        # first conv layer
        cnn.add(Conv2D(filters=32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)))
        # second layer
        cnn.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))  # filters is number of filters & kernel size is size of filter
        # max pooling layer
        cnn.add(MaxPooling2D(pool_size=(2, 2)))
        cnn.add(Conv2D(filters=128, kernel_size=(3, 3), activation="relu"))
        cnn.add(Conv2D(filters=256, kernel_size=(3, 3), activation="relu"))
        cnn.add(MaxPooling2D(pool_size=(2, 2)))
        cnn.add(Conv2D(filters=256, kernel_size=(3, 3), activation="relu"))
        cnn.add(MaxPooling2D(pool_size=(2, 2)))
        # Flatten layer
        cnn.add(Flatten())
        # Fully connected layer
        cnn.add(Dense(512, activation="relu"))
        cnn.add(Dense(10, activation="softmax"))
        prediction = self.compileCnn(cnn, trainFeatures, trainLabels, testFeatures, testLabels)
        return prediction

