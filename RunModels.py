from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, r2_score, precision_score, \
    recall_score
from Models import Models
from getFeatures import GetFeatures


class RunModels:

    @staticmethod
    def printAccuracy(predictionList, labelsList, accuracy):
        print("Accuracy =", accuracy, "%")
        print("Accuracy_Score =", accuracy_score(labelsList, predictionList) * 100, "%")
        print("R2_score =", r2_score(labelsList, predictionList) * 100, "%")
        print("Confusion matrix =", confusion_matrix(labelsList, predictionList))
        print("precision =", precision_score(labelsList, predictionList, average=None) * 100)
        print("Recall =", recall_score(labelsList, predictionList, average=None) * 100)
        print("Classification Report =", classification_report(labelsList, predictionList))

    def calculateAccuracy(self, predictionList, labelsList):
        counter = 0
        for i in range(len(predictionList)):
            if predictionList[i] == labelsList[i]:
                counter += 1
        accuracy = counter / len(predictionList) * 100
        self.printAccuracy(predictionList, labelsList, accuracy)

    def calculateModelsAccuracy(self, KnnPrediction, SvmPrediction,
                                ANNPrediction, CNNPrediction, testLabels):
        print('Wait for calculating accuracy...')
        print("KNN Model:")
        self.calculateAccuracy(KnnPrediction, testLabels)
        print("==============================================================")
        print("Svm Model:")
        self.calculateAccuracy(SvmPrediction, testLabels)
        print("==============================================================")
        print("ANN Model:")
        self.calculateAccuracy(ANNPrediction, testLabels)
        print("==============================================================")
        print("CNN Model:")
        self.calculateAccuracy(CNNPrediction, testLabels)

    def applyModels(self, trainFeatures, trainLabels, testFeatures, testLabels):
        hogTrainFeatures, hogTrainImages = GetFeatures.extractHOGFeatures(trainFeatures)
        hogTestFeatures, hogTestImages = GetFeatures.extractHOGFeatures(testFeatures)
        # Plot first hog image
        print('Hog is applied and one image was plotted')
        GetFeatures.plotImage(hogTrainImages[0], "hog")
        KnnPrediction = Models.applyKNN(hogTrainFeatures, trainLabels, hogTestFeatures)
        SvmPrediction = Models.applySVM(hogTrainFeatures, trainLabels, hogTestFeatures)
        ANNPrediction = Models.applyANN(hogTrainFeatures, trainLabels, hogTestFeatures)
        CNNPrediction = Models.applyCNN(trainFeatures, trainLabels, testFeatures, testLabels)
        # Calculate accuracy of models on test data
        self.calculateModelsAccuracy(KnnPrediction, SvmPrediction,
                                     ANNPrediction, CNNPrediction, testLabels)
