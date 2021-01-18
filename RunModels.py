from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, r2_score, precision_score, \
    recall_score
from Models import Models
from sklearn.metrics import f1_score
from getFeatures import GetFeatures


class RunModels:

    @staticmethod
    def printAccuracy(predictionList, labelsList):
        print("Accuracy_Score =", accuracy_score(labelsList, predictionList) * 100, "%")
        print("R2_score =", r2_score(labelsList, predictionList) * 100, "%")
        print("F1_Score =", f1_score(labelsList, predictionList, average='weighted') * 100, "%")
        print("Confusion matrix =", confusion_matrix(labelsList, predictionList))
        print("precision =", precision_score(labelsList, predictionList, average=None) * 100)
        print("Recall =", recall_score(labelsList, predictionList, average=None) * 100)
        print("Classification Report =", classification_report(labelsList, predictionList))


    def calculateModelsAccuracy(self, KnnPrediction, SvmPrediction,
                                ANNPrediction, CNNPrediction, testLabels):
        print('Wait for calculating accuracy...')
        print("KNN Model:")
        self.printAccuracy(KnnPrediction, testLabels)
        print("==============================================================")
        print("Svm Model:")
        self.printAccuracy(SvmPrediction, testLabels)
        print("==============================================================")
        print("ANN Model:")
        self.printAccuracy(ANNPrediction, testLabels)
        print("==============================================================")
        print("CNN Model:")
        self.printAccuracy(CNNPrediction, testLabels)

    def applyModels(self, trainFeatures, trainLabels, testFeatures, testLabels):
        hogTrainFeatures, hogTrainImages = GetFeatures.extractHOGFeatures(trainFeatures)
        hogTestFeatures, hogTestImages = GetFeatures.extractHOGFeatures(testFeatures)
        # Plot first hog image
        print('Hog is applied and one image was plotted')
        GetFeatures.plotImage(hogTrainImages[0], "hog")
        model = Models()
        KnnPrediction = model.applyKNN(hogTrainFeatures, trainLabels, hogTestFeatures)
        SvmPrediction = model.applySVM(hogTrainFeatures, trainLabels, hogTestFeatures)
        ANNPrediction = model.applyANN(hogTrainFeatures, trainLabels, hogTestFeatures)
        CNNPrediction = model.applyCNN(trainFeatures, trainLabels, testFeatures, testLabels)
        # Calculate accuracy of models on test data
        self.calculateModelsAccuracy(KnnPrediction, SvmPrediction,
                                     ANNPrediction, CNNPrediction, testLabels)
