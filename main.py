from RunModels import RunModels
from getFeatures import GetFeatures


def main():
    # Read data
    print('Wait for reading data...')
    testFeatures, testLabels = GetFeatures.readAndReshapeData('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte')
    trainFeatures, trainLabels = GetFeatures.readAndReshapeData('train-images.idx3-ubyte', 'train-labels.idx1-ubyte')
    print('Data was read and one image was plotted')
    # Plot first real image
    GetFeatures.plotImage(trainFeatures[0], "real")
    # Apply hog features
    print('Wait for applying hog...')
    # Test model
    run_models = RunModels()
    run_models.applyModels(trainFeatures, trainLabels, testFeatures, testLabels)


main()
