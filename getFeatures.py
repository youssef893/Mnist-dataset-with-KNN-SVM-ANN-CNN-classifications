import matplotlib.pyplot as pt
import numpy as np
from mlxtend.data import loadlocal_mnist
from skimage.feature import hog


class GetFeatures:
    @staticmethod
    def readAndReshapeData(imagesPath, labelsPath):
        features, labels = loadlocal_mnist(
            images_path=imagesPath,
            labels_path=labelsPath)
        features = features.reshape(len(features), 28, 28)
        return features, labels

    @staticmethod
    def extractHOGFeatures(imagesList):
        featuresList = []
        hog_imageList = []
        # Apply hog on every image
        for i in range(len(imagesList)):
            feature, hog_image = hog(imagesList[i], orientations=8, pixels_per_cell=(4, 4),
                                     cells_per_block=(1, 1), visualize=True, transform_sqrt=True)
            featuresList.append(feature)
            hog_imageList.append(hog_image)

        # Reshape data to can plot images
        featuresList = np.array(featuresList)
        hog_imageList = np.array(hog_imageList)
        return featuresList, hog_imageList

    @staticmethod
    def plotImage(image, imageType):
        # Plot first image
        if imageType == "hog":
            pt.imshow(image, cmap=pt.cm.gray)
            pt.show()
        else:
            pt.imshow(image, cmap=pt.rcParams["image.cmap"])
            pt.show()
