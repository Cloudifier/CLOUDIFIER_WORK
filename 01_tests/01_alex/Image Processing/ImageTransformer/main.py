import sys
import glob
import numpy as np
import pandas as pd
from PIL import Image #for getting image properties
from PyQt5.QtWidgets import (QWidget, QLabel, QLineEdit, QPushButton, QMessageBox,
    QTextEdit, QGridLayout, QApplication)

class ImageHelper:
    @staticmethod
    def getAllImages(filePath):
        filesFound = []
        fileTypes = ('*.jpg', '*.jpeg', '*.png')
        for fileType in fileTypes:
            filesFound.extend(glob.glob(filePath + '\\' + fileType))
        return filesFound

    @staticmethod
    def getImageSize(filePath):
        with Image.open(filePath) as img:
            type(img.size)
            width, height = img.size
            return str(width) + '__' + str(height)

    @staticmethod
    def toGreyscale(filePath):
        img = Image.open(filePath).convert('LA')
        return img

    @staticmethod
    def to1DArray(image):
        imageArray = np.array(image)
        flatArray = imageArray.ravel()
        return flatArray

    @staticmethod
    def getSizeDictionary(imageFiles):
        sizes = {};
        for imageFile in imageFiles:
            imageSize = ImageHelper.getImageSize(imageFile)
            sizes[imageFile] = imageSize
        return sizes




class Example(QWidget):


    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        source = QLabel('Source Path')
        destination = QLabel('Destination Path')
        label = QLabel('Label')

        self.sourceEdit = QLineEdit()
        self.destinationEdit = QLineEdit()
        self.labelEdit = QLineEdit()
        self.sourceEdit.setText('D:\Personal\Projects\Cloudifier\Buttons');
        self.destinationEdit.setText('D:\Personal\Projects\Cloudifier\Poze\Processed')
        self.labelEdit.setText('button')
        generateButton = QPushButton("Generate Dataset")
        generateButton.clicked.connect(self.generateDataset)

        grid = QGridLayout()
        grid.setSpacing(10)

        grid.addWidget(source, 1, 0)
        grid.addWidget(self.sourceEdit, 1, 1)

        grid.addWidget(destination, 2, 0)
        grid.addWidget(self.destinationEdit, 2, 1)

        grid.addWidget(label, 3, 0)
        grid.addWidget(self.labelEdit, 3, 1)

        grid.addWidget(generateButton, 4, 0)

        self.setLayout(grid)

        self.setGeometry(1000, 300, 1000, 300)
        self.setWindowTitle('Review')
        self.show()

    def showDialog(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)

        msg.setText("Successful dataset generation!")
        msg.setWindowTitle("Info")
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)

        retval = msg.exec_()


    def generateDataset(self):
        srcPath = self.sourceEdit.text()
        dstPath = self.destinationEdit.text()
        label = self.labelEdit.text()
        imageFiles = ImageHelper.getAllImages(srcPath)
        sizeDictionary = ImageHelper.getSizeDictionary(imageFiles)
        sizes = list(sizeDictionary .values()) #get just the values from dictionary. ex: 1200__800, 600__800, etc
        uniqueSizes = set(sizes) #set gives you unique values from a list by removing duplicates from collection: https://docs.python.org/2/library/sets.html

        for size in uniqueSizes: #now that you have the unique sizes, iterate over the dictionary and get the corresponding files for each size type
            df = pd.DataFrame()
            images = [imageFile for imageFile, imageSize in sizeDictionary.items() if imageSize == size] #you now have the files for this unique file size, you can get the 1D array
            for image in images:
                grayscale = ImageHelper.toGreyscale(image)
                arr = [label]
                arr = np.append(arr, ImageHelper.to1DArray(grayscale))
                #hacky stuff no 2, needed to bring the data to the same number of properties
                arr = np.append(arr, [0 for x in range(30000 - 1 - 1 - len(arr))])
                df = df.append(pd.Series(arr), ignore_index=True)

            # df.insert(0, '', [label for x in range(len(images))])

            #df[len(df.columns)] = [label for x in range(len(images))]
            #hacky stuff no 1, just to allow fit to work properly. remember that fit works with more than one class. besides 'button' we will introduce 'null'
            #df = df.append(np.zeros(len(df.columns)-1), ignore_index=True)
            zeroArray = np.append(['not_button'], [0 for x in range(30000 - 1)])
            print('array length: ' + str(len(zeroArray)) + ' range: ' + str(len(df.columns)))
            df = df.append(pd.Series(zeroArray), ignore_index=True)
            dest = dstPath + '\dataset_' + size + '.csv'
            df.to_csv(dest, sep=',')
        self.showDialog()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())