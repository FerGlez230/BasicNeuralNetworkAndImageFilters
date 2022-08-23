# Basic neural network and image filters
This project allows to upload an image and predict if its a rhino, elephant or zebra due to a simple neural network made with tensorflow. Also allows to apply several filters as fourier, fourier inverse, edge detection, Gaussian, linear blur processing, Laplacian filter. Finally it can shows an histogram for the result image with the filter previously applied.

## Technologies
Project is created with:
* Tensorflow.keras https://www.tensorflow.org/api_docs/python/tf/keras
* Tkinter https://docs.python.org/3/library/tkinter.html
* Matplotlib https://matplotlib.org/
* Numpy https://numpy.org/
* Python 3.8.8 https://www.python.org/downloads/release/python-388

## Sources
This project is inspired by Alejandro Puig [@puigalex](https://github.com/puigalex) 

## How to run this project?
* Have a directory called data, inside two directories (entrenamiento y validacion) and inside each of these two one directory for each animal with at least 300 animal's images. 

The data set was taken from https://www.kaggle.com/biancaferreira/african-wildlife. Made by [@Bianca Ferreira](https://www.kaggle.com/biancaferreira) 

![directory](https://user-images.githubusercontent.com/31389972/186055354-bea8e4a6-887e-4d07-a414-860e52bbd748.JPG)
* Run train.py file in order to create the files with trained model
* Run prediccion.py file 
* Upload an image and it will shows the prediction
* Optionally choose a filter to be applied to the image and select 'Histograma' to create and histogram of the resulting image

## Example of use
![filters](https://user-images.githubusercontent.com/31389972/186054432-bd161472-a5d9-4ca8-8e78-a8a8534ae871.JPG)
