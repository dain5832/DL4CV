# import the necessary packages
from DL4CV_PractitionerBundle.pyimagesearch.nn.conv import LeNet
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np

# grab the MNIST dataset (if this is your first time using this
# dataset then the llMB download may take a minute)
print("[INFO] accessing MNIST...")
((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()

# if we are using "channels first" ordering, then reshape the
# design matrix such that the matrix is:
# num_samples x depth x rows x columns
if K.image_data_format() == "channels_first":
    trainData = trainData.reshape((trainData.shape[0], 1, 28, 28))
    testData = testData.reshape((testData.shape[0], 1, 28, 28))

else:
    trainData = trainData.reshape(trainData.shape[0], 28, 28, 1)
    testData = testData.reshape(testData.shape[0], 28, 28, 1)

trainData = trainData.astype("float32") / 255.0
testData = testData.astype("float32") / 255.0

lb = LabelBinarizer()
trainLabels = lb.fit_transform(trainLabels)
testLabels = lb.transform(testLabels)

# compile
opt = SGD(lr=0.01)
model = LeNet.build(width=28, height=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt,\
              metrics=["accuracy"])

# train the network
print("[INFO] testing the network...")
H = model.fit(trainData, trainLabels, validation_data=(testData, testLabels),\
              batch_size=128, epochs=20, verbose=1)

# evaluate
print("[INFO] evaluating the network...")
pred = model.predict(testData, batch_size=128)
print(classification_report(testLabels.argmax(axis=1),\
            pred.argmax(axis=1),\
            target_names=[str(x) for x in lb.classes_]))

# plot
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 20), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 20), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 20), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 20), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()