import numpy as np
import cv2

#initialize the class and set the seed
labels = ["dog","cat","panda"]
np.random.seed(1)

#randomly initialize our weight matrix and bias vector
W = np.random.rand(3,3072)
b = np.random.rand(3)
orig = cv2.imread("/Users/Neto/Desktop/Aprendizados/2020/aprendizados/livros_codigos/deep_learning_for_computer_vison_with_python/StarterBundleTest-master/beagle.jpeg")
image = cv2.resize(orig, (32,32)).flatten()

#Compute the output scores by taking the dot product
scores = W.dot(image) + b

#loop over the scores + labels and display them
for(label,score) in zip(labels,scores):
    print("[INFO] {}: {:.2f}".format(label,score))

#draw the label with the highest score
cv2.putText(orig, "Label: {}".format(labels[np.argmax(scores)]),
            (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0),2)
cv2.imshow("Image",orig)
cv2.waitKey(0)
