import os
from face_recognition.face_recognition_cli import image_files_in_folder
import face_recognition
import math
import cv2
import pickle
import numpy as np
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    X = []
    y = []
    train_dir = '/home/genex/jnotebook/train_dirNew'
    model_save_path = 'knnSaved'
    # Loop through each person in the training set
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
              continue

        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = cv2.imread(img_path)
            #image = cv2.imread(img_path)
            #image = cv2.resize(image, (600,400), interpolation = cv2.INTER_AREA) 
            face_bounding_boxes = face_recognition.face_locations(image)
            if len(face_bounding_boxes) != 1:
                # If there are no people (or too many people) in a training image, skip the image.
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                # Add face encoding for current image to the training set
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes, num_jitters=0)[0])
                y.append(class_dir)
            
    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)
        param_grid = {'n_neighbors':np.arange(n_neighbors-10, n_neighbors+10)} # should be + - 5 in arange
        # Create and train the KNN classifier
        knn_clf = neighbors.KNeighborsClassifier(algorithm=knn_algo, weights='distance')
        #use gridsearch to test all values for n_neighbors
        knn_gscv = GridSearchCV(knn_clf, param_grid, cv=10) ## cv =5
        knn_gscv.fit(X, y)
        n_neighbors = knn_gscv.best_params_['n_neighbors']
    #fit model to data
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors,algorithm=knn_algo, weights='distance')
    knn_clf.fit(X,y)
    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)
  #return knn_gscv
    return knn_clf

if __name__ == '__main__':
    train('train_dirMany')
