# importing necessary modules
from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os

# constructing and parsing command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True, help="Path to your faces+images dataset")
ap.add_argument("-e", "--encodings", required=True, help="Path to your serialized database of encodings")
# use hog if you don't have gpu
ap.add_argument("-d", "--detection_method", type=str, default="hog", help="face detection model to use : hog or cnn")
args = vars(ap.parse_args())
# grabbing image paths to our dataset
print("[INFO] quantifying faces")
imagePaths = list(paths.list_images(args["dataset"]))
# intialize known encoding and known names
knownEncodings = []
knownNames = []

# start looping over the image paths
for (i, imagePath) in enumerate(imagePaths):
    # retrieve the person name from the path
    print("[INFO] Processing image {} of {}".format(i + 1, len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]
    # load the image from dataset and convert it to RGB color ordering
    # because OpenCV load images into BGR color ordering
    image = cv2.imread(imagePath)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # detect the rectangle(x,y) coordinates surrounding the faces
    rectangle = face_recognition.face_locations(rgb_image, model=args["detection_method"])
    # compute the facial embedding for the face
    encodings = face_recognition.face_encodings(rgb_image, rectangle)
    # loop over encodings
    for encoding in encodings:
        # append encoding and name to the list
        knownEncodings.append(encoding)
        knownNames.append(name)

# push the known encodings and names to the disk
print("[INFO] serializing encodings.... ")
data = {"encodings": knownEncodings, "names": knownNames}
f = open(args["encodings"], "wb")
f.write(pickle.dump(data))
f.close()
