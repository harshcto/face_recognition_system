# import the necessary packages
import face_recognition
import argparse
import pickle
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True, help="path to serialized db of facial encodings")
ap.add_argument("-i", "--image", required=True, help="path to input image")
# use hog for cpu and cnn for gpu support
ap.add_argument("-d", "--detection_method", type=str, default="hog", help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

# load the faces and embedings
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())
# load input image and convert it color ordering to BGR
image = cv2.imread(args["image"])
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# detect the (x,y) coordinates of rectangle for each of the face in the input image
print("[INFO recognizing faces...]")
rectangle = face_recognition.face_locations(rgb_image, model=args["detection_method"])
encodings = face_recognition.face_encodings(rgb_image, rectangle)
# intialize the names list
names = []

# loop over the facial embeddings
for encoding in encodings:
    # attempt to match each face in the input image to our known
    # encodings
    matches = face_recognition.compare_faces(data["encodings"], encoding)
    name = "Unknown"

    # check to see if we have found a match
    if True in matches:
        # find the indexes of all matched faces then initialize a
        # dictionary to count the total number of times each face
        # was matched
        matchedIdxs = [i for (i, b) in enumerate(matches) if b]
        counts = {}

        # loop over the matched indexes and maintain a count for
        # each recognized face face
        for i in matchedIdxs:
            name = data["names"][i]
            counts[name] = counts.get(name, 0) + 1

        # determine the recognized face with the largest number of
        # votes (note: in the event of an unlikely tie Python will
        # select first entry in the dictionary)
        name = max(counts, key=counts.get)

    # update the list of names
    names.append(name)

# loop over recognized faces
for ((top, right, bottom, left), name) in zip(rectangle, names):
    cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 2)
    y = top - 15 if top - 15 > 15 else top + 15
    cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

# show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)
