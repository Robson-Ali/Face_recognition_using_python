# Face_recognition_using_python
![face_recognition](https://github.com/Robson-Ali/Face_recognition_using_python/assets/99282807/81b8c171-f510-435a-90c2-d51a4d55c17a)

INTRODUCTION:

This project helps its users to build face detection and recognition tool using Python. I am the only contributor to this project. As a software engineer I am focusing on building a face recognition tool that could help companies and other vital bodies in the society, especially in the area of security and transportation. The level of insecurity in my area is becoming overwhelming. I was on the streets one day when I saw some unknown gunmen walk out of their cars just to rob different shops of their hard-earned money without being disturbed or recognized. When the victims affected by the ugly incident are asked if they could recognize these people they usually gave a negative answer. This issue of robbery continued to the extent that I got worried and I asked myself the reason I am in this world. I discovered that one of the reasons for existence on Earth is to solve problems. I started looking for a way to solve this problem of insecurity. I realized that this project will help me find a solution to this problem of insecurity. I hope to use this tool to help reduce the level of insecurity and capture the faces of those who are guilty of crimes.

# Installing
    import cv2 import dlib import numpy as np

# Load the pre-trained face detection model
    face_detector = dlib.get_frontal_face_detector()

# Load the pre-trained face landmark detection model
    landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Load the pre-trained face recognition model
    face_recognizer = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

# Load a sample image and extract its features
sample_image = cv2.imread('sample_image.jpg') sample_gray = cv2.cvtColor(sample_image, cv2.COLOR_BGR2GRAY) sample_faces = face_detector(sample_gray) sample_landmarks = [landmark_predictor(sample_gray, face) for face in sample_faces] sample_features = [face_recognizer.compute_face_descriptor(sample_gray, landmark) for landmark in sample_landmarks]

# Initialize the video capture
    video_capture = cv2.VideoCapture(0)

while True: # Read a frame from the video stream ret, frame = video_capture.read()

# Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Detect faces in the frame
    faces = face_detector(gray)

# Iterate over the detected faces
for face in faces:
    # Determine the facial landmarks
    landmarks = landmark_predictor(gray, face)

    # Extract the face features
    features = face_recognizer.compute_face_descriptor(gray, landmarks)

    # Compare the features with the sample image features
    distances = np.linalg.norm(np.array(sample_features) - np.array(features), axis=1)
    min_distance = np.min(distances)

    # Set a threshold to determine if the face belongs to the known person or is an unknown face
    if min_distance < 0.6:
        # Recognized face
        name = "Known Person"
    else:
        # Unknown face
        name = "Unknown"

    # Draw a rectangle around the face and display the name
    (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Display the resulting frame
    cv2.imshow('Face Recognition', frame)

# Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
    break
# Release the video capture and close the windows
    video_capture.release() cv2.destroyAllWindows()

# Description
Face Recognition is a very popular topic. It has lot of use cases in the filed of biometric security. Now a days with the help of Deep learning face recognition has become very feasible to people. As deep learning is a very data intensive task and we may always not have such huge amount of data to work in case of face recognition so with the advancement in One Shot Learning, face recognition has become more practical and feasible. This Python Package make it even more feasible, simple and easy to use. We have eliminated all the steps to download the supporting files and setting up the supporting files. You can simply installed the python package and start doing face detection and recognition.

# Steps Explanation
To learn more about the tasks which are being performed on the backend head over to link : Step by Step Face Recognition Code Implementation From Scratch In Python

# Using The Package
# Train Model
    import FaceReco.FaceReco as fr
    fr_object1 =  fr.FaceReco()
    fr_object1.train_model("lfw_selected/face")
# Test Model
    fr_object1.test_model("lfw_selected/face2/Johnny_Depp_0002.jpg")
# Load Saved Model
    fr_object2 =  fr.FaceReco()
    fr_object2.load_model("Model_Object_1") #folder of saved model
    fr_object2.test_model("lfw_selected/face2/Johnny_Depp_0002.jpg")
# Contributing
I am the only contributor to this project. But references were made from different projects.
