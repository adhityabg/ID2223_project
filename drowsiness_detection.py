import cv2
import os
from keras.models import load_model
from keras.applications.resnet50 import preprocess_input
import numpy as np
from pygame import mixer
import time
import tensorflow as tf
import shutil


config = tf.compat.v1.ConfigProto(gpu_options = 
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


mixer.init()
sound = mixer.Sound('alarm.wav')

# Configuration variables
current_dir = ""
haar_dir = current_dir + "haar_cascade_files/"
temp_dir = current_dir + "temp/"
model_dir = current_dir + "models/"

# Load the Haar Cascade files for eye detection
leye = cv2.CascadeClassifier(haar_dir + "haarcascade_lefteye_2splits.xml")
reye = cv2.CascadeClassifier(haar_dir + "haarcascade_righteye_2splits.xml")


model = load_model(model_dir + "eye_classifier_20201219.h5")
webcam_source = cv2.VideoCapture(0)
score = 0
alert_box = 2
rpred = [99]
lpred = [99]
threshold_score = 15

# Check and create the temp directory 
if not os.path.exists(temp_dir):
    os.mkdir(temp_dir)

# Video capture and monitoring continues infinitely until 'q' is pressed
while(True):
    ret, frame = webcam_source.read()
    height, width = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Actually detect left and right eyes from the frame
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)

    # Preprocess the right eye and run it through the model
    for (x, y, w, h) in right_eye:
        r_eye = gray[y:y+h, x:x+w]
        cv2.imwrite(temp_dir + "r_temp.jpg", r_eye)
        r_eye_read = cv2.imread(temp_dir + "r_temp.jpg")
        r_eye_preprocessed = cv2.resize(preprocess_input(r_eye_read.astype(np.float32)),(224,224))
        r_eye_preprocessed = np.expand_dims(r_eye_preprocessed, axis=0)
        preds = model.predict(r_eye_preprocessed)
        rpred = [p for p in np.argmax(preds, axis=1)]
        if(rpred == [1]):
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
            lbl = 'Open'
        else:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
            lbl = 'Closed'
        break

    # Preprocess the left eye and run it through the model
    for (x, y, w, h) in left_eye:
        l_eye = gray[y:y+h, x:x+w]
        cv2.imwrite(temp_dir + "l_temp.jpg", l_eye)
        l_eye_read = cv2.imread(temp_dir + "l_temp.jpg")
        l_eye_preprocessed = cv2.resize(preprocess_input(l_eye_read.astype(np.float32)),(224,224))
        l_eye_preprocessed = np.expand_dims(l_eye_preprocessed, axis=0)
        preds = model.predict(l_eye_preprocessed)
        lpred = [p for p in np.argmax(preds, axis=1)]
        if(lpred == [1]):
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
        else:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
        break

    cv2.rectangle(frame, (0, height-50), (200, height),(255, 255, 255), thickness=cv2.FILLED)
    if(rpred == [0] and lpred == [0]):
        score = score+1
        cv2.putText(frame, "Closed", (10, height-20), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    1, (0, 0, 0), 1, cv2.LINE_AA)
    else:
        score = score-1
        cv2.putText(frame, "Open", (10, height-20), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    1, (0, 0, 0), 1, cv2.LINE_AA)
    if(score < 0):
        score = 0
    
    cv2.putText(frame, 'Score:'+str(score), (100, height-20),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 1, cv2.LINE_AA)
    if(score > threshold_score):
        try:
            sound.play()
        except:
            pass
        if(alert_box < 16):
            alert_box = alert_box+2
        else:
            alert_box = alert_box-2
            if(alert_box < 2):
                alert_box = 2
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), alert_box)
    
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam_source.release()
cv2.destroyAllWindows()
shutil.rmtree(temp_dir, ignore_errors=True)