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
# device_count = {'GPU': 1}
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

face = cv2.CascadeClassifier(haar_dir + "haarcascade_frontalface_alt.xml")
leye = cv2.CascadeClassifier(haar_dir + "haarcascade_lefteye_2splits.xml")
reye = cv2.CascadeClassifier(haar_dir + "haarcascade_righteye_2splits.xml")


lbl = ['Close', 'Open']

model = load_model(model_dir + "eye_classifier_20201219.h5")
path = os.getcwd()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count = 0
score = 0
thicc = 2
rpred = [99]
lpred = [99]
classes = {1: "open", 0: "close"}

if not os.path.exists(temp_dir):
    os.mkdir(temp_dir)

while(True):
    ret, frame = cap.read()
    height, width = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)

    cv2.rectangle(frame, (0, height-50), (200, height),
                  (0, 0, 0), thickness=cv2.FILLED)

    # for (x, y, w, h) in faces:
    #     cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 100, 100), 1)

    for (x, y, w, h) in right_eye:
        r_eye = gray[y:y+h, x:x+w]
        cv2.imwrite(temp_dir + "r_temp.jpg", r_eye)
        r_eye_read = cv2.imread(temp_dir + "r_temp.jpg")
        count = count+1       
        r_eye_preprocessed = cv2.resize(preprocess_input(r_eye_read.astype(np.float32)),(224,224))
        r_eye_preprocessed = np.expand_dims(r_eye_preprocessed, axis=0)
        preds = model.predict(r_eye_preprocessed)
        rpred = [p for p in np.argmax(preds, axis=1)]
        if(rpred == [1]):
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 128, 0), 5)
            lbl = 'Open'
        else:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 128), 5)
            lbl = 'Closed'
        break

    for (x, y, w, h) in left_eye:
        l_eye = gray[y:y+h, x:x+w]
        cv2.imwrite(temp_dir + "l_temp.jpg", l_eye)
        l_eye_read = cv2.imread(temp_dir + "l_temp.jpg")
        count = count+1
        l_eye_preprocessed = cv2.resize(preprocess_input(l_eye_read.astype(np.float32)),(224,224))
        l_eye_preprocessed = np.expand_dims(l_eye_preprocessed, axis=0)
        preds = model.predict(l_eye_preprocessed)
        lpred = [p for p in np.argmax(preds, axis=1)]
        if(lpred == [1]):
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 128, 0), 5)
            lbl = 'Open'
        else:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 128), 5)
            lbl = 'Closed'
        break

    if(rpred == [0] and lpred == [0]):
        score = score+1
        cv2.putText(frame, "Closed", (10, height-20), font,
                    1, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        score = score-1
        cv2.putText(frame, "Open", (10, height-20), font,
                    1, (255, 255, 255), 1, cv2.LINE_AA)

    if(score < 0):
        score = 0
    cv2.putText(frame, 'Score:'+str(score), (100, height-20),
                font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    if(score > 15):
        # person is feeling sleepy so we beep the alarm
        cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
        try:
            sound.play()

        except:  # isplaying = False
            pass
        if(thicc < 16):
            thicc = thicc+2
        else:
            thicc = thicc-2
            if(thicc < 2):
                thicc = 2
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
shutil.rmtree(temp_dir, ignore_errors=True)