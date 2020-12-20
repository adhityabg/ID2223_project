import cv2
import os
from os import listdir


# Configuration variables
current_dir = ""
haar_dir = current_dir + "haar_cascade_files/"
frames_dir = current_dir + "saved_frames_webcam/"
videos = current_dir + "Fold1_part1/"

face = cv2.CascadeClassifier(haar_dir + "haarcascade_frontalface_alt.xml")
leye = cv2.CascadeClassifier(haar_dir + "haarcascade_lefteye_2splits.xml")
reye = cv2.CascadeClassifier(haar_dir + "haarcascade_righteye_2splits.xml")


if not os.path.exists(frames_dir):
    os.mkdir(frames_dir)
for name in listdir(videos):
  filename = videos + name
  vidcap = cv2.VideoCapture(filename)
  success, image = vidcap.read()
  count = 0
  while success and count<500:
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
      left_eye = leye.detectMultiScale(gray)
      right_eye = reye.detectMultiScale(gray)
      crop_face = gray
      crop_leye = gray
      crop_reye = gray
      for (x,y,w,h) in faces:
        crop_face = gray[y:y+h, x:x+w]
        cv2.imwrite(frames_dir + "face/"+name+"_frame%d.jpg" % count, crop_face)
      
      for (x,y,w,h) in left_eye:
        crop_leye = gray[y:y+h, x:x+w]
        cv2.imwrite(frames_dir + "leye/"+name+"_frame%d.jpg" % count, crop_leye)
      
      for (x,y,w,h) in right_eye:
        crop_reye = gray[y:y+h, x:x+w]
        cv2.imwrite(frames_dir + "reye/"+name+"_frame%d.jpg" % count, crop_reye)

      success, image = vidcap.read()
      print("Now reading: " + name + " frame: %d: " % count , success)
      count += 1

      cv2.imshow('face',crop_face)
      cv2.imshow('leye',crop_leye)
      cv2.imshow('reye',crop_reye)    
      # if cv2.waitKey(1) & 0xFF == ord('q'):
          # break
vidcap.release()
cv2.destroyAllWindows()
