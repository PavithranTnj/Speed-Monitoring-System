from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
import time
#import serial
#from mail import report_send_mail
#from mail import*


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from werkzeug.utils import secure_filename
'''ser = serial.Serial(
    port='COM6',
    baudrate=9600,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    bytesize=serial.EIGHTBITS,
    timeout=.1,
    rtscts=0
)'''

MODEL_PATH = 'keras_model.h5'

model = load_model(MODEL_PATH)


def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)

    x = x / 255
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)
    preds = np.argmax(preds, axis=1)
    if preds == 0:
        preds = "30KM Speed"
        #ser.write(b'0')
       # print(ser.write(b'1'))
        #report_send_mail(preds, 'image.jpg')
        time.sleep(3)
    elif preds==1:
        preds = "60KM Speed"
        #ser.write(b'1')
        #report_send_mail(preds, 'image.jpg')
       # print(ser.write(b'2'))
        time.sleep(3)
    elif preds==2:
        preds = "Speed Breaker"
        #ser.write(b'2')
        #report_send_mail(preds, 'image.jpg')
       # print(ser.write(b'2'))
        time.sleep(3)
    elif preds==3:
        preds = "School Zone"
        #ser.write(b'3')
        #report_send_mail(preds, 'image.jpg')
       # print(ser.write(b'2'))
        time.sleep(3)
    elif preds==4:
        preds = "80KM Speed"
        #ser.write(b'4')
        #report_send_mail(preds, 'image.jpg')
       # print(ser.write(b'2'))
        time.sleep(3)   
    elif preds==5:
        preds = "No Horn Zone"
        #ser.write(b'3')
        #report_send_mail(preds, 'image.jpg')
       # print(ser.write(b'2'))
        time.sleep(3)
    elif preds==6:
        preds = "No Parking"
        #ser.write(b'4')
        #report_send_mail(preds, 'image.jpg')
       # print(ser.write(b'2'))
        time.sleep(3)   
    elif preds==7:
        preds = "Pedestrain"
        #ser.write(b'3')
        #report_send_mail(preds, 'image.jpg')
       # print(ser.write(b'2'))
        time.sleep(3)
    elif preds==8:
        preds = "Road Work"
        #ser.write(b'4')
        #report_send_mail(preds, 'image.jpg')
       # print(ser.write(b'2'))
        time.sleep(3)         
    elif preds==9:
        preds = "Train Track"
        #ser.write(b'4')
        #report_send_mail(preds, 'image.jpg')
       # print(ser.write(b'2'))
        time.sleep(3)   
    return preds




import cv2

# Open the video capture
cap = cv2.VideoCapture(0)  # Use 0 for the default camera, or specify the video file path

# Check if the camera/video is opened successfully
if not cap.isOpened():
    print("Error opening video capture.")
    exit()

# Set the video capture duration (in seconds)
capture_duration = 10

# Set the frame rate of the video capture
frame_rate = 30  # Adjust as per your requirement

# Calculate the number of frames to capture
num_frames = int(capture_duration * frame_rate)

# Capture the frames
for i in range(num_frames):
    ret, frame = cap.read()  # Read a frame from the video capture

    if not ret:
        print("Error reading frame.")
        break

    cv2.imshow("Video Capture", frame)  # Display the frame

    # Wait for 1ms and check if the user pressed the 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save the last captured frame as an image
image_path = "image.jpg"  # Specify the path and filename for the image
cv2.imwrite(image_path, frame)

# Release the video capture and close any open windows
cap.release()
cv2.destroyAllWindows()

print("Image saved successfully.")



       

a = "image.jpg"

b=model_predict(a,model)
c = b 
print(c)


