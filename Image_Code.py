from __future__ import division, print_function
import cv2
import os
import numpy as np
import tensorflow as tf
import time
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tkinter import filedialog
from tkinter import Tk

#import serial

#from mail import report_send_mail
#from mail import*


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


'''
ser = serial.Serial(
    port='COM13',
    baudrate=9600,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    bytesize=serial.EIGHTBITS,
    timeout=.1,
    rtscts=0
)
'''

#MODEL_PATH = 'keras_model.h5'
model = load_model('keras_model.h5',compile=False)

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

# Initialize Tkinter
root = Tk()
root.withdraw()  # Hide the main window

# Use file dialog to select an image
img_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.jpg; *.jpeg; *.png")])
if img_path:
    # Load and display the selected image
    img = cv2.imread(img_path)
    cv2.imshow("Selected Image",img)
    image_path = "image.jpg"  # Specify the path and filename for the image
    cv2.imwrite(image_path, img)
    
if not img_path:
    print("No image selected. Exiting.")
    sys.exit()

# Call the model prediction function
result = model_predict(img_path, model)

# Display the result
print("Predicted Result:", result)
