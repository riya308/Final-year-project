# Importing the necessary librairies
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
import keras
from flask import Flask, render_template, request, send_from_directory
import os
from keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# flask frame work starts form here 
webapp=Flask(__name__)

@webapp.route('/')
def index():
    return render_template('index.html') #This function will redirect us to the home page of the web application

@webapp.route('/graph')
def graph():
    return render_template('graph.html') #This function will redirect us to the accuracy graph page  of the web application

@webapp.route('/about')
def about():
    return render_template('about.html') #This function will redirect us to the about page of the web application

@webapp.route("/prediction/<filename>")

def send_image(filename):
    return send_from_directory("static/img/",filename) # This function has been used to send the uploaded image from the directory 


data_dir=r"dataset/training_set/"    # The path of the training data set

mri=[]
for file in os.listdir(data_dir):    # This for loop is being used to print the each and every individual category 
    mri+=[file]                      # inside the training dataset and it is very much help full in mutliclass classification
print(mri)
print(len(mri))

mri = mri

@webapp.route("/Prediction",methods=["POST","GET"])   # This function has been written for the prediction page
def Prediction():
    if request.method=='POST':         # Here method is post because we are uploading or posting our image for prediction
        print("hdgkj")
        m = int(request.form["alg"])   # This will create a form where we can select our model 

        myfile = request.files['file']      # This will create a form at the webapp to upload our image
        fn = myfile.filename
        mypath = os.path.join("static/img/", fn)    # Our uploaded image will be saved in this path
        myfile.save(mypath)

       

        print("{} is the file name", fn)
        print("Accept incoming file:", fn)
        print("Save it to:", mypath)

        if m == 1:                                      # Here we need to select our model according to our wish
            print("bv1")
            new_model = load_model('mobilenet.h5')      # We have uploaded the pretrained model here 
            test_image = image.load_img(mypath, target_size=(224,224))     # Here we are feeding our uploaded image to model
            test_image = image.img_to_array(test_image)
            test_image = test_image/255                                     # It will rescale your uploaded image 
            test_image = image.img_to_array(test_image)                     # Here your uploaded image will be converted into array
        
        # test_image = np.expand_dims(test_image, axis=0)
        # result = new_model.predict(test_image)
        # preds = Birds[np.argmax(result)]

        else:
            print("bv1")
            new_model = load_model('SVM.h5')
            test_image = image.load_img(mypath, target_size=(64 , 64))
            test_image = image.img_to_array(test_image)
            test_image = test_image/255
            test_image = image.img_to_array(test_image)
        
        test_image = np.expand_dims(test_image, axis=0)
        result = new_model.predict(test_image)              # After selecting the model based on that model the result will get predicted 
        preds = mri[np.argmax(result)]                  


        return render_template("Prediction.html", text="The Uploaded Image Belongs to "+preds+"  Category", image_name=fn)
    return render_template("Prediction.html") # This will redirect you to the prediction page means you are getting the result on same page








# closing syntax for flask framework   
if __name__=='__main__':
    webapp.run(debug=True)       