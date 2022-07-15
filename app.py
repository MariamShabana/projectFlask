from flask import Flask,request,render_template
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input,decode_predictions,VGG16
import tensorflow as tf
import numpy as np
import cv2

model = tf.keras.models.load_model('effnet.h5')


app=Flask(__name__)
def names(number):

        if number == 1:
            return 'YES , It is glioma tumor'
        elif number == 2:
            return 'YES, It is  meningioma tumor'
        elif number == 0:
            return 'NO , It is no tumor'
        elif number == 3:
            return 'YES , It is pituitary tumor'
        else:
            return ' please enter MRI Image of brain tumor'


@app.route('/')
def index():
    return render_template("prediction.html",data="hey")
@app.route('/prediction', methods=["POST"])
def predict():

        #
        # # load an image from file
         imagefile=request.files['img']

         imagefile.save("img.jpg")
         image=cv2.imread("img.jpg")
         image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
         image=cv2.resize(image,(224,224))
        # image = load_img(imagefile, target_size=(224, 224))
        # # convert the image pixels to a numpy array
         image = img_to_array(image)
        # # reshape data for the model
         image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # # prepare the image for the VGG model
         image = preprocess_input(image)
         res = model.predict(image)
         classification = np.where(res == np.amax(res))[1][0]

         result= names(classification)
         return render_template('index.html',data=result)


if __name__=='__main__':
    app.run(port=3000,debug=True)