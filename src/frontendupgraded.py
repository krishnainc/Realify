from scipy.spatial import distance
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from flask import Flask, render_template, request , url_for , flash
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from PIL import Image
from keras.preprocessing import image
from keras.models import Model
import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
import random
from numba import cuda 

from werkzeug.utils import redirect

pickle_in = open("pickleforimage.p", "rb")
modelchild1 = pickle.load(pickle_in)
pathpic = r'static/'

app = Flask(__name__)
app.secret_key = 'Realify'
        
model = VGG16(weights='imagenet' , include_top = True)
feat_extractor = Model(inputs=model.input, outputs=model.get_layer("fc2").output)



@app.route('/')
def main():
   return render_template('new.html')

@app.route('/landing')
def land():
   return render_template('uploadd.html')


@app.route('/after', methods=['GET', 'POST'])
def similarity():
   if request.method == 'POST':

      if 'file' not in request.files:
         flash('No file part')
         return redirect(request.url)

      pic = request.files['file']
      nama = pic.filename

      if pic.filename == '':
         flash('No image selected for uploading')
         return redirect(url_for('main'))

      afterthat = Image.open(pic.stream)
      afterthat.save(pathpic + nama)

      imagepath = pathpic + nama

      
      #model
      def load_image(path):
         img = image.load_img(path, target_size=model.input_shape[1:3])
         x = image.img_to_array(img)
         x = np.expand_dims(x, axis=0)
         x = preprocess_input(x)
         return img, x

      def get_closest_images(query_image_idx, num_results=5):
         distances = [ distance.cosine(modelchild1[1][query_image_idx], feat) for feat in modelchild1[1] ]
         idx_closest = sorted(range(len(distances)), key=lambda k: distances[k])[1:num_results+1]
         return idx_closest

      def get_concatenated_images(indexes, thumb_height):
         thumbs = []
         for idx in indexes:
            img = image.load_img(modelchild1[0][idx])
            img = img.resize((int(img.width * thumb_height / img.height), thumb_height))
            thumbs.append(img)
         concat_image = np.concatenate([np.asarray(t) for t in thumbs], axis=1)
         return concat_image


      new_image, x = load_image(imagepath)
      new_features = feat_extractor.predict(x)

      new_pca_features = modelchild1[2].transform(new_features)[0]

      distances = [ distance.cosine(new_pca_features, feat) for feat in modelchild1[1] ]
      idx_closest = sorted(range(len(distances)), key=lambda k: distances[k])[0:5]  # grab first 5
      results_image = get_concatenated_images(idx_closest, 200)

      plt.figure(figsize = (5,5))
     
      plt.figure(figsize = (16,12))
     
      done = Image.fromarray(results_image)

   
      done.save(pathpic + "my_image.png")
      
      
      
      return render_template('uploadd.html' , babi = pic.filename)
      
   
   return render_template('uploadd.html')


if __name__ == '__main__':
   app.run()