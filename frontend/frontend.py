from scipy.spatial import distance
from flask import Flask, render_template, request , url_for , flash
from PIL import Image
from keras.preprocessing import image
import matplotlib.pyplot as plt
import pickle
import numpy as np
import os

from werkzeug.utils import redirect


app = Flask(__name__)
app.secret_key = 'Realify'

pickle_in = open("pickleforimage.p", "rb")
modelchild1 = pickle.load(pickle_in)
pathpic = r'static/'


@app.route('/')
def main():
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

      
      # let's display the image
      import random
      query_image_idx = int(len(imagepath))
      print(query_image_idx)
      img = image.load_img(imagepath)
      plt.imshow(img)


      similar_idx = [distance.cosine(modelchild1[1][query_image_idx], feat) for feat in modelchild1[1]]

      idx_closest = sorted(range(len(similar_idx)),key=lambda k: similar_idx[k])[1:6]

      thumbs = []
      for idx in idx_closest:
         img = image.load_img(modelchild1[0][idx])
         img = img.resize((int(img.width * 100 / img.height), 100))
         thumbs.append(img)

      concat_image = np.concatenate([np.asarray(t) for t in thumbs], axis=1)


      plt.figure(figsize=(16, 12))
      plt.imshow(concat_image)

      done = Image.fromarray(concat_image)

   
      done.save(pathpic + "my_image.png")

      return render_template('uploadd.html' , babi = pic.filename)
   return render_template('uploadd.html')

if __name__ == '__main__':
   app.run()