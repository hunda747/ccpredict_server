import tensorflow as tf
import numpy as np
import cv2
from flask import Flask, request, jsonify
from PIL import Image
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    # Handle the image upload and prediction logic here
    img = request.files['image']
    # Preprocess the image and make predictions

    base_model = tf.keras.applications.MobileNetV2(include_top=False, 
                                               weights='imagenet')

    base_model.trainable = False

    inputs = tf.keras.Input(shape=(160, 160, 3))
    data_augmentation = tf.keras.Sequential([
      tf.keras.layers.RandomFlip('horizontal'),
      tf.keras.layers.RandomRotation(0.2),
    ])
    x = data_augmentation(inputs)

    #rescale pixel values
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    x = preprocess_input(x)

    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    learning_rate=1e-4

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


    model_1=tf.keras.models.load_model('./mobile_net_cancer.h5')
    # print(tf.keras.models.load_model('mobile_net_cancer.h5').summary())
    features=[]
    # img = cv2.imread('./type3.jpg')
    # img = cv2.imread(image)
    image = Image.open(img)
    resized_img = image.resize((160, 160))
    # resized_img = cv2.resize(img, (160, 160))
            
    features.append(np.array(resized_img))
    preds=model_1.predict(np.array(features))
    result=preds.argmax()
    if result == 0:
      result='1'
    elif result == 1:
      result= '2'
    else:
      result ='3'
    print(result)

    # Return the predictions as a JSON response
    return jsonify({'probabilities': result})  # Replace with actual probabilities

if __name__ == '__main__':
    app.run(debug=True)
