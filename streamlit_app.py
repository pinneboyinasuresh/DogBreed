import os
import json
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = load_model(os.path.join(BASE_DIR, "dogbreed_model.h5"))

with open(os.path.join(BASE_DIR, "class_indices.json"), "r") as f:
    class_indices = json.load(f)

class_names = list(class_indices.keys())


@app.route("/", methods=["GET", "POST"])
def index():
    results = []

    if request.method == "POST":
        files = request.files.getlist("file")

        for file in files:
            if file:
                img_file = file.filename
                filepath = os.path.join(BASE_DIR, "static", img_file)
                file.save(filepath)

                img = load_img(filepath, target_size=(224, 224))
                x = img_to_array(img) / 255.0
                x = np.expand_dims(x, axis=0)

                pred = model.predict(x)
                prediction = class_names[np.argmax(pred)]

                results.append((img_file, prediction))

    return render_template("index.html", results=results)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

