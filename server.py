from flask import Flask, render_template, request, redirect
import numpy as np
import re
import base64
from PIL import Image
from imageio import imread
from PIL import Image
from keras.models import load_model
from prepare_data import normalize
import json
import csv
import os

app = Flask(__name__)

mlp = load_model("./models/animal_mlp.keras")
conv = load_model("./models/cnn-model-final.keras")
ANIMALS = {
    0: "cat",
    1: "elephant",
    2: "bear",
    3: "bird",
    4: "crab",
    5: "fish",
    6: "giraffe",
    7: "lion",
    8: "rabbit",
    9: "snake",
}


@app.route("/", methods=["GET", "POST"])
def ready():
    if request.method == "GET":
        return render_template("index1.html")
    if request.method == "POST":
        data = request.form["payload"].split(",")[1]
        net = request.form["net"]

        img = base64.b64decode(data)
        with open("temp.png", "wb") as output:
            output.write(img)
        x = imread("temp.png", mode="L")
        # resize input image to 28x28
        # x = imresize(x, (28, 28))
        x = Image.fromarray(x).resize(size=(28, 28))

        if net == "MLP":
            model = mlp
            # invert the colors
            x = np.invert(x)
            # flatten the matrix
            x = x.flatten()

            # brighten the image a bit (by 60%)
            for i in range(len(x)):
                if x[i] > 50:
                    x[i] = min(255, x[i] + x[i] * 0.60)

        if net == "ConvNet":
            model = conv
            x = np.expand_dims(x, axis=0)
            x = np.reshape(x, (28, 28, 1))
            # invert the colors
            x = np.invert(x)
            # brighten the image by 60%
            for i in range(len(x)):
                for j in range(len(x)):
                    if x[i][j] > 50:
                        x[i][j] = min(255, x[i][j] + x[i][j] * 0.60)

        # normalize the values between -1 and 1
        x = normalize(x)
        val = model.predict(np.array([x]))
        print(val)
        top_3 = sorted(range(len(val[0])), key=lambda i: val[0][i], reverse=True)[:3]
        print(top_3)
        classes = []
        preds = []
        something_else = 0.0
        for i in top_3:
            classes.append(ANIMALS[i])
            preds.append(val[0][i])
            something_else += val[0][i]
        something_else = 1 - something_else
        preds.append(something_else)
        classes.append("something else")
        print(classes)

        return render_template(
            "index1.html",
            preds=preds,
            classes=json.dumps(classes),
            chart=True,
            putback=request.form["payload"],
            net=net,
        )


@app.route("/save-img", methods=["POST"])
def saveImg():
    image = Image.open("temp.png")
    image = image.convert("L")
    image = image.resize((28, 28))
    pixel_values = list(image.getdata())
    label = request.form["doodle"]
    with open("data/doodle_data.csv", "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([label] + pixel_values)

    return redirect("/")


app.run()
