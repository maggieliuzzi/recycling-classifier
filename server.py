# Starts a server which can receive images via HTTP POST and return predictions

import argparse
import os
import numpy as np
import flask
from predict import prepare_model, predict_from_file, predict_from_pil


app = flask.Flask(__name__)
model = None

# Receive images via POST at /predict and respond with JSON containing the prediction vector
@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}
    print("\nReceived a POST request.")

    # Ensure there is an 'image' attribute in POST request
    if flask.request.files.get("image"):
        image = flask.request.files["image"].read()

        raw_prediction = predict_from_pil(model, image)

        # TODO (maggie.liuzzi): check if this is the order in which predictions for these materials are listed
        most_probable = np.argmax(raw_prediction)  # TODO (maggie.liuzzi): this function only returns first instance if the maximum value is repeated. Find one that doesn't
        if 0 == most_probable:
            prediction = "cardboard"
        if 1 == most_probable:
            prediction = "glass"
        if 2 == most_probable:
            prediction = "metal"
        if 3 == most_probable:
            prediction = "paper"
        if 4 == most_probable:
            prediction = "plastic"
        if 5 == most_probable:
            prediction = "trash"

        print("Probability of [cardboard, glass, metal, paper, plastic, trash]: " + str(raw_prediction))
        data["prediction"] = {"cardboard": float(raw_prediction[0]), "glass": float(raw_prediction[1]), 
                              "metal": float(raw_prediction[2]), "paper": float(raw_prediction[3]),
                              "plastic": float(raw_prediction[4]), "trash": float(raw_prediction[5]),
                              "guess": prediction}

        data["success"] = True
    else:
        print("Image attribute not found in POST request.")

    return flask.jsonify(data)

# When this script is ran directly, prepare the model + server to take requests
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Starts server to predict an object's material using a CNN model.")
    parser.add_argument('--model', required=True, help="Path to trained model file")
    parser.add_argument('--port', default=4000, help="Port occupied by server")
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        error("Could not find model file.")
        exit(1)
    try:
        n = int(args.port)
        if n < 1:
            error("Port number must be greater than or equal to 1.")
            exit(1)
    except ValueError:
        error("Port number must be an integer.")
        exit(1)

    print("\nLoading Keras model...")
    model = prepare_model(args.model)
    print("\nLoading Flask server...")
    app.run(host="0.0.0.0", port='4000')
