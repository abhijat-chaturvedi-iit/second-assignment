from flask import Flask
from flask import request
from joblib import load

app = Flask(__name__)
model_path = "model.joblib"
model = load(model_path)

@app.route("/predict", methods=['POST'])
def predict_digit():
    image1 = request.json['image1']
    image2 = request.json['image2']
    predicted1 = model.predict([image1])
    predicted2 = model.predict([image2])
    is_same = False
    if predicted1[0]==predicted2[0]:
        is_same=True
    return (str(is_same))

if __name__=="__main__":
    app.run(host='0.0.0.0',port=5000)