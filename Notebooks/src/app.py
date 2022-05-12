
from fastapi import FastAPI
import pickle

app = FastAPI()

model_loaded_data = {}

with open("../Models/model2.pkl", "rb") as f:
    model_loaded_data = pickle.loads(f.read())

def lable_predictor_from_export(txt="Hello World", vectorizer=None, model=None, classes=[]):

    assert(vectorizer is not None)
    assert(model is not None)

    input_vector = vectorizer.transform([txt])
    output_vector = model.predict(input_vector)
    assert(len(output_vector[0]) == len(classes))

    pred = {}
    for i, val in enumerate(output_vector[0]):
        pred[classes[i]] = int(val)
    
    return pred

@app.post("/")

def homepage_new():
    pred = lable_predictor_from_export("When does the kitchen close?", **model_loaded_data)
    return pred


