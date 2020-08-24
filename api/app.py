import os
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib

# Init app
app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))
# Database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///'+os.path.join(basedir, 'db.sqlite')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# Init db
db = SQLAlchemy(app)
# Init ma
ma = Marshmallow(app)

# Diabetes Class/Model
class Diabetes(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    pregnancies = db.Column(db.Integer)
    glucose = db.Column(db.Integer)
    blood_pressure = db.Column(db.Integer)
    skin_thickness = db.Column(db.Integer)
    insulin = db.Column(db.Integer)
    bmi = db.Column(db.Integer)
    diabetes_pedigree_func = db.Column(db.Integer)
    age = db.Column(db.Integer)
    outcome = db.Column(db.Integer)

    def __init__(self, pregnancies, glucose, blood_pressure,\
        skin_thickness, insulin, bmi, diabetes_pedigree_func,\
        age, outcome):
        self.pregnancies = pregnancies
        self.glucose = glucose
        self.blood_pressure = blood_pressure
        self.skin_thickness = skin_thickness
        self.insulin = insulin
        self.bmi = bmi
        self.diabetes_pedigree_func = diabetes_pedigree_func
        self.age = age
        self.outcome = outcome

# Diabetes Schema
class DiabetesSchema(ma.Schema):
    class Meta:
        fields = ('id', 'pregnancies', 'glucose', 'blood_pressure', 'skin_thickness',\
            'insulin', 'bmi', 'diabetes_pedigree_func', 'age', 'outcome')

# Init Schema
diabetes_schema = DiabetesSchema()

@app.route('/diabetes/predict', methods=['POST'])
def predict():
    pregnancies = request.json['pregnancies']
    glucose = request.json['glucose']
    blood_pressure = request.json['blood_pressure']
    skin_thickness = request.json['skin_thickness']
    insulin = request.json['insulin']
    bmi = request.json['bmi']
    diabetes_pedigree_func = request.json['diabetes_pedigree_func']
    age = request.json['age']

    model = joblib.load('finalized_model.pkl')
    sc_X = joblib.load('scaler.pkl')

    input_X = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,\
        insulin, bmi, diabetes_pedigree_func, age]])

    outcome = int(model.predict(sc_X.transform(input_X))[0])
    print(outcome)

    new_diabetes = Diabetes(pregnancies, glucose, blood_pressure, skin_thickness, insulin,\
        bmi, diabetes_pedigree_func, age, outcome)
    
    db.session.add(new_diabetes)
    db.session.commit()

    return diabetes_schema.jsonify(new_diabetes)

# Run Server
if __name__ == '__main__':
    app.run(debug=True)