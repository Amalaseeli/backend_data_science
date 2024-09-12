from app import app
from flask import render_template, jsonify, request
import os
import pandas as pd
import json
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

import pickle


DATASETS_DIR="datasets"
FEATURE_COLUMN_FILE="features.json"
SELECTED_MODEL_FILE="selected_model.json"

models={
    "LogisticRegression":{
        "Type":"Classification",
        "hyperparamaeters":{
            "c":1.0,
            "solver":"lbfgs"
            }
            },
    "SVC":{
        "Type":"Classification",
        "description":"SVC is a supervised learning algorithm for classification. It finds an optimal hyperplane in an N dimensional space",
        "hyperparameters":{
            "c":"float(defaulit =1.0)",
            "kernel":"str(default='rbf')",
            "gamma":"float(default='scale')"

        }
    },
    "MLPClassifier":{
        "Type":"Classification",
        "description":"MLP Classifier is a type of artificial neural network known as a multi layer perceptron. It can be used for classification and regression.",
        "hyperparameters":{
            "hidden_layer_sizes":"tuple(default(100,))",
            "activation":"str(default='relu')",
            "solver":"str(default='adam'",
            "alpha":"float(default=0.0001)"
        }
    },
    "GaussianNB":{
        "Type":"Classification",
        "description":"GausianNB is a simple classification algorithm based on Bayes' theorem with a gausian naive assumption. It assume that features are independent.",
        "hyperparameters":{}
    },
    "MultiNomialNB":{
        "Type":"Classification",
        "description":"MultiNomialNB is a variant of the naive Bayes algorithm, suitable for data with multinomial distribution, such as word counts in text data.It is widely used in NLP tasks.",
        "hyperparameters":{
            "alpha":"float(default=1.0)",
            "fit_prior":"bool(default=True)"
        }
    }
    
}

@app.route('/')
def Home():
    return render_template("index.html",title="Home", message="Hello User, Welcome!")

@app.route('/datasets', methods=['GET'])
def get_datasets():
    datasets=[f for f in os.listdir(DATASETS_DIR) if f.endswith('csv') ]
    return jsonify(datasets)

@app.route('/datasets', methods=['POST'])
def post_dataset():
    dataset_name=request.form['dataset_name']
    dataset_path=os.path.join(DATASETS_DIR, dataset_name)

    if os.path.exists(dataset_path):
        df=pd.read_csv(dataset_path)
        first_10_rows=df.head(10).to_json(orient='records')
        return jsonify({'data':first_10_rows})
    else:
        return jsonify({'error':'Dataset Not Found'}), 404
    
def load_features_columns():
    if os.path.exists(FEATURE_COLUMN_FILE):
        with open(FEATURE_COLUMN_FILE, 'r') as f:
            return json.load(f)
    return []

def save_feature_columns(columns):
    with open(FEATURE_COLUMN_FILE, 'w') as file:
        json.dump(columns, file)

@app.route('/features', methods=['POST'])
def save_columns():
    data=request.json
    selected_columns=data.get('selected',[])
    print('selected_columns',selected_columns)
    save_feature_columns(selected_columns)
    return jsonify({'success': True, 'features':selected_columns})

@app.route('/features', methods=['GET'])
def get_features():
    feature_columns=load_features_columns()
    return jsonify({'feature_columns': feature_columns})

@app.route('/get_unselected_columns', methods=['POST'])
def get_unselected_columns():
    dataset_name=request.form['dataset_name']
    dataset_path=os.path.join(DATASETS_DIR, dataset_name)

    if os.path.exists(dataset_path):
        df=pd.read_csv(dataset_path)
        all_columns=set(df.columns)
        selcted_columns=set(load_features_columns())
        unselcted_columns=list(all_columns-selcted_columns)
        print("unselcted_columns", unselcted_columns)
        return jsonify({'unselcted_columns':unselcted_columns})
    else:
        return jsonify({'error':'Dataset Not Found'}), 404
    
def save_selected_model(model_name):
    with open(SELECTED_MODEL_FILE, 'w') as f:
        json.dump({'model':model_name}, f)
        # f.write(json.dumps({"models":list(model_name)}))

def load_selcted_model():
    if os.path.exists(SELECTED_MODEL_FILE):
        with open(SELECTED_MODEL_FILE, 'r') as f:
            return json.load(f).get('model', '')
    return ''

@app.route('/models', methods=["GET"])
def get_models():
    return jsonify({"models":list(models.keys())})

@app.route('/models', methods=["POST"])
def select_model():
    data=request.json
    selected_model=data.get('model')
    print(selected_model)
    save_selected_model(selected_model)
    print("The selected model", selected_model)
    return jsonify({'Success':True, 'selcted_model':selected_model})

@app.route('/model_details', methods=['POST'])
def get_model_details():
    model_name=request.json.get('model')
    if model_name:
        if model_name in models:
            return jsonify(models[model_name])
        else:
            return jsonify({"error":"model not found"}), 404
    else:
        return jsonify({"error":"model_name not provided"}), 400
    
@app.route('/training',methods=['POST'])
def train():
    model=load_selcted_model()
    print(model)
    if model:
        if model== 'LogisticRegression':
            training_model=LogisticRegression()
        elif model == "MLPClassifier":
            training_model=MLPClassifier()
        elif model == "SVC":
            training_model=SVC()
    else:
        return jsonify({"error":"Model name not provided"}), 400
    print("training_model",training_model)
    if os.path.exists('datasets/heart.csv'):
        df=pd.read_csv('datasets/heart.csv')
        all_columns=set(df.columns)
        selected_columns=set(load_features_columns())
        print(all_columns, selected_columns)
        unselected_columns=list(all_columns-selected_columns)
    
    x_train=df.drop(columns=unselected_columns)
    y_train=df.drop(columns=selected_columns)
    x_train=pd.get_dummies(x_train)
    y_train=pd.get_dummies(y_train)

    training_model=training_model.fit(x_train, y_train)

    with open(str(model)+'.pkl','wb') as f:
        pickle.dump(training_model, f)

    return ''


@app.route('/testing', methods=["POST"])
def test():
    with open('LogisticRegression.pkl','rb') as f:
        testing_model=pickle.load(f)
    
    if os.path.exists('datasets/heart.csv'):
        df=pd.read_csv('datasets/heart.csv')
        all_columns=set(df.columns)
        selected_columns=set(load_features_columns())
        print(all_columns, selected_columns)
        unselected_columns=list(all_columns-selected_columns)
    
    x_train=df.drop(columns=unselected_columns)
    y_train=df.drop(columns=selected_columns)
    x_train=pd.get_dummies(x_train)
    y_train=pd.get_dummies(y_train)

    predicted= testing_model.predict(x_train)
    results=confusion_matrix(y_train, predicted)
    print("confusion_matrix:")
    print(results)
    print("Accuracy_score:", accuracy_score(y_train, predicted))
    print("classification_report:",classification_report(y_train, predicted))
    
    return jsonify({"success": True})


        
    

