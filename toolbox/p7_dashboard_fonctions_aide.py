from sklearn.linear_model import LogisticRegression
#from sklearn.preprocessing import OneHotEncoder
#from sklearn.preprocessing import MinMaxScaler
import sklearn_json as skljson
import urllib.parse
from urllib.request import urlopen
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas as pd
import warnings
import numpy as np
#import StringIO
import io
import base64
from io import BytesIO
#from PIL import Image
from flask import Flask, render_template, jsonify, request, flash, redirect, url_for
from flask_wtf import Form, validators  
from wtforms.fields import StringField
from wtforms import TextField, BooleanField, PasswordField, TextAreaField, validators
from wtforms.widgets import TextArea
import xgboost
import dill
import joblib

#Define a function that allows to load a binary object.
#def load_object(path_to_file):
#    with open(path_to_file, 'rb') as file:
#        obj = dill.load(file)
#    return obj

#Define a function that allows to load binary objects.
def load_object(path_to_file):
    ob = joblib.load(path_to_file)
    return ob

#Define a function that allows to convert a logistic regression classifier into a json file.
def logistic_regression_to_json(lrmodel, file=None):
    if file is not None:
        serialize = lambda x: json.dump(x, file)
    else:
        serialize = json.dumps
    data = {}
    data['init_params'] = lrmodel.get_params()
    data['model_params'] = mp = {}
    for p in ('coef_', 'intercept_','classes_', 'n_iter_'):
        mp[p] = getattr(lrmodel, p).tolist()
    return serialize(data)

#Define a function that allows to convert a json file into a logistic regression classifier.
def logistic_regression_from_json(jstring):
    data = json.loads(jstring)
    model = LogisticRegression(**data['init_params'])
    for name, p in data['model_params'].items():
        setattr(model, name, np.array(p))
    return model


