# -*- coding: utf-8 -*-
'''
Created on 2019年3月1日

@author: Zhukun Luo
Jiangxi university of finance and economics
'''
#将模型部署为api
from flask import  Flask
from flask_restful import reqparse,abort,Api,Resource
import pickle
import numpy as np
from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
from sklearn.externals import joblib
import numpy as np
from model import predictModel

app = Flask(__name__)
api = Api(app)

model = predictModel()

clf_path = './model/smote_sample_model.pkl'
with open(clf_path, 'rb') as f:
    model.svc = joblib.load(f)

# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('query')


class PredictPoverty(Resource):
    def get(self):
        # use parser and find the user's query
        args = parser.parse_args()
        user_query = args['query']
        print(user_query)
#         feature=user_query.split('&&')
#         print(feature)
        print(user_query)
        user_query=[21, 13, 5.0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0,0, 1, 0,0, 0, 0, 0, 0]
        prediction = model.predict(np.array([user_query]))
        # Output either 'Negative' or 'Positive' along with the score
        if prediction == 1:
            pred_text = '贫困学生家庭'
        elif prediction==2:
            pred_text = '一般家庭'
        else :
            pred_text ='条件较好家庭'


        # create JSON object
        output = {'prediction': pred_text}

        return output
'''
HTTP/1.0 200 OK
Content-Length: 61
Content-Type: application/json
Date: Fri, 01 Mar 2019 09:22:21 GMT
Server: Werkzeug/0.14.1 Python/3.6.3

{
    "prediction": "条件较好家庭"
}
'''

# Setup the Api resource routing here
# Route the URL to the resource
api.add_resource(PredictPoverty, '/')


if __name__ == '__main__':
    app.run(debug=True)