#!/usr/bin/env python3
#+-----------------------------------------------------------------------------------------------+
#| @ Author : Padma Polash Paul @ Created on June 2020                                           |
#| @ Manage Orgs on MLP @ API: v.1                                                               |
#| @ -- mlOS - DEPLOY MODEL                                                                      |
#+-----------------------------------------------------------------------------------------------+
from flask import Flask, jsonify, request,current_app
import numpy as np
import os
import time
import socket
import json
import pickle
import pandas as pd
import sys
import uuid
from keras.models import load_model
import base64
import cv2
import numpy as np
from PIL import Image 
import fasttext
from io import BytesIO
import re
import texthero as hero
import xgboost
import mlos


app = Flask(__name__)

# INIT mlOS MODEL AND API HANDLER
mlosapi=mlos.mlosapi()
mlosmodel = mlos.mlosmodel()
mloslogs =mlos.logs()
invproc = mlos.invproc()

# VARIABLE DETAILS
is_model_loaded=False # DO NOT MODITY THIS LINE
print("App starning ... ", flush=True)
modelinfo= mlosmodel.getmodelinfo() # DO NOT MODITY THIS LINE

# EXAMPLE HOW TO ACCESS modelinfo 
model_key=modelinfo["model_key"]
model_gener=modelinfo["model_gener"]
model_ext=modelinfo["model_ext"]
formattestdata=modelinfo["formattestdata"]
model_base=modelinfo["model_base"]
model_full_path=modelinfo["model_full_path"]
transformation_dir= modelinfo["transformation_dir"]
feature_data_types=modelinfo["feature_data_types"]
feature_data_types= modelinfo["feature_data_types"]
logpath=  modelinfo["logpath"]
logfile= modelinfo["logfile"]
realtime_dir =modelinfo["realtime_dir"]
realtime_data_file=modelinfo["realtime_data_file"]
apistatpath=os.path.join("volumedata", 'apistat' )
api_status_countfile=os.path.join("volumedata", 'apistat',"count.json" )
api_status_dir= modelinfo["api_status_dir"] 
api_status_file= modelinfo["api_status_file"] 
realtime_row_count_file = modelinfo["realtime_row_count_file"] 
dbset = modelinfo["dbset"]
feimp= modelinfo["feimp"]

# LOADING THE MODEL 
# is_model_loaded - if the model is in the memory
# model - is loaded in this variable 
is_model_loaded, model = mlosmodel.loadmodel(modelinfo)  # DO NOT MODITY THIS LINE
is_expmodel_loaded,explain_info, explainermodel,trn_data=mlosmodel.getexplainermodel(modelinfo)

if(is_model_loaded):
    print("Model in memory... ", flush=True)
else:
    print("Cant load model. ", flush=True)
# FEATUE DATA TYPES FOR INPUT DATA PROCESSING
feture_data_types = modelinfo["feature_data_types"]

# GET MODEL INFO IN DASHBOARD 
@app.route("/getinfo", methods=['POST'])
def getinfo():
    rq = request.json.get("rq") # DO NOT MODITY THIS LINE
    result= mlosapi.getinfo(rq,is_model_loaded,modelinfo) # DO NOT MODITY THIS LINE
    # ==================================================================================
    #  Please do not assign value directly to result variable
    #  if you want to return more info from this apiendpoint, you can use add additional attribute to result,
    #  result["more_custom_info_1"] = your_variable_1_here
    #  result["more_custom_info_1"] = your_variable_2_here 
    # ==================================================================================
    return jsonify(result), 200 # DO NOT MODITY THIS LINE

# GET THE FEATURES FOR THE MODEL
@app.route("/features", methods=['POST'])
def getfeatures():
    result= mlosapi.getfeatures(is_model_loaded,modelinfo) # DO NOT MODITY THIS LINE
    # ==================================================================================
    #  Please do not assign value directly to result variable
    #  if you want to return more info from this apiendpoint, you can use add additional attribute to result,
    #  result["more_custom_info_1"] = your_variable_1_here
    #  result["more_custom_info_1"] = your_variable_2_here 
    # ==================================================================================
    return jsonify(result), 200 # DO NOT MODITY THIS LINE

# GET THE SAMPLE FOR THE MODEL
@app.route("/samples", methods=['POST'])
def getpayloadsample():
    result= mlosapi.getsamples(is_model_loaded,modelinfo) # DO NOT MODITY THIS LINE
    # ==================================================================================
    #  Please do not assign value directly to result variable
    #  if you want to return more info from this apiendpoint, you can use add additional attribute to result,
    #  result["more_custom_info_1"] = your_variable_1_here
    #  result["more_custom_info_1"] = your_variable_2_here 
    # ==================================================================================
    return jsonify(result), 200 # DO NOT MODITY THIS LINE

# STATUS OF THE DEPLOYED MODEL
# IF THE MODEL IS RUNNING 
@app.route("/status", methods=['GET'])
def appstatus():
    result=mlosapi.getapistatus(is_model_loaded,model_key) # DO NOT MODITY THIS LINE
    # ==================================================================================
    #  Please do not assign value directly to result variable
    #  if you want to return more info from this apiendpoint, you can use add additional attribute to result,
    #  result["more_custom_info_1"] = your_variable_1_here
    #  result["more_custom_info_1"] = your_variable_2_here 
    # ==================================================================================
    return jsonify(result), 200  # DO NOT MODITY THIS LINE

# THIS IS A TEST POST API
# this is an api endpoint to test your custom code
@app.route("/testpost", methods=['POST'])
def testpost():
    #  DUMMY API PLESE USE TO TEST YOUR CUSTOM API 
    if is_model_loaded==False:
        vt={
            "msg":"In memory model is not found for the api call. Please slect a model and redeploy."
        }
        result={'results':vt,"status":"OK","success":True,"error":False}
        return jsonify(result), 200
    vt={
            "msg":"Model is running. This is a test POST api call."
    }
    result={'results':vt,"status":"OK","success":True,"error":False}
    return jsonify(result), 200

#  PREDICTION SERVICE 
#  GET DATA AND RETURN PREDICTED RESPONSE 
@app.route("/predict", methods=['POST'])
def predict():
    if is_model_loaded==False:
        vt={
            "success":False,
            "error":True,
            "msg":"In memory model is not found for the api call. Please slect a model and redeploy."
        }
        mlosapi.saveapicallstatus(api_status_countfile, 'predict','error',0)
        result={'results':vt,"status":"OK","success":True,"error":False}
        return jsonify(result), 200
    try:
        # Check API Execustion time
        tm = time.time()
        # HANDLE DATA PAYLOAD FROM API REQUEST
        data=None
        try:
            # GET DATA FROM API REQUEST
            data = request.json.get("data")
        except:
            # IF DATA HAS ERROR
            vt={
                "error":True,
                "success":False,
                "msg":"No data provided for prediction."
            }
            result={'results':vt,"status":"OK","success":True,"error":False}
            return jsonify(result), 200 
        #  SAVE LOG FILE TO SEE 
        logfile = modelinfo["logfile"]
        # LOG FILE TO SEE 
        mloslogs.logtext(logfile,data,False) 
        if data == None:
            vt={
                "success":False,
                "error":True,
                "msg":"No data provided for prediction."
            }
            result={'results':vt,"status":"OK","success":True,"error":False}
            return jsonify(result), 200 
        
        if model == None:
            vt={
                "success":False,
                "error":True,
                "msg":"Model is not loaded yet."
            }
            result={'results':vt,"status":"OK","success":True,"error":False}
            return jsonify(result), 200 

        # PREPARE DATA FOR PREDICTION 
        _saveX_test,X_test= mlosmodel.prepare_test_data(data,modelinfo) # DO NOT MODIFY THIS LINE
        if( len(X_test)==0):
            vt = {
                "success":False,
                "error":True,
                "msg":"Invalid data type provided",
            }
            result={'results':vt,"status":"OK","success":True,"error":False}
            return jsonify(result), 200

        # PREDICT RESPONSE FROM PAYLOAD
        y_pred=[]
        y_prob=[]
        response= mlosmodel.predict_testdata(X_test,model,modelinfo) # DO NOT MODIFY THIS LINE
        if(response["results"]["success"]==False):
            return jsonify(response), 200
        else:
            y_pred =response["results"]["prediction"]
            y_prob= response["results"]["probability"]
        itype=[]
        explain=[]
        if(model_gener=="clsify" or model_gener=="regr"):
            if(is_expmodel_loaded):
                itype,explain= mlosmodel.explainmodel (explainermodel,explain_info,trn_data, X_test,dbset["trn"],feimp, "Future" )
            else:
                itype,explain= mlosmodel.explainmodel (model,explain_info,trn_data, X_test,dbset["trn"],feimp, "Future" )

        mloslogs.logtext(logfile,"Prediction Completed",False)

        # Restore and Save Realtime predicted data
        reconstrcted_response=mlosmodel.prepare_restored_response(y_pred,_saveX_test,modelinfo) # DO NOT MODIFY THIS LINE
        print(reconstrcted_response)
        mloslogs.logtext(logfile,reconstrcted_response,False)
        mloslogs.logtext(logfile,"Prediction done . Saving response",False)
        elsp=time.time()-tm
        mlosapi.saveapicallstatus(api_status_countfile,'predict','success',elsp)
        # ==================================================================================
        #  If you want to return more variable you can add to vt variable 
        # ==================================================================================
        vt = {
            "success":True,
            "error":False,
            "msg":"Predicted Result",
            "response_time":elsp,
            "prediction":y_pred.tolist(),
            "probability":y_prob.tolist(),
            "class_label":modelinfo["class_label"],
            "explanation_type":itype,
            "explanation":explain,
            "prediction_transformed":reconstrcted_response
        }
        # ==================================================================================
        #  Please do not assign value directly to result variable
        #  if you want to return more info from this apiendpoint, you can use add additional attribute to result,
        #  result["more_custom_info_1"] = your_variable_1_here
        #  result["more_custom_info_1"] = your_variable_2_here 
        # ==================================================================================
        result={'results':vt,"status":"OK","success":True,"error":False}
        return jsonify(result), 200
    except:
        vt={
            "success":False,
            "error":True,
            "msg":"API execution error. Please check provied data to predict"
        }
        mlosapi.saveapicallstatus(api_status_countfile,'predict','error',0)
        result={'results':vt,"status":"OK","success":True,"error":False}
        return jsonify(result), 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80, debug=True)
    #app.run( port=5002, debug=True)