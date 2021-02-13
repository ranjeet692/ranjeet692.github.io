from flask import Flask, request,make_response, current_app
from flask_failsafe import failsafe
import flask
from flask_cors import CORS, cross_origin
import json
import hashlib
import re
from os import path
from flask import render_template


root = "/home/sol315/server/uniquemachine/"
app = Flask(__name__)
CORS(app)

mask = []
mac_mask = []

with open("mask.txt", 'r') as f:
    mask = json.loads(f.read())
with open("mac_mask.txt", 'r') as fm:
    mac_mask = json.loads(fm.read())

@app.route("/")
def fingerprint():
    return render_template('index.html')

@app.route('/features', methods=['POST'])
def features():
    agent = ""
    accept = ""
    encoding = ""
    language = ""
    IP = ""

    try:
        agent = request.headers.get('User-Agent')
        accpet = request.headers.get('Accept')
        encoding = request.headers.get('Accept-Encoding')
        language = request.headers.get('Accept-Language')
        IP = request.remote_addr
    except:
        pass

    feature_list = [
            "agent",
            "accept",
            "encoding",
            "language",
            "langsDetected",
            "resolution",
            "fonts",
            "WebGL", 
            "inc", 
            "gpu", 
            "gpuImgs", 
            "timezone", 
            "plugins", 
            "cookie", 
            "localstorage", 
            "adBlock", 
            "cpu_cores", 
            "canvas_test", 
            "audio"]

    cross_feature_list = [
            "timezone",
            "fonts",
            "langsDetected",
            "audio"
            ]
    

    result = request.get_json()
    single_hash = "single"
    cross_hash = "cross"


    fonts = list(result['fonts'])

    cnt = 0
    for i in range(len(mask)):
        fonts[i] = str(int(fonts[i]) & mask[i] & mac_mask[i])
        if fonts[i] == '1':
            cnt += 1

    result['agent'] = agent
    result['accept'] = accept
    result['encoding'] = encoding
    result['language'] = language
    
    print(agent)
           
    feature_str = "IP"
    value_str = "'" + IP + "'"

    for feature in feature_list:
        
        if result[feature] is not "":
            value = result[feature]
        else:
            value = "NULL"

        feature_str += "," + feature

        if feature == "gpuImgs":
            value = ",".join('%s_%s' % (k,v) for k,v in value.items())
        else:
            value = str(value)


        if feature == 'cpu_cores':
            value = int(value)

        if feature == 'langsDetected':
            value = str("".join(value))
            value = value.replace(" u'", "")
            value = value.replace("'", "")
            value = value.replace(",", "_")
            value = value.replace("[", "")
            value = value.replace("]", "")
            value = value[1:]
        
        value_str += ",'" + str(value) + "'"

    result['fonts'] = fonts
    for feature in cross_feature_list:
        cross_hash += str(result[feature])
        hash_object = hashlib.md5((str(result[feature])).encode('utf-8'))

    hash_object = hashlib.md5(value_str.encode('utf-8'))
    single_hash = hash_object.hexdigest()

    hash_object = hashlib.md5(cross_hash.encode('utf-8'))
    cross_hash = hash_object.hexdigest()

    print (single_hash, cross_hash)
    return flask.jsonify({"single": single_hash, "cross": cross_hash})
