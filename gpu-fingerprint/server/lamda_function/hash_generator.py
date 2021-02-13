
import json
import hashlib
import re
from os import path

def lambda_handler(event, context):
    agent = ""
    accept = ""
    encoding = ""
    language = ""
    IP = ""

    try:
        agent = event['User-Agent']
        accpet = event['Accept']
        encoding = event['Accept-Encoding']
        language = event['Accept-Language']
        IP = event['remote_addr']
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
    return {"single": single_hash, "cross": cross_hash}
