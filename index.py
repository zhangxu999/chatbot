# -*- coding: utf-8 -*-
import  logging
import requests
from flask import Flask, request
import json
import xmltodict
import time
logger = logging.getLogger(__name__)
app = Flask(__name__, static_url_path="")
template = """
<xml>
  <ToUserName><![CDATA[{toUser}]]></ToUserName>
  <FromUserName><![CDATA[{fromUser}]]></FromUserName>
  <CreateTime>{createTime}</CreateTime>
  <MsgType><![CDATA[text]]></MsgType>
  <Content><![CDATA[{Content}]]></Content>
</xml>
"""
def response_by_rasa(user_sentences):
    url = 'http://localhost:6006/model/parse/'
    respones = requests.post(url, json={"text": user_sentences})
    bot_utter = json.loads(respones.text)
    return bot_utter['response'], bot_utter['best_match']
import  datetime
@app.route('/bot/', methods=['GET', 'POST'])
def index():
    data = request.data.decode('utf8')
    data = json.loads(data)
    user_message = data['text']
    response_template = {
        'status':'success',
        'response':'',
        'best_match':'',
        'response_time': str(datetime.datetime.now())[:-7],
        "text": user_message
    }
    logger.info(user_message)
    try:
        response, best_match = response_by_rasa(user_message)
        response_template['response'] = response
        response_template['best_match'] = best_match
    except Exception as e:
        response_template['status'] = 'fail'
        response_template['response'] = '发生错误，暂时不能提供服务'
        logger.error(e)
    return response_template
@app.route('/', methods=['GET','POST'])
def hashtag():
    if request.method == 'POST':
        data = request.data.decode('utf8')
        dict_data = xmltodict.parse(data)
        content = dict_data['xml']['Content']
        toUser = dict_data['xml']['FromUserName']
        fromUser = 'gh_73137ca9d188'
        createTime = int(time.time())
        Content = response_by_rasa(content)
        response = template.format(
            toUser=toUser,fromUser=fromUser,createTime=createTime,Content=Content)
        return response
    token = 'hdsjsql'
    timestamp = request.args.get('timestamp')
    nonce = request.args.get('nonce')
    parameters = {
    'token':'hdsjsql',
    'timestamp': request.args.get('timestamp'),
    'nonce': request.args.get('nonce')
    }
    add_str = ''.join([parameters[s] for s in sorted(['token','timestamp','nonce'])])
    import hashlib
    add_str = add_str.encode('utf8')
    print(add_str)
    hashed_str = hashlib.sha1(add_str).hexdigest()
    print(hashed_str)
    if request.args.get('signature') == hashed_str:
        return request.args.get('echostr')
    else:
        return request.args.get('echostr')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)