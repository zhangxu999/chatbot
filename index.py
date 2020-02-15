# -*- coding: utf-8 -*-


# import time
# import numpy as np
# import pkuseg
# from gensim.models import Word2Vec,KeyedVectors
# import torch
# from torch import nn
# from sklearn.metrics.pairwise import cosine_similarity
# data_path = 'data/temp_data/'
# from models import BiRNN

# tencent_vector = KeyedVectors.load_word2vec_format(data_path+'small_ailab_embedding.txt')

# with open(data_path+'label.npy','rb') as f:
#     qa_label = np.load(f,allow_pickle=True)
# with open(data_path+'answer.txt','r',encoding='utf8') as f_answer:
#     answer_list = f_answer.read().split('*$%')[:-1]
#     answer_list = np.array(answer_list)

# with open(data_path+'vocab.bin','rb') as f:
#     vocab =  torch.load( f)

# embed_size, num_hiddens, num_layers = 200,100,2
# net = BiRNN(vocab,embed_size,num_hiddens,num_layers)

# with open(data_path+'intent_classification.model','rb') as f:
#     net.load_state_dict(torch.load( f))

# seg = pkuseg.pkuseg()

# def tokenizer(text):
#     text = text.lower()
#     text = seg.cut(text)
#     return_text = []
#     for words in text:
# #             if words in stopwords:
# #                 continue
#         if words in tencent_vector:
#             return_text.append(words)
#         else:
#             return_text  += list(words)
#     return return_text

# def get_tokenized_qa_words(data):
#     return [tokenizer(question) for question  in data]

# def query(sentence):
#     words =  [w for w in sentence if  (w in tencent_vector)]
#     if not words:
#         return np.zeros(tencent_vector.vector_size)
#     vectors = np.mean([tencent_vector[w] for w in words], axis=0)
#     return vectors
    

# with open(data_path+'vectors_by_intent.npy','rb') as f:    
#     vectors_by_intent = np.load(f,allow_pickle=True)
# with open(data_path+'vector_position_by_intent.npy','rb') as f:    
#     vector_position_by_intent = np.load(f,allow_pickle=True)

# base_intents = ['finance',
#  'wikipedia',
#  'short_chat',
#  'complain',
#  'normal_chat',
#  'idiom',
#  'english',
#  'bless']

# def predict_sentiment(net, vocab, sentence):
#     """sentence是词语的列表"""
#     device = list(net.parameters())[0].device
#     sentence = torch.tensor([vocab.stoi[word] for word in sentence], device=device)
#     label = torch.argmax(net(sentence.view((1, -1))), dim=1)[0].tolist()
#     #print(base_intents[label])
#     return label

# log_file = open('task.log','a',encoding='utf8')
# def get_response(user_sentences):
#     start_time = time.time()
#     cuted_words = tokenizer(user_sentences)
#     pred_intent = predict_sentiment(net,vocab,cuted_words)
#     user_qu_tensor = np.array([query(cuted_words)])
#     sim_mat = cosine_similarity(user_qu_tensor,vectors_by_intent[pred_intent])
#     top10_posi = np.argsort(sim_mat,axis=1)[0][-1:]
#     top_sim_ques = vector_position_by_intent[pred_intent][top10_posi]
#     for answer_posi in top_sim_ques[::-1]:
#         response = answer_list[answer_posi]
#         print(user_sentences,":::::", response,'-------',time.time()-start_time,file=log_file)
#         log_file.flush()
#         return response

import requests
from flask import Flask, request, render_template
import json
import xmltodict
import time
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
    start_time = time.time()
    url = 'http://localhost:6006/model/parse/'
    post_data = '{{"text":"{}"}}'.format(user_sentences)
    print(post_data)
    respones = requests.post(url,json={"text":user_sentences})
    bot_utter = json.loads(respones.text)
    print(bot_utter)
    return bot_utter['response']

@app.route('/bot/<sentence>/', methods=['GET', 'POST'])
def index(sentence):
    response = response_by_rasa(sentence)
    return response
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
    app.run(host='0.0.0.0', port=80, debug=True)