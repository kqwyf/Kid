#!/usr/bin/env python3

import os
import sys
import json
import math
import shutil
import pickle
#import sqlite3
from collections import OrderedDict, Counter

import numpy as np
from tqdm import tqdm
#生成与 __file__同一目录下p的文件的路径
def with_path(p):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, p)

DIM=500
MAX_SENTENCE_NUMBER=200000
BUCKET_FILES=[('cx.m.train','cx.t.train'),('cx.m.valid','cx.t.valid'),('wb.m.train','wb.t.train')]
DICTIONARY_PATH = 'dictionary.json'
EOS = '<eos>'
UNK = '<unk>'
PAD = '<pad>'
GO = '<go>'

buckets = [
    #(5, 15),
    (10, 20),
    (15, 25),
    (20, 30),
    (25, 35),
    (30, 40),
    (35, 45)
]

def time(s):
    ret = ''
    if s >= 60 * 60:
        h = math.floor(s / (60 * 60))
        ret += '{}h'.format(h)
        s -= h * 60 * 60
    if s >= 60:
        m = math.floor(s / 60)
        ret += '{}m'.format(m)
        s -= m * 60
    if s >= 1:
        s = math.floor(s)
        ret += '{}s'.format(s)
    return ret

def load_dictionary():
    if os.path.exists(DICTIONARY_PATH):
        with open(with_path(DICTIONARY_PATH), 'r' , encoding="UTF-8") as fp:
            dictionary = [EOS, UNK, PAD, GO] + json.load(fp)
            index_word = OrderedDict()
            word_index = OrderedDict()
            for index, word in enumerate(dictionary):
                index_word[index] = word
                word_index[word] = index
            dim = len(dictionary)
        return dim, dictionary, index_word, word_index
    else:
        return DIM,None,{0:EOS,1:UNK,2:PAD,3:GO},{EOS:0,UNK:1,PAD:2,GO:3}


"""
def save_model(sess, name='model.ckpt'):
    import tensorflow as tf
    if not os.path.exists('model'):
        os.makedirs('model')
    saver = tf.train.Saver()
    saver.save(sess, with_path('model/' + name))

def load_model(sess, name='model.ckpt'):
    import tensorflow as tf
    saver = tf.train.Saver()
    saver.restore(sess, with_path('model/' + name))
"""

dim, dictionary, index_word, word_index = load_dictionary()

EOS_ID = word_index[EOS]
UNK_ID = word_index[UNK]
PAD_ID = word_index[PAD]
GO_ID = word_index[GO]

class BucketData(object):

    #def __init__(self, buckets_dir, encoder_size, decoder_size):
    def __init__(self, encoder_size, decoder_size):
        self.encoder_size = encoder_size
        self.decoder_size = decoder_size
        #self.name = 'bucket_%d_%d.db' % (encoder_size, decoder_size)
        #self.path = os.path.join(buckets_dir, self.name)
        #self.conn = sqlite3.connect(self.path)
        #self.cur = self.conn.cursor()
        #sql = '''SELECT MAX(ROWID) FROM conversation;'''
        #self.size = self.cur.execute(sql).fetchall()[0][0] # 问答数
        self.size=0
        self.asks=[]
        self.answers=[]
        """for line in fpanswer:
            if decoder_size-5<len(line.split())<=decoder_size:
                self.answers.append(line)
        """

    def all_answers(self, ask): # 搜索ask的所有answer
        #sql = '''
        #SELECT answer FROM conversation
        #WHERE ask = '{}';
        #'''.format(ask.replace("'", "''"))
        ret = []
        #for s in self.cur.execute(sql):
        #    ret.append(s[0])
        for i in xrange(len(self.asks)):
            if(self.asks[i]==ask):
                ret.append(self.answers[i])
        return list(set(ret))

    def random(self): # 随机返回一个问答
        rowid = np.random.randint(0, self.size)
        return self.asks[rowid],self.answers[rowid]
        #while True:
            #rowid = np.random.randint(1, self.size + 1)
            #sql = '''
            #SELECT ask, answer FROM conversation
            #WHERE ROWID = {};
            #'''.format(rowid)
            #ret = self.cur.execute(sql).fetchall()
            #if len(ret) == 1:
            #    ask, answer = ret[0]
            #    if ask is not None and answer is not None:
            #        return ask, answer

bucket_dbs=[BucketData(i,j) for i,j in buckets]
# 将读取文件和分割buckets操作放在此函数内，以支持读取多个文件
def read_bucket_dbs(askfile, answerfile): # 读取问答文件
    global dim
    fpask=open(askfile,'r',encoding="UTF-8")
    fpanswer=open(answerfile,'r',encoding="UTF-8")
    count=0
    for l1,l2 in zip(fpask,fpanswer):
        #if encoder_size-5<len(l1.split())<=encoder_size and len(l2.split())<=decoder_size:
        len1,len2=len(l1.split()),len(l2.split())
        tmp=min([100] + [i for i in range(len(buckets)) if len1<=buckets[i][0] and len2<=buckets[i][1]])
        if tmp!=100:
            bucket_dbs[tmp].asks.append(l1)
            bucket_dbs[tmp].answers.append(l2)
            bucket_dbs[tmp].size+=1
            count+=1
            for word in l1.split():
                if not word in word_index:
                    index_word[len(index_word)]=word
                    word_index[word]=len(word_index)
            for word in l2.split():
                if not word in word_index:
                    index_word[len(index_word)]=word
                    word_index[word]=len(word_index)
            if count>MAX_SENTENCE_NUMBER:
                break
    dim = max(dim,len(index_word))
    return bucket_dbs



def sentence_indice(sentence): # embedding
    global dim,index_word,word_index
    ret = []
    for  word in sentence.split():
        if word in word_index:
            ret.append(word_index[word])
        else:
            ret.append(1)
    return ret



def indice_sentence(indice): # imbedding
    global index_word
    ret = []
    for index in indice:
        word = index_word[index] if index in index_word else UNK
        if word == EOS:
            break
        if word != UNK and word != GO and word != PAD:
            ret.append(word)
    return ''.join(ret)



def vector_sentence(vector): 
    return indice_sentence(vector.argmax(axis=1))


"""
def generate_bucket_dbs(
        input_dir,
        output_dir,
        buckets,
        tolerate_unk=1
    ):
    pool = {}
    word_count = Counter()
    def _get_conn(key):
        if key not in pool:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            name = 'bucket_%d_%d.db' % key
            path = os.path.join(output_dir, name)
            conn = sqlite3.connect(path)
            cur = conn.cursor()
            cur.execute("CREATE TABLE IF NOT EXISTS conversation (ask text, answer text);)
            conn.commit()
            pool[key] = (conn, cur)
        return pool[key]
    all_inserted = {}
    for encoder_size, decoder_size in buckets:
        key = (encoder_size, decoder_size)
        all_inserted[key] = 0
    # 从input_dir列出数据库列表
    db_paths = []
    for dirpath, _, filenames in os.walk(input_dir):
        for filename in (x for x in sorted(filenames) if x.endswith('.db')):
            db_path = os.path.join(dirpath, filename)
            db_paths.append(db_path)
    # 对数据库列表中的数据库挨个提取
    for db_path in db_paths:
        print('读取数据库: {}'.format(db_path))
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        def is_valid(s):
            unk = 0
            for w in s:
                if w not in word_index:
                    unk += 1
                    if unk > tolerate_unk:
                        return False
            return True
        # 读取最大的rowid，如果rowid是连续的，结果就是里面的数据条数
        # 比SELECT COUNT(1)要快
        total = c.execute('''SELECT MAX(ROWID) FROM conversation;''').fetchall()[0][0]
        ret = c.execute('''SELECT ask, answer FROM conversation;''')
        wait_insert = []
        def _insert(wait_insert):
            if len(wait_insert) > 0:
                for encoder_size, decoder_size, ask, answer in wait_insert:
                    key = (encoder_size, decoder_size)
                    conn, cur = _get_conn(key)
                    cur.execute(""#""
                    INSERT INTO conversation (ask, answer) VALUES ('{}', '{}');
                    ""#".format(ask.replace("'", "''"), answer.replace("'", "''")))
                    all_inserted[key] += 1
                for conn, _ in pool.values():
                    conn.commit()
                wait_insert = []
            return wait_insert
        for ask, answer in tqdm(ret, total=total):
            if is_valid(ask) and is_valid(answer):
                for i in range(len(buckets)):
                    encoder_size, decoder_size = buckets[i]
                    if len(ask) <= encoder_size and len(answer) < decoder_size:
                        word_count.update(list(ask))
                        word_count.update(list(answer))
                        wait_insert.append((encoder_size, decoder_size, ask, answer))
                        if len(wait_insert) > 10000000:
                            wait_insert = _insert(wait_insert)
                        break
    word_count_arr = [(k, v) for k, v in word_count.items()]
    word_count_arr = sorted(word_count_arr, key=lambda x: x[1], reverse=True)
    wait_insert = _insert(wait_insert)
    return all_inserted, word_count_arr



if __name__ == '__main__':
    print('generate bucket dbs')
    # 来源数据库目录
    db_path = ''
    if len(sys.argv) >= 2 and os.path.exists(sys.argv[1]):
        db_path = sys.argv[1]
        if not os.path.isdir(db_path):
            print('invalid db source path, not dir')
            exit(1)
    elif os.path.exists('./db'):
        db_path = './db'
    else:
        print('invalid db source path')
        exit(1)

    # 输出目录
    target_path = './bucket_dbs'
    # 不存在就建
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    elif os.path.exists(target_path) and not os.path.isdir(target_path):
        print('invalid target path, exists but not dir')
        exit(1)
    elif os.path.exists(target_path) and os.path.isdir(target_path):
        shutil.rmtree(target_path)
        os.makedirs(target_path)

    # 生成
    all_inserted, word_count_arr = generate_bucket_dbs(
        db_path,
        target_path,
        buckets,
        1
    )
    # 导出字典
    # print('一共找到{}个词'.format(len(word_count_arr)))
    # with open('dictionary_detail.json', 'w') as fp:
    #     json.dump(word_count_arr, fp, indent=4, ensure_ascii=False)
    # with open('dictionary.json', 'w') as fp:
    #     json.dump([x for x, _ in word_count_arr], fp, indent=4, ensure_ascii=False)
    # 输出词库状况
    for key, inserted_count in all_inserted.items():
        print(key)
        print(inserted_count)
    print('done')
"""