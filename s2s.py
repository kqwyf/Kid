#!/usr/bin/env python3

import os
import sys
import math
import time

import numpy as np
import tensorflow as tf

import data_utils
import s2s_model
#tf定义了tf.app.flags，用于支持接受命令行传递参数，相当于接受argv。
#第一个是参数名称，第二个参数是默认值，第三个是参数描述
#http://blog.csdn.net/leiting_imecas/article/details/72367937

tf.app.flags.DEFINE_float(
    'learning_rate',
    0.0003,
    '学习率'
)
tf.app.flags.DEFINE_float(
    'max_gradient_norm',
    5.0,
    '梯度最大阈值'
)
tf.app.flags.DEFINE_float(
    'dropout',
    1.0,
    '每层输出DROPOUT的大小'
)
tf.app.flags.DEFINE_integer(
    'batch_size',
    64,
    '批量梯度下降的批量大小'
)
tf.app.flags.DEFINE_integer(
    'size',
    512,
    'LSTM每层神经元数量'
)
tf.app.flags.DEFINE_integer(
    'num_layers',
    2,
    'LSTM的层数'
)
tf.app.flags.DEFINE_integer(
    'num_epoch',
    2,
    '训练几轮'
)
tf.app.flags.DEFINE_integer(
    'num_samples',
    512,
    '分批softmax的样本量'
)
tf.app.flags.DEFINE_integer(
    'num_per_epoch',
    3200,
    '每轮训练多少随机样本'
)
tf.app.flags.DEFINE_string(
    'buckets_dir',
    './bucket_dbs',
    'sqlite3数据库所在文件夹'
)
tf.app.flags.DEFINE_string(
    'model_dir',
    './model',
    '模型保存的目录'
)
tf.app.flags.DEFINE_string(
    'model_name',
    'model',
    '模型保存的名称'
)
tf.app.flags.DEFINE_boolean(
    'use_fp16',
    False,
    '是否使用16位浮点数（默认32位）'
)
tf.app.flags.DEFINE_integer(
    'bleu',
    -1,
    '是否测试bleu'
)
tf.app.flags.DEFINE_boolean(
    'test',
    True,
    '是否在测试'
)

FLAGS = tf.app.flags.FLAGS
buckets = data_utils.buckets

def create_model(session, forward_only):
    """建立模型"""
    print("dim={}".format(data_utils.dim))
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    model = s2s_model.S2SModel(
        data_utils.dim,
        data_utils.dim,
        buckets,
        FLAGS.size,
        FLAGS.dropout,
        FLAGS.num_layers,
        FLAGS.max_gradient_norm,
        FLAGS.batch_size,
        FLAGS.learning_rate,
        FLAGS.num_samples,
        forward_only,
        dtype
    )
    return model

def train():
    print("进入训练模式")
    """训练模型"""
    # 准备数据
    print('准备数据')
    for askfile,answerfile in data_utils.BUCKET_FILES:
        bucket_dbs=data_utils.read_bucket_dbs(askfile,answerfile) # BucketData列表
    print("dim={}".format(data_utils.dim))
    bucket_sizes = []
    for i in range(len(data_utils.buckets)):
        bucket_size = bucket_dbs[i].size
        bucket_sizes.append(bucket_size)
        print('bucket {} 中有数据 {} 条'.format(i, bucket_size))
    total_size = sum(bucket_sizes)
    buckets_scale = [ # buckets大小前缀和
        sum(bucket_sizes[:i + 1]) / total_size
        for i in range(len(bucket_sizes))
    ]
    print('共有数据 {} 条'.format(total_size))
    print("开始training")
    # 开始建模与训练
    with tf.Session() as sess:
        #　构建模型
        print("构建模型")
        model = create_model(sess, False)
        print("构建完毕")
        # 初始化变量
        print("初始化变量")
        sess.run(tf.global_variables_initializer())
        print("初始化完毕")
        # 开始训练
        print("开始训练")
        metrics = '  '.join([
            '\r[{}]',
            '{:.1f}%',
            '{}/{}',
            'loss={:.3f}',
            '{}/{}'
        ])
        bars_max = 20
        for epoch_index in range(1, FLAGS.num_epoch + 1):
            print('Epoch {}:'.format(epoch_index))
            time_start = time.time()
            epoch_trained = 0
            batch_loss = []
            while True:
                # 选择一个要训练的bucket
                random_number = np.random.random_sample()
                #print("random number:{}".format(random_number))
                bucket_id = min([
                    i for i in range(len(buckets_scale))
                    if buckets_scale[i] > random_number
                ])
                print("bucket_id:{}".format(bucket_id))
                data, data_in = model.get_batch_data(
                    bucket_dbs,
                    bucket_id
                )
                #print("data:")
                #print(data)
                encoder_inputs, decoder_inputs, decoder_weights = model.get_batch(
                    bucket_dbs,
                    bucket_id,
                    data
                )
                #print("encoder_inputs:")
                #print(encoder_inputs)
                #print("decoder_inputs:")
                #print(decoder_inputs)
                #print("decoder_weights:")
                #print(decoder_weights)
                _, step_loss, output = model.step(
                    sess,
                    encoder_inputs,
                    decoder_inputs,
                    decoder_weights,
                    bucket_id,
                    False
                )
                epoch_trained += FLAGS.batch_size
                batch_loss.append(step_loss)
                time_now = time.time()
                time_spend = time_now - time_start
                time_estimate = time_spend / (epoch_trained / FLAGS.num_per_epoch)
                percent = min(100, epoch_trained / FLAGS.num_per_epoch) * 100
                bars = math.floor(percent / 100 * bars_max)
                sys.stdout.write(metrics.format(
                    '=' * bars + '-' * (bars_max - bars),
                    percent,
                    epoch_trained, FLAGS.num_per_epoch,
                    np.mean(batch_loss),
                    data_utils.time(time_spend), data_utils.time(time_estimate)
                ))
                sys.stdout.flush()
                if epoch_trained >= FLAGS.num_per_epoch:
                    break
            print('\n')

            if not os.path.exists(FLAGS.model_dir):
                os.makedirs(FLAGS.model_dir)
            print("保存模型数据中……")
            model.saver.save(sess, os.path.join(FLAGS.model_dir, FLAGS.model_name))
        print("程序正在退出……")


def test_bleu(count):
    print("进入bleu测试模式")
    """测试bleu"""
    from nltk.translate.bleu_score import sentence_bleu
    from tqdm import tqdm
    # 准备数据
    print('准备数据')
    bucket_dbs = data_utils.read_bucket_dbs(FLAGS.buckets_dir)
    bucket_sizes = []
    for i in range(len(buckets)):
        bucket_size = bucket_dbs[i].size
        bucket_sizes.append(bucket_size)
        print('bucket {} 中有数据 {} 条'.format(i, bucket_size))
    total_size = sum(bucket_sizes)
    print('共有数据 {} 条'.format(total_size))
    # bleu设置0的话，默认对所有样本采样
    if count <= 0:
        count = total_size
    buckets_scale = [
        sum(bucket_sizes[:i + 1]) / total_size
        for i in range(len(bucket_sizes))
    ]
    with tf.Session() as sess:
        #　构建模型
        model = create_model(sess, True)
        model.batch_size = 1
        # 初始化变量
        sess.run(tf.global_variables_initializer())
        model.saver.restore(sess, os.path.join(FLAGS.model_dir, FLAGS.model_name))

        total_score = 0.0
        for i in tqdm(range(count)):
            # 选择一个要训练的bucket
            random_number = np.random.random_sample()
            bucket_id = min([
                i for i in range(len(buckets_scale))
                if buckets_scale[i] > random_number
            ])
            data, _ = model.get_batch_data(
                bucket_dbs,
                bucket_id
            )
            encoder_inputs, decoder_inputs, decoder_weights = model.get_batch(
                bucket_dbs,
                bucket_id,
                data
            )
            _, _, output_logits = model.step(
                sess,
                encoder_inputs,
                decoder_inputs,
                decoder_weights,
                bucket_id,
                True
            )
            outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
            ask, _ = data[0]
            all_answers = bucket_dbs[bucket_id].all_answers(ask)
            ret = data_utils.indice_sentence(outputs)
            if not ret:
                continue
            references = [list(x) for x in all_answers]
            score = sentence_bleu(
                references,
                list(ret),
                weights=(1.0,)
            )
            total_score += score
        print('BLUE: {:.2f} in {} samples'.format(total_score / count * 10, count))

#测试模型
def test():
    print("进入测试模式")
    class TestBucket(object):
        def __init__(self, sentence):
            self.sentence = sentence
        def random(self):
            return sentence, ''
    with tf.Session() as sess:
        #　构建模型
        print("创建模型")
        model = create_model(sess, True)
        model.batch_size = 1
        # 初始化变量
        print("初始化变量")
        sess.run(tf.global_variables_initializer())
        print("加载模型数据")
        model.saver.restore(sess, os.path.join(FLAGS.model_dir, FLAGS.model_name))
        print("开始测试")
        sys.stdout.write("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()
        while sentence:
            bucket_id = min([
                b for b in range(len(buckets))
                if buckets[b][0] > len(sentence)
            ])
            data, _ = model.get_batch_data(
                {bucket_id: TestBucket(sentence)},
                bucket_id
            )
            encoder_inputs, decoder_inputs, decoder_weights = model.get_batch(
                {bucket_id: TestBucket(sentence)},
                bucket_id,
                data
            )
            _, _, output_logits = model.step(
                sess,
                encoder_inputs,
                decoder_inputs,
                decoder_weights,
                bucket_id,
                True
            )
            outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
            ret = data_utils.indice_sentence(outputs)
            print(ret)
            print("> ", end="")
            sys.stdout.flush()
            sentence = sys.stdin.readline()

def main(_):
    if FLAGS.bleu > -1:
        test_bleu(FLAGS.bleu)
    elif FLAGS.test:
        test()
    else:
        train()
if __name__ == '__main__':
    np.random.seed(0)
    tf.set_random_seed(0)
    tf.app.run()
