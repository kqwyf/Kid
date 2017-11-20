import pdb
import random
import copy

import numpy as np
import tensorflow as tf

import data_utils





# seq2seq_f
def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
    tmp_cell = copy.deepcopy(cell)
    return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
        encoder_inputs, # tensor of inputs
        decoder_inputs, #tensor of outputs
        tmp_cell, #自定义的cell
        num_encoder_symbols=source_vocab_size, #encode输入数据的大小
        num_decoder_symbols=target_vocab_size, #decode输入数据的大小
        embedding_size=size, #embedding 维度
        output_projection=output_projection, #
        feed_previous=do_decode,
        dtype=dtype
    )


    #输入变量的定义
    # Feeds for inputs.
    self.encoder_inputs = []
    self.decoder_inputs = []
    self.target_weights = []
    for i in xrange(buckets[-1][0]):  # Last bucket is the biggest one.
        self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                    name="encoder{0}".format(i)))
    #encoder_inputs 这个列表对象中的每一个元素表示一个占位符，其名字分别为encoder0, encoder1,…,encoder39，encoder{i}的几何意义是编码器在时刻i的输入。
    # 输出比输入大 1，这是为了保证下面的targets可以向左shift 1位
    for i in xrange(buckets[-1][1] + 1):
        self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                    name="decoder{0}".format(i)))
        self.target_weights.append(tf.placeholder(dtype, shape=[None],
                                                  name="weight{0}".format(i)))
    #target_weights 是一个与 decoder_outputs 大小一样的 0-1 矩阵。该矩阵将目标序列长度以外的其他位置填充为标量值 0。
    # Our targets are decoder inputs shifted by one.
    targets = [self.decoder_inputs[i + 1]
                for i in xrange(len(self.decoder_inputs) - 1)]
    # 跟language model类似，targets变量是decoder inputs平移一个单位的结果，




    # 区别在于seq2seq_f函数的参数feed previous是True还是false
    if forward_only: # 测试阶段
        self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(#？？
            self.encoder_inputs, self.decoder_inputs, targets,
            self.target_weights, buckets, lambda x, y: seq2seq_f(
                x, y, True),
            softmax_loss_function=softmax_loss_function)
        # If we use output projection, we need to project outputs for
        # decoding.
        if output_projection is not None:
            for b in xrange(len(buckets)):
                self.outputs[b] = [
                    tf.matmul(output, output_projection[
                              0]) + output_projection[1]
                    for output in self.outputs[b]
                ]
    else:#训练阶段
        self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
            self.encoder_inputs, self.decoder_inputs, targets,
            self.target_weights, buckets,
            lambda x, y: seq2seq_f(x, y, False),
            softmax_loss_function=softmax_loss_function)





    params = tf.trainable_variables()
    if not forward_only:# 只有训练阶段才需要计算梯度和参数更新
        self.gradient_norms = []
        self.updates = []
        opt = tf.train.GradientDescentOptimizer(self.learning_rate) # 用梯度下降法优化
        for b in xrange(len(buckets)):
            gradients = tf.gradients(self.losses[b], params) #计算损失函数关于参数的梯度
            clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                             max_gradient_norm)# clip gradients 防止梯度爆炸
            self.gradient_norms.append(norm)
            self.updates.append(opt.apply_gradients(
                zip(clipped_gradients, params), global_step=self.global_step))#更新参数