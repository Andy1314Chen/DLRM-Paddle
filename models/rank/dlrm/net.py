# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import math
import numpy as np

MIN_FLOAT = np.finfo(np.float32).min / 100.0


class DLRMLayer(nn.Layer):
    def __init__(self,
                 dense_feature_dim,
                 bot_layer_sizes,
                 sparse_feature_number,
                 sparse_feature_dim,
                 top_layer_sizes,
                 num_field,
                 sync_mode=None):
        super(DLRMLayer, self).__init__()
        self.dense_feature_dim = dense_feature_dim
        self.bot_layer_sizes = bot_layer_sizes
        self.sparse_feature_number = sparse_feature_number
        self.sparse_feature_dim = sparse_feature_dim
        self.top_layer_sizes = top_layer_sizes
        self.num_field = num_field

        self.bot_mlp = MLPLayer(input_shape=dense_feature_dim,
                                units_list=bot_layer_sizes,
                                last_action="relu")

        self.top_mlp = MLPLayer(input_shape=int(num_field * (num_field + 1) / 2) + sparse_feature_dim,
                                units_list=top_layer_sizes)

        self.embedding = paddle.nn.Embedding(num_embeddings=self.sparse_feature_number,
                                             embedding_dim=self.sparse_feature_dim,
                                             sparse=True,
                                             weight_attr=paddle.ParamAttr(
                                                 name="SparseFeatFactors",
                                                 initializer=paddle.nn.initializer.Uniform()))

    def forward(self, sparse_inputs, dense_inputs):
        # (batch_size, sparse_feature_dim)
        x = self.bot_mlp(dense_inputs)

        # interact dense and sparse feature
        batch_size, d = x.shape

        sparse_embs = []
        for s_input in sparse_inputs:
            emb = self.embedding(s_input)
            emb = paddle.reshape(emb, shape=[-1, self.sparse_feature_dim])
            sparse_embs.append(emb)

        T = paddle.reshape(paddle.concat(x=sparse_embs + [x], axis=1), (batch_size, -1, d))
        Z = paddle.bmm(T, paddle.transpose(T, perm=[0, 2, 1]))

        Zflat = paddle.triu(Z, 1) + paddle.tril(paddle.ones_like(Z) * MIN_FLOAT, 0)
        Zflat = paddle.reshape(paddle.masked_select(Zflat,
                                                    paddle.greater_than(Zflat, paddle.ones_like(Zflat) * MIN_FLOAT)),
                               (batch_size, -1))

        R = paddle.concat([x] + [Zflat], axis=1)

        y = self.top_mlp(R)
        return y


class MLPLayer(nn.Layer):
    def __init__(self, input_shape, units_list=None, l2=0.01, last_action=None, **kwargs):
        super(MLPLayer, self).__init__(**kwargs)

        if units_list is None:
            units_list = [128, 128, 64]
        units_list = [input_shape] + units_list

        self.units_list = units_list
        self.l2 = l2
        self.mlp = []
        self.last_action = last_action

        for i, unit in enumerate(units_list[:-1]):
            if i != len(units_list) - 1:
                dense = paddle.nn.Linear(in_features=unit,
                                         out_features=units_list[i + 1],
                                         weight_attr=paddle.ParamAttr(
                                             initializer=paddle.nn.initializer.Normal(std=1.0 / math.sqrt(unit))))
                self.mlp.append(dense)
                self.add_sublayer('dense_%d' % i, dense)

                relu = paddle.nn.ReLU()
                self.mlp.append(relu)
                self.add_sublayer('relu_%d' % i, relu)

                norm = paddle.nn.BatchNorm1D(units_list[i + 1])
                self.mlp.append(norm)
                self.add_sublayer('norm_%d' % i, norm)
            else:
                dense = paddle.nn.Linear(in_features=unit,
                                         out_features=units_list[i + 1],
                                         weight_attr=paddle.nn.initializer.Normal(std=1.0 / math.sqrt(unit)))
                self.mlp.append(dense)
                self.add_sublayer('dense_%d' % i, dense)

                if last_action is not None:
                    relu = paddle.nn.ReLU()
                    self.mlp.append(relu)
                    self.add_sublayer('relu_%d' % i, relu)

    def forward(self, inputs):
        outputs = inputs
        for n_layer in self.mlp:
            outputs = n_layer(outputs)
        return outputs
