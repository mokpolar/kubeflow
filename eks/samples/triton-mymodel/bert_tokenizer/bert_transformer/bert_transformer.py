# Copyright 2020 kubeflow.org.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import kfserving
from typing import Dict
import numpy as np
from . import tokenization
from . import data_processing
from tensorrtserver.api import *



class BertTransformer(kfserving.KFModel):
    def __init__(self, name: str, predictor_host: str):
        super().__init__(name)
        # 이건 Bert용
        self.short_paragraph_text = "The Apollo program was the third United States human spaceflight program. First conceived as a three-man spacecraft to follow the one-man Project Mercury which put the first Americans in space, Apollo was dedicated to President John F. Kennedy's national goal of landing a man on the Moon. The first manned flight of Apollo was in 1968. Apollo ran from 1961 to 1972 followed by the Apollo-Soyuz Test Project a joint Earth orbit mission with the Soviet Union in 1975."
        
        # 필요한 것
        self.predictor_host = predictor_host
        self.tokenizer = tokenization.FullTokenizer(vocab_file="/mnt/models/vocab.txt", do_lower_case=True)
        # 여기 모델 명이 있군. 필요한 것 
        self.model_name = "bert_tf_v2_large_fp16_128_v2"
        # 모델 버젼이 왜 음수일까
        self.model_version = -1
        # 아래에 전달할 프로토콜 타입인가봐 
        self.protocol = ProtocolType.from_str('http')
        # 이것도 아마도 Bert
        self.infer_ctx = None
    # 이 부분은 똑같이 넣어두면 될 것 같고
    def preprocess(self, inputs: Dict) -> Dict:
        self.doc_tokens = data_processing.convert_doc_tokens(self.short_paragraph_text)
        self.features = data_processing.convert_examples_to_features(self.doc_tokens, inputs["instances"][0], self.tokenizer, 128, 128, 64)
        return self.features
    # 아 이거봐라 predict가 여기 들어있네 
    # 자체 모델에서 나온 predict를 여기서 하는구나. 
    def predict(self, features: Dict) -> Dict:
        # 이 InferContext가 어디서 온 건지를 모르겠다. 얘는 어디서 왔니 . 
        # 그럼 python api에서 찾은 함수중 하나를 여기 놓고  api에 두면 될 것 같은데?
        # custom image에서는 predict api를 직접 구현하는데 이 모델도 그런 것 같다. 
        # 이것 처럼 하면 될 것 같다.
        # 근데 서버에 대한 내용은 어디있지? 
        # 아 그 정보가 InferContext에 같이 들어가네 
        # infer_ctx는 InferContext 의 인스턴스이고, 
        # 여기에는 Triton 서버에 대한 내용이 들어가고
        # InferContext에는 run이 있네. 
        # 이 run의 내용은 실제 모델에 대한 input이 되겠구나. 
        if not self.infer_ctx:

            self.infer_ctx = InferContext(self.predictor_host, self.protocol, self.model_name, self.model_version, http_headers='', verbose=True)
        # infer_ctx.run 에들어가는 input들은 {} 로 감싸주고 config.pbtxt의 유형과 같다. 
        batch_size = 1
        unique_ids = np.int32([1])
        segment_ids = features["segment_ids"]
        input_ids = features["input_ids"]
        input_mask = features["input_mask"] 
        result = self.infer_ctx.run({'unique_ids': (unique_ids,),
                                     'segment_ids': (segment_ids,),
                                     'input_ids': (input_ids,),
                                     'input_mask': (input_mask,)},
                                    {'end_logits': InferContext.ResultFormat.RAW,
                                     'start_logits': InferContext.ResultFormat.RAW}, batch_size)
        return result 
    # predict에서 나온 결과물을 여기서 그대로 갖고 해석하는 것 같다. 
    # 그냥 기본 mobilenet에서 나온거 갖다 쓰면 될 것 같다. 
    # result에서 나온 걸 그대로 받아 쓰네 이렇게 구현하면 될 것 같음. 
    def postprocess(self, result: Dict) -> Dict:
        end_logits = result['end_logits'][0]
        start_logits = result['start_logits'][0]
        n_best_size = 20

        # The maximum length of an answer that can be generated. This is needed 
        #  because the start and end predictions are not conditioned on one another
        max_answer_length = 30

        (prediction, nbest_json, scores_diff_json) = \
           data_processing.get_predictions(self.doc_tokens, self.features, start_logits, end_logits, n_best_size, max_answer_length)
        return {"predictions": prediction, "prob": nbest_json[0]['probability'] * 100.0}
