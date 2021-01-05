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
import argparse
from .bert_transformer import BertTransformer
# 별도로 구현한 BertTransformer 클래스를 사용한다. 
# 기본 모델 이름 정하고
# 다른 Predictor를 구현하는 것 처럼 아예 kfserving 서버를 실행해버리네
# 여기서 BertTransformer 클래스에 모델 네임을 args 로 넣고 predictor host도 넣어준다. 
# 그리고 kfserving서버를 스타트해버린다. 
# 이건 보통.. predictor파일에 넣던거 아닌가?
# 아니구나 transformer파일에 넣던 것은 맞네
# 근데 그동안은 inference.py 파일에 넣고 
# preprocess, postprocess 앞단에 만들어 두고 그걸 실행했는데 그걸 다만 별도의 파일에 있는 클래스로 뺐을 뿐

DEFAULT_MODEL_NAME = "model"

parser = argparse.ArgumentParser(parents=[kfserving.kfserver.parser])
parser.add_argument('--model_name', default=DEFAULT_MODEL_NAME,
                    help='The name that the model is served under.')
parser.add_argument('--predictor_host', help='The URL for the model predict function', required=True)

args, _ = parser.parse_known_args()

if __name__ == "__main__":
    transformer = BertTransformer(args.model_name, predictor_host=args.predictor_host)
    kfserver = kfserving.KFServer()
    kfserver.start(models=[transformer])
