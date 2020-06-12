import base64
import io
from typing import Dict

import kfserving
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class KFServingSampleModel(kfserving.KFModel):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.ready = False

    def load(self):
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        net = Net()
        state_dict = torch.load("model.pt")
        net.load_state_dict(state_dict)
        net.eval()
        self.model = net

        self.ready = True

    def predict(self, request: Dict) -> Dict:
        inputs = request["instances"]

        data = inputs[0]["image"]["b64"]

        raw_img_data = base64.b64decode(data)
        input_image = Image.open(io.BytesIO(raw_img_data))

        preprocess = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)

        output = self.model(input_batch)

        scores = torch.nn.functional.softmax(output, dim=1)[0]

        _, top_3 = torch.topk(output, 3)

        results = {}
        for idx in top_3[0]:
            results[self.classes[idx]] = scores[idx].item()

        return {"predictions": results}


if __name__ == "__main__":
    model = KFServingSampleModel("custom-model")
    model.load()
    kfserving.KFServer(workers=1).start([model])