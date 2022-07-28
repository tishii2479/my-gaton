import torch
from torch import nn
import torch.nn.functional as F
from random import randint, random


torch.set_printoptions(precision=3, sci_mode=False)


class Model(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int
    ):
        super().__init__()
        self.f1 = nn.Linear(in_dim, hidden_dim)
        self.f2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, input):
        h = self.f1.forward(input)
        h = F.relu(h)
        h = self.f2.forward(h)
        out = F.softmax(h, dim=1)
        return out


def create_toydata(
    num_topic: int,
    min_data: int,
    max_data: int,
    threshold: float = 0.8
):
    data_size = []
    data = []
    labels = []
    for topic in range(num_topic):
        size = randint(min_data, max_data)
        data_size.append(size)
        for _ in range(size):
            vec = [0] * num_topic
            for i in range(num_topic):
                if i == topic:
                    vec[i] = random() * 0.2 + threshold
                else:
                    vec[i] = random() * 0.2
            data.append(vec)
            labels.append(topic)

    return torch.tensor(data), torch.tensor(labels)


num_topic = 5
min_data, max_data = 10, 10

data, labels = create_toydata(num_topic, min_data, max_data)
test_data, test_labels = create_toydata(num_topic, 5, 5, 0.6)

hidden_dim = 30
model = Model(num_topic, hidden_dim, num_topic)
optimizer = torch.optim.SGD(model.parameters(), lr=1)

epochs = 100000


def fit():
    for epoch in range(epochs):
        model.train()
        out = model.forward(data)
        loss = F.cross_entropy(out, labels) / len(data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            model.eval()
            out = model.forward(test_data)
            test_loss = F.cross_entropy(out, test_labels) / len(data)

            print(f'{epoch}, {loss}, {test_loss}')


def eval():
    model.eval()
    out = model.forward(data)
    print(out)


fit()
eval()
