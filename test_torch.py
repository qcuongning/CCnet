import torch
from torch import nn
import tensorflow.keras


class simple_model(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(simple_model, self).__init__()
        self.fc1 = nn.Linear(D_in, H)
        self.fc2 = nn.Linear(H, D_out)

    def forward(self, x):
        h_relu = self.fc1(x).clamp(min=0)
        y_pred = self.fc2(h_relu)
        return y_pred


N, D_in, D_out, H = 100, 320, 2, 64
x = torch.randn(N, D_in)
print(x.shape)
y = torch.randn(N, D_out)
print(y.shape)
model = simple_model(D_in, H, D_out)
loss = nn.MSELoss(reduction="sum")
opt = torch.optim.SGD(model.parameters(), lr=1e-4)
for t in range(100):
    y_pred = model(x)
    lo = loss(y_pred, y)
    print(t, lo.data.numpy())
    opt.zero_grad()
    lo.backward()
    opt.step()
