from typing import Tuple

from tinygrad import Tensor, nn
from tinygrad.nn.state import get_state_dict, get_parameters
from tinygrad.nn.optim import SGD

from datasets import fetch_mnist
from dump import dump

class Net:
  def __init__(self):
    self.l1 = nn.Linear(196, 64, bias=False)
    self.l2 = nn.Linear(64, 10, bias=False)

  def __call__(self, x):
    x = x.reshape(-1, 1, 28, 28)
    x = x.max_pool2d(kernel_size=2, stride=2)
    x = x.reshape(-1, 196)
    return self.l2(self.l1(x).relu())

X_train, Y_train, X_test, Y_test = fetch_mnist()
with Tensor.train(False):
  net = Net()
  def eval_step(x: Tensor, y: Tensor) -> Tensor:
    return ((net(x).argmax(axis=-1) == y).mean() * 100).realize()

  batch = Tensor(X_test[:1]) # type: ignore
  labels = Tensor(Y_test[:1]) # type: ignore
  h, c, weights, weight_map = dump(eval_step, get_parameters(net), [batch, labels], "net", True, False)
  state_dict = get_state_dict(net)
  for offset, (size, bid) in weight_map.items():
    # find a match in the state dict
    for name, param in state_dict.items():
      if id(param.lazydata.realized) == bid:
        print(f"{name} -> off: {offset}, len: {size}")
        break
    else: print(f"not found: {offset}, {size}")

  print("len(weights):", len(weights))
  with open("./esp32/main/net.h", "w") as f: f.write(h)
  with open("./esp32/main/net.c", "w") as f: f.write(c)
with Tensor.train():
  net = Net()
  opt = SGD(get_parameters(net), lr=3e-4)
  def train_step(x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
    opt.zero_grad()
    loss = net(x).sparse_categorical_crossentropy(y).backward()
    opt.step()
    acc = (net(x).argmax(axis=-1) == y).mean()
    return loss.realize(), acc.realize()

  batch = Tensor(X_train[:4]) # type: ignore
  labels = Tensor(Y_train[:4]) # type: ignore
  loss, acc = train_step(batch, labels)
  print(f"loss: {loss.item():6.2f} acc: {acc.item()*100:5.2f}%")

  h, c, weights, weight_map = dump(train_step, get_parameters(opt), [batch, labels], "train", False, True)
  state_dict = get_state_dict({"net": net, "opt": opt})
  for offset, (size, bid) in weight_map.items():
    # find a match in the state dict
    for name, param in state_dict.items():
      if id(param.lazydata.realized) == bid:
        print(f"{name} -> off: {offset}, len: {size}")
        break
    else: print(f"not found: {offset}, {size}")

  print("len(weights):", len(weights))
  with open("./esp32/main/train.h", "w") as f: f.write(h)
  with open("./esp32/main/train.c", "w") as f: f.write(c)

  # write weights as c array
  with open("./esp32/main/weights.h", "w") as f:
    f.write("#pragma once\n")
    f.write(f"#define WEIGHTS_LEN {len(weights)}\n")
    f.write("const unsigned char weights[WEIGHTS_LEN] = {\n")
    for i, b in enumerate(weights):
      f.write(f"  {b}")
      if i != len(weights) - 1: f.write(",")
      f.write("\n")
    f.write("};\n")
