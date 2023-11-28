from typing import cast
from socketserver import TCPServer, BaseRequestHandler
import select, struct

import numpy as np

from datasets import fetch_mnist

X_train, Y_train, X_test, Y_test = fetch_mnist(tensors=False)

class Handler(BaseRequestHandler):
  def handle(self):
    print(f"got connection from {self.client_address}")
    while True:
      if select.select([self.request], [], [], 10)[0]:
        # read a int32
        index, bs = struct.unpack("II", self.request.recv(8))
        print(f"got index {index} with batch size {bs}")

        # send the data
        if bs != 0:
          X, Y = cast(np.ndarray, X_train[index:index+bs]), cast(np.ndarray, Y_train[index:index+bs])
          sendbytes = X.tobytes() + Y.tobytes()
          print(f"sending {len(sendbytes)} bytes")
          self.request.sendall(sendbytes)
        else:
          X, Y = cast(np.ndarray, X_test[index:index+1]), cast(np.ndarray, Y_test[index:index+1])
          sendbytes = X.tobytes() + Y.tobytes()
          print(f"sending {len(sendbytes)} bytes")
          self.request.sendall(sendbytes)
      else: break

HOST, PORT = "0.0.0.0", 29999
with TCPServer((HOST, PORT), Handler, bind_and_activate=False) as server:
  server.allow_reuse_address = True
  server.server_bind()
  server.server_activate()
  server.serve_forever()
