import pyevo
import numpy as np

# rt = pyevo.RunTime("onnx")
# rt.load("../../tests/model/mnist_8/model.onnx")
# rt.run()
# rt.dump_graph()

# data1 = np.array()
# data2 = np.array([2, 4, 6, 8])

t1 = pyevo.Tensor([[3.3, 2, 5, 7], [1, 2, 3, 1]])
t2 = pyevo.Tensor([[2, 1, 3, 3,1], [0, 4, 2, 2]])
t3 = t1 + t2