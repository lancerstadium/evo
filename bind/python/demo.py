import pyevo

rt = pyevo.RunTime("onnx")
rt.load("../../tests/model/mnist_8/model.onnx")
rt.run()
rt.dump_graph()