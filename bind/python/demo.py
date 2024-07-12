import pyevo

model = pyevo.Evo("onnx", "../../tests/model/mnist_8/model.onnx")
model.run()
model.display()