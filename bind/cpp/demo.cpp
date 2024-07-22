#include "Evo.hpp"
#include <iostream>

int main() {
    // --- cpp
    Evo::RunTime rt = Evo::RunTime("onnx");
    rt.load("../../tests/model/mnist_8/model.onnx");
    rt.load("../../tests/model/mobilenet_v2_7/model.onnx");
    Evo::Tensor* t0 = rt.load_tensor("../../tests/model/mnist_8/test_data_set_0/input_0.pb");
    t0->dump();
    rt.set_tensor("Input3", t0);
    rt.run();
    rt.dump_graph();
    return 0;
}