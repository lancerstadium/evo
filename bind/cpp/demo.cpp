#include "Evo.hpp"

int main() {
    Evo e = Evo("onnx", "../../tests/model/mnist_8/model.onnx");
    e.run();
    e.display();
}