// pywrap.cpp
#include <pybind11/pybind11.h>
#include "pyevo.hpp"

namespace py = pybind11;

int add(int i, int j) {
    return i + j;
}

PYBIND11_MODULE(pyevo, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring
    m.def("add", &add, "A function that adds two numbers");
    py::class_<MyClass>(m, "MyClass")
    .def(py::init<double, double>())  
    .def("run", &MyClass::run, py::call_guard<py::gil_scoped_release>())
    ;
}