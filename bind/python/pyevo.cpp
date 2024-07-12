#include <pybind11/pybind11.h>
#include "../cpp/Evo.hpp"

namespace py = pybind11;

PYBIND11_MODULE(pyevo, m) {
    m.doc() = "Libevo's python bind: use pybind11.";
    py::class_<Evo>(m, "Evo")
    .def(py::init<>())
    .def(py::init<const char*>())
    .def(py::init<const char*, const char*>())
    .def("load", &Evo::load, py::call_guard<py::gil_scoped_release>())
    .def("unload", &Evo::unload, py::call_guard<py::gil_scoped_release>())
    .def("run", &Evo::run, py::call_guard<py::gil_scoped_release>())
    .def("display", &Evo::display, py::call_guard<py::gil_scoped_release>())
    ;
}