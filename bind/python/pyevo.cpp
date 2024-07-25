#define EVO_PYBIND11
#include "../cpp/Evo.hpp"
#include <pybind11/stl.h>

PYBIND11_MODULE(pyevo, m) {
    m.doc() = "Libevo's python bind: use pybind11.";

    Evo::internal_mdl = Evo::Model::from(internal_model_init());
    py::class_<Evo::Tensor>(m, "Tensor")
        .def(py::init<>())
        .def(py::init<const char*, Evo::TensorType>())
        .def(py::init<py::array_t<double>>())
        .def(py::init<py::array_t<int>>())
        .def(py::init<py::array_t<bool>>())
        .def_static("from", &Evo::Tensor::from, py::return_value_policy::reference)
        .def("proto", &Evo::Tensor::proto)
        .def("dump", static_cast<void (Evo::Tensor::*)()>(&Evo::Tensor::dump))
        .def("dump", static_cast<void (Evo::Tensor::*)(int)>(&Evo::Tensor::dump))
        .def("__add__", &Evo::Tensor::operator+)
        .def("__mul__", &Evo::Tensor::operator*);

    py::class_<Evo::Model>(m, "Model")
        .def(py::init<>())
        .def(py::init<model_t*>())
        .def_static("from", &Evo::Model::from, py::return_value_policy::reference)
        .def("proto", &Evo::Model::proto);

    py::class_<Evo::Graph>(m, "Graph")
        .def(py::init<>())
        .def(py::init<Evo::Model&>())
        .def_static("from", &Evo::Graph::from, py::return_value_policy::reference)
        .def("proto", &Evo::Graph::proto);

    py::class_<Evo::Node>(m, "Node")
        .def(py::init<>())
        .def(py::init<Evo::Graph&, const char*, Evo::OpType>())
        .def_static("from", &Evo::Node::from, py::return_value_policy::reference)
        .def("proto", &Evo::Node::proto)
        .def("in", &Evo::Node::in, py::return_value_policy::reference)
        .def("out", &Evo::Node::out, py::return_value_policy::reference);

    py::class_<Evo::RunTime>(m, "RunTime")
        .def(py::init<>())
        .def(py::init<const char*>())
        .def("proto", &Evo::RunTime::proto)
        .def("model", &Evo::RunTime::model, py::return_value_policy::reference)
        .def("load", &Evo::RunTime::load, py::return_value_policy::reference)
        .def("unload", &Evo::RunTime::unload)
        .def("load_tensor", &Evo::RunTime::load_tensor, py::return_value_policy::reference)
        .def("set_tensor", &Evo::RunTime::set_tensor)
        .def("get_tensor", &Evo::RunTime::get_tensor, py::return_value_policy::reference)
        .def("run", &Evo::RunTime::run)
        .def("dump_graph", &Evo::RunTime::dump_graph);

    
    py::class_<Evo::Image>(m, "Image")
        .def(py::init<>())
        .def(py::init<const char*>())
        .def("proto", &Evo::Image::proto)
        .def("dump_raw", static_cast<void (Evo::Image::*)()>(&Evo::Image::dump_raw))
        .def("dump_raw", static_cast<void (Evo::Image::*)(int)>(&Evo::Image::dump_raw))
        .def("to_tensor", static_cast<Evo::Tensor* (Evo::Image::*)()>(&Evo::Image::to_tensor), py::return_value_policy::reference)
        .def("to_tensor", static_cast<Evo::Tensor* (Evo::Image::*)(int)>(&Evo::Image::to_tensor), py::return_value_policy::reference)
        .def("save", &Evo::Image::save)
        .def_static("load_mnist", &Evo::Image::load_mnist, py::return_value_policy::reference)
        .def("__getitem__", static_cast<Evo::Image*(Evo::Image::*)(int)>(&Evo::Image::operator[]), py::return_value_policy::reference)
        .def("__getitem__", static_cast<Evo::Image*(Evo::Image::*)(const std::vector<int>&)>(&Evo::Image::operator[]), py::return_value_policy::reference);

    m.def("add", &Evo::add);
    m.def("mul", &Evo::mul);
}