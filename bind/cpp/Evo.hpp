#include <evo.h>
#include <vector>
#include <string>


namespace Evo {

typedef op_type_t OpType;
typedef tensor_type_t TensorType;

class Tensor;
class Node;
class Graph;
class Model;
class RunTime;

class Tensor {
private:
    tensor_t *_ts;

public:
    Tensor(tensor_t *ts) {
        this->_ts = ts;
    }
    Tensor(const char *name, TensorType type) {
        this->_ts = tensor_new(name, type);
    }
    ~Tensor() {
        tensor_free(this->_ts);
    }

    tensor_t* proto() {
        return this->_ts;
    }
    void dump() {
        tensor_dump(this->_ts);
    }
};

class Model {
private:
    model_t* _mdl;

public:
    Model(model_t* mdl) {
        this->_mdl = mdl;
    }
    Model(const char *name) {
        this->_mdl = model_new(name);
    }
    ~Model() {
        model_free(this->_mdl);
    }

    model_t* proto() {
        return this->_mdl;
    }
};

class Graph {
private:
    graph_t* _g;

public:
    Graph(Model &m) {
        this->_g = graph_new(m.proto());
    }
    ~Graph() {
        graph_free(this->_g);
    }

    graph_t* proto() {
        return this->_g;
    }
};

class Node {
private:
    node_t *_nd;

public:
    Node(Graph &g, const char* name, OpType type) {
        this->_nd = node_new(g.proto(), name, type);
    }
    ~Node() {
        node_free(this->_nd);
    }

    node_t* proto() {
        return this->_nd;
    }
};

class RunTime {
private:
    runtime_t* _rt;

public:
    RunTime(const char* fmt) {
        this->_rt  = runtime_new(fmt);
    }
    ~RunTime() {
        runtime_free(this->_rt);
    }

    runtime_t* proto() {
        return this->_rt;
    }
    Model load(const char *path) {
        return Model(runtime_load(this->_rt, path));
    }
    void unload() {
        runtime_unload(this->_rt);
    }
    Tensor load_tensor(const char *path) {
        return Tensor(runtime_load_tensor(this->_rt, path));
    }
    void set_tensor(const char *name, Tensor &ts) {
        runtime_set_tensor(this->_rt, name, ts.proto());
    }
    Tensor get_tensor(const char* name) {
        return Tensor(runtime_get_tensor(this->_rt, name));
    }
    void run() {
        runtime_run(this->_rt);
    }
    void dump_graph() {
        runtime_dump_graph(this->_rt);
    }
};

}

