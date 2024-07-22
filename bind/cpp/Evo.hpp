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
    Tensor() {}
    Tensor(const char *name, TensorType type) {
        this->_ts = tensor_new(name, type);
    }
    ~Tensor() {
        if(this->_ts) {
            tensor_free(this->_ts);
        }
    }

    static Tensor* from(tensor_t *ts) {
        Tensor *t = new Tensor();
        t->_ts = ts;
        return t;
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
    Model() {}
    Model(model_t* mdl) {
        this->_mdl = mdl;
    }
    Model(const char *name) {
        this->_mdl = model_new(name);
    }
    ~Model() {
        if(this->_mdl) {
            model_free(this->_mdl);
        }
    }

    static Model* from(model_t *mdl) {
        Model *m = new Model();
        m->_mdl = mdl;
        return m;
    }
    model_t* proto() {
        return this->_mdl;
    }
};

class Graph {
private:
    graph_t* _g;

public:
    Graph() {}
    Graph(Model &m) {
        this->_g = graph_new(m.proto());
    }
    ~Graph() {
        if(this->_g) {
            graph_free(this->_g);
        }
    }

    static Graph* from(graph_t *g) {
        Graph *g_ = new Graph();
        g_->_g = g;
        return g_;
    }
    graph_t* proto() {
        return this->_g;
    }
};

class Node {
private:
    node_t *_nd;

public:
    Node() {}
    Node(Graph &g, const char* name, OpType type) {
        this->_nd = node_new(g.proto(), name, type);
    }
    ~Node() {
        node_free(this->_nd);
    }

    static Node* from(node_t *nd) {
        Node *n = new Node();
        n->_nd = nd;
        return n;
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
    Model* model() {
        return Model::from(this->_rt->mdl);
    }
    Model* load(const char *path) {
        runtime_load(this->_rt, path);
        return this->model();
    }
    void unload() {
        runtime_unload(this->_rt);
    }
    Tensor* load_tensor(const char *path) {
        return Tensor::from(runtime_load_tensor(this->_rt, path));
    }
    void set_tensor(const char *name, Tensor *ts) {
        runtime_set_tensor(this->_rt, name, ts->proto());
    }
    Tensor* get_tensor(const char* name) {
        return Tensor::from(runtime_get_tensor(this->_rt, name));
    }
    void run() {
        runtime_run(this->_rt);
    }
    void dump_graph() {
        runtime_dump_graph(this->_rt);
    }
};

}

