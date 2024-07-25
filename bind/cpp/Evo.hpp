#include <evo.h>
#include <vector>
#include <string>

#ifdef EVO_PYBIND11
#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>

namespace py = pybind11;
using namespace py::literals;
#endif

namespace Evo {

typedef op_type_t OpType;
typedef enum {
    UNDEFINED = TENSOR_TYPE_UNDEFINED,
    BOOL = TENSOR_TYPE_BOOL,
    INT8 = TENSOR_TYPE_INT8,
    INT16 = TENSOR_TYPE_INT16,
    INT32 = TENSOR_TYPE_INT32,
    INT64 = TENSOR_TYPE_INT64,
    UINT8 = TENSOR_TYPE_UINT8,
    UINT16 = TENSOR_TYPE_UINT16,
    UINT32 = TENSOR_TYPE_UINT32,
    UINT64 = TENSOR_TYPE_UINT64,
    BFLOAT16 = TENSOR_TYPE_BFLOAT16,
    FLOAT16 = TENSOR_TYPE_FLOAT16,
    FLOAT32 = TENSOR_TYPE_FLOAT32,
    FLOAT64 = TENSOR_TYPE_FLOAT64,
    COMPLEX64 = TENSOR_TYPE_COMPLEX64,
    COMPLEX128 = TENSOR_TYPE_COMPLEX128,
    STRING = TENSOR_TYPE_STRING,
} TensorType;

class Tensor;
class Node;
class Graph;
class Model;
class RunTime;

static Model * internal_mdl = nullptr;

static Tensor * add(Tensor *a, Tensor *b);
static Tensor * mul(Tensor *a, Tensor *b);

class Tensor {
private:
    tensor_t *_ts;

public:
    Tensor() : _ts(nullptr) {}
    Tensor(const char *name, TensorType type) {
        this->_ts = tensor_new(name, (tensor_type_t)type);
    }
#ifdef EVO_PYBIND11
    template<typename T>
    Tensor(py::array_t<T> arr) {
        py::buffer_info buf = arr.request();
        int ndim = buf.ndim;
        int dims[ndim];
        for(int i = 0; i < ndim; i++) {
            dims[i] = buf.shape[i];
        }
        size_t ndata = buf.size;
        auto *datas = static_cast<T*>(buf.ptr);
        TensorType type;
        if (std::is_same<T, double>::value) {
            type = TensorType::FLOAT64;
        } else if (std::is_same<T, int>::value) {
            type = TensorType::INT64;
        } else if (std::is_same<T, bool>::value) {
            type = TensorType::BOOL;
        } else {
            throw std::runtime_error("Unsupported type");
        }
        this->_ts = tensor_new("Tensor", (tensor_type_t)type);
        tensor_reshape(this->_ts, ndim, dims);
        memcpy(this->_ts->datas, datas, ndata * sizeof(T));
        this->_ts->ndata = ndata;
    }
#endif
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
        if(this->_ts)
            return this->_ts;
        return NULL;
    }
    static Tensor* zero() {
        Tensor *t = new Tensor();
        t->_ts = tensor_new("Tensor", TENSOR_TYPE_INT64);
        return t;
    }
    void dump() {
        if(this->_ts)
            tensor_dump(this->_ts);
    }
    void dump(int level) {
        switch(level) {
            case 1:
            if(this->_ts) { tensor_dump2(this->_ts); } return;
            case 0:
            default: dump(); return;
        }
    }

    Tensor * operator+(Tensor * other) {
        return add(this, other);
    }
    Tensor * operator*(Tensor * other) {
        return mul(this, other);
    }
};

class Model {
private:
    model_t* _mdl;

public:
    Model() : _mdl(nullptr) {}
    Model(model_t* mdl) {
        this->_mdl = mdl;
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
    Graph() : _g(nullptr) {}
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
    Node() : _nd(nullptr) {}
    Node(OpType type, std::vector<Tensor*> inputs, std::vector<Tensor*> outputs) : _nd(node_new(internal_mdl ? internal_mdl->proto()->graph : NULL, op_name((op_type_t)type), type)) {
        Tensor** in = inputs.data();
        int nin = inputs.size();
        this->_nd->nin = nin;
        this->_nd->in = (tensor_t**)malloc(sizeof(tensor_t*) * nin);
        for(int i = 0; i < nin; i++) {
            this->_nd->in[i] = in[i]->proto();
        }
        Tensor** out = outputs.data();
        int nout = outputs.size();
        this->_nd->nout = nout;
        this->_nd->out = (tensor_t**)malloc(sizeof(tensor_t*) * nout);
        for(int i = 0; i < nout; i++) {
            this->_nd->out[i] = out[i]->proto();
        }
        if(_nd->graph && _nd->graph->dev) {
            this->_nd->op = device_find_op(_nd->graph->dev, (op_type_t) type);
        }
    }
    Node(Graph &g, const char* name, OpType type) {
        this->_nd = node_new(g.proto(), name, type);
    }
    ~Node() {
        if(this->_nd) {
            node_free(this->_nd);
        }
    }

    static Node* from(node_t *nd) {
        Node *n = new Node();
        n->_nd = nd;
        return n;
    }
    node_t* proto() {
        return this->_nd;
    }
    void forward() {
        if(this->_nd && this->_nd->op && this->_nd->op->run)
            this->_nd->op->run(this->_nd);
    }
    Tensor * in(int i) {
        if(i < 0 || i >= this->_nd->nin)
            return nullptr;
        return Tensor::from(this->proto()->in[i]);
    }
    Tensor * out(int i) {
        if(i < 0 || i >= this->_nd->nout)
            return nullptr;
        return Tensor::from(this->proto()->out[i]);
    }
};

class RunTime {
private:
    runtime_t* _rt;

public:
    RunTime() : _rt(nullptr) {}
    RunTime(const char* fmt) {
        this->_rt  = runtime_new(fmt);
    }
    ~RunTime() {
        if(this->_rt) {
            runtime_free(this->_rt);
        }
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
        if(this->_rt)
            runtime_unload(this->_rt);
    }
    Tensor* load_tensor(const char *path) {
        if(this->_rt && path)
            return Tensor::from(runtime_load_tensor(this->_rt, path));
        else 
            return nullptr;
    }
    void set_tensor(const char *name, Tensor *ts) {
        if(this->_rt && name && ts)
            runtime_set_tensor(this->_rt, name, ts->proto());
    }
    Tensor* get_tensor(const char* name) {
        if(this->_rt && name)
            return Tensor::from(runtime_get_tensor(this->_rt, name));
        else 
            return nullptr;
    }
    void run() {
        if(this->_rt)
            runtime_run(this->_rt);
    }
    void dump_graph() {
        if(this->_rt)
            runtime_dump_graph(this->_rt);
    }
};


class Image;

class Image {
private:
    image_t *_img;

public:
    Image() : _img(nullptr) {}
    Image(const char *path) : _img(image_load(path)) {}
    ~Image() { 
        if(this->_img) {
            image_free(this->_img);
        }
    }

    static Image* from(image_t *img) { 
        Image *i = new Image();
        i->_img = img;
        return i;
    }
    image_t* proto() {
        if(this->_img) {
            return this->_img;
        } else {
            return nullptr;
        }
    }
    void dump_raw() {
        if(this->_img) {
            image_dump_raw(this->_img, -1);
        }
    }
    void dump_raw(int i) {
        if(this->_img) {
            image_dump_raw(this->_img, i);
        }
    }
    Tensor* to_tensor() {
        if(this->_img && this->_img->raw) {
            return Tensor::from(this->_img->raw);
        } else {
            return nullptr;
        }
    }
    Tensor* to_tensor(int i) {
        if(this->_img && this->_img->raw) {
            return Tensor::from(image_get_raw(this->_img, i));
        } else {
            return nullptr;
        }
    }
    static Image* load_mnist(const char *pics, const char* labels) {
        return Image::from(image_load_mnist(pics, labels));
    }
    void save(const char* path) {
        image_save(this->_img, path);
    }
    Image* operator[](const int idx) {
        return Image::from(image_get(this->_img, idx));
    }
    Image* operator[](const std::vector<int>& idxs) { 
        return Image::from(image_get_batch(this->_img, idxs.size(), (int *)idxs.data()));
    }
};

static Tensor * add(Tensor *a, Tensor *b) {
    std::vector<Tensor*> inputs = {a, b};
    std::vector<Tensor*> outputs = {Tensor::zero()};
    Node nd = Node(OP_TYPE_ADD, inputs, outputs);
    nd.forward();
    return nd.out(0);
}

static Tensor * mul(Tensor *a, Tensor *b) {
    std::vector<Tensor*> inputs = {a, b};
    std::vector<Tensor*> outputs = {Tensor::zero()};
    Node nd = Node(OP_TYPE_MUL, inputs, outputs);
    nd.forward();
    return nd.out(0);
}
}

