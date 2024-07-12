#include <evo.h>
#include <string>

class Evo {
    const char *model_path;
    serializer_t *sez;
    context_t *ctx;
public:
    Evo() {
        device_reg("cpu");
        this->model_path = NULL;
        this->sez = NULL;
        this->ctx = NULL;
    }
    Evo(const char* model_format) {
        device_reg("cpu");
        this->model_path = model_path;
        this->sez = serializer_new(model_format);
        this->ctx = NULL;
    }
    Evo(const char *model_format, const char *model_path) {
        device_reg("cpu");
        this->model_path = model_path;
        this->sez = serializer_new(model_format);
        this->ctx = this->load(model_path);
    }
    ~Evo() {
        serializer_free(this->sez);
        device_unreg("cpu");
    }

    context_t * load(const char *model_path) {
        this->model_path = model_path;
        this->ctx = this->sez->load_model(this->sez, model_path);
        return this->ctx;
    }

    void unload() {
        if(this->ctx) {
            this->sez->unload(this->ctx);
        }
    }

    void run() {
        graph_prerun(this->ctx->graph);
        graph_run(this->ctx->graph);
        graph_posrun(this->ctx->graph);
    }

    void display() {
        graph_dump(this->ctx->graph);
    }

};