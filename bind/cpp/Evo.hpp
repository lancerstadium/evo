#include <evo.h>
#include <string>

class Evo {
    const char *model_path;
    serializer_t *sez;
    model_t *mdl;
public:
    Evo() {
        device_reg("cpu");
        this->model_path = NULL;
        this->sez = NULL;
        this->mdl = NULL;
    }
    Evo(const char* model_format) {
        device_reg("cpu");
        this->model_path = model_path;
        this->sez = serializer_new(model_format);
        this->mdl = NULL;
    }
    Evo(const char *model_format, const char *model_path) {
        device_reg("cpu");
        this->model_path = model_path;
        this->sez = serializer_new(model_format);
        this->mdl = this->load(model_path);
    }
    ~Evo() {
        serializer_free(this->sez);
        device_unreg("cpu");
    }

    model_t * load(const char *model_path) {
        this->model_path = model_path;
        this->mdl = this->sez->load_model(this->sez, model_path);
        return this->mdl;
    }

    void unload() {
        if(this->mdl) {
            this->sez->unload(this->mdl);
        }
    }

    void run() {
        graph_prerun(this->mdl->graph);
        graph_run(this->mdl->graph);
        graph_posrun(this->mdl->graph);
    }

    void display() {
        graph_dump(this->mdl->graph);
    }

};