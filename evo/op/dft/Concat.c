#include <string.h>
#include "../../util/sys.h"
#include "../../core/resolver.h"

typedef struct {
    int axis;
    int caxis;
} operator_pdata_t;

static void Concat_operator(node_t *nd) {
    operator_pdata_t *pdat = (operator_pdata_t *)nd->priv;
    tensor_t *y = nd->out[0];
    tensor_t *x;
    int ybase;
    int ypitch;
    int xpitch;
    int i, j, k;
    int idx;
    size_t o, l;

    if (nd->in[0]->type == TENSOR_TYPE_STRING) {
        char **py = (char **)y->datas;
        char **px;
        for (i = y->ndim - 1, ypitch = 1; i >= pdat->caxis; i--)
            ypitch *= y->dims[i];
        for (idx = 0, ybase = 0; idx < nd->nin; idx++) {
            x = nd->in[idx];
            px = (char **)x->datas;
            for (i = x->ndim - 1, xpitch = 1; i >= pdat->caxis; i--)
                xpitch *= x->dims[i];
            for (o = 0, j = 0, k = ybase, l = x->ndata; o < l; o++) {
                if (py[k + o])
                    free(py[k + o]);
                py[k + o] = sys_strdup(px[o]);
                if (++j == xpitch) {
                    k += (ypitch - xpitch);
                    j = 0;
                }
            }
            ybase += xpitch;
        }
    } else {
        char *py = (char *)y->datas;
        char *px;
        int sz = tensor_type_sizeof(nd->in[0]->type);
        for (i = y->ndim - 1, ypitch = 1; i >= pdat->caxis; i--)
            ypitch *= y->dims[i];
        for (idx = 0, ybase = 0; idx < nd->nin; idx++) {
            x = nd->in[idx];
            px = (char *)x->datas;
            for (i = x->ndim - 1, xpitch = 1; i >= pdat->caxis; i--)
                xpitch *= x->dims[i];
            for (o = 0, j = 0, k = ybase, l = x->ndata; o < l; o++) {
                memcpy(py + (k + o) * sz, px + o * sz, sz);
                if (++j == xpitch) {
                    k += (ypitch - xpitch);
                    j = 0;
                }
            }
            ybase += xpitch;
        }
    }
}

void op_Concat_dft(node_t *nd) {
    // 1. Concat init
    if (!nd || !nd->in || nd->in[0]->type == TENSOR_TYPE_UNDEFINED) {
        return;
    }
    if (!(nd->nin >= 1) || !(nd->nout == 1) || (nd->in[0]->ndim == 0)) {
        return;
    }
    operator_pdata_t *pdat = malloc(sizeof(operator_pdata_t));
    if (pdat) {
        pdat->axis = node_get_attr_int(nd, "axis", 1);
        nd->priv = pdat;
    }
    // 2. Concat reshape
    tensor_t *x = nd->in[0];
    tensor_t *y = nd->out[0];
    int ndim = x->ndim;
    int dims[ndim];
    int *pdims;
    int i, j, s;
    pdat->caxis = pdat->axis;
    if (pdat->caxis < 0)
        pdat->caxis += ndim;
    if (pdat->caxis < 0 || pdat->caxis >= ndim)
        return;
    s = x->dims[pdat->caxis];
    for (i = 1; i < nd->nin; i++) {
        pdims = nd->in[i]->dims;
        for (j = 0; j < ndim; j++) {
            if (j == pdat->caxis)
                s += pdims[j];
            else if (x->dims[j] != pdims[j])
                return;
            dims[j] = pdims[j];
        }
    }
    dims[pdat->caxis] = s;
    y->type = x->type;
    tensor_reshape(y, ndim, dims);
    // 3. Concat run
    Concat_operator(nd);
    // 4. Concat exit
    if (pdat)
        free(pdat);
    return;
}