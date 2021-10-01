import sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
sys.path.append('../../../../')

from django.shortcuts import render
import os
from Explain.Explainer import get_data_and_explainer
from VOTERS.base_function import ctx_prepare_local
from VOTERS.base_function import ctx_prepare_global as ctx_prepare_global
# from VOTERS.base_function import ctx_prepare_global
from Functions.ColorGen import ColorGen

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

data, explainer = get_data_and_explainer("CENSINCOME", "VOTERS")
model = explainer.model
layers = explainer.model.deep_layers
n_label = explainer.model.n_label
hiddens = [[explainer.get_node_name(0, u) for u in range(model.n_cont + model.n_disc)]] + \
          [[explainer.get_node_name(depth + 1, u) for u in range(layer_now)] for depth, layer_now in enumerate(layers)]
# color_gen = ColorGen(minval=-1, maxval=1, colors_hex=['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#ffffbf', '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026'])
color_gen = ColorGen(minval=-1, maxval=1, colors_hex=['#87CEFA', '#FFFF00', 'FF4500'])


def global_init_page(request):
    ctx = {'model_name': 'cens'}

    # ------------------------------ 前端输入 -------------------------------------------------------
    if request.POST:
        ctx['link_std_thres'] = float(request.POST['link_std_thres'])
        ctx['link_val_ratio_thres'] = float(request.POST['link_val_ratio_thres'])
        ctx['label_id'] = request.POST['label_id']
    else:
        ctx['link_std_thres'] = 0.6
        ctx['link_val_ratio_thres'] = 0.1
        ctx['label_id'] = 'all'

    ctx = ctx_prepare_global(ctx, data, explainer, model, layers, n_label, hiddens, color_gen)

    return render(request, 'network_global.html', ctx)


def local_init_page(request):
    ctx = {'model_name': 'kdc'}

    # ------------------------------ 前端输入 -------------------------------------------------------
    if request.POST:
        ctx['eid'] = int(request.POST['eid_input'])
        ctx['link_thres'] = float(request.POST['link_thres'])
        ctx['label_id'] = request.POST['label_id']
    else:
        ctx['eid'] = 0
        ctx['link_thres'] = 0.6
        ctx['label_id'] = 'all'

    ctx_prepare_local(ctx, data, explainer, model, layers, n_label, hiddens, color_gen)
    return render(request, 'network.html', ctx)
