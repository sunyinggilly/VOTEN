from path_global import explain_global
import os
from Explain.Explainer import get_data_and_explainer
from CONFIG import HOME_PATH
from voting_function_draw import voting_function


def solve(data_name, model_name, threshold, threshold_ratio, label_id, name):
    save_path = "%s/Explain/out/%s/%s/casestudy/%s" % (HOME_PATH, data_name, model_name, name) 
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    data, explainer = get_data_and_explainer(data_name, model_name)
    links_color, links_show, nodes_show = explain_global(data, explainer, threshold, threshold_ratio=threshold_ratio, label_id=label_id)

    # draw function
    for d, hid_from, hid_to, info in links_show:
        voting_function(explainer, data, d, hid_from, hid_to, explainer.model.voting_mask_one, save_path, avg_draw=True)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    solve("KDC", "VOTERS", 1, threshold_ratio=0.01, label_id=1, name="M1KDCL1-global")
