from path import explain_sample
import os
from Explain.Explainer import get_data_and_explainer
from CONFIG import HOME_PATH
from voting_function_draw import voting_function


def solve(data_name, model_name, sample_id, threshold, label_id, name):
    save_path = "%s/Explain/out/%s/%s/casestudy/%s" % (HOME_PATH, data_name, model_name, name) # sample_id, threshold, label_id)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    data, explainer = get_data_and_explainer(data_name, model_name)

    links_color, links_show, nodes_show = explain_sample(data, sample_id, explainer, threshold, label_id=label_id)

    # draw function
    for d, hid_from, hid_to, info in links_show:
        pin_y = info[1]
        for node in nodes_show:
            if node[0] == d and node[1] == hid_from:
                pin_x = node[2][0]
        voting_function(explainer, data, d, hid_from, hid_to, explainer.model.voting_mask_one, save_path, pin_point=[[pin_x, pin_y]])


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    # solve("COVTYPE", "VOTERS", 99493, 0.3, 2, name='S1COV99493L2')
    # solve("KDC", "VOTERS", 6297, 0.6, 1, name="M1KDC2097L1")
    # solve("IJCAI18X", "VOTERS", 49422, 0.4, 0, name="S2IJCAI349422L0")
    # solve("KDC", "VOTERS", 6297, 1, 1, name="M1KDC2097L1")
    solve("KDC", "VOTERS", 6297, 0.5, 1, name="M1KDC2097L1")
