import cv2
import os
import torch
import agent
import numpy as np
from copy import deepcopy
from parameters import Parameters
import util


p = Parameters()

def Demoing():
    print('Get agent')
    # lane_agent = agent.Agent()
    # lane_agent.load_state_dict(
    #     torch.load('savefile/804_tensor(0.5786)_lane_detection_network.pkl', map_location='cpu'), False)
    lane_agent = agent.Agent()
    lane_agent.load_weights(804, "tensor(0.5786)")
    if torch.cuda.is_available():
        lane_agent.cuda()
    lane_agent.evaluate_mode()

    # check model with a picture
    # test_image = cv2.imread("test_curves/000c698734a78dce648bdb0d26f24b4f.jpg")
    # test_image = cv2.resize(test_image, (512,256))/255.0
    # test_image = np.rollaxis(test_image, axis=2, start=0)
    # _, _, ti = test(lane_agent, np.array([test_image]))
    # cv2.imshow("test", ti[0])
    # cv2.waitKey(0)
    imgs_dir = "../dataset/powerline/imgs"
    imgs_save_dir = imgs_dir + "_rsts"
    os.makedirs(imgs_save_dir, exist_ok=True)
    for cur_f in os.listdir(imgs_dir):
        cur_img = os.path.join(imgs_dir, cur_f)
        cur_img_dst = os.path.join(imgs_save_dir, cur_f)
        test_image = cv2.imread(cur_img)
        test_image = cv2.resize(test_image, (512,256))/255.0
        test_image = np.rollaxis(test_image, axis=2, start=0)
        _, _, ti = test(lane_agent, np.array([test_image]))
        cv2.imwrite(cur_img_dst, ti[0])


############################################################################
## test on the input test image
############################################################################
def test(lane_agent, test_images, thresh=p.threshold_point, index=-1):
    result = lane_agent.predict_lanes_test(test_images)
    confidences, offsets, instances = result[index]

    num_batch = len(test_images)

    out_x = []
    out_y = []
    out_images = []

    for i in range(num_batch):
        # test on test data set
        image = deepcopy(test_images[i])
        image = np.rollaxis(image, axis=2, start=0)
        image = np.rollaxis(image, axis=2, start=0) * 255.0
        image = image.astype(np.uint8).copy()

        confidence = confidences[i].view(p.grid_y, p.grid_x).cpu().data.numpy()

        offset = offsets[i].cpu().data.numpy()
        offset = np.rollaxis(offset, axis=2, start=0)
        offset = np.rollaxis(offset, axis=2, start=0)

        instance = instances[i].cpu().data.numpy()
        instance = np.rollaxis(instance, axis=2, start=0)
        instance = np.rollaxis(instance, axis=2, start=0)

        # generate point and cluster
        raw_x, raw_y = generate_result(confidence, offset, instance, thresh)

        # eliminate fewer points
        in_x, in_y = eliminate_fewer_points(raw_x, raw_y)

        # sort points along y
        in_x, in_y = util.sort_along_y(in_x, in_y)

        result_image = util.draw_points(in_x, in_y, deepcopy(image))

        out_x.append(in_x)
        out_y.append(in_y)
        out_images.append(result_image)

    return out_x, out_y, out_images



############################################################################
## eliminate result that has fewer points than threshold
############################################################################
def eliminate_fewer_points(x, y):
    # eliminate fewer points
    out_x = []
    out_y = []
    for i, j in zip(x, y):
        if len(i)>2:
            out_x.append(i)
            out_y.append(j)
    return out_x, out_y


############################################################################
## generate raw output
############################################################################
def generate_result(confidance, offsets, instance, thresh):
    mask = confidance > thresh

    grid = p.grid_location[mask]
    offset = offsets[mask]
    feature = instance[mask]

    lane_feature = []
    x = []
    y = []
    for i in range(len(grid)):
        if (np.sum(feature[i] ** 2)) >= 0:
            point_x = int((offset[i][0] + grid[i][0]) * p.resize_ratio)
            point_y = int((offset[i][1] + grid[i][1]) * p.resize_ratio)
            if point_x > p.x_size or point_x < 0 or point_y > p.y_size or point_y < 0:
                continue
            if len(lane_feature) == 0:
                lane_feature.append(feature[i])
                x.append([point_x])
                y.append([point_y])
            else:
                flag = 0
                index = 0
                min_feature_index = -1
                min_feature_dis = 10000
                for feature_idx, j in enumerate(lane_feature):
                    dis = np.linalg.norm((feature[i] - j) ** 2)
                    if min_feature_dis > dis:
                        min_feature_dis = dis
                        min_feature_index = feature_idx
                if min_feature_dis <= p.threshold_instance:
                    lane_feature[min_feature_index] = (lane_feature[min_feature_index] * len(x[min_feature_index]) +
                                                       feature[i]) / (len(x[min_feature_index]) + 1)
                    x[min_feature_index].append(point_x)
                    y[min_feature_index].append(point_y)
                elif len(lane_feature) < 12:
                    lane_feature.append(feature[i])
                    x.append([point_x])
                    y.append([point_y])
    return x, y


if __name__ == '__main__':
    Demoing()
