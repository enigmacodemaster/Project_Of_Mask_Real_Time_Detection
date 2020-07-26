import numpy as np

def NMS(lists, thre):
    x1 = lists[:, 0]
    y1 = lists[:, 1]
    x2 = lists[:, 2]
    y2 = lists[:, 3]

    scores = lists[:, 4]

    # 每一个检测框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    res = []

    while order.size > 0:
        i = order[0]
        res.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        width = np.maximum(0.0, xx2 - xx1 + 1)
        height = np.maximum(0.0, yy2 - yy1 + 1)

        interaction = width * height

        over = interaction / (areas[i] + areas[order[1:]] - interaction)

        # 保留IoU小于阈值的box
        inds = np.where(over <= thre)[0]
        order = order[inds + 1]

    return res
