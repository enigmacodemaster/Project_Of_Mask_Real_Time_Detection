import numpy as np
# import matplotlib.pyplot as plt
import cv2

def anchor2hw(anchor):
	 # anchor表示方式为 [left,top,right,bottom]
	w = anchor[2] - anchor[0] + 1
	h = anchor[3] - anchor[1] + 1
	cx = anchor[0] + 0.5 * (w - 1)
	cy = anchor[1] + 0.5 * (h - 1)
	return w, h, cx, cy

def makeAnchors(ws, hs, cx, cy):
	# 将计算好的w，h，中心x和中心y转化回坐标系表示，用以画图
	ws = ws[:, np.newaxis]
	hs = hs[:, np.newaxis]
	# 在水平方向上堆叠，有多少个长宽，生成多少个anchors
	anchors = np.hstack((cx - 0.5*(ws - 1),
						 cy - 0.5*(hs - 1),
						 cx + 0.5*(ws - 1),
						 cy + 0.5*(hs - 1)))
	return anchors


def generateAnchors_ratios(anchor, ratios):
	w, h, cx, cy = anchor2hw(anchor)
	size = w * h
	size_ratios = size / ratios
	ws = np.round(np.sqrt(size_ratios))
	hs = np.round(ws * ratios)
	anchors = makeAnchors(ws, hs, cx, cy)
	return anchors

def generateAnchors_scales(anchor, scales):
	w, h, cx, cy = anchor2hw(anchor)
	ws = w * scales
	hs = h * scales
	anchors = makeAnchors(ws, hs, cx, cy)
	return anchors


def generateAnchors(base_size = 16, ratios = [0.5, 1, 2], scales=np.array([8,16,32])):
	base_anchor = np.array([0, 0, base_size - 1, base_size - 1]) + 600
	ratio_anchors = generateAnchors_ratios(base_anchor, ratios)
	anchors = np.vstack([generateAnchors_scales(ratio_anchors[i,:], scales)
						for i in range(ratio_anchors.shape[0])])

	return anchors


if __name__ == '__main__':
	img = cv2.imread('sample_cp1.jpg')
	colors = [(0,0,255), (0,255,0), (255, 0, 0)]
	anchors = generateAnchors()
	for i, rec in enumerate(anchors):
		color = colors[int(i // 3)]
		cv2.rectangle(img, (int(rec[0]),int(rec[1])), (int(rec[2]),int(rec[3])), color, 1)

cv2.imwrite('homework_week3.jpg', img)