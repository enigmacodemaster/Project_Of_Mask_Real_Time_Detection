def iou(boxA, boxB):
	'''
	box的表示方法：左上角坐标x1,y1 + 右下角坐标x2,y2
	- boxA = [x1,y1,x2,y2]
	- boxB = [x1',y1',x2',y2']
	'''
	left_max = max(boxA[0], boxB[0])
	top_max = max(boxA[1], boxB[1])
	right_min = min(boxA[2], boxB[2])
	bottom_min = min(boxA[3], boxB[3])

	# 计算重合部分的面积
	interaction = max(0, (right_min - left_max)*(bottom_min - top_max))
	# 分别计算两个box的面积
	sa = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
	sb = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
	# 计算总面积
	union = sa + sb - interaction
	# iou
	iou = interaction / union
	return iou