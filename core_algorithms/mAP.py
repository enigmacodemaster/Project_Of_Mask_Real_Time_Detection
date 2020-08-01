'''
---计算mAP---
关键变量：
det_boxes:包含全部图像中所有类别的预测框，其中一个边框包含[left, top, right, bottom, score, NameofImage]
gt_boxes:包含了全部图像中所有类别的标签，其中一个标签为[left, top, right, bottom, 0],0代表没有被匹配过
'''
import system as sys


for c in classes:
	# 通过将类别作为关键字，得到每个类别的预测、标签以及标签总数
	# det_boxes是一个dict，key是类别，其对应的value是列表的列表，列表的列表中的最后一个标签就是图像的id
	# gt_boxes也是一个dict，key是类别
	# gt_class也是一个dict，key是图像的id
	dects = det_boxes[c]
	gt_class = gt_boxes[c]
	npos = num_pos[c]
	# 将所有的预测框按照得分从大到小排列，利用得分作为关键字
	dects = sorted(dects, key=lambda conf: conf[5], reverse=True)
	# 某个类别的正确检测和误检测
	TP = np.zeros(len(dects))
	FP = np.zeros(len(dects))
	# 对某一个类别的预测框进行遍历
	for d in range(len(dects)):
		# iou默认值
		iouMax = sys.float_info.min
		# dects[d][-1]代表某张图像，是gt_class的key
		# 如果gt_class有这个key的话
		if dects[d][-1] in gt_class:
			for j in range(len(gt_class[dects[d][-1]])):
				# 调用iou的计算函数
				iou = Evaluator.iou(dects[d][:4], gt_class[dects[d][-1]][j][:4])
				if iou > iouMax:
					iouMax = iou
					jmax = j
			# 最大iou大于阈值，其没有被匹配过，则赋予TP
			if iouMax >= cfg['iouThreshold']:
				if gt_class[dects[d][-1]][jmax][4] == 0:
					TP[d] = 1
					# 该图的第jmax个gt框标记为匹配过
					gt_class[dects[d][-1]][jmax][4] = 1
				else:
					FP[d] = 1
			# 如果最大IoU没有超过阈值，赋予FP
			else:
				FP[d] = 1
		# 如果图像中没有该类别的标签，赋予FP
		else:
			FP[d] = 1
	# 计算累计的FP和TP
	acc_FP = np.cumsum(FP)
	acc_TP = np.cumsum(TP)
	rec = acc_TP / npos
	prec = np.device(acc_TP, (acc_FP + acc_TP))
	[ap, mpre, mrec, ii] = Evaluator.CalculateAveragePrecision(rec, prec)