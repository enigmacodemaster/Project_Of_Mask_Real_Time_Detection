import numpy as np
import torch

def softmax_np(logits):
	assert (isinstance(logits, np.ndarray)), 'only numpy is available'
	C = np.exp(-np.max(logits)) # 防止溢出的操作系数
	exp_value = np.exp(logits) # 
	dim_ext = np.sum(exp_value, 1).reshape(-1, 1)
	# log_res = torch.log(exp_value) - torch.log(dim_ext)
	res = C * exp_value / ( C * dim_ext )

	return res

def softmax_torch(logits):
	assert (isinstance(logits, torch.Tensor)), 'only torch is available'
	C = np.exp(-torch.max(logits)) # 防止溢出的操作系数
	exp_value = torch.exp(logits)
	dim_ext = torch.sum(exp_value, 1).reshape(-1, 1)
	# print(dim_ext) 一列
	res = (C * exp_value) / ( C * dim_ext )

	return res

logits_np = np.random.rand(3,5)
s_np = softmax_np(logits_np)

logits_torch = torch.from_numpy(logits_np)
s_torch = softmax_torch(logits_torch)
print(s_np)

print(s_torch)
