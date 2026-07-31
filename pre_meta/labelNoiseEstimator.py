import numpy as np



import torch

from torch import nn

import torch.nn.functional as F

import torch.optim as optim





from dataSplit_LN import get_cifar10, get_default_data_transforms, CustomImageDataset, get_data_loaders



# from model import ToyCifarNet



# GPU settings

torch.backends.cudnn.benchmark = True

use_cuda = torch.cuda.is_available()

device = torch.device("cuda" if use_cuda else "cpu")



class AverageMeter(object):

    """ Computes and stores the average and current value """

    def __init__(self):

        self.reset()



    def reset(self):

        self.val = 0

        self.avg = 0

        self.sum = 0

        self.count = 0



    def update(self, val, n = 1):

        self.val = val

        self.sum += val * n

        self.count += n

        self.avg = self.sum / self.count



def get_auxiliary_data(testset_extract=True, data_size=32):

	''' return the numpy array of auxiliary data, x: (320, 3, 32, 32), y: (320, ) (class labels from 0 to 9 sorted)'''

	if testset_extract:

		_, _, x, y = get_cifar10()

	else:

		x, y, _, _ = get_cifar10()



	x_aux, y_aux = [], []



	for i in range(10):

		for (sample, label) in zip(x, y):

			if (label == i) and (len(x_aux) + 1 <= data_size * (i + 1)):

				x_aux.append(sample)

				y_aux.append(label)



	x_aux = np.array(x_aux)

	y_aux = np.array(y_aux)



	return x_aux, y_aux



def get_auxiliary_data_loader(testset_extract=True, data_size=32):

	x_aux, y_aux = get_auxiliary_data(testset_extract, data_size)



	# No matter you choose trainset or testset to generate auxiliary data

	# use the transforms on testdata

	_, transforms = get_default_data_transforms(verbose=False)



	# batch_size = data_size, don't shuffle

	auxiliary_data_loader = torch.utils.data.DataLoader(

		CustomImageDataset(x_aux, y_aux, transforms),

		batch_size=data_size, shuffle=False

		)



	return auxiliary_data_loader





def accuracy(output, target, topk=(1,)):

	maxk = max(topk)

	minibatch_size = len(target)

	_, pred = output.topk(maxk, 1, True, True)

	pred = pred.t()

	correct = pred.eq(target.view(1, -1).expand_as(pred))



	res = []

	for k in topk:

		correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)

		res.append(correct_k.mul_(100.0 / minibatch_size))

	return res





def validate(model, eval_loader):

	class_criterion = nn.CrossEntropyLoss().cuda()

	top1 = AverageMeter()



	# switch to evaluate mode

	model.eval()



	for i, (input, target) in enumerate(eval_loader):

		with torch.no_grad():

			input_var, target_var = input.cuda(), target.cuda()



		minibatch_size = len(target_var)



		# compute output

		output1 = model(input_var)

		softmax1 = F.softmax(output1, dim=1)

		class_loss = class_criterion(output1, target_var)



		# measure accuracy and record loss

		prec1 = accuracy(output1.data, target_var.data)



		top1.update(prec1[0], minibatch_size)



	return top1.avg





def estimate_label_noise_ratio(client_model_update, aux_loader, num_classes=10):

	test_acc = validate(client_model_update, aux_loader)
	test_acc = test_acc.item() / 100.

	# print(test_acc)



	test_acc = max(1 / num_classes, test_acc)

	test_acc = min(test_acc, 1.)



	a = num_classes / (num_classes - 1)

	b = -2

	c = 1 - test_acc



	delta = b ** 2 - 4 * a * c

	eps1 = (-b + np.sqrt(delta)) / (2 * a)

	eps2 = (-b - np.sqrt(delta)) / (2 * a)



	eps = min(eps1, eps2)

	# print("eps: ", eps)



	return eps





def compute_ratio_per_client_update(client_models, client_idx, aux_loader, num_classes=10):

	ra_dict = {}

	for i, client_model_update in enumerate(client_models):

		ra_all = estimate_label_noise_ratio(client_model_update, aux_loader)

		# [TODO] whether I need to minus ra from prev_global_model

		# may add code here

		ra_dict[client_idx[i]] = 1 - ra_all



	return ra_dict



# def main():

# 	client_train_loader, test_loader, data_size_per_client = get_data_loaders(100, 128, True)

# 	aux_loader = get_auxiliary_data_loader(testset_extract=True, data_size=32)

# 	client_models = [ToyCifarNet(init_weights=False).to(device) for _ in range(20)]

# 	ra_dict = compute_ratio_per_client_update(client_models, np.random.permutation(100)[:20], aux_loader)



# 	print('test')



# if __name__ == '__main__':

#	main()



