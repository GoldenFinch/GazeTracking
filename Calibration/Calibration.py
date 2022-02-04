import torch
from torch.autograd import Variable


def create(M, N, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    return Variable(torch.randn(M, N), requires_grad=True)


def load(calibration_file):
    calibration = torch.load(calibration_file)

    return calibration['calibration_parameters']


def save(calibration_parameters, calibration_file):
    calibration = {
        'calibration_parameters': calibration_parameters
    }
    torch.save(calibration, calibration_file)


def index2parameters(calibration_parameters, index):
    stack = calibration_parameters[index[0]]
    stack = stack.reshape(1, -1)
    for i in range(1, index.shape[0]):
        parameter = calibration_parameters[index[i]]
        parameter = parameter.reshape(1, -1)
        stack = torch.cat((stack, parameter), 0)

    return stack


def stack_parameters(calibration_parameters, num):
    stack = calibration_parameters[0]
    stack = stack.reshape(1, -1)
    for i in range(num-1):
        parameter = calibration_parameters[0]
        parameter = parameter.reshape(1, -1)
        stack = torch.cat((stack, parameter), 0)

    return stack
