from ResNet18 import ResNet18
import torch
import Calibration.Calibration as Calibration
from Model import Model


root_directory = 'C:'
batch_size = 64
First_Train = False
LR = 0.01
Candidates = 15

N = 10
M = 10
Normalized = True


def validate(first_train, network, lr, normalized, epochs, num_cali_para, num_cali_samp, test_set_index):
    model_file = root_directory + '/PyCharmProjects/GazeTracking/Network/ResNet18_{}_{}para_{}samp_fold{}.pth.zip'.format("normalized" if normalized else "unnormalized", num_cali_para, num_cali_samp, test_set_index)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Net = network(num_cali_para)
    optimizer_Net = torch.optim.Adam(Net.parameters(), lr=lr)
    if not first_train:
        # calibration_parameters = Calibration.load(parameters_file)
        calibration_parameters = Calibration.create(Candidates, num_cali_para)
    else:
        calibration_parameters = Calibration.create(Candidates, num_cali_para)
    optimizer_Calibration = torch.optim.SGD([calibration_parameters], lr=lr*10)
    model = Model(device, Net, optimizer_Net, optimizer_Calibration, calibration_parameters, num_cali_para, first_train, model_file)
    
    string = model_file.split('/')[-1]+'  epochs:'+str(model.epoch)

    print(string)
    print('train angle error: ')
    print(model.train_angle_error)
    print('validation: ')
    print(str(model.validation))
    print('test angle error: ')
    print(str(model.test_angle_error))
    i = len(model.validation['test_angle_error'])
    mean = (model.validation['test_angle_error'][i-6]+model.validation['test_angle_error'][i-5]+model.validation['test_angle_error'][i-4]+model.validation['test_angle_error'][i-3]+model.validation['test_angle_error'][i-2]) / 5
    print(mean, format(model.validation['test_angle_error'][i-1], '.2f'), format(mean, '.2f'))
    print(float(format(model.validation['test_angle_error'][i-1], '.2f')) >= float(format(mean, '.2f')))


if __name__ == '__main__':
    for TestSet_Index in range(1):
        validate(First_Train, ResNet18, LR, Normalized, 80, N, M, TestSet_Index)
