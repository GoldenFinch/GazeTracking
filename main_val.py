import argparse
from ResNet18 import ResNet18
import torch
import Calibration.Calibration as Calibration
from Model import Model

parser = argparse.ArgumentParser()
parser.add_argument('--N', type=int, help='number of calibration parameters')
parser.add_argument('--M', type=int, help='number of calibration samples')
args = parser.parse_args()

root_directory = '/home/ms21/lzhang665'
batch_size = 64
First_Train = False
LR = 0.01
Candidates = 15

N = args.N
M = args.M
Normalized = True


def validate(first_train, network, lr, normalized, epochs, num_cali_para, num_cali_samp, test_set_index):
    dataset_root = root_directory + '/MPIIFaceGaze_{}'.format("normalized" if normalized else "unnormalized")
    dataset_file = root_directory + '/PyCharmProjects/GazeTracking/Dataset/Dataset_{}.data'.format("normalized" if normalized else "unnormalized")
    parameters_file = root_directory + '/PyCharmProjects/GazeTracking/Calibration/MPIIFazeGaze_{}_{}para_{}samp_fold{}.para'.format("normalized" if normalized else "unnormalized", num_cali_para, num_cali_samp, test_set_index)
    model_file = root_directory + '/PyCharmProjects/GazeTracking/Network/ResNet18_{}_{}para_{}samp_fold{}.pth'.format("normalized" if normalized else "unnormalized", num_cali_para, num_cali_samp, test_set_index)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Net = network(num_cali_para)
    optimizer_Net = torch.optim.Adam(Net.parameters(), lr=lr)
    if not first_train:
        calibration_parameters = Calibration.load(parameters_file)
    else:
        calibration_parameters = Calibration.create(Candidates, num_cali_para)
    optimizer_Calibration = torch.optim.SGD([calibration_parameters], lr=lr*10)
    model = Model(device, Net, optimizer_Net, optimizer_Calibration, calibration_parameters, num_cali_para, first_train, model_file)
    
    string = model_file.split('/')[-1]+'  epochs:'+str(model.epoch)+'  angle error:'+str(model.test_angle_error)

    with open(root_directory+'/model.txt', 'a+') as f:
        f.write(string+'\n')


if __name__ == '__main__':
    for TestSet_Index in range(15):
        validate(First_Train, ResNet18, LR, Normalized, 80, N, M, TestSet_Index)


