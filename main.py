import argparse
from ResNet18 import ResNet18
import torch
from torch.utils.data import DataLoader
import Calibration.Calibration as Calibration
from Model import Model
from Dataset import MyDataset, transform, dataset_division, val_set_division, test_set_division

parser = argparse.ArgumentParser()
parser.add_argument('--N', type=int, help='number of calibration parameters')
parser.add_argument('--M', type=int, help='number of calibration samples')
args = parser.parse_args()

root_directory = '/home/ms21/lzhang665'
batch_size = 64
First_Train = True
LR = 0.01
Candidates = 15

N = args.N
M = args.M
Normalized = True


def train(first_train, network, lr, normalized, epochs, num_cali_para, num_cali_samp, test_set_index):
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

    train_set, val_set, test_set = dataset_division(dataset_root, dataset_file, test_set_index)
    train_dataset = MyDataset(train_set, transform=transform)

    val_cali_set, val_val_set, val_test_set = val_set_division(val_set, num_cali_samp)
    test_cali_set, test_test_set = test_set_division(test_set, num_cali_samp)
    val = {
        'val_cali_loader': DataLoader(dataset=MyDataset(val_cali_set, transform=transform), batch_size=batch_size, pin_memory=True),
        'val_val_loader': DataLoader(dataset=MyDataset(val_val_set, transform=transform), batch_size=batch_size, pin_memory=True),
        'val_test_loader': DataLoader(dataset=MyDataset(val_test_set, transform=transform), batch_size=batch_size, pin_memory=True),
    }
    test = {
        'test_cali_loader': DataLoader(dataset=MyDataset(test_cali_set, transform=transform), batch_size=batch_size, pin_memory=True),
        'test_test_loader': DataLoader(dataset=MyDataset(test_test_set, transform=transform), batch_size=batch_size, pin_memory=True),
    }

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=4)

    model.train(epochs, train_dataloader, val, test)
    Calibration.save(calibration_parameters, parameters_file)


if __name__ == '__main__':
    for TestSet_Index in range(15):
        train(First_Train, ResNet18, LR, Normalized, 80, N, M, TestSet_Index)


