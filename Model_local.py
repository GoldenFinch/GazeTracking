import torch
from torch.autograd import Variable
import torch.nn.functional as F
import os
import math
import util
import Calibration as Calibration


class Model:
    def __init__(self, device, Net, optimizer_Net, optimizer_Calibration, calibration_parameters, num_cali_para, first_train=True, model_path=None):
        self.device = device
        self.Net = Net.to(device)
        self.optimizer_Net = optimizer_Net
        self.optimizer_Calibration = optimizer_Calibration
        self.calibration_parameters = calibration_parameters
        self.num_cali_para = num_cali_para
        self.model_path = model_path

        self.epoch = 0
        self.train_angle_error = []
        self.validation = {
            'best_epoch_cali': [],
            'test_angle_error': []
        }
        self.test_angle_error = 0

        if not first_train:
            self.load(model_path)

    def load(self, model_path):
        if os.path.exists(model_path):
            model = torch.load(model_path, map_location=self.device)
            self.Net.load_state_dict(model['Net'])
            self.optimizer_Net.load_state_dict(model['optimizer_Net'])
            self.optimizer_Calibration.load_state_dict(model['optimizer_Calibration'])

            self.epoch = model['epoch']
            self.train_angle_error = model['train_angle_error']
            self.validation = model['validation']
            self.test_angle_error = model['test_angle_error']
        else:
            raise FileNotFoundError('Model file not found!')

    def save(self):
        model = {
            'Net': self.Net.state_dict(),
            'optimizer_Net': self.optimizer_Net.state_dict(),
            'optimizer_Calibration': self.optimizer_Calibration.state_dict(),
            'epoch': self.epoch,
            'train_angle_error': self.train_angle_error,
            'validation': self.validation,
            'test_angle_error': self.test_angle_error
        }
        torch.save(model, self.model_path)
        print("Model saved!")

    def train(self, epochs, train_loader, val, test):
        for i in range(self.epoch + 1, self.epoch + 1 + epochs):
            self.Net.train()
            total_angle_error = 0
            for batch_idx, (data, target, index) in enumerate(train_loader):
                data, target, parameters = Variable(data).to(self.device), Variable(target).to(self.device), Calibration.index2parameters(self.calibration_parameters, index).to(self.device)
                AL = self.Net.forward(data, parameters)
                loss = F.mse_loss(AL, target)
                angle_error = util.angle_calculation_avg(AL.cpu().detach().numpy(), target.cpu().detach().numpy())
                total_angle_error += angle_error
                self.optimizer_Net.zero_grad()
                self.optimizer_Calibration.zero_grad()
                loss.backward()
                self.optimizer_Net.step()
                self.optimizer_Calibration.step()
                if batch_idx % 50 == 0:
                    print('Batch: {}  Loss: {}  Angle Error: {}'.format(batch_idx+1, loss, angle_error))
            for params in self.optimizer_Net.param_groups:
                params['lr'] = 0.01 * math.pow(0.1, math.floor((self.epoch + 1) / 35))
            for params in self.optimizer_Calibration.param_groups:
                params['lr'] = 0.1 * math.pow(0.1, math.floor((self.epoch + 1) / 35))
            total_angle_error /= (batch_idx + 1)
            self.train_angle_error.append(total_angle_error)
            print('Epoch: {}  Average Angle Error: {}'.format(i, total_angle_error))
            validation = self.validate(val['val_cali_loader'], val['val_val_loader'], val['val_test_loader'])
            self.validation['best_epoch_cali'].append(validation['best_epoch_cali'])
            self.validation['test_angle_error'].append(validation['test_angle_error'])
            if i > 5:
                mean_last_5 = (self.validation['test_angle_error'][i - 2] + self.validation['test_angle_error'][i - 3] + self.validation['test_angle_error'][i - 4] +
                               self.validation['test_angle_error'][i - 5] + self.validation['test_angle_error'][i - 6]) / 5
                if float(format(self.validation['test_angle_error'][i-1], '.2f')) >= float(format(mean_last_5, '.2f')):
                    break
        best_epoch = self.validation['test_angle_error'].index(min(self.validation['test_angle_error']))
        best_epoch_cali = self.validation['best_epoch_cali'][best_epoch]
        test_parameters = self.calibrate(test['test_cali_loader'], best_epoch_cali)
        test_angle_error = self.test(test['test_test_loader'], test_parameters)
        self.test_angle_error = test_angle_error
        self.epoch += i
        self.save()

    def validate(self, val_cali_loader, val_val_loader, val_test_loader):
        self.Net.eval()
        calibration_parameter = Calibration.create(1, self.num_cali_para, self.num_cali_para)
        optimizer_Calibration = torch.optim.SGD([calibration_parameter], lr=0.1)
        scale = 50
        epoch = 0
        val_angle_error = []
        calibration_parameters = []
        for batch_idx, (data, target, index) in enumerate(val_cali_loader):
            data, target, parameters = Variable(data).to(self.device), Variable(target).to(self.device), Calibration.stack_parameters(calibration_parameter, len(index)).to(self.device)
            data, target, parameters = torch.cat((data, data), 0), torch.cat((target, target), 0), torch.cat((parameters, parameters), 0)
        while True:
            AL = self.Net.forward(data, parameters)
            loss = F.mse_loss(AL, target)
            optimizer_Calibration.zero_grad()
            loss.backward()
            optimizer_Calibration.step()
            epoch += 1
            if epoch % scale == 0:
                epoch_scale = math.floor(epoch / scale)
                angle_error = self.test(val_val_loader, calibration_parameter)
                val_angle_error.append(angle_error)
                calibration_parameters.append(calibration_parameter)
                if epoch_scale % 5 == 0:
                    print('Cali Epoch: {}  Angle Error: {}'.format(epoch_scale+1, angle_error))
                if epoch_scale > 5:
                    mean_last_5 = (val_angle_error[epoch_scale-2]+val_angle_error[epoch_scale-3]+val_angle_error[epoch_scale-4]+val_angle_error[epoch_scale-5]+val_angle_error[epoch_scale-6]) / 5
                    if float(format(val_angle_error[epoch_scale-1], '.2f')) >= float(format(mean_last_5, '.2f')):
                        break
        best_epoch_cali = val_angle_error.index(min(val_angle_error))
        test_angle_error = self.test(val_test_loader, calibration_parameters[best_epoch_cali].reshape(1, -1))
        print('Bes Cali Epoch: {}  Test Angle Error: {}'.format(best_epoch_cali, test_angle_error))
        return {
            'best_epoch_cali': (best_epoch_cali+1) * scale,
            'test_angle_error': test_angle_error
        }

    def calibrate(self, cali_loader, epochs):
        """
        return the calibration parameters of the data from one candidate.
        :param cali_loader: container of the data
        :param epochs: calibration epochs
        :return: calibration parameters after training
        """
        self.Net.eval()
        calibration_parameter = Calibration.create(1, self.num_cali_para, self.num_cali_para)
        optimizer_Calibration = torch.optim.SGD([calibration_parameter], lr=0.1)
        for batch_idx, (data, target, index) in enumerate(cali_loader):
            data, target, parameters = Variable(data).to(self.device), Variable(target).to(self.device), Calibration.stack_parameters(calibration_parameter, len(index)).to(self.device)
            data, target, parameters = torch.cat((data, data), 0), torch.cat((target, target), 0), torch.cat((parameters, parameters), 0)
        for i in range(epochs):
            AL = self.Net.forward(data, parameters)
            loss = F.mse_loss(AL, target)
            optimizer_Calibration.zero_grad()
            loss.backward()
            optimizer_Calibration.step()

        return calibration_parameter

    def test(self, loader, calibration_parameters):
        """
        return the mean angle error of the data from one candidate.
        :param loader: container of the data
        :param calibration_parameters: the calibration parameters for this candidate
        :return: mean angle error
        """
        self.Net.eval()
        total_angle_error = 0
        for batch_idx, (data, target, index) in enumerate(loader):
            data, target, parameters = Variable(data).to(self.device), Variable(target).to(self.device), Calibration.stack_parameters(calibration_parameters, len(index)).to(self.device)
            AL = self.Net.forward(data, parameters)
            angle_error = util.angle_calculation_avg(AL.cpu().detach().numpy(), target.cpu().detach().numpy())
            total_angle_error += angle_error

        return total_angle_error / (batch_idx + 1)

