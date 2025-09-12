import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms.functional as TF
import numpy as np
import os
import math
import random
import logging
import logging.handlers
from matplotlib import pyplot as plt
from scipy.ndimage import zoom
import SimpleITK as sitk
from medpy import metric
import cv2
from PIL import Image
from torch import Tensor
from thop import profile


def set_seed(seed):
    # for hash
    os.environ['PYTHONHASHSEED'] = str(seed)
    # for python and numpy
    random.seed(seed)
    np.random.seed(seed)
    # for cpu gpu
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # for cudnn
    cudnn.benchmark = False
    cudnn.deterministic = True


def get_logger(name, log_dir):
    '''
    Args:
        name(str): name of logger
        log_dir(str): path of log
    '''

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    info_name = os.path.join(log_dir, '{}.info.log'.format(name))
    info_handler = logging.handlers.TimedRotatingFileHandler(info_name,
                                                             when='D',
                                                             encoding='utf-8')
    info_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    info_handler.setFormatter(formatter)

    logger.addHandler(info_handler)

    return logger


def log_config_info(config, logger):
    config_dict = config.__dict__
    log_info = f'#----------Config info----------#'
    logger.info(log_info)
    for k, v in config_dict.items():
        if k[0] == '_':
            continue
        else:
            log_info = f'{k}: {v},'
            logger.info(log_info)



def get_optimizer(config, model):
    assert config.opt in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop', 'SGD'], 'Unsupported optimizer!'

    if config.opt == 'Adadelta':
        return torch.optim.Adadelta(
            model.parameters(),
            lr = config.lr,
            rho = config.rho,
            eps = config.eps,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'Adagrad':
        return torch.optim.Adagrad(
            model.parameters(),
            lr = config.lr,
            lr_decay = config.lr_decay,
            eps = config.eps,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'Adam':
        return torch.optim.Adam(
            model.parameters(),
            lr = config.lr,
            betas = config.betas,
            eps = config.eps,
            weight_decay = config.weight_decay,
            amsgrad = config.amsgrad
        )
    elif config.opt == 'AdamW':
        return torch.optim.AdamW(
            model.parameters(),
            lr = config.lr,
            betas = config.betas,
            eps = config.eps,
            weight_decay = config.weight_decay,
            amsgrad = config.amsgrad
        )
    elif config.opt == 'Adamax':
        return torch.optim.Adamax(
            model.parameters(),
            lr = config.lr,
            betas = config.betas,
            eps = config.eps,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'ASGD':
        return torch.optim.ASGD(
            model.parameters(),
            lr = config.lr,
            lambd = config.lambd,
            alpha  = config.alpha,
            t0 = config.t0,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'RMSprop':
        return torch.optim.RMSprop(
            model.parameters(),
            lr = config.lr,
            momentum = config.momentum,
            alpha = config.alpha,
            eps = config.eps,
            centered = config.centered,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'Rprop':
        return torch.optim.Rprop(
            model.parameters(),
            lr = config.lr,
            etas = config.etas,
            step_sizes = config.step_sizes,
        )
    elif config.opt == 'SGD':
        return torch.optim.SGD(
            model.parameters(),
            lr = config.lr,
            momentum = config.momentum,
            weight_decay = config.weight_decay,
            dampening = config.dampening,
            nesterov = config.nesterov
        )
    else: # default opt is SGD
        return torch.optim.SGD(
            model.parameters(),
            lr = 0.01,
            momentum = 0.9,
            weight_decay = 0.05,
        )


def get_scheduler(config, optimizer):
    assert config.sch in ['StepLR', 'MultiStepLR', 'ExponentialLR', 'CosineAnnealingLR', 'ReduceLROnPlateau',
                        'CosineAnnealingWarmRestarts', 'WP_MultiStepLR', 'WP_CosineLR'], 'Unsupported scheduler!'
    if config.sch == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size = config.step_size,
            gamma = config.gamma,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones = config.milestones,
            gamma = config.gamma,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma = config.gamma,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max = config.T_max,
            eta_min = config.eta_min,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode = config.mode,
            factor = config.factor,
            patience = config.patience,
            threshold = config.threshold,
            threshold_mode = config.threshold_mode,
            cooldown = config.cooldown,
            min_lr = config.min_lr,
            eps = config.eps
        )
    elif config.sch == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0 = config.T_0,
            T_mult = config.T_mult,
            eta_min = config.eta_min,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'WP_MultiStepLR':
        lr_func = lambda epoch: epoch / config.warm_up_epochs if epoch <= config.warm_up_epochs else config.gamma**len(
                [m for m in config.milestones if m <= epoch])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
    elif config.sch == 'WP_CosineLR':
        lr_func = lambda epoch: epoch / config.warm_up_epochs if epoch <= config.warm_up_epochs else 0.5 * (
                math.cos((epoch - config.warm_up_epochs) / (config.epochs - config.warm_up_epochs) * math.pi) + 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)

    return scheduler






def test_single_volume(image, label, net, classes, patch_size=[256, 256],
                    test_save_path=None, case=None, z_spacing=1, val_or_test=False):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None and val_or_test is True:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list


def save_model(model, save_path, optimizer=None, epoch=None):
    """
    Save the trained model and the optimiser state (if provided).

    :param model: trained PyTorch model
    :param save_path: Save path
    :param optimizer: optimiser (optional)
    :param epoch: current number of training rounds (optional)
    """

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if optimizer is not None and epoch is not None:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, save_path)
    else:
        torch.save(model.state_dict(), save_path)

    print(f"模型已保存到 {save_path}")


def load_model(model, load_path, optimizer=None):
    """
    Loads a saved model from the specified path.

    :param model: Initialised model
    :param load_path: file path of the model
    :param optimizer: Optimiser (optional)
    :return: loaded model and optimiser (if provided)
    """
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint.get('epoch', None)
    print(f"Load model from {load_path}, restore to epoch {epoch}")
    return model, optimizer, epoch


# 用于计算和显示模型的性能
def calculate_dice_coefficient(y_true, y_pred):
    """
    Calculates the Dice coefficient, which is used to evaluate the performance of the model.

    :param y_true: true labels (tensor)
    :param y_pred: model prediction result (tensor)
    :return: Dice coefficient
    """
    smooth = 1e-6
    intersection = torch.sum(y_true * y_pred)
    union = torch.sum(y_true) + torch.sum(y_pred)
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.item()



def plot_loss_curve(losses, save_path="loss_curve.png"):
    """
    Plots the loss curve during training and saves it as an image file.

    :param losses: list of loss values
    :param save_path: save path
    """
    plt.plot(losses)
    plt.title('Training Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"The loss curves have been saved to {save_path}")


# 用于保存预测结果和真实标签的可视化图像
def visualize_predictions(predictions, ground_truth, save_path="predictions", prefix="sample"):
    """
    Visualise the predictions and true labels of the model.

    :param predictions: Segmentation results predicted by the model
    :param ground_truth: truth label
    :param save_path: directory where predictions are saved
    :param prefix: Prefix for image saving
    """
    os.makedirs(save_path, exist_ok=True)

    for i in range(len(predictions)):
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        ax[0].imshow(predictions[i, 0, :, :, predictions[i].shape[3] // 2], cmap='gray')
        ax[0].set_title(f"Predicted - {prefix}_{i}")

        ax[1].imshow(ground_truth[i, 0, :, :, ground_truth[i].shape[3] // 2], cmap='gray')
        ax[1].set_title(f"Ground Truth - {prefix}_{i}")

        plt.savefig(f"{save_path}/{prefix}_{i}.png")
        plt.close(fig)
    print(f"Predictions have been saved to {save_path}")

def evaluate_dice(model, data_loader, device):
    """
    Computes and outputs the average Dice coefficient on the test set.

    :param model: trained model
    :param data_loader: test set DataLoader
    :param device: device ('cuda' or 'cpu')
    :return: Average Dice coefficient
    """
    model.eval()
    dice_coeffs = []

    with torch.no_grad():
        for data in data_loader:
            inputs, labels = data['image'].to(device), data['label'].to(device)
            outputs = model(inputs)

            dice = calculate_dice_coefficient(labels, outputs)
            dice_coeffs.append(dice)

    average_dice = np.mean(dice_coeffs)
    print(f"Mean Dice: {average_dice:.4f}")
    return average_dice


def log_metrics(epoch, loss, dice_score, log_file="training_log.txt"):
    """
    Records metrics such as losses and Dice coefficients during training.

    :param epoch: current training rounds
    :param loss: current training loss
    :param dice_score: Dice coefficient of current training.
    :param log_file: path to log file
    """
    with open(log_file, "a") as f:
        f.write(f"Epoch {epoch}: Loss = {loss:.4f}, Dice Coefficient = {dice_score:.4f}\n")
    print(f"The metrics for round {epoch} have been logged to {log_file}")


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.bce_loss = nn.BCELoss()
    def forward(self, pred, target):
        """
        Calculate the Dice coefficient loss

        :param pred: model's prediction, shape (batch_size, C, H, W)
        :param target: true label, shape (batch_size, C, H, W)
        :return: Dice coefficient loss
        """
        smooth = 1e-6
        pred_ = pred.view(pred.size(0), -1)
        target_ = target.view(target.size(0), -1)

        intersection = pred_ * target_

        dice_score = (2 * intersection.sum(1) + smooth) / (pred_.sum(1) + target_.sum(1) + smooth)

        dice_loss = 1 - dice_score.mean()
        return dice_loss