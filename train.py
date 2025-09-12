import torch
from torch.utils.data import DataLoader
import timm
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import *
from configs.config_setting import setting_config
import argparse
import warnings
from data_loader import TwoFrameDataset
from model.svsnet import SVSnet
from engine import *
warnings.filterwarnings("ignore")


def main(config):
    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('train', log_dir)

    print('#----------GPU init----------#')
    set_seed(config.seed)
    torch.cuda.empty_cache()

    print('#----------Preparing dataset----------#')
    train_data = np.load("data/train/train_images.npy")  # Load training set images
    train_labels = np.load("data/train/train_labels.npy")  # Load training set labels
    train_dataset = TwoFrameDataset(train_data, train_labels)
    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=config.num_workers)

    val_data = np.load("data/val/val_images.npy")  # Loading validation set images
    val_labels = np.load("data/val/val_labels.npy")  # Loading validation set labels
    val_dataset = TwoFrameDataset(val_data, val_labels)
    val_loader = DataLoader(val_dataset,
                              batch_size=1,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=config.num_workers)

    test_data = np.load("data/test/test_images.npy")  # Loading test set images
    test_labels = np.load("data/test/test_labels.npy")  # Loading test set labels
    test_dataset = TwoFrameDataset(test_data, test_labels)
    test_loader = DataLoader(test_dataset,
                              batch_size=1,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=config.num_workers)

    print('#----------Prepareing Model----------#')
    device = torch.device("cuda:1")
    model = SVSnet()
    model = model.to(device)

    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)

    print('#----------Set other params----------#')
    min_loss = 999
    start_epoch = 1
    min_epoch = 1

    if os.path.exists(resume_model):
        print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        min_loss, min_epoch, loss = checkpoint['min_loss'], checkpoint['min_epoch'], checkpoint['loss']

        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, min_loss: {min_loss:.4f}, min_epoch: {min_epoch}, loss: {loss:.4f}'
        logger.info(log_info)

    step = 0
    print('#----------Training----------#')
    for epoch in range(start_epoch, config.epochs + 1):

        torch.cuda.empty_cache()

        step = train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            step,
            logger,
            config,
            device
        )

        loss = val_one_epoch(
            val_loader,
            model,
            criterion,
            epoch,
            logger,
            config,
            device
        )

        if loss < min_loss:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            min_loss = loss
            min_epoch = epoch

        torch.save(
            {
                'epoch': epoch,
                'min_loss': min_loss,
                'min_epoch': min_epoch,
                'loss': loss,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(checkpoint_dir, 'latest.pth'))

    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
        print('#----------Testing----------#')
        best_weight = torch.load(config.work_dir + 'checkpoints/best.pth', map_location=torch.device('cpu'))
        model.load_state_dict(best_weight)
        loss = test_one_epoch(
            test_loader,
            model,
            criterion,
            logger,
            config,
            device
        )
        os.rename(
            os.path.join(checkpoint_dir, 'best.pth'),
            os.path.join(checkpoint_dir, f'best-epoch{min_epoch}-loss{min_loss:.4f}.pth')
        )


if __name__ == '__main__':
    config = setting_config
    main(config)