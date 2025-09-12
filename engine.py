import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix


def train_one_epoch(train_loader, model, criterion, optimizer, scheduler, epoch, step, logger, config, device):
    model.train()
    total_loss_list = []

    for iter, data in enumerate(train_loader):
        step += iter
        optimizer.zero_grad()

        images = data["image"].to(device, non_blocking=True).float()
        targets = data["label"].to(device, non_blocking=True).float()


        out = model(images)
        total_loss= criterion(out, targets)
        total_loss.backward()
        optimizer.step()

        total_loss_list.append(total_loss.item())

        now_lr = optimizer.state_dict()['param_groups'][0]['lr']

        if iter % config.print_interval == 0:
            log_info = f'train: epoch {epoch}, iter:{iter}, total_loss: {np.mean(total_loss_list):.4f}, lr: {now_lr}'
            print(log_info)
            logger.info(log_info)

    scheduler.step()

    return step


def val_one_epoch(test_loader, model, criterion, epoch, logger, config, device):
    model.eval()
    preds = []
    gts = []
    total_loss_list= []

    with torch.no_grad():
        for data in tqdm(test_loader):
            img = data["image"].to(device, non_blocking=True).float()
            msk = data["label"].to(device, non_blocking=True).float()


            out = model(img)
            total_loss = criterion(out, msk)
            total_loss_list.append(total_loss.item())

            gts.append(msk.squeeze(1).cpu().detach().numpy())
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out)

    if epoch % config.val_interval == 0:
        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds >= config.threshold, 1, 0)
        y_true = np.where(gts >= 0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        log_info = f'val epoch: {epoch}, total_loss: {np.mean(total_loss_list):.4f}  ' \
                   f'miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, ' \
                   f'specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
        print(log_info)
        logger.info(log_info)

    else:
        log_info = f'val epoch: {epoch}, total_loss: {np.mean(total_loss_list):.4f} '
        print(log_info)
        logger.info(log_info)

    return np.mean(total_loss_list)


def test_one_epoch(test_loader, model, criterion, logger, config, device, test_data_name=None):
    model.eval()
    preds = []
    gts = []
    total_loss_list= []
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):

            img = data["image"].to(device, non_blocking=True).float()
            msk = data["label"].to(device, non_blocking=True).float()

            out = model(img)
            total_loss= criterion(out, msk)
            total_loss_list.append(total_loss.item())
            msk = msk.squeeze(1).cpu().detach().numpy()
            gts.append(msk)
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out)


        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds >= config.threshold, 1, 0)
        y_true = np.where(gts >= 0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        if test_data_name is not None:
            log_info = f'test_datasets_name: {test_data_name}'
            print(log_info)
            logger.info(log_info)
        log_info = f'test of best model, total_loss: {np.mean(total_loss_list):.4f}' \
                   f'miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, ' \
                   f'specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
        print(log_info)
        logger.info(log_info)

    return np.mean(total_loss_list)

