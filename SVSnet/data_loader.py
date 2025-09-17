import numpy as np
import random
from skimage import transform
import torch
from torch.utils.data import Dataset, DataLoader


# ================= Tool Function =================

def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


def random_channel_shift(x1, x2, intensity, channel_index=0):
    x1 = np.rollaxis(x1, channel_index, 0)
    x2 = np.rollaxis(x2, channel_index, 0)
    min_x1, max_x1 = np.min(x1), np.max(x1)
    min_x2, max_x2 = np.min(x2), np.max(x2)
    shift = np.random.uniform(-intensity, intensity)
    channel_images1 = [np.clip(x_channel + shift, min_x1, max_x1) for x_channel in x1]
    channel_images2 = [np.clip(x_channel + shift, min_x2, max_x2) for x_channel in x2]
    x1 = np.stack(channel_images1, axis=0)
    x1 = np.rollaxis(x1, 0, channel_index + 1)
    x2 = np.stack(channel_images2, axis=0)
    x2 = np.rollaxis(x2, 0, channel_index + 1)
    return x1, x2


# ================= TwoFrameDataset =================

class TwoFrameDataset(Dataset):
    def __init__(self, data, labels, crop_size=(400, 400), intensity=0.2, do_aug=True, normalize=True):
        """
        :param data: input image (N, 1, 2, H, W)
        :param labels: label (N, 1, 2, H, W)
        :param crop_size: Random crop size
        :param intensity: Channel Offset Intensity
        :param do_aug: Whether or not to do data enhancement
        :param normalize: Is it normalised to [0,1]
        """
        self.data = data
        self.labels = labels
        self.crop_size = crop_size
        self.intensity = intensity
        self.do_aug = do_aug
        self.normalize = normalize

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Take two frames (H, W), data format (1,2,H,W)
        img_x1 = self.data[idx][0, 0]  # First Frame
        img_x2 = self.data[idx][0, 1]  # Second Frame

        # Labels take only the last frame
        img_y = self.labels[idx][0, 1]  # (H, W)

        # Expand to (1,H,W)
        img_x1 = np.expand_dims(img_x1, 0)
        img_x2 = np.expand_dims(img_x2, 0)
        img_y = np.expand_dims(img_y, 0)

        if self.do_aug:
            # channel offset
            if random.random() > 0.5:
                img_x1, img_x2 = random_channel_shift(img_x1, img_x2, self.intensity)

            # random flip
            if random.random() > 0.5:
                img_x1, img_x2, img_y = flip_axis(img_x1, 1), flip_axis(img_x2, 1), flip_axis(img_y, 1)
            if random.random() > 0.5:
                img_x1, img_x2, img_y = flip_axis(img_x1, 2), flip_axis(img_x2, 2), flip_axis(img_y, 2)

            # random rotate
            if random.random() > 0.5:
                angle = np.random.randint(-10, 10)
                img_x1 = np.reshape(transform.rotate(img_x1[0], angle), [1, *img_x1.shape[1:]])
                img_x2 = np.reshape(transform.rotate(img_x2[0], angle), [1, *img_x2.shape[1:]])
                img_y = np.reshape(transform.rotate(img_y[0], angle), [1, *img_y.shape[1:]])

            # Random crop + resize back to original size
            if random.random() > 0.5:
                H, W = img_x1.shape[1], img_x1.shape[2]
                h_s = random.randint(0, H - self.crop_size[0])
                w_s = random.randint(0, W - self.crop_size[1])

                img1_crop = img_x1[0, h_s:h_s + self.crop_size[0], w_s:w_s + self.crop_size[1]]
                img2_crop = img_x2[0, h_s:h_s + self.crop_size[0], w_s:w_s + self.crop_size[1]]
                img_y_crop = img_y[0, h_s:h_s + self.crop_size[0], w_s:w_s + self.crop_size[1]]

                img_x1 = np.reshape(transform.resize(img1_crop, (H, W)), [1, H, W])
                img_x2 = np.reshape(transform.resize(img2_crop, (H, W)), [1, H, W])
                img_y = np.reshape(transform.resize(img_y_crop, (H, W)), [1, H, W])

        # Heap in depth dimension → (1,2,H,W)
        img_x = np.stack([img_x1[0], img_x2[0]], axis=0)  # (2,H,W)
        img_x = np.expand_dims(img_x, axis=0)             # (1,2,H,W)

        # Label binarisation (1,H,W)
        img_y = (img_y > 0.5).astype(np.float32)

        # normalisation
        if self.normalize:
            img_x = img_x.astype(np.float32) / 255.0

        return {
            "image": torch.from_numpy(img_x).float(),  # (1,2,H,W)
            "label": torch.from_numpy(img_y).float()   # (1,H,W)
        }

