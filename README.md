# PyTorch-Medical Image Segmentation

<p align="center"><img src="project.png" width="480"\></p>

**This repository contains the PyTorch implementation of a medical image segmentation model proposed in a research paper. While the implementation may differ slightly from the original architecture described in the paper, the focus here is on capturing and realizing the core ideas presented. I warmly welcome issues and suggestions from the community. If you would like to become a contributor to this project, please feel free to contact me at: wgyhhh001@gmail.com.**

## Implementations   
### SVSnet
Sequential vessel segmentation via deep channel attention network

#### Authors
Dongdong Hao, Song Ding, Linwei Qiu, Yisong Lv, Baowei Fei, Yueqi Zhu, Binjie Qin

#### Abstract
Accurately segmenting contrast-filled vessels from X-ray coronary angiography (XCA) image sequence is an essential step for the diagnosis and therapy of coronary artery disease. However, developing automatic vessel segmentation is particularly challenging due to the overlapping structures, low contrast and the presence of complex and dynamic background artifacts in XCA images. This paper develops a novel encoder–decoder deep network architecture which exploits the several contextual frames of 2D+t sequential images in a sliding window centered at current frame to segment 2D vessel masks from the current frame. The architecture is equipped with temporal–spatial feature extraction in encoder stage, feature fusion in skip connection layers and channel attention mechanism in decoder stage. In the encoder stage, a series of 3D convolutional layers are employed to hierarchically extract temporal–spatial features. Skip connection layers subsequently fuse the temporal–spatial feature maps and deliver them to the corresponding decoder stages. To efficiently discriminate vessel features from the complex and noisy backgrounds in the XCA images, the decoder stage effectively utilizes channel attention blocks to refine the intermediate feature maps from skip connection layers for subsequently decoding the refined features in 2D ways to produce the segmented vessel masks. Furthermore, Dice loss function is implemented to train the proposed deep network in order to tackle the class imbalance problem in the XCA data due to the wide distribution of complex background artifacts. Extensive experiments by comparing our method with other state-of-the-art algorithms demonstrate the proposed method’s superior performance over other methods in terms of the quantitative metrics and visual validation.

[[Paper]](https://www.sciencedirect.com/science/article/pii/S0893608020301672)
