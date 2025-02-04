from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.image import imread

import torch
import torch.nn as nn

from torchvision import tv_tensors
from torchvision.io import read_image
from torchvision.transforms import v2
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights


class EfficientNet_B2(nn.Module):
    def __init__(self, weights: Optional[EfficientNet_B2_Weights] = None, num_classes=500):
        super(EfficientNet_B2, self).__init__()
        self.model = efficientnet_b2(weights=weights)
        # Change out_features of last FC-layer
        dim_feats = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(dim_feats, num_classes, bias=False)
        print('Model defined succesfully!')
    
    def forward(self, x):
        embeddings = self.model.avgpool(self.model.features(x))
        embeddings = embeddings.squeeze()
        output = self.model.classifier(embeddings)
        last_layer_weight = self.model.classifier[1].weight

        return output, embeddings, last_layer_weight
    
    def load(self, path, **kwargs):
        self.load_state_dict(
            torch.load(path, **kwargs)
        )

def render_tensor(img_path, device):
    img = read_image(img_path).to(device)
    if (str(device) == 'cuda'):
        plt.imshow(torch.permute(img, (1, 2, 0)).cpu())
    else:
        plt.imshow(torch.permute(img, (1, 2, 0)))
    plt.axis('off')
    plt.show()
    return

# _____Extract&Align Faces block_____

def detect_faces(image, th=0.98):
    """Функция принимает на вход картинку (путь к ней ИЛИ непосредственно numpy-тензор)
        и возвращает боксы, вероятности и ключевые точки обнаруженных лиц на ней"""
    from facenet_pytorch import MTCNN
    import warnings
    warnings.filterwarnings('ignore')
    detector = MTCNN()

    if isinstance(image, str):
        np_img = imread(image)
    if isinstance(image, np.ndarray):
        np_img = image
        
    # Detect faces
    boxes, probs, landmarks = detector.detect(np_img, landmarks=True)
    
    detected = pd.DataFrame({'boxes': boxes.tolist(), 'probs': probs.tolist(), 'landmarks': landmarks.tolist()})
    detected = detected[detected['probs'] > th]

    return boxes, probs, landmarks

def extract_face(img, box, landmark):
    new_landmark = []
    for dot in landmark:
        new_x = int(dot[0]-box[0])
        new_y = int(dot[1]-box[1])
        new_landmark.append([new_x, new_y])

    box = list(map(int, box))

    if isinstance(img, np.ndarray):
        face = img[box[1]:box[3], box[0]:box[2], :]
        face = torch.Tensor(face).type(torch.uint8)
    elif isinstance(img, torch.Tensor):
        face = img[:, box[1]:box[3], box[0]:box[2]]
        face = face.permute((1,2,0))
    else:
        raise Exception("Got not torch.Tensor nor numpy.ndarray")

    new_landmark = torch.Tensor(new_landmark)
    new_landmark = new_landmark.repeat(1, 2)
    new_landmark = tv_tensors.BoundingBoxes(
        new_landmark, format='XYXY', canvas_size=face.shape[:2]
    )
    
    return {'image': face, 'landmarks': new_landmark}

# Write some functions for align. Their names speak for themselves
def calculate_angle_to_rotate(face):
    """
    face is _always_ a dict with to keys: 'image', 'landmarks'
    face['image'] -> torch.Tensor [HxWxC]
    face['landmarks'] -> tv_tensors.BoundingBoxes [5x4] (format 'XYXY')
    """
    eye1_x, eye1_y = face['landmarks'][0][0], face['landmarks'][0][1]
    eye2_x, eye2_y = face['landmarks'][1][0], face['landmarks'][1][1]

    coside = eye2_x - eye1_x
    verside = eye2_y - eye1_y
    tan = verside/ coside
    angle = torch.atan(tan)

    angle_in_degrees = 180 * angle / torch.pi
    
    return angle_in_degrees
    
def align_face(face):
    """
    face is _always_ a dict with to keys: 'image', 'landmarks'
    face['image'] -> torch.Tensor [HxWxC]
    face['landmarks'] -> tv_tensors.BoundingBoxes [5x4] (format 'XYXY')
    """
    angle = calculate_angle_to_rotate(face)
    rotation = v2.RandomRotation(degrees=(angle, angle))
    aligned_face_tuple = rotation(face['image'].permute(2,0,1), face['landmarks'])
    aligned_face = {
        'image': aligned_face_tuple[0].permute(1,2,0),
        'landmarks': aligned_face_tuple[1]
    }

    return aligned_face  