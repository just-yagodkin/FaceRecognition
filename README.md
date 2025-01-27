# Face Recognition Project

## Description📜

This project is dedicated to the development of a **face recognition pipeline** that can process images containing one or multiple faces and output feature embeddings for each detected face. These embeddings enable effective differentiation between individuals, making the system versatile for various applications, such as access control and identity verification.

### Motivation
Imagine you own a company and want to replace traditional access cards with a face recognition system for seamless employee entry. Such a system should process video footage from cameras and identify individuals in various conditions:

- Faces can be captured in close-up or from a distance.
- Faces may appear at an angle or slightly tilted.
- Multiple people may appear in a single frame.

To meet these requirements, the face recognition system must be robust and adaptable.

## How it works🙈
A typical face recognition pipeline consists of **three** main **stages**:

1. **Face Detection**\
Locating and identifying the positions of faces in the input image.

2. **Face Alignment**\
Transforming each detected face into a standardized orientation. This preprocessing step significantly enhances recognition accuracy, as misaligned faces make the task much harder for the model.

3. **Face Recognition**\
A neural network processes the aligned face image and generates a fixed-size feature vector (embedding). Embeddings of the same person should be close based on a similarity metric (e.g., cosine similarity), while embeddings of different individuals should be far apart.

### Why not use standard classification?
Unlike traditional classification models, which learn to predict a fixed set of predefined classes, this system must generalize to unseen individuals. Instead of directly classifying faces, it produces embeddings that can represent new, previously unseen individuals. This approach makes the system flexible and capable of recognizing faces outside the training dataset.

## Features of the project✨
- **Complete Pipeline**:\
The system integrates all key stages of face recognition, from detection and alignment to embedding generation.

- **Research and Practical Focus**:\
Analyzed and studied relevant literature on face recognition techniques.
Built a fully functional solution that is ready for real-world applications.

## Sources📚
- ArcFace Loss function [paper](https://arxiv.org/pdf/1801.07698)
- Triplet Loss [paper](https://arxiv.org/pdf/1503.03832)
- EfficientNet [paper](https://proceedings.mlr.press/v97/tan19a/tan19a.pdf) (3rd-stage model within the pipeline)
- MTCNN [paper](https://kpzhang93.github.io/MTCNN_face_detection_alignment/paper/spl.pdf) (a ready-made solution for the face detection task, model for the 1st-stage)
- MTCNN-PyTorch [github](https://github.com/timesler/facenet-pytorch)
- Triplet Loss(triplet mining) [article](https://omoindrot.github.io/triplet-loss#triplet-mining)
- Data Augmentation [article](https://rumn.medium.com/ultimate-guide-to-fine-tuning-in-pytorch-part-3-deep-dive-to-pytorch-data-transforms-53ed29d18dde) (examples of data transformation)
---
> *Interesting fact*: **Face**book has been using facial recognition to identify users in published photos and suggest tagging them since 2010
