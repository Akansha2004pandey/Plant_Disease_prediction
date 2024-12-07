Plant Disease Classification Using Deep Learning
Project Overview
This project focuses on the classification of plant diseases using deep learning models. The dataset used consists of 87,000 data points across 38 different plant disease classes. We have implemented two approaches for plant disease classification: a custom CNN model and transfer learning using pretrained models such as ResNet-50, DenseNet-121, and VGG-16. Our goal was to achieve high accuracy in classifying plant diseases and to explore the effectiveness of transfer learning in this domain.

University
University Name: Netaji Subhas University of Technology
Under the Guidance of: Professor Gaurav Singhal

Project Details
Dataset
Dataset Name: New Plant Disease Dataset
Total Data Points: 87,000
Number of Classes: 38
Source: [https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset]
Approaches Used
Custom CNN Model

Built a Convolutional Neural Network (CNN) with 5 convolutional layers.
Applied pooling, activation functions, and fully connected layers for classification.
Optimized using the Adam optimizer.
Transfer Learning with Pretrained Models

ResNet-50: Used to leverage deeper feature extraction capabilities.
DenseNet-121: Applied for improved feature reuse and reduced overfitting.
VGG-16: A simpler architecture for effective fine-tuning.
Model Architecture
Custom CNN Model:

5 Convolutional layers.
ReLU activation function for non-linearity.
Max pooling layers for dimensionality reduction.
Softmax output layer for multi-class classification.
Adam optimizer for training the model.
Transfer Learning:

Fine-tuned pretrained models: ResNet-50, DenseNet-121, and VGG-16.
Transfer learning used to improve model performance by leveraging ImageNet weights.
Performance
Achieved Accuracy: 97%
Evaluation Metric: Model accuracy was evaluated on the validation set, with transfer learning models outperforming the custom CNN approach.
Future Work
Optimization: Further model improvements through hyperparameter tuning, data augmentation, and ensemble learning.
Real-World Application: Deployment for real-time plant disease detection.
Interpretability: Implementing methods like Grad-CAM for better model decision interpretation.

Contributing
Feel free to open an issue or submit a pull request if you'd like to contribute to the project. We welcome contributions related to model optimization, dataset improvements, or any other relevant changes.

License
This project is licensed under the MIT License - see the LICENSE file for details.

This README provides a structured overview of your project, guiding users on how to set it up and contribute. You can customize the sections as per your specific project details.






