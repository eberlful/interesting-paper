1. Pruning Deep Convolutional Neural Network Using Conditional Mutual Information


## Pruning Deep Convolutional Neural Network Using Conditional Mutual Information

Paper: https://arxiv.org/abs/2411.18578

Date: 27.11.2024

##### Abstract
Convolutional Neural Networks (CNNs) achieve high performance in image classification tasks but are challenging to deploy on resource-limited hardware due to their large model sizes. To address this issue, we leverage Mutual Information, a metric that provides valuable insights into how deep learning models retain and process information through measuring the shared information between input features or output labels and network layers. In this study, we propose a structured filter-pruning approach for CNNs that identifies and selectively retains the most informative features in each layer. Our approach successively evaluates each layer by ranking the importance of its feature maps based on Conditional Mutual Information (CMI) values, computed using a matrix-based Renyi {\alpha}-order entropy numerical method. We propose several formulations of CMI to capture correlation among features across different layers. We then develop various strategies to determine the cutoff point for CMI values to prune unimportant features. This approach allows parallel pruning in both forward and backward directions and significantly reduces model size while preserving accuracy. Tested on the VGG16 architecture with the CIFAR-10 dataset, the proposed method reduces the number of filters by more than a third, with only a 0.32% drop in test accuracy.
