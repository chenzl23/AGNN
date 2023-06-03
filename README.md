# Alternating Graph-Regularized Neural Networks (AGNN)

## Introduction
This is an implement of AGNN with PyTorch, which was run on a machine with AMD R9-5900HX CPU, RTX 3080 16G GPU and 32G RAM.

## Paper
Zhaoliang Chen, Zhihao Wu, Zhenghong Lin, Shiping Wang, Claudia Plant, and Wenzhong Guo. [AGNN: Alternating Graph-Regularized Neural Networks to Alleviate Over-Smoothing](https://ieeexplore.ieee.org/abstract/document/10138925). IEEE Transactions on Neural Networks and Learning Systems.


## Requirements
- torch: 1.11.0 + cu115
- numpy: 1.20.1
- scipy: 1.6.2
- scikit-learn: 1.1.2

## Running Examples
  - You can run the model with predefined hyperparameters (rho, lambda, dimension of hidden units, etc.). For example, on ACM dataset, we can run the following command (Avg. Accuracy = 90.3%):
    ```
    python main.py --dataset-name ACM --lamda=0.9  --rho=0.1 --layer-num=8
    ```
  - Another example on Flickr dataset (Avg. Accuracy = 58.4%):
    ```
    python main.py --dataset-name Flickr --lamda=0.8  --rho=0.5 --hidden-dim=16 --dropout=0.0
    ```
  - In some cases with deep layer, you may need to use a tiny learning rate (e.g., 0.005) for better performance.

## Reference
```
@ARTICLE{10138925,
  author={Chen, Zhaoliang and Wu, Zhihao and Lin, Zhenghong and Wang, Shiping and Plant, Claudia and Guo, Wenzhong},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={AGNN: Alternating Graph-Regularized Neural Networks to Alleviate Over-Smoothing}, 
  year={2023},
  pages={1-13},
  doi={10.1109/TNNLS.2023.3271623}}
```
