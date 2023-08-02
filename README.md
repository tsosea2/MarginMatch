This is the repository for the paper `MarginMatch: Improving Semi-Supervised Learning with Pseudo-Margins`. If you found this repository helpful, consider citing our paper:

```
@inproceedings{sosea2023marginmatch,
  title={MarginMatch: Improving Semi-Supervised Learning with Pseudo-Margins},
  author={Sosea, Tiberiu and Caragea, Cornelia},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={15773--15782},
  year={2023}
}
```
Our code builds upon TorchSSL (https://github.com/TorchSSL/TorchSSL) and we reuse various parts such as data loading
(with minor changes such as using indexed datasets for AUM) and network building
(WideResNets).


To run our code for MarginMatch, anaconda must be installed:
https://www.anaconda.com/products/distribution

To install the required packages, please run:

```
sh install.sh
```

To configure the parameters of MarginMatch, please specify a configuration
in the "config" directory. The config yaml file contains all the parameters
of our MarginMatch, such as learning rate, AUM threshold or AUM smoothing value.
To train using MarginMatch, simply run:

```
python marginmatch.py --c config/<config_file.yaml>
```

Training metrics such as mask rate or unlabeled data impurity are automatically
logged using tensorboard in the main repository directory. To start a tensorboard
session run:

tensorboard --logdir .

We evaluate the performance of our MarginMatch on four computer vision benchmarks:
CIFAR-10, CIFAR-100, SVHN, and STL-10 using various low-resource scenarios (4 labels
per class, 25 labels per class, 400 labels per class). We also test MarginMatch on 
ImageNet and WebVision.


If you encounter any problems running the code please create an issue in this repository.
