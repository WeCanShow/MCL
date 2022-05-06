# Model-Contrastive Learning

This is an implementation demo of our 2021 paper Model-Contrastive Learning for Backdoor Defense in PyTorch.

![Python 3.8](https://img.shields.io/badge/python-3.8-DodgerBlue.svg?style=plastic)
![Pytorch 1.9](https://img.shields.io/badge/pytorch-1.9-DodgerBlue.svg?style=plastic)
![CUDA 11.2](https://img.shields.io/badge/cuda-11.2-DodgerBlue.svg?style=plastic)
![License CC BY-NC](https://img.shields.io/badge/license-CC_BY--NC-DodgerBlue.svg?style=plastic)

## MCL: Quick start with pretrained model
We have already uploaded the `all2one` pretrained backdoor model(i.e. gridTrigger WRN-16-1, target label 5).

For evaluating the performance of  MCL, you can easily run command:

```bash
$ python main.py 
```
where the default parameters are shown in `config.py`.

The trained model will be saved at the path `weight/erasing_net/<s_name>.tar`

Please carefully read the `main.py` and `configs.py`, then change the parameters for your experiment.


| Dataset  | Baseline ACC | Baseline ASR | MCL ACC | MCL ASR |
| -------- | ------------ | ------------ | ------- | ------- |
| CIFAR-10 | 83.01        | 99.64        | 82.19   | 2.22    |

---

## Training your own backdoored model
We have provided a `DatasetBD` Class in `data_loader.py` for generating training set of different backdoor attacks. 

For implementing backdoor attack(e.g. GridTrigger attack), you can run the below command:

```bash
$ python train_badnet.py
```
This command will train the backdoored model and print clean accuracies and attack rate. You can also select the other backdoor triggers reported in the paper. 

Please carefully read the `train_badnet.py` and `configs.py`, then change the parameters for your experiment.  

## Acknowledgements
Much of the code in this repository was adapted from code in **[this paper](https://github.com/bboylyg/NAD)** by Yige Li et al.

## Other source of backdoor attacks
#### Attack

**CL:** Clean-label backdoor attacks

- [Paper](https://people.csail.mit.edu/madry/lab/cleanlabel.pdf)
- [pytorch implementation](https://github.com/hkunzhe/label_consistent_attacks_pytorch)

**SIG:** A New Backdoor Attack in CNNS by Training Set Corruption Without Label Poisoning

- [Paper](https://ieeexplore.ieee.org/document/8802997/footnotes)

#### Defense


**Fine-tuning && Neural Attention Distillation**: Erasing Backdoor Triggers from Deep Neural Networks

- [Paper](https://openreview.net/pdf?id=9l0K4OM-oXE)
- [Pytorch implementation](https://github.com/bboylyg/NAD)

**I-BAU**: Adversarial Unlearning of Backdoors via Implicit Hypergradient.

- [Paper](https://arxiv.org/pdf/2110.03735.pdf)
- [Pytorch implementation](https://github.com/YiZeng623/I-BAU)

**Neural Cleanse**: Identifying and Mitigating Backdoor Attacks in Neural Networks

- [Paper](https://people.cs.uchicago.edu/~ravenben/publications/pdf/backdoor-sp19.pdf)
- [Tensorflow implementation](https://github.com/Abhishikta-codes/neural_cleanse)
- [Pytorch implementation1](https://github.com/lijiachun123/TrojAi)
- [Pytorch implementation2](https://github.com/VinAIResearch/input-aware-backdoor-attack-release/tree/master/defenses)

#### Library

`Note`: TrojanZoo provides a universal pytorch platform to conduct security researches (especially backdoor attacks/defenses) of image classification in deep learning.

Backdoors 101 — is a PyTorch framework for state-of-the-art backdoor defenses and attacks on deep learning models. 

BackdoorBox — is a Python toolbox for backdoor attacks and defenses.

- [TrojanZoo](https://github.com/ain-soph/trojanzoo)
- [backdoors101](https://github.com/ebagdasa/backdoors101)
- [BackdoorBox](https://github.com/THUYimingLi/BackdoorBox)


## Contacts

If you have any questions, leave a message below with GitHub.

