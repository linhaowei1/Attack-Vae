# Attack-Vae
We will first focus on MNIST.

![real](https://github.com/linhaowei1/Attack-Vae/blob/main/checkpoint/pic/real_img_500epoch.png)$\quad$![recon](https://github.com/linhaowei1/Attack-Vae/blob/main/checkpoint/pic/recons_500epoch.png)

左边是real_image，右边是vae重建的图片。

## vae训练

```python
batchsize = 144
epochs = 500
lr = 1e-4
optim = Adam
```
训练时长：`2021-05-19 23:37:59.600` 开始， `2021-05-20 00:17:56.000` 结束。

## VAE-generated Images v.s. Standard Images classifying Accuracy

We evaluate the acc on a pretrained small CNN model.

|  data   | accuracy %  |
|  ----  | ----  |
| standard test set  | 99.53 |
| gen-500  | 94.07 |
| gen-400  | 93.73 |
| gen-300  | 93.52 |
| gen-200  | 92.90 |
| gen-100  | 92.13 |
| gen-50   | 90.82 |
| gen-0    | 11.34 |

`gen-epochs` denotes we train the VAE `epochs` iterations and generate pictures using VAE. This could be used as a evaluation for VAE's generation ability.

## Hard images Visualization

Guess what are them? (top: clean example; bottom: generated)

$\quad\quad\quad\quad\quad$![0](https://github.com/linhaowei1/Attack-Vae/blob/main/hardimgs/0/1.png)![1](https://github.com/linhaowei1/Attack-Vae/blob/main/hardimgs/1/1.png)![2](https://github.com/linhaowei1/Attack-Vae/blob/main/hardimgs/2/25.png)![3](https://github.com/linhaowei1/Attack-Vae/blob/main/hardimgs/3/3.png)![4](https://github.com/linhaowei1/Attack-Vae/blob/main/hardimgs/4/27.png)![5](https://github.com/linhaowei1/Attack-Vae/blob/main/hardimgs/5/23.png)![6](https://github.com/linhaowei1/Attack-Vae/blob/main/hardimgs/6/35.png)![7](https://github.com/linhaowei1/Attack-Vae/blob/main/hardimgs/7/16.png)![8](https://github.com/linhaowei1/Attack-Vae/blob/main/hardimgs/8/40.png)![9](https://github.com/linhaowei1/Attack-Vae/blob/main/hardimgs/9/50.png)

## Latent Attack

The best performing attack on VAEs in the current literature is a latent space attack (Tabacof et al., 2016; Gondim-Ribeiro et al., 2018; Kos et al., 2018), where an adversary perturbs input $x_o$ to have a posterior $q_\phi$ similar to that of the target $x_t$, optimizing

\[\arg\min_{\delta:||\delta||\le c} KL(q_{\phi}(z|x_o+\delta)||q_{\phi}(z|x_t))\]


## logs
- 5.19: Get vanilla vae.
- 5.21: Exploration : classifier accuracy with VAE generated examples.
- 5.22: Try Latent Attack

## Tasks
### Reproduce
- [x] vanilla vae
- [ ] latent attack
- [ ] classifer attack
- [ ] VAE attack

### Exploration
- [ ] untargeted attack
- [ ] FID/Inception Score/Evaluation protocols
- [ ] Transferability

