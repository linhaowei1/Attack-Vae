# Attack-Vae
We will first focus on MNIST.

![real](https://github.com/linhaowei1/Attack-Vae/blob/main/checkpoint/pic/real_img_500epoch.png)$\quad$![recon](https://github.com/linhaowei1/Attack-Vae/blob/main/checkpoint/pic/recons_500epoch.png)

左边是real_image，右边是vae重建的图片。

vae训练配置：

```python
batchsize = 144
epochs = 500
lr = 1e-4
optim = Adam
```

VAE-generated Images v.s. Standard Images classifying Accuracy

|  data   | accuracy%  |
|  ----  | ----  |
| testset  | 99.53 |
| gen-500  | 94.07 |
| gen-400  | 93.73 |
| gen-300  | 93.52 |
| gen-200  | 92.90 |
| gen-100  | 92.13 |
| gen-50   | 90.82 |
| gen-0    | 11.34 |

## logs
- 5.19: Get vanilla vae.
- 5.21: Exploration : classifier accuracy with VAE generated examples.

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

