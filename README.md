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


## logs
- 5.19: Get vanilla vae.

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

