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

$$
\arg\min_{\delta:||\delta||\le c} KL(q_{\phi}(z|x_o+\delta)||q_{\phi}(z|x_t))
$$

事实上，VAE是十分鲁棒的（在结构比较简单的情况下），而且这个编码的latent space呈现一定的连续性，有可能我们的攻击只是使得编码类别发生了变化，变成了另外一个类的图片.一些例子如下

![0](https://github.com/linhaowei1/Attack-Vae/blob/main/latent_attack/9/3.png)$\quad$![0](https://github.com/linhaowei1/Attack-Vae/blob/main/latent_attack/9/7.png)$\quad$![0](https://github.com/linhaowei1/Attack-Vae/blob/main/latent_attack/9/48.png)$\quad$![0](https://github.com/linhaowei1/Attack-Vae/blob/main/latent_attack/9/25.png)$\quad$![0](https://github.com/linhaowei1/Attack-Vae/blob/main/latent_attack/5/4.png)$\quad$![0](https://github.com/linhaowei1/Attack-Vae/blob/main/latent_attack/5/7.png)$\quad$![0](https://github.com/linhaowei1/Attack-Vae/blob/main/latent_attack/5/12.png)$\quad$![0](https://github.com/linhaowei1/Attack-Vae/blob/main/latent_attack/5/13.png)$\quad$![0](https://github.com/linhaowei1/Attack-Vae/blob/main/latent_attack/5/18.png)$\quad$![0](https://github.com/linhaowei1/Attack-Vae/blob/main/latent_attack/5/22.png)$\quad$![0](https://github.com/linhaowei1/Attack-Vae/blob/main/latent_attack/5/23.png)$\quad$![0](https://github.com/linhaowei1/Attack-Vae/blob/main/latent_attack/5/63.png)$\quad$![0](https://github.com/linhaowei1/Attack-Vae/blob/main/latent_attack/1/21.png)$\quad$![0](https://github.com/linhaowei1/Attack-Vae/blob/main/latent_attack/1/25.png)$\quad$![0](https://github.com/linhaowei1/Attack-Vae/blob/main/latent_attack/1/29.png)$\quad$![0](https://github.com/linhaowei1/Attack-Vae/blob/main/latent_attack/1/33.png)$\quad$![0](https://github.com/linhaowei1/Attack-Vae/blob/main/latent_attack/1/34.png)$\quad$![0](https://github.com/linhaowei1/Attack-Vae/blob/main/latent_attack/1/38.png)$\quad$![0](https://github.com/linhaowei1/Attack-Vae/blob/main/latent_attack/1/41.png)$\quad$![0](https://github.com/linhaowei1/Attack-Vae/blob/main/latent_attack/1/82.png)

注意：攻击的epsilon = 0.3, stepsize = 3/255, iteration = 40.
## Untargeted Attack

The Untargeted-Attack is rather successful in VAE.

We just maximize the Loss (reconstruction loss + kl divergence loss) using PGD, and get very good results.

epsilon = 0.2, stepsize = 1/150, iteration = 30.

![0](https://github.com/linhaowei1/Attack-Vae/blob/main/untargeted_attack/1/39.png)$\quad$![0](https://github.com/linhaowei1/Attack-Vae/blob/main/untargeted_attack/1/58.png)$\quad$![0](https://github.com/linhaowei1/Attack-Vae/blob/main/untargeted_attack/1/91.png)$\quad$![0](https://github.com/linhaowei1/Attack-Vae/blob/main/untargeted_attack/2/6.png)$\quad$![0](https://github.com/linhaowei1/Attack-Vae/blob/main/untargeted_attack/2/7.png)$\quad$![0](https://github.com/linhaowei1/Attack-Vae/blob/main/untargeted_attack/2/3.png)$\quad$![0](https://github.com/linhaowei1/Attack-Vae/blob/main/untargeted_attack/2/19.png)$\quad$![0](https://github.com/linhaowei1/Attack-Vae/blob/main/untargeted_attack/4/30.png)$\quad$![0](https://github.com/linhaowei1/Attack-Vae/blob/main/untargeted_attack/4/38.png)$\quad$![0](https://github.com/linhaowei1/Attack-Vae/blob/main/untargeted_attack/4/42.png)$\quad$![0](https://github.com/linhaowei1/Attack-Vae/blob/main/untargeted_attack/4/58.png)$\quad$![0](https://github.com/linhaowei1/Attack-Vae/blob/main/untargeted_attack/2/22.png)$\quad$![0](https://github.com/linhaowei1/Attack-Vae/blob/main/untargeted_attack/2/25.png)$\quad$![0](https://github.com/linhaowei1/Attack-Vae/blob/main/untargeted_attack/2/27.png)$\quad$![0](https://github.com/linhaowei1/Attack-Vae/blob/main/untargeted_attack/2/50.png)$\quad$![0](https://github.com/linhaowei1/Attack-Vae/blob/main/untargeted_attack/2/61.png)$\quad$![0](https://github.com/linhaowei1/Attack-Vae/blob/main/untargeted_attack/2/70.png)$\quad$![0](https://github.com/linhaowei1/Attack-Vae/blob/main/untargeted_attack/2/79.png)$\quad$![0](https://github.com/linhaowei1/Attack-Vae/blob/main/untargeted_attack/0/48.png)$\quad$![0](https://github.com/linhaowei1/Attack-Vae/blob/main/untargeted_attack/0/41.png)$\quad$![0](https://github.com/linhaowei1/Attack-Vae/blob/main/untargeted_attack/0/30.png)$\quad$![0](https://github.com/linhaowei1/Attack-Vae/blob/main/untargeted_attack/0/21.png)$\quad$![0](https://github.com/linhaowei1/Attack-Vae/blob/main/untargeted_attack/0/11.png)$\quad$![0](https://github.com/linhaowei1/Attack-Vae/blob/main/untargeted_attack/0/5.png)$\quad$![0](https://github.com/linhaowei1/Attack-Vae/blob/main/untargeted_attack/3/3.png)$\quad$![0](https://github.com/linhaowei1/Attack-Vae/blob/main/untargeted_attack/3/4.png)$\quad$![0](https://github.com/linhaowei1/Attack-Vae/blob/main/untargeted_attack/3/19.png)$\quad$![0](https://github.com/linhaowei1/Attack-Vae/blob/main/untargeted_attack/3/25.png)$\quad$![0](https://github.com/linhaowei1/Attack-Vae/blob/main/untargeted_attack/3/65.png)$\quad$![0](https://github.com/linhaowei1/Attack-Vae/blob/main/untargeted_attack/4/3.png)$\quad$![0](https://github.com/linhaowei1/Attack-Vae/blob/main/untargeted_attack/4/8.png)$\quad$![0](https://github.com/linhaowei1/Attack-Vae/blob/main/untargeted_attack/4/11.png)
## logs
- 5.19: Get vanilla vae.
- 5.21: Exploration : classifier accuracy with VAE generated examples.
- 5.22: Try Latent Attack

## Tasks
### Reproduce
- [x] vanilla vae
- [x] latent attack
- [ ] classifer attack
- [ ] VAE attack

### Exploration
- [x] untargeted attack
- [ ] FID/Inception Score/Evaluation protocols
- [ ] Transferability
- [ ] Can attack boost one class detection? (based on untargeted attack)

