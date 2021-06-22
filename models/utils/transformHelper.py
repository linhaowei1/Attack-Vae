import torch
import torch.nn.functional as F
import numpy as np


def rot_img(x, theta):
    theta = torch.tensor(theta)
    x = x.to("cpu")
    rot_mat = torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                            [torch.sin(theta), torch.cos(theta), 0]], dtype=torch.float)
    rot_mat = rot_mat[None, ...].repeat(x.shape[0], 1, 1)
    grid = F.affine_grid(rot_mat, x.size()).type(torch.float)
    output = F.grid_sample(x, grid)

    return output


def cut_perm(x, aug_index):
    _, _, h, w = x.size()
    h_mid = int(h / 2)
    w_mid = int(w / 2)

    if aug_index == -1: # rotate in clockwise
        x_left = x[:, :, :, 0:w_mid]
        x_right = x[:, :, :, w_mid:]

        x_new_left = torch.cat((x_left[:, :, h_mid:, :], x_right[:, :, h_mid:, :]), dim=2)
        x_new_right = torch.cat((x_left[:, :, 0:h_mid, :], x_right[:, :, 0:h_mid, :]),dim=2)

        result = torch.cat((x_new_left, x_new_right), dim=3)

    else:
        result = x
        if aug_index // 2 == 1:
            result = torch.cat((result[:, :, h_mid:, :], result[:, :, 0:h_mid, :]), dim=2)
        if aug_index % 2 == 1:
            result = torch.cat((result[:, :, :, w_mid:], result[:, :, :, 0:w_mid]), dim=3)

    return result
