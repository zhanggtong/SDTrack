import os

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt


############## used for visulize eliminated tokens #################
def get_keep_indices(decisions):
    keep_indices = []
    for i in range(3):
        if i == 0:
            keep_indices.append(decisions[i])
        else:
            keep_indices.append(keep_indices[-1][decisions[i]])
    return keep_indices


def gen_masked_tokens(tokens, indices, alpha=0.2):
    # indices = [i for i in range(196) if i not in indices]
    indices = indices[0].astype(int)
    tokens = tokens.copy()
    tokens[indices] = alpha * tokens[indices] + (1 - alpha) * 255
    return tokens


def recover_image(tokens, H, W, Hp, Wp, patch_size):
    # image: (C, 196, 16, 16)
    image = tokens.reshape(Hp, Wp, patch_size, patch_size, 3).swapaxes(1, 2).reshape(H, W, 3)
    return image


def pad_img(img):
    height, width, channels = img.shape
    im_bg = np.ones((height, width + 8, channels)) * 255
    im_bg[0:height, 0:width, :] = img
    return im_bg


def gen_visualization(image, mask_indices, patch_size=16):
    # image [224, 224, 3]
    # mask_indices, list of masked token indices

    # mask mask_indices need to cat
    # mask_indices = mask_indices[::-1]
    num_stages = len(mask_indices)
    for i in range(1, num_stages):
        mask_indices[i] = np.concatenate([mask_indices[i-1], mask_indices[i]], axis=1)

    # keep_indices = get_keep_indices(decisions)
    image = np.asarray(image)
    H, W, C = image.shape
    Hp, Wp = H // patch_size, W // patch_size
    image_tokens = image.reshape(Hp, patch_size, Wp, patch_size, 3).swapaxes(1, 2).reshape(Hp * Wp, patch_size, patch_size, 3)

    stages = [
        recover_image(gen_masked_tokens(image_tokens, mask_indices[i]), H, W, Hp, Wp, patch_size)
        for i in range(num_stages)
    ]
    imgs = [image] + stages
    imgs = [pad_img(img) for img in imgs]
    viz = np.concatenate(imgs, axis=1)
    return viz


def visualize_patch_attention(image_tensor, patch_attention, save_dir='./maskpics'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  # 创建目录
        existing_files = []
    else:
        existing_files = [f for f in os.listdir(save_dir) if f.startswith('attention_') and f.endswith('.png')]

    # existing_files = [f for f in os.listdir(save_dir) if f.startswith('attention_') and f.endswith('.png')]

    # 提取现有编号并找到最大值
    max_num = 0
    for file in existing_files:
        try:
            # 从 "attention_001.png" 中提取数字
            num = int(file.split('_')[1].split('.')[0])
            max_num = max(max_num, num)
        except (IndexError, ValueError):
            continue

    # 新文件编号
    new_num = max_num + 1
    save_path = os.path.join(save_dir, f'attention_{new_num:03d}.png')

    if isinstance(image_tensor, torch.Tensor):
        image = image_tensor.squeeze(0).detach().cpu().numpy()
    else:
        image = image_tensor.squeeze(0)

    if isinstance(patch_attention, torch.Tensor):
        patch_attention = patch_attention.detach().cpu().numpy()

    patch_attention = patch_attention.squeeze()
    assert len(patch_attention) == 256

    # reshape和归一化注意力
    attention_grid = patch_attention.reshape(16, 16)
    attention_grid_norm = (attention_grid - attention_grid.min()) / (attention_grid.max() - attention_grid.min() + 1e-8)
    attention_map = cv2.resize(attention_grid_norm, (256, 256), interpolation=cv2.INTER_LINEAR)

    # 智能处理图像数据范围
    image_display = np.transpose(image, (1, 2, 0))

    # 诊断图像数据范围
    print("=== 图像数据诊断 ===")
    print(f"原始数据类型: {image_display.dtype}")
    print(f"原始数据范围: [{image_display.min():.3f}, {image_display.max():.3f}]")

    # 自动处理数据范围
    if image_display.dtype in [np.float32, np.float64]:
        if image_display.max() > 10.0:  # 可能是0-255的浮点数
            image_display = image_display / 255.0
            print("检测到0-255浮点数，已归一化到0-1")
        elif image_display.max() <= 1.0:  # 已经是0-1范围
            print("数据已在0-1范围内")
        else:
            # 其他情况，归一化到0-1
            image_display = (image_display - image_display.min()) / (image_display.max() - image_display.min())
            print("已归一化到0-1范围")

        # 确保在有效范围内
        image_display = np.clip(image_display, 0.0, 1.0)
    else:
        # 整数类型，转换为0-1浮点数
        image_display = image_display.astype(np.float32) / 255.0
        print("整数类型已转换为0-1浮点数")

    print(f"处理后数据范围: [{image_display.min():.3f}, {image_display.max():.3f}]")

    # 创建可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. 原始图像
    axes[0].imshow(image_display)
    axes[0].set_title('Original Image\n(自动归一化)')
    axes[0].axis('off')

    # 2. 注意力热力图
    im = axes[1].imshow(attention_map, cmap='hot')
    axes[1].set_title('Attention Heatmap')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])

    # 3. 叠加效果
    axes[2].imshow(image_display)
    axes[2].imshow(attention_map, cmap='jet', alpha=0.5)
    axes[2].set_title('Attention Overlay\n(alpha=0.5)')
    axes[2].axis('off')

    # 添加patch网格
    for i in range(1, 16):
        axes[2].axhline(i * 16, color='white', linewidth=1, alpha=0.8)
        axes[2].axvline(i * 16, color='white', linewidth=1, alpha=0.8)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

    return attention_map

# 使用示例
# attention_map = visualize_patch_attention(image_tensor, patch_attention, 'patch_attention_result.png')