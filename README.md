# 3D-Aware Indoor Scene Synthesis with Depth Priors

![image](./docs/assets/teaser.jpg)
**Figure:** *Left: High-fidelity images with the corresponding depths generated by DepthGAN. Right: The 3D reconstruction results.*

> **3D-Aware Indoor Scene Synthesis with Depth Priors** <br>
> Zifan Shi, Yujun Shen, Jiapeng Zhu, Dit-Yan Yeung, Qifeng Chen <br>
> *arXiv preprint arXiv:TODO*

[[Paper](TODO)]
[[Project Page](https://shizifan.github.io/DepthGAN)]
[[Demo](https://youtu.be/RMmIso5Oxno)]

Existing methods fail to learn high-fidelity 3D-aware indoor scene synthesis merely from 2D images. In this work, we fill in this gap by proposing **DepthGAN**, which incorporates depth as a 3D prior. Concretely, we propose a *dual-path generator*, where one path is responsible for depth generation, whose intermediate features are injected into the other path as the condition for appearance rendering. Such a design eases the 3D-aware synthesis with explicit geometry information. Meanwhile, we introduce a *switchable discriminator* both to differentiate real *v.s.* fake domains, and to predict the depth from a given input. In this way, the discriminator can take the spatial arrangement into account and advise the generator to learn an appropriate depth condition. Extensive experimental results suggest that our approach is capable of synthesizing indoor scenes with impressively good quality and 3D consistency, significantly outperforming state-of-the-art alternatives.

## Results

Qualitative comparison between **DepthGAN** and existing alternatives.

![image](./docs/assets/result1.jpg)

Diverse synthesis via varying the appearance latent code, conditioned on the same depth image.

![image](./docs/assets/depth_fixed.jpg)

Diverse geometries via varying the depth latent code, rendered with the same appearance style.

![image](./docs/assets/appearance_fixed.jpg)

## Code Coming Soon

## BibTeX

```bibtex
@article{...,
  title   = {3D-Aware Indoor Scene Synthesis with Depth Priors},
  author  = {Shi, Zifan and Shen, Yujun and Zhu, Jiapeng and Yeung, Dit-Yan and Chen, Qifeng},
  article = {TODO},
  year    = {2022}
}
```