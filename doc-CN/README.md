# FATE

"FATE: Full-head Gaussian Avatar with Textural Editing from Monocular Video" (FateAvatar)的实现代码。

[张嘉伟](https://zjwsite.github.io/)<sup>1</sup>, [巫子健](https://github.com/Zijian-Wu)<sup>1</sup>, [梁智扬](https://github.com/ZhiyangLiang?tab=repositories)<sup>1</sup>, [龚奕成](https://github.com/Gong-Yicheng)<sup>1</sup>, [胡东方]()<sup>2</sup>, [姚遥](https://yoyo000.github.io/)<sup>1</sup>, [曹汛](https://cite.nju.edu.cn/People/Faculty/20190621/i5054.html)<sup>1</sup>,  [朱昊](http://zhuhao.cc/home/)<sup>1+</sup>

<sup>1</sup>南京大学, <sup>2</sup>OPPO

<sup>+</sup>通讯作者

[[论文](https://arxiv.org/abs/2411.15604)] | [[EN](../README.md)] | [[项目主页](https://zjwfufu.github.io/FATE-page/)]

> 从易于捕捉的单目视频中重建高保真、可动画化的3D头部虚拟形象是一个至关重要但极具挑战性的任务。尽管在渲染性能和表现力方面已经取得了显著进展，但仍面临诸多挑战，包括不完整的重建和低效的高斯表示。为了解决这些问题，我们提出了FATE——一种从单目视频重建可编辑的，完整的3D人头的方法。FATE结合了一种基于采样的致密化策略，确保点的最佳位置分布，从而提高渲染效率。我们还引入了一种神经烘焙技术，将离散的高斯表示转化为连续的属性图，从而实现直观的外观编辑。此外，我们提出了一个通用的补全框架，以恢复非正面外观，最终实现完整的3D头部虚拟形象。FATE在定性和定量评估中均优于以往的方法，达到了当前最先进的性能。据我们所知，FATE是首个可驱动并支持360°重建的从单目视频中重建3D人头的重建方法。

<div align=center>
  <img src="../assets/teaser.jpg">
</div>

## 安装

出于兼容性考虑，我们推荐在 Linux 系统上运行此仓库。我们也在 Windows 10 上进行测试，但您可能会遇到一些问题，[例如](https://github.com/cleardusk/3DDFA_V2/issues/12)。我们已经在 RTX3090、RTX3080 和 V100 GPU 上进行了测试和开发。请确保您的设备支持 CUDA 工具包 11.6 或更高版本。

- 克隆此仓库并且配置conda环境

  ```
  git clone https://github.com/zjwfufu/FateAvatar.git --recursive
  
  conda env create -f environment.yml
  ```

- 安装3DGS依赖

  ```
  cd submodules
  pip install ./diff-gaussian-rasterization
  pip install ./simple-knn
  cd ..
  ```

- 安装PyTorch3D (实验时版本为PyTorch3D 0.7.7)

  ```
  git clone https://github.com/facebookresearch/pytorch3d.git
  cd pytorch3d && pip install .
  ```

- 一些其他的功能性依赖

  ```
  # for completion framework
  pip install cmake dlib
  
  cd submodules/3DDFA_V2
  bash build.sh	# you may need install manually in Win10
  cd ..
  
  # [Optional] for monogaussianavatar baseline
  pip install functorch==0.2.0
  
  # [Optional] for splattingavatar baseline
  pip install libigl packaging pybind11
  
  cd submodules/simple_phongsurf
  pip install .
  cd ..
  ```

- 相关权重

  | Model                        | Links                                                        |
  | ---------------------------- | ------------------------------------------------------------ |
  | FLAME2020                    | [generic_model.pkl](https://flame.is.tue.mpg.de/)            |
  | SphereHead                   | [spherehead-ckpt-025000.pkl](https://cuhko365-my.sharepoint.com/:u:/g/personal/223010106_link_cuhk_edu_cn/EUU4STpe1p5HhLKIYgxuN6YBWNxVKac0WCXzoDLSJPn4RA?e=pOyGkK) |
  | GFPGAN1.3                    | [GFPGANv1.3.pth](https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth) |
  | VGG16 for Perpetual Loss     | [vgg16.pt](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt) |
  | Landmark Detection from Dlib | [shape_predictor_68_face_landmarks.dat](https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat) |
  | MODNet                       | [modnet_webcam_portrait_matting.ckpt](https://drive.google.com/drive/folders/1umYmlCulvIFNaqPjwod1SayFmSRHziyR) |
  | Face Parsing                 | [79999_iter.pth](https://drive.google.com/file/d/154JgKpzCPW82qINcVieuPH3fZ2e0P812/view?usp=drive_open) |
  | Parsing Net                  | [parsing_parsenet.pth](https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth) |

## 数据

我们支持来自 [IMAvatar](https://github.com/zhengyuf/IMavatar) 和 [INSTA](https://github.com/Zielon/INSTA) 预处理管线的数据集。有关 IMAvatar 数据集的布局，请参阅 [这里](https://github.com/zjwfufu/FateAvatar/blob/b39642aa34e451350eba85996dc0735192c5473e/train/dataset.py#L127)，有关 INSTA 数据集的布局，请参阅 [这里](https://github.com/Zielon/INSTA/blob/3dc6f865e36c7a64632b85f99cc170891e51d44a/scripts/transforms.py#L44)。`./data` 文件夹应具有以下结构：

```
.
|-- insta
|   |-- bala
|   |-- obama
|	|-- ....
|-- imavatar
    |-- yufeng
    |-- person1
    |-- ...
```

> [!NOTE]
> 对于 IMAvatar 数据集，我们进一步应用了 [face-parsing](https://github.com/zllrunning/face-parsing.PyTorch) 来移除衣服。对于 INSTA 数据集，我们修改了 [这个脚本](https://github.com/Zielon/INSTA/blob/3dc6f865e36c7a64632b85f99cc170891e51d44a/scripts/transforms.py#L44) 来导出与 FLAME 相关的属性。

我们使用了来自 [INSTA](https://github.com/Zielon/INSTA)、[PointAvatar](https://github.com/zhengyuf/PointAvatar/tree/master)、[Emotalk3D](https://nju-3dv.github.io/projects/EmoTalk3D/) 和 [NerFace](https://github.com/gafniguy/4D-Facial-Avatars) 的公开数据集。要访问 Emotalk3D 数据集，请在 [这里](https://nju-3dv.github.io/projects/EmoTalk3D/static/license/LicenseAgreement_EmoTalk3D.pdf) 申请。

*我们提供两个受试者预处理好的数据集的[下载链接](https://box.nju.edu.cn/f/5a502b0628bb477b8618/)：INSTA中的bala和IMAvatar中的yufeng。~~更多的预处理数据集将很快发布。~~*

## NeRSemble Benchmark

[2025/5/18 更新] 我们提供了用于[NeRSemble Benchmark](https://kaldir.vc.in.tum.de/nersemble_benchmark/benchmark/mono_flame_avatar)的脚本 `run_nersemble_benchmark.sh` 。 如果需要尝试, 只需运行 `pip install nersemble_benchmark` 并将 `FLAME2023.pkl` 连同两个顶点索引文件一起，放入 `./weights` 目录下。

##  预训练

~~*我们将很快发布预训练模型。*~~

## 用法

使用以下命令从单目视频中重建人头：

```
python train_mono_avatar.py \
    --model_name <MODEL_NAME> \
    --config <CONFIG_PATH> \ 
    --root_path <DATASET_PATH> \
    --workspace <EXP_DIR> \
    --name <EXP_NAME>
    [--resume]
```

- `--model_name`

  我们提供的头部头像重建方法有：`["FateAvatar", "GaussianAvatars", "FlashAvatar", "SplattingAvatar", "MonoGaussianAvatar"]`。

- `--config`

  配置文件可以在 `./config` 文件夹中找到，我们为每种方法提供了 YAML 文件。

- `--root_path`

  数据集路径请确保路径字符串包含 `imavatar` 或 `insta`，以指定加载哪种数据集类型。

- `--workspace`

  实验保存路径。训练日志、快照、检查点以及各种媒体文件将被组织在该目录下。

- `--name`

	每次实验的标识符。

- `--resume`

	如果启用，将从 `workspace` 中的现有检查点继续训练。

使用以下命令从补全框架生成伪数据：

```
python train_generate_pseudo.py \
    --model_name <MODEL_NAME> \
    --config <CONFIG_PATH> \ 
    --workspace <EXP_DIR> \
    --name <EXP_NAME>
```

使用以下命令对单目视频中重建的人头进行补全：

```
python train_full_avatar.py \
    --model_name <MODEL_NAME> \
    --config <CONFIG_PATH> \ 
    --root_path <DATASET_PATH> \
    --workspace <EXP_DIR> \
    --name <EXP_NAME>
```

使用以下命令对重建好的头部头像进行神经烘焙：

```
python train_neural_baking.py \
    --config <CONFIG_PATH> \ 
    --root_path <DATASET_PATH> \
    --workspace <EXP_DIR> \
    --name <EXP_NAME> \
    [--use_full_head_resume]
```

> [!NOTE]
> 神经烘焙仅在 `FateAvatar` 中可用。

- `--use_full_head_resume`

  如果为 True，它将从 `--workspace` 中烘焙通过补全框架处理的检查点。

以下是一个训练示例：

```bash
MODEL_NAME="FateAvatar"
CONFIG_PATH="./config/fateavatar.yaml"
DATASET_PATH="./data/insta/bala"
EXP_DIR="./workspace/insta/bala"
EXP_NAME="fateavatar_insta_bala"

python train_mono_avatar.py --model_name $MODEL_NAME --config $CONFIG_PATH --root_path $DATASET_PATH \
    --workspace $EXP_DIR --name $EXP_NAME

python train_generate_pseudo.py --model_name $MODEL_NAME --config $CONFIG_PATH --workspace $EXP_DIR \
    --name $EXP_NAME

python train_full_avatar.py --model_name $MODEL_NAME --config $CONFIG_PATH --root_path $DATASET_PATH \
    --workspace $EXP_DIR --name $EXP_NAME

python train_neural_baking.py --config $CONFIG_PATH --root_path $DATASET_PATH \
    --workspace $EXP_DIR --name $EXP_NAME
```

对于烘焙后的人头，我们提供了一个纹理编辑脚本。您还可以在 `./edit_assets` 中自定义额外的编辑项目。

```
python avatar_edit_baked.py \
    --config <CONFIG_PATH> \ 
    --root_path <DATASET_PATH> \
    --workspace <EXP_DIR> \
    --name <EXP_NAME> \
    [--use_full_head_resume]
```

- `--use_full_head_resume`

  如果为 True，它将对来自 `--workspace` 的 **烘焙后的** 全头检查点进行编辑。

我们还提供了一个cross-reenact脚本：

```
python avatar_reenact.py \
    --config <CONFIG_PATH> \
    --model_name <MODEL_NAME> \
    --dst_path <DATASET_PATH> \
    --workspace <EXP_DIR> \
    --name <EXP_NAME> \
```

- `--dst_path`:

  目标头部头像数据集的路径。

> [!NOTE]
> Cross-reenactment仅支持来自 **相同数据集类型** 的源头部头像和目标头部头像之间的操作。

## 查看器

我们定制化了一个本地GUI，用于与重建后的头部化身进行交互：

```
python avatar_gui.py \
	--model_name <MODEL_NAME> \
	--config <CONFIG_PATH> \
	--workspace <EXP_DIR> \
	--name <EXP_NAME> \
	[--ckpt_path <CKPT_PATH>] \
	[--use_full_head_resume] \
	[--use_baked_resume]
```

- `--ckpt_path`:

	如果提供了 `ckpt_path`，它将从其指定的路径加载检查点。

- `--use_full_head_resume`:

	如果为 True，它将自动从 `--workspace` 加载全头检查点。

- `--use_baked_resume`:

	如果为 True，它将从 `--workspace` 加载烘焙后的检查点。当 `use_full_head_resume` 和 `use_baked_resume` 都为 True 时，它将加载神经烘焙后的全头检查点（如果存在）。

#### 全头补全

<div align=center>
  <img src="../assets/render.gif">
</div>

#### 贴图编辑

<div align=center>
  <img src="../assets/lty.gif">
</div>

#### 风格迁移

<div align=center>
  <img src="../assets/the_wave.gif">
</div>

## 致谢

本仓库基于 [3DGS](https://github.com/gafniguy/4D-Facial-Avatars/issues/57#issuecomment-1744790191) 构建，并结合了多个优秀的开源项目：[3DDFA_V2](https://github.com/gafniguy/4D-Facial-Avatars/issues/57#issuecomment-1744790191)、[SphereHead](https://lhyfst.github.io/spherehead/)、[MODNet](https://github.com/ZHKKKe/MODNet)、[Face-parsing](https://github.com/zllrunning/face-parsing.PyTorch) 和 [GFPGAN](https://github.com/ZHKKKe/MODNet)。GUI 的灵感来自 [DreamGaussian](https://github.com/dreamgaussian/dreamgaussian) 和 [GaussianAvatars](https://github.com/ShenhanQian/GaussianAvatars?tab=readme-ov-file)。

我们感谢 [FlashAvatar](https://github.com/USTC3DV/FlashAvatar-code)、[SplattingAvatar](https://github.com/initialneil/SplattingAvatar)、[MonoGaussianAvatar](https://github.com/yufan1012/MonoGaussianAvatar) 和 [GaussianAvatars](https://github.com/ShenhanQian/GaussianAvatars) 发布的代码，其为我们的实验提供了很大的帮助。

感谢所有作者的杰出工作。

## 许可证

本代码根据 MIT 许可证分发。请注意，我们的代码依赖于其他仓库，每个仓库都有其各自的许可证，也必须遵守（*例如* [3DGS 许可证](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md)）。

请注意，`./model/baseline` 目录下的代码基于相应论文的原始实现（[FlashAvatar](https://github.com/USTC3DV/FlashAvatar-code)、[SplattingAvatar](https://github.com/initialneil/SplattingAvatar)、[MonoGaussianAvatar](https://github.com/yufan1012/MonoGaussianAvatar) 和 [GaussianAvatars](https://github.com/ShenhanQian/GaussianAvatars)）。

## 引用

如果您在研究中发现我们的论文或代码有帮助，请使用以下 BibTex 条目进行引用：

```bibtex
@inproceedings{zhang2025fate,
      title={FATE: Full-head Gaussian Avatar with Textural Editing from Monocular Video}, 
      author={Zhang, Jiawei and Wu, Zijian and Liang, Zhiyang and Gong, Yicheng and Hu, Dongfang and Yao, Yao and Cao, Xun and Zhu, Hao},
      journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition}
      year={2025},
}
```

如有问题、意见或错误报告，请邮件 jiaweizhang DOT fufu AT gmail DOT com，或在 GitHub 上提出issue。
