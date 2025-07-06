# NestQuant: Post-Training Integer-Nesting Quantization for On-Device DNN

## Environment

The `nestquant` demo requires PyTorch and the `squant function` of `SQuant`.

For CNNs quantization, we adopt the `pytorchcv` library, and for ViTs quantization, we adopt the `timm` library:

```bash
pip install pytorchcv==0.0.51
pip install timm==0.6.11
```

## Usage

CNN-based quantization with part-bit model inference evaluation.

```bash
python main_nestquant_cnn.py --model resnet18 --dataset_path /Datasets/ILSVRC2012/ --mode squant_ -wb 8 -ab 8 -nb 4 -s 25 --batch_size 128 --nest --adaround --packbits --nestquant --eval_nestquant --savepath ./nestquant/qmodel/
```

ViT-based quantization with part-bit model inference evaluation.

```bash
python main_nestquant_vit.py --model swin_large_patch4_window7_224 --dataset_path /Datasets/ILSVRC2012/ --mode squant_ -wb 8 -ab 8 -nb 4 -s 25 --batch_size 16 --nest --adaround --packbits --nestquant --eval_nestquant --savepath ./nestquant/qmodel/
```

You can recursively set all `nest` hyperparameters to `False` by `.set_nest` in the `NestLinearQuantizer` and `NestConv2dQuantizer` of `quantized_DNN_model_nestquant` for upgrading to full-bit model for simulation.

## Publication

Jianhang Xie, Chuntao Ding, Xiaqing Li, Shenyuan Ren, Yidong Li, Zhichao Lu, “NestQuant: Post-Training Integer-Nesting Quantization for On-Device DNN,” ***IEEE Transactions on Mobile Computing***, pp.1-15, 2025.

URL: [DOI](https://doi.org/10.1109/TMC.2025.3582583), [ArXiv](https://arxiv.org/abs/2506.17870)

```bibtex
@article{xie.tmc2025nestquant,
  author={Xie, Jianhang and Ding, Chuntao and Li, Xiaqing and Ren, Shenyuan and Li, Yidong and Lu, Zhichao},
  journal={IEEE Transactions on Mobile Computing}, 
  title={NestQuant: Post-Training Integer-Nesting Quantization for On-Device DNN}, 
  year={2025},
  volume={},
  number={},
  pages={1-15},
  doi={10.1109/TMC.2025.3582583}
}
```

## Acknowledgement

The implementation is built on top of https://github.com/clevercool/SQuant

The pack-bits tensor modified from https://github.com/Felix-Petersen/distquant
