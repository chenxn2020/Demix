# Demix
Code and datasets for paper "Negative Sampling with Adaptive Denoising
Mixup for Knowledge Graph Embedding" accepted by ISWC'23.

## Setup
We check the reproducibility under this environment.

+ Python 3.9.0
+ CUDA 10.1
+ pytorch-lightning 1.6.5

To run the codes, you need to install the requirements:

```bash
git clone https://github.com/DeMix2023/Demix.git
cd Demix

conda create -n demix python=3.9
conda activate demix
pip install -r requirements.txt
```

## Train Demix
You can try our code easily by runing the scripts in ./script, for example:
```bash
bash ./script/run_transe_fb.sh
```
The training process, validation results, and final test results will be printed and saved in the corresponding log file. After training, you can find training logs in ``./wandb``.
We put the trained model state dicts in ``./output``.

## Acknowledgment

The repository benefits greatly from [NeuralKG](https://github.com/zjukg/NeuralKG). Thanks a lot for their excellent work.

## Citation
Please cite our paper if you use our model in your work:
```latex
@inproceedings{Demix,
  title     = {Negative Sampling with Adaptive Denoising Mixup for Knowledge Graph Embedding},
  author    = {Chen, Xiangnan and Zhang, Wen and Yao, Zhen and Chen, Mingyang and Tang, Siliang},
  booktitle = {{ISWC}},
  series    = {Lecture Notes in Computer Science},
  pages     = {253--270},
  publisher = {Springer},
  year      = {2023}
}
```