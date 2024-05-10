# AI tape detect
AI automatic detection for wine box.
### 环境 
1. requirement

```
  Python >=3.8
  CUDA >=11.2
  torch 1.9.1
  torchaudio 0.9.1
  torchvision 0.10.1
```

2. 创建conda环境

```
指定目录创建conda环境 --prefix
conda create –-prefix=path/to/your/name python==3.8
conda activate name
```

3. 安装依赖

```
# 无GPU版本
# pip install torch==1.9.1+cpu torchvision==0.10.1+cpu torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

# GPU版本，对应CUDA=11.1
# pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

# GPU版本，对应CUDA=10.2
pip install torch==1.9.1+cu102 torchvision==0.10.1+cu102 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
