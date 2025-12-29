# llm-code-memorization

## EDISS node instruction:

!!Important, create a conda environment in /ediss_data/ partition, otherwise there's not enough space
```
conda create --prefix=pytorch-gpu python=3.10
```

Active the environment created
Install these libraries
```
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install mkl=2024.0.0
pip install -r requirements.txt 
```
