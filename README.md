# Target-Conditioned Gating for Many-to-Many Argument Assignment in Document-Level Event Extraction
Code for the paper: 
["Target-Conditioned Gating for Many-to-Many Argument Assignment in Document-Level Event Extraction"]

## Dependency

```
torch
transformers
numpy
tqdm
```
## Dataset

Please unzip `data.zip` at `Data/` as:

```bash
>> cd Data
>> unzip data.zip
```

`data.zip` is from [Data.zip](https://github.com/dolphin-zs/Doc2EDAG/raw/master/Data.zip) of [Doc2EDAG](https://github.com/dolphin-zs/Doc2EDAG).

## Model
Please download the model at `chinese_roberta_wwm_ext/`.
Moedl is from [Huggingface](https://huggingface.co/hfl/chinese-roberta-wwm-ext)

## Usage

To train the model:

```bash
>> python run.py
```


