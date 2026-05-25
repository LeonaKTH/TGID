# Joint Learning Event-specific Probe and Argument Library with Differential Optimization for Document-Level Multi-Event Extraction
Code for the paper: 
["Joint Learning Event-specific Probe and Argument Library with Differential Optimization for Document-Level Multi-Event Extraction"]

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

To evaluate a saved checkpoint:

```bash
>> python run.py --mode eval --checkpoint exp_du3_50.pkl
```

## Experiment 0: EPAL Representation Separability

The original MoE-style Experiment 0 measures routing overlap and routing entropy. EPAL does not expose an MoE router, so this repository provides an EPAL-specific version that measures empirical M2M argument entanglement with:

- cross-type multi-target argument statistics
- subset-level F1 on M2M / non-M2M / single-event / multi-event documents
- proxy representation separability with cosine similarity, a linear probe, and optional t-SNE coordinates

Run dataset statistics only:

```bash
>> python3 scripts/experiment0_epal.py --data-name ChFinAnn --split test --device cpu
```

Run subset F1 from an evaluation `raw.pt`:

```bash
>> python3 scripts/experiment0_epal.py --data-name ChFinAnn --split test --raw-result raw.pt
```

Run representation separability from a checkpoint:

```bash
>> python3 scripts/experiment0_epal.py --data-name ChFinAnn --split test --checkpoint exp_du3_50.pkl --true-bio
```

Outputs are written to `Result/experiment0_epal/summary.json`. If `scikit-learn` is installed, the script also reports linear-probe macro-F1 and writes `proxy_tsne.csv`.
