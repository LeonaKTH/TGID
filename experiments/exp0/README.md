# Experiment 0: TGID Gate Comparison and Routing Analysis

This directory contains the minimal EPAL + target-conditioned MoE
instrumentation and analysis scripts for the TGID-Sigmoid vs TGID-Softmax
gate comparison on M2M arguments.

## Train the equal-parameter gate pair

Main TGID run:

```bash
python run.py \
  --mode train \
  --data-name ChFinAnn \
  --model-save-name tgid_chfinann_sigmoid \
  --use-tgid-router \
  --tgid-gate sigmoid \
  --tgid-activation threshold \
  --tgid-variant full \
  --num-experts 4
```

Controlled softmax-threshold run:

```bash
python run.py \
  --mode train \
  --data-name ChFinAnn \
  --model-save-name tgid_chfinann_softmax_t \
  --use-tgid-router \
  --tgid-gate softmax \
  --tgid-activation threshold \
  --tgid-threshold 0.5 \
  --tgid-variant full \
  --num-experts 4
```

Controlled softmax-top-k run:

```bash
python run.py \
  --mode train \
  --data-name ChFinAnn \
  --model-save-name tgid_chfinann_softmax_k \
  --use-tgid-router \
  --tgid-gate softmax \
  --tgid-activation topk \
  --route-top-k 2 \
  --tgid-variant full \
  --num-experts 4
```

Core ablations:

```bash
python run.py --mode train --data-name ChFinAnn --model-save-name tgid_chfinann_core --use-tgid-router --tgid-gate sigmoid --tgid-activation threshold --tgid-variant core --num-experts 4
python run.py --mode train --data-name ChFinAnn --model-save-name tgid_chfinann_no_tc --use-tgid-router --tgid-gate sigmoid --tgid-activation threshold --tgid-variant no_tc --num-experts 4
python run.py --mode train --data-name ChFinAnn --model-save-name tgid_chfinann_type_only --use-tgid-router --tgid-gate sigmoid --tgid-activation threshold --tgid-variant type_only --num-experts 4
python run.py --mode train --data-name ChFinAnn --model-save-name tgid_chfinann_no_init --use-tgid-router --tgid-gate sigmoid --tgid-activation threshold --tgid-variant no_init --num-experts 4
```

If the DuEE-Fin-format data is stored under `Data/Pseudo_Doc2EDAG`, use `Pseudo_Doc2EDAG` as `--data-name`:

```bash
python run.py --mode train --data-name Pseudo_Doc2EDAG --model-save-name tgid_dueefin_sigmoid --use-tgid-router --tgid-gate sigmoid --tgid-activation threshold --tgid-variant full --num-experts 4
python run.py --mode train --data-name Pseudo_Doc2EDAG --model-save-name tgid_dueefin_softmax_t --use-tgid-router --tgid-gate softmax --tgid-activation threshold --tgid-threshold 0.5 --tgid-variant full --num-experts 4
python run.py --mode train --data-name Pseudo_Doc2EDAG --model-save-name tgid_dueefin_softmax_k --use-tgid-router --tgid-gate softmax --tgid-activation topk --route-top-k 2 --tgid-variant full --num-experts 4
python run.py --mode train --data-name Pseudo_Doc2EDAG --model-save-name tgid_dueefin_core --use-tgid-router --tgid-gate sigmoid --tgid-activation threshold --tgid-variant core --num-experts 4
python run.py --mode train --data-name Pseudo_Doc2EDAG --model-save-name tgid_dueefin_no_tc --use-tgid-router --tgid-gate sigmoid --tgid-activation threshold --tgid-variant no_tc --num-experts 4
python run.py --mode train --data-name Pseudo_Doc2EDAG --model-save-name tgid_dueefin_type_only --use-tgid-router --tgid-gate sigmoid --tgid-activation threshold --tgid-variant type_only --num-experts 4
python run.py --mode train --data-name Pseudo_Doc2EDAG --model-save-name tgid_dueefin_no_init --use-tgid-router --tgid-gate sigmoid --tgid-activation threshold --tgid-variant no_init --num-experts 4
```

Replace `ChFinAnn` with the actual dataset folder name and adjust `--model-save-name` for other main table runs.

```bash
python run.py \
  --mode train \
  --data-name Pseudo_Doc2EDAG \
  --model-save-name tgid_pseudo_doc2edag_sigmoid \
  --use-tgid-router \
  --tgid-gate sigmoid \
  --num-experts 4
```

```bash
python run.py \
  --mode train \
  --data-name Pseudo_Doc2EDAG \
  --model-save-name tgid_pseudo_doc2edag_softmax \
  --use-tgid-router \
  --tgid-gate softmax \
  --num-experts 4
```

During evaluation, route probabilities can be dumped with:

```bash
python run.py \
  --mode eval \
  --checkpoint tgid_pseudo_doc2edag_sigmoid_50.pkl \
  --data-name Pseudo_Doc2EDAG \
  --use-tgid-router \
  --tgid-gate sigmoid \
  --dump-route-prob \
  --route-dump-path outputs/exp0/tgid_sigmoid_routes.pt
```

The route dump records are dictionaries with `doc_id`, `arg_key`, `arg_text`,
`gold_event_type_ids`, M2M bucket flags, `category`, and `route_prob`.

## M2M-All F1 from raw.pt

```bash
python experiments/exp0/eval_raw_m2m_f1.py \
  --raw-result raw.pt \
  --data-name Pseudo_Doc2EDAG \
  --split test \
  --output outputs/exp0/pseudo_doc2edag_sigmoid_m2m_f1.json
```

## Data statistics

```bash
python experiments/exp0/stat_multi_target_args.py \
  --gold_data Data/ChFinAnn/test.json \
  --output outputs/exp0/data_stats.json
```

## Routing overlap and entropy

```bash
python experiments/exp0/compute_routing_overlap.py \
  --route_dump outputs/exp0/tgid_sigmoid_routes.pt \
  --gold_data Data/Pseudo_Doc2EDAG/test.json \
  --output outputs/exp0/tgid_sigmoid_routing.json \
  --top_k 2
```

This writes routing overlap, TargetSensitivity, SoftOverlap, Top1Collision, and
route entropy. Same-type multi-instance routing diagnostics still require the
later instance/probe-conditioned TGID decoder; this first gate comparison
diagnoses event-type target pairs.

## Subset F1

```bash
python experiments/exp0/eval_subset_f1.py \
  --gold_data Data/ChFinAnn/test.json \
  --pred_data outputs/epal_softmax_moe/pred_test.json \
  --output outputs/exp0/subset_f1.json
```

Unmatched false positives are counted only in `OVERALL` in this first pass,
because they do not have an unambiguous gold argument category.
