import argparse
import logging
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sanitize_omp_num_threads():
    omp_num_threads = os.environ.get("OMP_NUM_THREADS")
    if omp_num_threads is None:
        return
    try:
        if int(omp_num_threads) > 0:
            return
    except ValueError:
        pass
    os.environ["OMP_NUM_THREADS"] = "1"


def parse_args():
    parser = argparse.ArgumentParser(description="Dump Experiment 0 TGID/Softmax-MoE route probabilities.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_name", default=None)
    parser.add_argument("--output", default="outputs/exp0/softmax_moe_routes.pt")
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--tgid_gate", choices=["sigmoid", "softmax"], default="softmax")
    parser.add_argument("--true_bio", action="store_true")
    return parser.parse_args()


def main():
    sanitize_omp_num_threads()
    import torch
    from procnet.data_processor.DocEE_processor import DocEEProcessor
    from procnet.data_preparer.DocEE_preparer import DocEEPreparer
    from procnet.metric.DocEE_metric import DocEEMetric
    from procnet.optimizer.basic_optimizer import BasicOptimizer
    from procnet.trainer.DocEE_proxy_node_trainer import DocEETrainer
    from run import get_config, resolve_checkpoint_path

    args = parse_args()
    config = get_config()
    if args.data_name is not None:
        config.data_name = args.data_name
    config.use_tgid_router = True
    config.tgid_gate = args.tgid_gate
    config.num_experts = args.num_experts
    config.dump_route_prob = True
    config.route_dump_path = args.output

    processor = DocEEProcessor(config.data_name)
    preparer = DocEEPreparer(config=config, processor=processor)
    train_loader, dev_loader, test_loader = preparer.get_loader_for_flattened_fragment_before_event()
    metric = DocEEMetric(preparer=preparer)

    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    model = torch.load(checkpoint_path, map_location=config.device)
    model.to(config.device)
    model.device = config.device
    if hasattr(model, "config"):
        model.config.dump_route_prob = True
        model.config.route_dump_path = args.output

    optimizer = BasicOptimizer(config=config, model=model)
    trainer = DocEETrainer(
        config=config,
        model=model,
        optimizer=optimizer,
        preparer=preparer,
        metric=metric,
        train_loader=train_loader,
        dev_loader=dev_loader,
        test_loader=test_loader,
    )
    trainer.eval(test_loader=test_loader, true_bio=args.true_bio)
    logging.info("Saved route dump to %s", args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")
    main()
