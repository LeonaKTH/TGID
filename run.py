import importlib
import os
from pathlib import Path
import logging
import argparse

OMP_NUM_THREADS_WAS_FIXED = False


def sanitize_omp_num_threads():
    global OMP_NUM_THREADS_WAS_FIXED
    omp_num_threads = os.environ.get('OMP_NUM_THREADS')
    if omp_num_threads is None:
        return

    try:
        if int(omp_num_threads) > 0:
            return
    except ValueError:
        pass

    os.environ['OMP_NUM_THREADS'] = '1'
    OMP_NUM_THREADS_WAS_FIXED = True


sanitize_omp_num_threads()

import torch
from procnet.data_processor.DocEE_processor import DocEEProcessor
from procnet.data_preparer.DocEE_preparer import DocEEPreparer
from procnet.model.DocEE_proxy_node_model import DocEEProxyNodeModel
from procnet.optimizer.basic_optimizer import BasicOptimizer
from procnet.trainer.DocEE_proxy_node_trainer import DocEETrainer
from procnet.metric.DocEE_metric import DocEEMetric
from procnet.conf.DocEE_conf import DocEEConfig
from procnet.conf.global_config_manager import GlobalConfigManager

importlib.reload(logging)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO, datefmt='%I:%M:%S')
if OMP_NUM_THREADS_WAS_FIXED:
    logging.warning('Invalid OMP_NUM_THREADS detected. Resetting it to 1.')


def get_config() -> DocEEConfig:
    config = DocEEConfig()
    config.model_save_name = 'exp_du3'
    config.node_size = 512
    config.max_epochs = 50
    config.data_loader_shuffle = True
    config.model_name = "hfl/chinese-roberta-wwm-ext"
    config.device = torch.device('cuda')
    config.max_len = 510

    config.learning_rate_slow = 2e-5
    config.learning_rate_fast = 5e-5
    config.gradient_accumulation_steps = 8
    config.temperature = 0.5
    config.data_name = 'ChFinAnn' # ChFinAnn, DuEE_Fin

    return config


def build_model(config: DocEEConfig, dee_pre: DocEEPreparer) -> DocEEProxyNodeModel:
    model = DocEEProxyNodeModel(config=config, preparer=dee_pre)

    # Increase vocabulary for the added uppercase tokens.
    model.language_model.resize_token_embeddings(new_num_tokens=len(dee_pre.tokenizer.vocab) + 26)
    return model


def resolve_checkpoint_path(checkpoint: str) -> Path:
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.is_absolute():
        return checkpoint_path
    return GlobalConfigManager.current_path / checkpoint_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'eval'], default='train')
    parser.add_argument('--data-name', type=str, default=None,
                        help='Dataset name under Data/, e.g. pseudo_FNDEE, DuEE_Fin, ChFinAnn.')
    parser.add_argument('--model-save-name', type=str, default=None,
                        help='Prefix used for checkpoints and Result/ outputs.')
    parser.add_argument('--device', type=str, default=None,
                        help='Override device, e.g. cpu, cuda, or cuda:0.')
    parser.add_argument('--max-epochs', type=int, default=None,
                        help='Override training epochs.')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint path for evaluation, e.g. Result/exp_ch3/exp_ch3_58.pkl')
    parser.add_argument('--true-bio', action='store_true',
                        help='Use gold BIO labels during evaluation. Only valid in eval mode.')
    parser.add_argument('--use-softmax-moe', action='store_true',
                        help='Enable Experiment 0 EPAL + Softmax-MoE baseline.')
    parser.add_argument('--use-tgid-router', action='store_true',
                        help='Enable target-conditioned MoE router for TGID gate comparison.')
    parser.add_argument('--tgid-gate', choices=['sigmoid', 'softmax'], default='softmax',
                        help='Gate function for --use-tgid-router. Use sigmoid for TGID-Sigmoid and softmax for the equal-parameter baseline.')
    parser.add_argument('--tgid-activation', choices=['threshold', 'topk'], default='threshold',
                        help='Expert activation strategy after gate scoring.')
    parser.add_argument('--tgid-threshold', type=float, default=None,
                        help='Threshold for --tgid-activation threshold.')
    parser.add_argument('--disable-learnable-tgid-threshold', action='store_true',
                        help='Keep the TGID threshold fixed instead of learning it after initialization.')
    parser.add_argument('--tgid-variant', choices=['full', 'core', 'no_tc', 'type_only', 'no_init'], default='full',
                        help='TGID ablation variant.')
    parser.add_argument('--tgid-tc-weight', type=float, default=None,
                        help='Weight for TGID target-contrastive routing regularizer.')
    parser.add_argument('--tgid-tc-temperature', type=float, default=None,
                        help='Temperature for the TGID target-contrastive routing regularizer.')
    parser.add_argument('--tgid-warmup-epochs', type=int, default=None,
                        help='Number of TGID curriculum warm-up epochs without target-conditioned routing.')
    parser.add_argument('--disable-tgid-curriculum', action='store_true',
                        help='Disable TGID two-stage curriculum warm-up.')
    parser.add_argument('--event-type-threshold', type=float, default=None,
                        help='Inference threshold for keeping a non-null event type. Use >0 to improve precision.')
    parser.add_argument('--role-threshold', type=float, default=None,
                        help='Inference threshold for keeping a non-null role argument. Use >0 to improve precision.')
    parser.add_argument('--num-experts', type=int, default=None,
                        help='Number of TGID/Softmax-MoE experts.')
    parser.add_argument('--route-top-k', type=int, default=None,
                        help='Top-k used by routing analysis scripts.')
    parser.add_argument('--dump-route-prob', action='store_true',
                        help='Dump Experiment 0 route probabilities during eval/train evaluation.')
    parser.add_argument('--route-dump-path', type=str, default=None,
                        help='Path for dumped route probabilities.')
    return parser.parse_args()


def run():
    sanitize_omp_num_threads()
    args = parse_args()
    config = get_config()
    if args.data_name is not None:
        config.data_name = args.data_name
    if args.model_save_name is not None:
        config.model_save_name = args.model_save_name
    if args.device is not None:
        config.device = torch.device(args.device)
    if args.max_epochs is not None:
        config.max_epochs = args.max_epochs
    config.use_softmax_moe = args.use_softmax_moe
    config.use_tgid_router = args.use_tgid_router or args.use_softmax_moe
    config.tgid_gate = 'softmax' if args.use_softmax_moe else args.tgid_gate
    config.tgid_activation = args.tgid_activation
    config.tgid_variant = args.tgid_variant
    if args.tgid_threshold is not None:
        config.tgid_threshold = args.tgid_threshold
    if args.disable_learnable_tgid_threshold:
        config.tgid_learnable_threshold = False
    if args.tgid_tc_weight is not None:
        config.tgid_tc_weight = args.tgid_tc_weight
    if args.tgid_tc_temperature is not None:
        config.tgid_tc_temperature = args.tgid_tc_temperature
    if config.tgid_variant in {'core', 'no_tc'}:
        config.tgid_tc_weight = 0.0
    if args.tgid_warmup_epochs is not None:
        config.tgid_warmup_epochs = args.tgid_warmup_epochs
    if args.disable_tgid_curriculum:
        config.tgid_curriculum = False
    if config.tgid_variant == 'core':
        config.tgid_curriculum = False
    if args.event_type_threshold is not None:
        config.event_type_threshold = args.event_type_threshold
    if args.role_threshold is not None:
        config.role_threshold = args.role_threshold
    if args.num_experts is not None:
        config.num_experts = args.num_experts
    if args.route_top_k is not None:
        config.route_top_k = args.route_top_k
    config.dump_route_prob = args.dump_route_prob
    if args.route_dump_path is not None:
        config.route_dump_path = args.route_dump_path
    logging.info('save_name = {}'.format(config.model_save_name))
    dee_pro = DocEEProcessor(config.data_name)
    dee_pre = DocEEPreparer(config=config, processor=dee_pro)
    train_loader, dev_loader, test_loader = dee_pre.get_loader_for_flattened_fragment_before_event()
    metric = DocEEMetric(preparer=dee_pre)

    if args.mode == 'eval':
        if args.checkpoint is None:
            raise ValueError('--checkpoint is required when --mode eval is used')
        checkpoint_path = resolve_checkpoint_path(args.checkpoint)
        logging.info('Loading checkpoint from %s', checkpoint_path)
        model = torch.load(checkpoint_path, map_location=config.device)
    else:
        model = build_model(config=config, dee_pre=dee_pre)

    model.to(config.device)

    optimizer = BasicOptimizer(config=config, model=model)

    trainer = DocEETrainer(config=config,
                        model=model,
                        optimizer=optimizer,
                        preparer=dee_pre,
                        metric=metric,
                        train_loader=train_loader,
                        dev_loader=dev_loader,
                        test_loader=test_loader,
                        )
    if args.mode == 'train':
        trainer.train()
    else:
        score_result, _ = trainer.eval(true_bio=args.true_bio)
        logging.info('Eval score result = %s', score_result)


if __name__ == '__main__':
    run()
