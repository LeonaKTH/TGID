from dataclasses import dataclass
from procnet.conf.basic_conf import BasicConfig


@dataclass
class DocEEConfig(BasicConfig):
    node_size: int = None
    proxy_slot_num: int = None
    use_softmax_moe: bool = False
    use_tgid_router: bool = False
    tgid_gate: str = "softmax"
    tgid_activation: str = "threshold"
    tgid_variant: str = "full"
    tgid_threshold: float = None
    tgid_learnable_threshold: bool = True
    tgid_tc_weight: float = 0.01
    tgid_tc_temperature: float = 0.5
    tgid_curriculum: bool = True
    tgid_warmup_epochs: int = 5
    event_type_threshold: float = 0.0
    role_threshold: float = 0.0
    num_experts: int = 4
    route_top_k: int = 2
    dump_route_prob: bool = False
    route_dump_path: str = "outputs/exp0/softmax_moe_routes.pt"
