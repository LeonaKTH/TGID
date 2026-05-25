import torch
from torch import nn
import torch.nn.functional as F


class TargetConditionedMoERouter(nn.Module):
    """Type/role-conditioned MoE router for TGID gate comparisons.

    The module is intentionally small so it can be inserted after EPAL's
    argument/proxy representation without changing the original decoding code.
    """

    def __init__(
        self,
        hidden_size: int,
        num_event_types: int,
        num_event_roles: int,
        num_experts: int,
        dropout: float = 0.1,
        gate: str = "softmax",
        activation: str = "threshold",
        threshold: float = None,
        learnable_threshold: bool = True,
        top_k: int = 2,
        use_role_condition: bool = True,
        use_schema_condition: bool = True,
        event_role_mask=None,
    ):
        super().__init__()
        if gate not in {"softmax", "sigmoid"}:
            raise ValueError("gate must be 'softmax' or 'sigmoid'")
        if activation not in {"threshold", "topk"}:
            raise ValueError("activation must be 'threshold' or 'topk'")
        self.hidden_size = hidden_size
        self.num_event_types = num_event_types
        self.num_event_roles = num_event_roles
        self.num_experts = num_experts
        self.gate = gate
        self.activation = activation
        threshold_value = 1.0 / float(num_experts) if threshold is None else float(threshold)
        if learnable_threshold:
            self.threshold = nn.Parameter(torch.tensor(threshold_value, dtype=torch.float))
        else:
            self.register_buffer("threshold", torch.tensor(threshold_value, dtype=torch.float))
        self.top_k = top_k
        self.use_role_condition = use_role_condition
        self.use_schema_condition = use_schema_condition

        self.type_embedding = nn.Embedding(num_event_types, hidden_size)
        self.role_embedding = nn.Embedding(num_event_roles, hidden_size)
        self.condition_norm = nn.LayerNorm(hidden_size)
        self.router = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_experts),
        )
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, hidden_size),
                )
                for _ in range(num_experts)
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_size)
        if event_role_mask is None:
            event_role_mask = torch.zeros(num_event_types, num_event_roles, dtype=torch.float)
        else:
            event_role_mask = torch.as_tensor(event_role_mask, dtype=torch.float)
        if event_role_mask.shape != (num_event_types, num_event_roles):
            raise ValueError(
                "event_role_mask must have shape ({}, {}), got {}".format(
                    num_event_types,
                    num_event_roles,
                    tuple(event_role_mask.shape),
                )
            )
        self.register_buffer("event_role_mask", event_role_mask)

    def build_target_condition(self):
        type_ids = torch.arange(self.num_event_types, device=self.event_role_mask.device)
        type_emb = self.type_embedding(type_ids)
        if not self.use_role_condition:
            return self.condition_norm(type_emb)

        role_ids = torch.arange(self.num_event_roles, device=self.event_role_mask.device)
        role_emb = self.role_embedding(role_ids)
        if self.use_schema_condition:
            role_weight = self.event_role_mask
        else:
            role_weight = torch.ones_like(self.event_role_mask)
        role_count = role_weight.sum(dim=-1, keepdim=True).clamp_min(1.0)
        role_condition = role_weight @ role_emb / role_count
        return self.condition_norm(type_emb + role_condition)

    def compute_gate_score(self, route_logits: torch.Tensor):
        if self.gate == "softmax":
            return F.softmax(route_logits, dim=-1)
        return torch.sigmoid(route_logits)

    def apply_activation(self, gate_score: torch.Tensor):
        if self.activation == "topk":
            k = min(max(int(self.top_k), 1), gate_score.shape[-1])
            topk_index = torch.topk(gate_score, k=k, dim=-1).indices
            mask = torch.zeros_like(gate_score)
            mask.scatter_(-1, topk_index, 1.0)
        else:
            threshold = self.threshold.clamp(0.0, 1.0)
            mask = (gate_score >= threshold).float()
            has_active = mask.sum(dim=-1, keepdim=True) > 0
            top1_index = torch.argmax(gate_score, dim=-1, keepdim=True)
            fallback = torch.zeros_like(gate_score)
            fallback.scatter_(-1, top1_index, 1.0)
            mask = torch.where(has_active, mask, fallback)

        activated = gate_score * mask
        normalizer = activated.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        if self.gate == "sigmoid":
            normalizer = normalizer.detach()
        return activated / normalizer

    def forward(self, arg_repr: torch.Tensor):
        """Route argument representations.

        Args:
            arg_repr: [batch_size, num_args, hidden_size]

        Returns:
            routed_repr: [batch_size, num_args, num_event_types, hidden_size]
            route_prob: [batch_size, num_args, num_event_types, num_experts]
            route_logits: [batch_size, num_args, num_event_types, num_experts]
        """
        batch_size, num_args, _ = arg_repr.shape
        target_condition = self.build_target_condition().to(arg_repr.device)
        target_condition = target_condition.view(1, 1, self.num_event_types, self.hidden_size)
        target_condition = target_condition.expand(batch_size, num_args, -1, -1)

        arg_by_type = arg_repr.unsqueeze(2).expand(-1, -1, self.num_event_types, -1)
        route_input = torch.cat([arg_by_type, target_condition], dim=-1)
        route_logits = self.router(route_input)
        gate_score = self.compute_gate_score(route_logits)
        route_prob = self.apply_activation(gate_score)

        expert_outputs = torch.stack([expert(arg_repr) for expert in self.experts], dim=2)
        expert_outputs = expert_outputs.unsqueeze(2)
        routed_repr = torch.sum(route_prob.unsqueeze(-1) * expert_outputs, dim=3)
        routed_repr = self.output_norm(routed_repr + arg_by_type)
        return routed_repr, route_prob, route_logits


class SoftmaxMoERouter(TargetConditionedMoERouter):
    """Backward-compatible name for the original Experiment 0 Softmax router."""

    def __init__(self, hidden_size: int, num_event_types: int, num_experts: int, dropout: float = 0.1):
        super().__init__(
            hidden_size=hidden_size,
            num_event_types=num_event_types,
            num_event_roles=1,
            num_experts=num_experts,
            dropout=dropout,
            gate="softmax",
            activation="topk",
            event_role_mask=torch.zeros(num_event_types, 1),
        )
