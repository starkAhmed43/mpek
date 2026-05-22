import torch
from torch import nn

from MTLKcatKM.layers import MLP, PLE


class CachedMPEKRegressor(nn.Module):
    def __init__(
        self,
        protein_dim: int = 1024,
        ligand_dim: int = 300,
        expert_dim: int = 768,
        expert_layers: int = 1,
        num_experts: int = 4,
        ple_layers: int = 1,
        dropout: float = 0.2,
        tower_layers: int = 3,
        tower_hidden: int = 128,
        tower_dropout: float = 0.0,
    ):
        super().__init__()
        self.protein_projection = nn.Linear(protein_dim, ligand_dim)
        self.multi_block = PLE(
            experts_in=ligand_dim * 2,
            experts_out=expert_dim,
            experts_hidden=expert_dim,
            expert_hid_layer=expert_layers,
            dropout_rate=dropout,
            num_experts=num_experts,
            num_tasks=1,
            num_ple_layers=ple_layers,
        )
        self.tower = MLP(
            in_size=expert_dim,
            hidden_size=tower_hidden,
            out_size=1,
            layer_num=tower_layers,
            dropout_rate=tower_dropout,
        )

    def forward(self, protein, ligand):
        protein = self.protein_projection(protein)
        x = torch.cat([ligand, protein], dim=1)
        tower_input = self.multi_block(x)[0]
        return self.tower(tower_input)
