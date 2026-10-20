"""
PyTorch Neural Network architecture definition for Ban Quality Classification.
MLP with ReLU activation and single logit output.
"""

from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn


class BanQualityMLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) for tire quality classification.
    
    Architecture:
        Input(5) -> Linear(5, 32) -> ReLU -> Linear(32, 16) -> ReLU -> Dropout(0.10) -> Linear(16, 1)
    """
    def __init__(
        self,
        input_dim: int = 5,
        hidden_layers: Optional[List[int]] = None,
        dropout: float = 0.10,
    ):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [32, 16]
            
        layers: List[nn.Module] = []
        current_dim = input_dim
        
        for i, h_dim in enumerate(hidden_layers):
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.ReLU())
            current_dim = h_dim
            
        if dropout > 0.0:
            layers.append(nn.Dropout(p=dropout))
            
        layers.append(nn.Linear(current_dim, 1))
        
        self.network = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning raw logits.
        x: Tensor of shape (batch_size, input_dim) with dtype float32.
        Returns: Tensor of shape (batch_size, 1).
        """
        return self.network(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute probability of class 1 (Perlu Perbaikan) via Sigmoid.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
        return probs

    def compute_activations(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Compute step-by-step activations and pre-activations through the network layers.

        Args:
            x: Tensor of shape (1, input_dim) or (input_dim,)

        Returns:
            Dictionary with activations, pre-activations, biases, and firing status.
        """
        self.eval()
        if x.dim() == 1:
            x = x.unsqueeze(0)

        with torch.no_grad():
            curr = x
            layer_records = []
            hidden_idx = 1

            i = 0
            while i < len(self.network):
                mod = self.network[i]
                if isinstance(mod, nn.Linear):
                    next_mod = self.network[i + 1] if (i + 1) < len(self.network) else None
                    if isinstance(next_mod, nn.ReLU):
                        # Linear followed by ReLU
                        z = mod(curr)
                        a = next_mod(z)
                        pre_vals = [round(float(v), 4) for v in z[0].tolist()]
                        post_vals = [round(float(v), 4) for v in a[0].tolist()]
                        active_cnt = int((a[0] > 0).sum().item())
                        total_cnt = int(mod.out_features)
                        layer_records.append({
                            "layer_index": hidden_idx,
                            "layer_name": f"Hidden {hidden_idx} (ReLU)",
                            "layer_type": "hidden",
                            "units": total_cnt,
                            "pre_activations": pre_vals,
                            "post_activations": post_vals,
                            "active_count": active_cnt,
                            "total_count": total_cnt,
                            "active_percentage": round((active_cnt / total_cnt) * 100, 1),
                            "biases": [round(float(v), 4) for v in mod.bias.tolist()] if mod.bias is not None else [],
                        })
                        curr = a
                        hidden_idx += 1
                        i += 2
                        continue
                    else:
                        # Output Linear layer
                        z = mod(curr)
                        p = torch.sigmoid(z)
                        pre_vals = [round(float(v), 4) for v in z[0].tolist()]
                        post_vals = [round(float(v), 4) for v in p[0].tolist()]
                        prob_val = float(p[0, 0].item())
                        layer_records.append({
                            "layer_index": hidden_idx,
                            "layer_name": "Output Layer (Sigmoid)",
                            "layer_type": "output",
                            "units": mod.out_features,
                            "pre_activations": pre_vals,
                            "post_activations": post_vals,
                            "active_count": 1 if prob_val >= 0.5 else 0,
                            "total_count": int(mod.out_features),
                            "active_percentage": 100.0 if prob_val >= 0.5 else 0.0,
                            "biases": [round(float(v), 4) for v in mod.bias.tolist()] if mod.bias is not None else [],
                        })
                        curr = z
                        i += 1
                        continue
                elif isinstance(mod, nn.Dropout):
                    # Dropout in eval mode is identity
                    curr = mod(curr)
                    i += 1
                    continue
                else:
                    curr = mod(curr)
                    i += 1

            prob = float(torch.sigmoid(curr[0, 0]).item())
            logit = float(curr[0, 0].item())

            return {
                "input_values": [round(float(v), 4) for v in x[0].tolist()],
                "layers": layer_records,
                "output_logit": round(logit, 4),
                "probability_bad": round(prob, 4),
                "probability_good": round(1.0 - prob, 4),
            }

    def get_weights_summary(self) -> Dict[str, Any]:
        """
        Extract weights and biases for all linear layers to support synaptic visualization.
        """
        layers_info = []
        l_idx = 0
        for mod in self.network:
            if isinstance(mod, nn.Linear):
                layers_info.append({
                    "layer_index": l_idx,
                    "in_features": mod.in_features,
                    "out_features": mod.out_features,
                    "weights": [[round(float(w), 4) for w in row] for row in mod.weight.detach().tolist()],
                    "biases": [round(float(b), 4) for b in mod.bias.detach().tolist()] if mod.bias is not None else [],
                })
                l_idx += 1
        return {"layers": layers_info}


def build_model(config: Optional[Dict[str, Any]] = None) -> BanQualityMLP:
    """
    Factory function to instantiate BanQualityMLP from a configuration dictionary.
    """
    if config is None:
        return BanQualityMLP()
        
    arch = config.get("architecture", {})
    input_dim = arch.get("input_dim", 4)
    hidden_layers = arch.get("hidden_layers", [32, 16])
    dropout = arch.get("dropout", 0.10)
    
    return BanQualityMLP(
        input_dim=input_dim,
        hidden_layers=hidden_layers,
        dropout=dropout,
    )
