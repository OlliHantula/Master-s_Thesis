import torch
import torch.nn as nn

def summarize_model(model: nn.Module, show_submodules=False):
    """
    Prints a concise summary of a PyTorch model including:
    - Total number of parameters
    - Number of trainable parameters
    - Number of frozen parameters
    - Optionally suppresses duplicate submodules

    Args:
        model (nn.Module): The PyTorch model to summarize.
        show_submodules (bool): If True, prints all submodules. If False, suppresses duplicate structures.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print("Model Summary:")
    print("-" * 60)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters:    {frozen_params:,}")
    print("-" * 60)

    if show_submodules:
        print("Submodule structure:")
        print(model)
    else:
        printed = set()
        for name, module in model.named_children():
            mod_str = str(module)
            if mod_str not in printed:
                print(f"{name}: {module.__class__.__name__}")
                printed.add(mod_str)
            else:
                print(f"{name}: {module.__class__.__name__} (duplicate structure suppressed)")
