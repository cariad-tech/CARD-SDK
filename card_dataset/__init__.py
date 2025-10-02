# card_dataset/__init__.py
__all__ = ["UnifiedCardDataset"]

def __getattr__(name):
    if name == "UnifiedCardDataset":
        try:
            # Import on demand so importing the package doesn't require torch immediately.
            from .dataloader import UnifiedCardDataset
        except ModuleNotFoundError as e:
            if e.name == "torch":
                raise ModuleNotFoundError(
                    "UnifiedCardDataset requires PyTorch. Install with:\n\n"
                    "  pip install card-sdk[torch]\n"
                    "or install a specific CUDA build of torch from pytorch.org."
                ) from e
            raise
        return UnifiedCardDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
