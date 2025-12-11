# src/config/arch_space.py

ARCH_SPACE = {
    "opt3_tiny": {
        "num_channels": [16, 16, 32],
        "num_blocks":   [2, 1, 1],
        "multi_scale":  False,
    },
    "opt3_base": {
        "num_channels": [16, 32, 64],
        "num_blocks":   [2, 2, 2],
        "multi_scale":  False,
    },
    "opt3_ms": {
        "num_channels": [16, 32, 64],
        "num_blocks":   [2, 2, 2],
        "multi_scale":  True,
    },
    "opt3_deep": {
        "num_channels": [16, 32, 64],
        "num_blocks":   [3, 3, 3],
        "multi_scale":  True,
    },
    "opt4_tiny": {
        "num_channels": [16, 16, 32, 32],
        "num_blocks":   [1, 1, 1, 1],
        "multi_scale":  False,
    },
    "opt4_base": {
        "num_channels": [16, 32, 64, 128],
        "num_blocks":   [2, 2, 2, 2],
        "multi_scale":  False,
    },
    "opt4_ms": {
        "num_channels": [16, 32, 64, 128],
        "num_blocks":   [2, 2, 2, 2],
        "multi_scale":  True,
    },
    "opt4_deep": {
        "num_channels": [16, 32, 64, 128],
        "num_blocks":   [3, 3, 3, 3],
        "multi_scale":  True,
    },
}
