from box import Box

config = {
    "device": "cuda:3",
    "seed": 12306,
    "root": "data",
    "backbone": "resnet50",
    "datasets": [
        "office31",
        "OfficeHome"
    ],
    "pretrain": {
        "enable": True,
        "epochs": {
        "default": 160,
        "ex": {
            "amazon": 120
        }
        },
        "switch_epoch": 40,
        "batch": 31,
        "opt": {
        "lr": 0.001,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "ema": 0.9,
        "alpha": 1
        }
    },
    "finetune": {
        "enable": True,
        "epochs": {
        "default": 160,
        "ex": {
            "amazon": 120
        }
        },
        "opt": {
        "lr": 0.001,
        "momentum": 0.9,
        "weight_decay": 0.0005
        },
        "alpha": 0.5,
        "shot": 3
    }
}

cfg = Box(config)
device = cfg.device
