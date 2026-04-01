from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterable


FULL_EXPERIMENT_DATASETS = [
    "ETTh1",
    "ETTh2",
    "ETTm1",
    "ETTm2",
    "solar_AL",
    "weather",
    "exchange_rate",
    "electricity",
]
FULL_EXPERIMENT_HORIZONS = [96, 192, 336, 720]

DATASET_NAME_ALIASES = {
    "etth1": "ETTh1",
    "etth2": "ETTh2",
    "ettm1": "ETTm1",
    "ettm2": "ETTm2",
    "solar": "solar_AL",
    "solar_al": "solar_AL",
    "weather": "weather",
    "exchange": "exchange_rate",
    "exchange_rate": "exchange_rate",
    "exchange-rate": "exchange_rate",
    "electricity": "electricity",
}

DLINEAR_SCRIPT_URLS = {
    "ETTh1": "https://raw.githubusercontent.com/honeywell21/DLinear/main/scripts/EXP-LongForecasting/Linear/etth1.sh",
    "ETTh2": "https://raw.githubusercontent.com/honeywell21/DLinear/main/scripts/EXP-LongForecasting/Linear/etth2.sh",
    "ETTm1": "https://raw.githubusercontent.com/honeywell21/DLinear/main/scripts/EXP-LongForecasting/Linear/ettm1.sh",
    "ETTm2": "https://raw.githubusercontent.com/honeywell21/DLinear/main/scripts/EXP-LongForecasting/Linear/ettm2.sh",
    "weather": "https://raw.githubusercontent.com/honeywell21/DLinear/main/scripts/EXP-LongForecasting/Linear/weather.sh",
    "exchange_rate": "https://raw.githubusercontent.com/honeywell21/DLinear/main/scripts/EXP-LongForecasting/Linear/exchange_rate.sh",
    "electricity": "https://raw.githubusercontent.com/honeywell21/DLinear/main/scripts/EXP-LongForecasting/Linear/electricity.sh",
}

PATCHTST_SCRIPT_URLS = {
    "ETTh1": "https://raw.githubusercontent.com/yuqinie98/PatchTST/main/PatchTST_supervised/scripts/PatchTST/etth1.sh",
    "ETTh2": "https://raw.githubusercontent.com/yuqinie98/PatchTST/main/PatchTST_supervised/scripts/PatchTST/etth2.sh",
    "ETTm1": "https://raw.githubusercontent.com/yuqinie98/PatchTST/main/PatchTST_supervised/scripts/PatchTST/ettm1.sh",
    "ETTm2": "https://raw.githubusercontent.com/yuqinie98/PatchTST/main/PatchTST_supervised/scripts/PatchTST/ettm2.sh",
    "weather": "https://raw.githubusercontent.com/yuqinie98/PatchTST/main/PatchTST_supervised/scripts/PatchTST/weather.sh",
    "electricity": "https://raw.githubusercontent.com/yuqinie98/PatchTST/main/PatchTST_supervised/scripts/PatchTST/electricity.sh",
}

TQNET_SCRIPT_URLS = {
    "ETTh1": "https://raw.githubusercontent.com/ACAT-SCUT/TQNet/master/scripts/TQNet/etth1.sh",
    "ETTh2": "https://raw.githubusercontent.com/ACAT-SCUT/TQNet/master/scripts/TQNet/etth2.sh",
    "ETTm1": "https://raw.githubusercontent.com/ACAT-SCUT/TQNet/master/scripts/TQNet/ettm1.sh",
    "ETTm2": "https://raw.githubusercontent.com/ACAT-SCUT/TQNet/master/scripts/TQNet/ettm2.sh",
    "weather": "https://raw.githubusercontent.com/ACAT-SCUT/TQNet/master/scripts/TQNet/weather.sh",
    "electricity": "https://raw.githubusercontent.com/ACAT-SCUT/TQNet/master/scripts/TQNet/electricity.sh",
    "solar_AL": "https://raw.githubusercontent.com/ACAT-SCUT/TQNet/master/scripts/TQNet/solar.sh",
}

ITRANSFORMER_SCRIPT_URLS = {
    "ETTh1": "https://raw.githubusercontent.com/thuml/iTransformer/main/scripts/multivariate_forecasting/ETT/iTransformer_ETTh1.sh",
    "ETTh2": "https://raw.githubusercontent.com/thuml/iTransformer/main/scripts/multivariate_forecasting/ETT/iTransformer_ETTh2.sh",
    "ETTm1": "https://raw.githubusercontent.com/thuml/iTransformer/main/scripts/multivariate_forecasting/ETT/iTransformer_ETTm1.sh",
    "ETTm2": "https://raw.githubusercontent.com/thuml/iTransformer/main/scripts/multivariate_forecasting/ETT/iTransformer_ETTm2.sh",
    "exchange_rate": "https://raw.githubusercontent.com/thuml/iTransformer/main/scripts/multivariate_forecasting/Exchange/iTransformer.sh",
    "weather": "https://raw.githubusercontent.com/thuml/iTransformer/main/scripts/multivariate_forecasting/Weather/iTransformer.sh",
    "electricity": "https://raw.githubusercontent.com/thuml/iTransformer/main/scripts/multivariate_forecasting/ECL/iTransformer.sh",
    "solar_AL": "https://raw.githubusercontent.com/thuml/iTransformer/main/scripts/multivariate_forecasting/SolarEnergy/iTransformer.sh",
}

MODERNTCN_SCRIPT_URLS = {
    "ETTh1": "https://raw.githubusercontent.com/luodhhh/ModernTCN/main/ModernTCN-Long-term-forecasting/scripts/ETTh1.sh",
    "ETTh2": "https://raw.githubusercontent.com/luodhhh/ModernTCN/main/ModernTCN-Long-term-forecasting/scripts/ETTh2.sh",
    "ETTm1": "https://raw.githubusercontent.com/luodhhh/ModernTCN/main/ModernTCN-Long-term-forecasting/scripts/ETTm1.sh",
    "ETTm2": "https://raw.githubusercontent.com/luodhhh/ModernTCN/main/ModernTCN-Long-term-forecasting/scripts/ETTm2.sh",
    "exchange_rate": "https://raw.githubusercontent.com/luodhhh/ModernTCN/main/ModernTCN-Long-term-forecasting/scripts/Exchange.sh",
    "weather": "https://raw.githubusercontent.com/luodhhh/ModernTCN/main/ModernTCN-Long-term-forecasting/scripts/weather.sh",
    "electricity": "https://raw.githubusercontent.com/luodhhh/ModernTCN/main/ModernTCN-Long-term-forecasting/scripts/ECL.sh",
}

TIMEMIXER_SCRIPT_URLS = {
    "ETTh1": "https://raw.githubusercontent.com/kwuking/TimeMixer/main/scripts/long_term_forecast/ETT_script/TimeMixer_ETTh1_unify.sh",
    "ETTh2": "https://raw.githubusercontent.com/kwuking/TimeMixer/main/scripts/long_term_forecast/ETT_script/TimeMixer_ETTh2_unify.sh",
    "ETTm1": "https://raw.githubusercontent.com/kwuking/TimeMixer/main/scripts/long_term_forecast/ETT_script/TimeMixer_ETTm1_unify.sh",
    "ETTm2": "https://raw.githubusercontent.com/kwuking/TimeMixer/main/scripts/long_term_forecast/ETT_script/TimeMixer_ETTm2_unify.sh",
    "weather": "https://raw.githubusercontent.com/kwuking/TimeMixer/main/scripts/long_term_forecast/Weather_script/TimeMixer_unify.sh",
    "electricity": "https://raw.githubusercontent.com/kwuking/TimeMixer/main/scripts/long_term_forecast/ECL_script/TimeMixer_unify.sh",
    "solar_AL": "https://raw.githubusercontent.com/kwuking/TimeMixer/main/scripts/long_term_forecast/Solar_script/TimeMixer_unify.sh",
}

TIMEMIXERPP_REFERENCE_URL = "https://github.com/kwuking/TimeMixer"

BACKBONE_PROFILE_LIBRARY: dict[str, dict[str, Any]] = {
    "DLinear": {
        "lookback": 336,
        "model_params": {
            "individual": False,
        },
        "runtime": {
            "epochs": 10,
            "patience": 3,
        },
        "datasets": {
            "ETTh1": {
                "runtime": {"batch_size": 32, "lr": 0.005},
                "source_url": DLINEAR_SCRIPT_URLS["ETTh1"],
                "source_kind": "official_script",
                "source_note": "Exact official ETTh1 DLinear script.",
            },
            "ETTh2": {
                "runtime": {"batch_size": 32, "lr": 0.05},
                "source_url": DLINEAR_SCRIPT_URLS["ETTh2"],
                "source_kind": "official_script",
                "source_note": "Exact official ETTh2 DLinear script.",
            },
            "ETTm1": {
                "runtime": {"batch_size": 8, "lr": 0.0001},
                "source_url": DLINEAR_SCRIPT_URLS["ETTm1"],
                "source_kind": "official_script",
                "source_note": "Exact official ETTm1 DLinear script.",
            },
            "ETTm2": {
                "runtime": {"batch_size": 32, "lr": 0.001},
                "runtime_by_horizon": {
                    336: {"lr": 0.01},
                    720: {"lr": 0.01},
                },
                "source_url": DLINEAR_SCRIPT_URLS["ETTm2"],
                "source_kind": "official_script",
                "source_note": "Exact official ETTm2 DLinear script, including horizon-specific LR overrides.",
            },
            "weather": {
                "runtime": {"batch_size": 16, "lr": 0.0001},
                "source_url": DLINEAR_SCRIPT_URLS["weather"],
                "source_kind": "official_script",
                "source_note": "Official weather script leaves LR implicit; repo default is 1e-4.",
            },
            "exchange_rate": {
                "runtime": {"batch_size": 8, "lr": 0.0005},
                "runtime_by_horizon": {
                    336: {"batch_size": 32},
                    720: {"batch_size": 32},
                },
                "source_url": DLINEAR_SCRIPT_URLS["exchange_rate"],
                "source_kind": "official_script",
                "source_note": "Exact official exchange_rate DLinear script, including horizon-specific batch size overrides.",
            },
            "electricity": {
                "runtime": {"batch_size": 16, "lr": 0.001},
                "source_url": DLINEAR_SCRIPT_URLS["electricity"],
                "source_kind": "official_script",
                "source_note": "Exact official electricity DLinear script.",
            },
            "solar_AL": {
                "inherits": "weather",
                "source_url": DLINEAR_SCRIPT_URLS["weather"],
                "source_kind": "official_adaptation",
                "source_note": "Adapted from the official DLinear weather script; kept the custom multivariate profile and changed only dataset-specific dimensions.",
            },
        },
    },
    "PatchTST": {
        "lookback": 336,
        "model_params": {
            "individual": False,
            "e_layers": 3,
            "n_heads": 16,
            "d_model": 128,
            "d_ff": 256,
            "dropout": 0.2,
            "fc_dropout": 0.2,
            "head_dropout": 0.0,
            "patch_len": 16,
            "stride": 8,
            "padding_patch": "end",
            "revin": True,
            "affine": True,
            "subtract_last": False,
            "decomposition": False,
            "kernel_size": 25,
        },
        "runtime": {
            "epochs": 100,
            "patience": 100,
            "lr": 0.0001,
            "batch_size": 128,
        },
        "datasets": {
            "ETTh1": {
                "model_params": {
                    "n_heads": 4,
                    "d_model": 16,
                    "d_ff": 128,
                    "dropout": 0.3,
                    "fc_dropout": 0.3,
                },
                "source_url": PATCHTST_SCRIPT_URLS["ETTh1"],
                "source_kind": "official_script",
                "source_note": "Exact official ETTh1 PatchTST script; patience falls back to repo default 100.",
            },
            "ETTh2": {
                "model_params": {
                    "n_heads": 4,
                    "d_model": 16,
                    "d_ff": 128,
                    "dropout": 0.3,
                    "fc_dropout": 0.3,
                },
                "source_url": PATCHTST_SCRIPT_URLS["ETTh2"],
                "source_kind": "official_script",
                "source_note": "Exact official ETTh2 PatchTST script; patience falls back to repo default 100.",
            },
            "ETTm1": {
                "runtime": {"patience": 20},
                "source_url": PATCHTST_SCRIPT_URLS["ETTm1"],
                "source_kind": "official_script",
                "source_note": "Exact official ETTm1 PatchTST script.",
            },
            "ETTm2": {
                "runtime": {"patience": 20},
                "source_url": PATCHTST_SCRIPT_URLS["ETTm2"],
                "source_kind": "official_script",
                "source_note": "Exact official ETTm2 PatchTST script.",
            },
            "weather": {
                "runtime": {"patience": 20},
                "source_url": PATCHTST_SCRIPT_URLS["weather"],
                "source_kind": "official_script",
                "source_note": "Exact official weather PatchTST script.",
            },
            "electricity": {
                "runtime": {"batch_size": 32, "patience": 10},
                "source_url": PATCHTST_SCRIPT_URLS["electricity"],
                "source_kind": "official_script",
                "source_note": "Exact official electricity PatchTST script.",
            },
            "exchange_rate": {
                "inherits": "ETTh1",
                "source_url": PATCHTST_SCRIPT_URLS["ETTh1"],
                "source_kind": "official_adaptation",
                "source_note": "Official PatchTST repo does not ship an exchange_rate script; reused the low-dimensional ETTh1 profile as the closest official reference.",
            },
            "solar_AL": {
                "inherits": "weather",
                "source_url": PATCHTST_SCRIPT_URLS["weather"],
                "source_kind": "official_adaptation",
                "source_note": "Official PatchTST repo does not ship a solar_AL script; reused the multivariate weather profile as the closest official reference.",
            },
        },
    },
    "TQNet": {
        "lookback": 96,
        "model_params": {
            "model_type": "TQNet",
            "d_model": 512,
            "dropout": 0.0,
            "use_revin": True,
        },
        "runtime": {
            "epochs": 30,
            "patience": 5,
            "batch_size": 128,
            "lr": 0.0001,
        },
        "datasets": {
            "ETTh1": {
                "model_params": {"cycle": 24, "dropout": 0.5},
                "runtime": {"batch_size": 256, "lr": 0.001},
                "source_url": TQNET_SCRIPT_URLS["ETTh1"],
                "source_kind": "official_script",
                "source_note": "Exact official ETTh1 TQNet script plus repo defaults for d_model/use_revin.",
            },
            "ETTh2": {
                "model_params": {"cycle": 24, "dropout": 0.5},
                "runtime": {"batch_size": 256, "lr": 0.001},
                "source_url": TQNET_SCRIPT_URLS["ETTh2"],
                "source_kind": "official_script",
                "source_note": "Exact official ETTh2 TQNet script plus repo defaults for d_model/use_revin.",
            },
            "ETTm1": {
                "model_params": {"cycle": 96, "dropout": 0.5},
                "runtime": {"batch_size": 256, "lr": 0.001},
                "source_url": TQNET_SCRIPT_URLS["ETTm1"],
                "source_kind": "official_script",
                "source_note": "Exact official ETTm1 TQNet script plus repo defaults for d_model/use_revin.",
            },
            "ETTm2": {
                "model_params": {"cycle": 96, "dropout": 0.5},
                "runtime": {"batch_size": 256, "lr": 0.001},
                "source_url": TQNET_SCRIPT_URLS["ETTm2"],
                "source_kind": "official_script",
                "source_note": "Exact official ETTm2 TQNet script plus repo defaults for d_model/use_revin.",
            },
            "weather": {
                "model_params": {"cycle": 144, "dropout": 0.5},
                "runtime": {"batch_size": 64, "lr": 0.001},
                "source_url": TQNET_SCRIPT_URLS["weather"],
                "source_kind": "official_script",
                "source_note": "Exact official weather TQNet script plus repo defaults for d_model/use_revin.",
            },
            "electricity": {
                "model_params": {"cycle": 168, "dropout": 0.5},
                "runtime": {"batch_size": 32, "lr": 0.003},
                "source_url": TQNET_SCRIPT_URLS["electricity"],
                "source_kind": "official_script",
                "source_note": "Exact official electricity TQNet script plus repo defaults for d_model/use_revin.",
            },
            "solar_AL": {
                "model_params": {"cycle": 144, "dropout": 0.5, "use_revin": False},
                "runtime": {"batch_size": 64, "lr": 0.003},
                "source_url": TQNET_SCRIPT_URLS["solar_AL"],
                "source_kind": "official_script",
                "source_note": "Exact official solar_AL TQNet script plus repo default d_model.",
            },
            "exchange_rate": {
                "inherits": "ETTh1",
                "model_params": {"cycle": 7},
                "source_url": TQNET_SCRIPT_URLS["ETTh1"],
                "source_kind": "official_adaptation",
                "source_note": "Official TQNet repo does not ship an exchange_rate script; reused the low-dimensional ETTh1 profile and adapted the cycle to a weekly daily-frequency prior.",
            },
        },
    },
    "iTransformer": {
        "lookback": 96,
        "model_params": {
            "d_model": 128,
            "d_ff": 128,
            "n_heads": 8,
            "e_layers": 2,
            "factor": 1,
            "dropout": 0.1,
            "activation": "gelu",
            "use_norm": True,
        },
        "runtime": {
            "epochs": 10,
            "patience": 3,
            "batch_size": 32,
            "lr": 0.0001,
        },
        "datasets": {
            "ETTh1": {
                "model_params": {"d_model": 256, "d_ff": 256},
                "model_params_by_horizon": {
                    336: {"d_model": 512, "d_ff": 512},
                    720: {"d_model": 512, "d_ff": 512},
                },
                "source_url": ITRANSFORMER_SCRIPT_URLS["ETTh1"],
                "source_kind": "official_script",
                "source_note": "Exact official iTransformer ETTh1 script, including the wider H336/H720 projector width.",
            },
            "ETTh2": {
                "source_url": ITRANSFORMER_SCRIPT_URLS["ETTh2"],
                "source_kind": "official_script",
                "source_note": "Exact official iTransformer ETTh2 script.",
            },
            "ETTm1": {
                "source_url": ITRANSFORMER_SCRIPT_URLS["ETTm1"],
                "source_kind": "official_script",
                "source_note": "Exact official iTransformer ETTm1 script.",
            },
            "ETTm2": {
                "source_url": ITRANSFORMER_SCRIPT_URLS["ETTm2"],
                "source_kind": "official_script",
                "source_note": "Exact official iTransformer ETTm2 script.",
            },
            "weather": {
                "model_params": {"e_layers": 3, "d_model": 512, "d_ff": 512},
                "source_url": ITRANSFORMER_SCRIPT_URLS["weather"],
                "source_kind": "official_script",
                "source_note": "Exact official iTransformer weather script.",
            },
            "exchange_rate": {
                "source_url": ITRANSFORMER_SCRIPT_URLS["exchange_rate"],
                "source_kind": "official_script_with_minor_unified_fix",
                "source_note": "Mostly follows the official exchange_rate script; kept the standard 10-epoch training budget instead of inheriting the repo-side H336 one-epoch anomaly into the unified retraining pool.",
            },
            "electricity": {
                "model_params": {"e_layers": 3, "d_model": 512, "d_ff": 512},
                "runtime": {"batch_size": 16, "lr": 0.0005},
                "source_url": ITRANSFORMER_SCRIPT_URLS["electricity"],
                "source_kind": "official_script",
                "source_note": "Exact official iTransformer electricity script.",
            },
            "solar_AL": {
                "model_params": {"d_model": 512, "d_ff": 512},
                "runtime": {"lr": 0.0005},
                "source_url": ITRANSFORMER_SCRIPT_URLS["solar_AL"],
                "source_kind": "official_script",
                "source_note": "Exact official iTransformer solar script; batch size falls back to the repo default 32 because the script leaves it implicit.",
            },
        },
    },
    "ModernTCN": {
        "lookback": 336,
        "model_params": {
            "patch_size": 8,
            "patch_stride": 4,
            "downsample_ratio": 2,
            "ffn_ratio": 1,
            "num_blocks": [1],
            "large_size": [51],
            "small_size": [5],
            "dims": [64, 64, 64, 64],
            "dw_dims": [256, 256, 256, 256],
            "head_dropout": 0.0,
            "dropout": 0.3,
            "revin": True,
            "affine": False,
            "subtract_last": False,
            "decomposition": False,
            "kernel_size": 25,
        },
        "runtime": {
            "epochs": 100,
            "patience": 20,
            "batch_size": 128,
            "lr": 0.0001,
        },
        "datasets": {
            "ETTh1": {
                "runtime": {"batch_size": 512},
                "source_url": MODERNTCN_SCRIPT_URLS["ETTh1"],
                "source_kind": "official_script",
                "source_note": "Exact official ModernTCN ETTh1 script.",
            },
            "ETTh2": {
                "runtime": {"batch_size": 512},
                "model_params_by_horizon": {
                    720: {"head_dropout": 0.5, "dropout": 0.8},
                },
                "source_url": MODERNTCN_SCRIPT_URLS["ETTh2"],
                "source_kind": "official_script",
                "source_note": "Exact official ModernTCN ETTh2 script, including the heavier H720 dropout.",
            },
            "ETTm1": {
                "model_params": {"ffn_ratio": 8, "num_blocks": [3, 3, 3]},
                "model_params_by_horizon": {
                    192: {"head_dropout": 0.1},
                    336: {"head_dropout": 0.1},
                    720: {"head_dropout": 0.1},
                },
                "runtime": {"batch_size": 512},
                "runtime_by_horizon": {
                    192: {"lookback_override": 192},
                    720: {"lookback_override": 672},
                },
                "source_url": MODERNTCN_SCRIPT_URLS["ETTm1"],
                "source_kind": "official_script",
                "source_note": "Exact official ModernTCN ETTm1 script, including its horizon-specific lookback schedule.",
            },
            "ETTm2": {
                "model_params": {"ffn_ratio": 8, "num_blocks": [3, 3, 3]},
                "model_params_by_horizon": {
                    96: {"head_dropout": 0.2},
                    192: {"head_dropout": 0.2},
                    336: {"head_dropout": 0.2, "dropout": 0.8},
                    720: {"head_dropout": 0.1},
                },
                "runtime": {"batch_size": 512},
                "runtime_by_horizon": {
                    96: {"lookback_override": 192},
                    192: {"lookback_override": 336},
                    336: {"lookback_override": 336},
                    720: {"lookback_override": 960},
                },
                "source_url": MODERNTCN_SCRIPT_URLS["ETTm2"],
                "source_kind": "official_script",
                "source_note": "Exact official ModernTCN ETTm2 script, including its horizon-specific lookback schedule.",
            },
            "weather": {
                "model_params": {"ffn_ratio": 8, "dropout": 0.4},
                "runtime": {"batch_size": 256},
                "runtime_by_horizon": {
                    336: {"batch_size": 512},
                    720: {"batch_size": 512, "lookback_override": 720},
                },
                "source_url": MODERNTCN_SCRIPT_URLS["weather"],
                "source_kind": "official_script",
                "source_note": "Exact official ModernTCN weather script, including the larger batch size for H336/H720.",
            },
            "exchange_rate": {
                "model_params": {
                    "patch_size": 1,
                    "patch_stride": 1,
                    "ffn_ratio": 1,
                    "dropout": 0.2,
                    "head_dropout": 0.6,
                },
                "runtime": {"batch_size": 128, "lookback_override": 6},
                "runtime_by_horizon": {
                    336: {"batch_size": 512},
                    720: {"batch_size": 512},
                },
                "source_url": MODERNTCN_SCRIPT_URLS["exchange_rate"],
                "source_kind": "official_script",
                "source_note": "Exact official ModernTCN exchange_rate script.",
            },
            "electricity": {
                "model_params": {"ffn_ratio": 8, "dropout": 0.9},
                "runtime": {"batch_size": 32, "patience": 10, "lookback_override": 720},
                "source_url": MODERNTCN_SCRIPT_URLS["electricity"],
                "source_kind": "official_script",
                "source_note": "Exact official ModernTCN electricity script.",
            },
            "solar_AL": {
                "inherits": "electricity",
                "source_url": MODERNTCN_SCRIPT_URLS["electricity"],
                "source_kind": "official_adaptation",
                "source_note": "Official ModernTCN repo does not ship a solar_AL script; reused the official high-dimensional electricity profile as the closest energy-domain adaptation.",
            },
        },
    },
    "TimeMixer": {
        "lookback": 96,
        "model_params": {
            "d_model": 16,
            "d_ff": 32,
            "e_layers": 2,
            "dropout": 0.1,
            "moving_avg": 25,
            "channel_independence": 1,
            "decomp_method": "moving_avg",
            "top_k": 5,
            "use_norm": True,
            "down_sampling_layers": 3,
            "down_sampling_window": 2,
            "down_sampling_method": "avg",
        },
        "runtime": {
            "epochs": 100,
            "patience": 15,
            "batch_size": 16,
            "lr": 0.001,
        },
        "datasets": {
            "ETTh1": {
                "runtime": {"epochs": 10, "patience": 10, "batch_size": 128, "lr": 0.01},
                "source_url": TIMEMIXER_SCRIPT_URLS["ETTh1"],
                "source_kind": "official_script",
                "source_note": "Exact official TimeMixer ETTh1 unified script.",
            },
            "ETTh2": {
                "runtime": {"lr": 0.01},
                "source_url": TIMEMIXER_SCRIPT_URLS["ETTh2"],
                "source_kind": "official_script",
                "source_note": "Official TimeMixer ETTh2 script leaves train_epochs/patience/batch_size implicit; reused the repo defaults together with the explicit LR and multiscale settings from the script.",
            },
            "ETTm1": {
                "runtime": {"batch_size": 16, "lr": 0.01},
                "source_url": TIMEMIXER_SCRIPT_URLS["ETTm1"],
                "source_kind": "official_script",
                "source_note": "Official TimeMixer ETTm1 script leaves train_epochs/patience implicit; reused the repo defaults and copied the explicit batch size and multiscale settings.",
            },
            "ETTm2": {
                "model_params": {"d_model": 32},
                "runtime": {"batch_size": 128, "lr": 0.01},
                "source_url": TIMEMIXER_SCRIPT_URLS["ETTm2"],
                "source_kind": "official_script",
                "source_note": "Official TimeMixer ETTm2 script leaves train_epochs/patience implicit; reused the repo defaults and copied the explicit batch size and multiscale settings.",
            },
            "weather": {
                "model_params": {"e_layers": 3},
                "runtime": {"epochs": 20, "patience": 10, "batch_size": 128, "lr": 0.01},
                "source_url": TIMEMIXER_SCRIPT_URLS["weather"],
                "source_kind": "official_script",
                "source_note": "Exact official TimeMixer weather script.",
            },
            "exchange_rate": {
                "inherits": "ETTh1",
                "source_url": TIMEMIXER_SCRIPT_URLS["ETTh1"],
                "source_kind": "official_adaptation",
                "source_note": "Official TimeMixer repo does not ship an exchange_rate script; reused the low-dimensional ETTh1 unified profile as the closest official reference.",
            },
            "electricity": {
                "model_params": {"e_layers": 3},
                "runtime": {"epochs": 20, "patience": 10, "batch_size": 32, "lr": 0.01},
                "source_url": TIMEMIXER_SCRIPT_URLS["electricity"],
                "source_kind": "official_script",
                "source_note": "Exact official TimeMixer electricity script.",
            },
            "solar_AL": {
                "model_params": {
                    "e_layers": 3,
                    "d_model": 512,
                    "d_ff": 2048,
                    "channel_independence": 0,
                    "use_norm": False,
                    "down_sampling_layers": 2,
                },
                "runtime": {"epochs": 10, "patience": 3, "batch_size": 32, "lr": 0.001},
                "source_url": TIMEMIXER_SCRIPT_URLS["solar_AL"],
                "source_kind": "official_script",
                "source_note": "Exact official TimeMixer solar script.",
            },
        },
    },
    "TimeMixerPP": {
        "lookback": 96,
        "model_params": {
            "d_model": 256,
            "expert_hidden": 384,
            "head_rank": 32,
            "patch_len": 8,
            "patch_stride": 4,
            "n_blocks": 4,
            "n_resolutions": 3,
            "n_heads": 8,
            "ffn_ratio": 4,
            "dropout": 0.1,
            "stochastic_depth": 0.1,
            "num_experts": 4,
        },
        "runtime": {
            "epochs": 12,
            "patience": 4,
            "batch_size": 32,
            "lr": 0.0002,
            "weight_decay": 0.05,
        },
        "datasets": {
            "ETTh1": {
                "runtime": {"batch_size": 64},
                "source_url": TIMEMIXERPP_REFERENCE_URL,
                "source_kind": "official_release_adaptation",
                "source_note": "Official TimeMixer repo README announces TimeMixer++ but does not expose runnable long-horizon scripts on GitHub; this vanilla TimeMixer++ baseline keeps the TimeMixer++-style shared trunk used by AIF-Plus/AEF-Plus while removing artifact-aware modules.",
            },
            "ETTh2": {
                "runtime": {"batch_size": 64},
                "source_url": TIMEMIXERPP_REFERENCE_URL,
                "source_kind": "official_release_adaptation",
                "source_note": "Same vanilla TimeMixer++ trunk profile as ETTh1; official GitHub release note exists but no dedicated ETTh2 forecasting script is published.",
            },
            "ETTm1": {
                "runtime": {"batch_size": 32},
                "source_url": TIMEMIXERPP_REFERENCE_URL,
                "source_kind": "official_release_adaptation",
                "source_note": "Adapted from the official TimeMixer repository release note for TimeMixer++; used the same vanilla multiscale trunk as the paper-aligned AIF/AEF shared backbone.",
            },
            "ETTm2": {
                "runtime": {"batch_size": 32},
                "source_url": TIMEMIXERPP_REFERENCE_URL,
                "source_kind": "official_release_adaptation",
                "source_note": "Adapted from the official TimeMixer repository release note for TimeMixer++; kept the same vanilla trunk and optimizer settings as the unified retrainable pool.",
            },
            "weather": {
                "runtime": {"batch_size": 32},
                "source_url": TIMEMIXERPP_REFERENCE_URL,
                "source_kind": "official_release_adaptation",
                "source_note": "Vanilla TimeMixer++ weather profile using the TimeMixer++-style trunk capacity from newplan; official repo currently exposes the release note but not weather-specific scripts.",
            },
            "exchange_rate": {
                "runtime": {"batch_size": 64},
                "source_url": TIMEMIXERPP_REFERENCE_URL,
                "source_kind": "official_release_adaptation",
                "source_note": "Low-dimensional exchange_rate setting using the same vanilla TimeMixer++ trunk family; no official exchange_rate script is available for TimeMixer++.",
            },
            "electricity": {
                "runtime": {"batch_size": 16},
                "source_url": TIMEMIXERPP_REFERENCE_URL,
                "source_kind": "official_release_adaptation",
                "source_note": "High-dimensional electricity setting for the vanilla TimeMixer++ baseline; batch size is reduced for memory while trunk capacity stays aligned with the AIF/AEF shared backbone family.",
            },
            "solar_AL": {
                "runtime": {"batch_size": 16},
                "source_url": TIMEMIXERPP_REFERENCE_URL,
                "source_kind": "official_release_adaptation",
                "source_note": "High-dimensional solar_AL setting using the same vanilla TimeMixer++ trunk family; official repo currently exposes only the TimeMixer++ release announcement.",
            },
        },
    },
}


@dataclass(frozen=True)
class ResolvedBackboneConfig:
    dataset_name: str
    backbone_name: str
    horizon: int
    lookback: int
    model_params: dict[str, Any]
    runtime: dict[str, Any]
    source_url: str
    source_kind: str
    source_note: str


def canonicalize_dataset_name(name: str) -> str:
    token = str(name).strip()
    if not token:
        return token
    normalized = token.replace("-", "_").lower()
    return DATASET_NAME_ALIASES.get(normalized, token)


def canonicalize_dataset_names(values: Iterable[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        canonical = canonicalize_dataset_name(value)
        if not canonical or canonical in seen:
            continue
        ordered.append(canonical)
        seen.add(canonical)
    return ordered


def _merge_dict(base: dict[str, Any] | None, override: dict[str, Any] | None) -> dict[str, Any]:
    merged = deepcopy(base or {})
    for key, value in (override or {}).items():
        merged[key] = deepcopy(value)
    return merged


def _merge_profile_entry(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key in ["model_params", "runtime", "model_params_by_horizon", "runtime_by_horizon"]:
        merged[key] = _merge_dict(base.get(key), override.get(key))

    for key in ["lookback", "source_url", "source_kind", "source_note"]:
        if key in override:
            merged[key] = deepcopy(override[key])
    return merged


def _resolve_dataset_profile(backbone_name: str, dataset_name: str) -> dict[str, Any]:
    canonical_dataset = canonicalize_dataset_name(dataset_name)
    backbone_spec = BACKBONE_PROFILE_LIBRARY[backbone_name]
    dataset_specs = backbone_spec["datasets"]
    if canonical_dataset not in dataset_specs:
        raise KeyError(f"Unsupported dataset `{canonical_dataset}` for backbone `{backbone_name}`")

    raw_entry = deepcopy(dataset_specs[canonical_dataset])
    inherits = raw_entry.pop("inherits", None)
    if inherits is None:
        return raw_entry
    parent_entry = _resolve_dataset_profile(backbone_name, str(inherits))
    return _merge_profile_entry(parent_entry, raw_entry)


def _resolve_horizon_map(mapping: dict[Any, dict[str, Any]] | None, horizon: int) -> dict[str, Any]:
    if not mapping:
        return {}
    if horizon in mapping:
        return deepcopy(mapping[horizon])
    if str(horizon) in mapping:
        return deepcopy(mapping[str(horizon)])
    return {}


def resolve_backbone_experiment(
    backbone_cfg: dict[str, Any],
    dataset_name: str,
    horizon: int,
    runtime_defaults: dict[str, Any] | None = None,
) -> ResolvedBackboneConfig:
    backbone_name = str(backbone_cfg["name"])
    if backbone_name not in BACKBONE_PROFILE_LIBRARY:
        raise KeyError(f"Unsupported backbone `{backbone_name}`")

    profile_spec = BACKBONE_PROFILE_LIBRARY[backbone_name]
    dataset_profile = _resolve_dataset_profile(backbone_name, dataset_name)
    canonical_dataset = canonicalize_dataset_name(dataset_name)

    model_params = _merge_dict(backbone_cfg.get("params", {}), profile_spec.get("model_params", {}))
    model_params = _merge_dict(model_params, dataset_profile.get("model_params", {}))
    model_params = _merge_dict(model_params, _resolve_horizon_map(dataset_profile.get("model_params_by_horizon"), horizon))

    runtime = _merge_dict(runtime_defaults or {}, profile_spec.get("runtime", {}))
    runtime = _merge_dict(runtime, dataset_profile.get("runtime", {}))
    runtime = _merge_dict(runtime, _resolve_horizon_map(dataset_profile.get("runtime_by_horizon"), horizon))
    global_lookback_override = None if runtime_defaults is None else runtime_defaults.get("lookback_override")
    if global_lookback_override is not None:
        runtime["lookback_override"] = int(global_lookback_override)

    batch_size = int(runtime.get("batch_size", 64))
    if int(runtime.get("eval_batch_size", 0) or 0) <= 0:
        runtime["eval_batch_size"] = min(max(batch_size, 128), 512)

    lookback_override = runtime.get("lookback_override")
    if lookback_override is not None:
        lookback = int(lookback_override)
    else:
        lookback = int(dataset_profile.get("lookback", profile_spec.get("lookback", runtime.get("lookback", 96))))
    source_url = str(dataset_profile.get("source_url", ""))
    source_kind = str(dataset_profile.get("source_kind", "internal_default"))
    source_note = str(dataset_profile.get("source_note", ""))
    return ResolvedBackboneConfig(
        dataset_name=canonical_dataset,
        backbone_name=backbone_name,
        horizon=int(horizon),
        lookback=lookback,
        model_params=model_params,
        runtime=runtime,
        source_url=source_url,
        source_kind=source_kind,
        source_note=source_note,
    )


def collect_required_lookbacks(
    backbone_cfgs: Iterable[dict[str, Any]],
    datasets: Iterable[str],
    horizons: Iterable[int],
    runtime_defaults: dict[str, Any] | None = None,
) -> list[int]:
    resolved: set[int] = set()
    canonical_datasets = canonicalize_dataset_names(datasets)
    horizon_list = [int(item) for item in horizons]
    for dataset_name in canonical_datasets:
        for backbone_cfg in backbone_cfgs:
            for horizon in horizon_list[:1] or [96]:
                resolved.add(
                    resolve_backbone_experiment(
                        backbone_cfg=backbone_cfg,
                        dataset_name=dataset_name,
                        horizon=horizon,
                        runtime_defaults=runtime_defaults,
                    ).lookback
                )
    return sorted(resolved)


def describe_resolved_profile(resolved: ResolvedBackboneConfig) -> dict[str, Any]:
    return {
        "dataset_name": resolved.dataset_name,
        "backbone": resolved.backbone_name,
        "horizon": resolved.horizon,
        "lookback": resolved.lookback,
        "batch_size": resolved.runtime.get("batch_size"),
        "eval_batch_size": resolved.runtime.get("eval_batch_size"),
        "epochs": resolved.runtime.get("epochs"),
        "patience": resolved.runtime.get("patience"),
        "lr": resolved.runtime.get("lr"),
        "weight_decay": resolved.runtime.get("weight_decay"),
        "source_kind": resolved.source_kind,
        "source_url": resolved.source_url,
        "source_note": resolved.source_note,
    }
