# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""INT8 quantization config for diffusion transformers."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Optional

import torch
from torch.nn import Module
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
)
from vllm.model_executor.parameter import (
    ChannelQuantScaleParameter,
    ModelWeightParameter,
)
from vllm.model_executor.utils import replace_parameter

from vllm_omni.platforms import current_omni_platform

from .base import DiffusionQuantizationConfig

if TYPE_CHECKING:
    from vllm.model_executor.models.utils import WeightsMapper

# Dynamic quantization is supported first.
ACTIVATION_SCHEMES = ["dynamic"]

logger = init_logger(__name__)


def create_int8_weight_parameter(
    output_size_per_partition: int,
    input_size_per_partition: int,
    weight_loader: Callable | None,
) -> torch.nn.Parameter:
    """
    Create int8 weight parameter.
    """
    from vllm.model_executor.parameter import ModelWeightParameter

    return ModelWeightParameter(
        data=torch.empty(
            output_size_per_partition,
            input_size_per_partition,
            dtype=torch.int8,
        ),
        input_dim=1,
        output_dim=0,
        weight_loader=weight_loader,
    )


def create_int8_scale_parameter(
    parameter_type: torch.nn.Parameter,
    output_partition_sizes: list[int],
    input_size_per_partition: int,
    block_size: list[int] | None,
    weight_loader: Callable | None,
    params_dtype: torch.dtype,
) -> torch.nn.Parameter:
    """
    Create scale parameter based on quantization strategy
    """
    if parameter_type == ChannelQuantScaleParameter:
        scale = parameter_type(
            data=torch.empty((sum(output_partition_sizes), 1), dtype=torch.float32),
            output_dim=0,
            weight_loader=weight_loader,
        )
    else:
        raise ValueError(f"Unknown parameter type: {parameter_type}")

    return scale


class Int8Config(QuantizationConfig):
    """
    Config class for Int8.
    """

    def __init__(
        self,
        is_checkpoint_int8_serialized: bool = False,
        activation_scheme: str = "dynamic",
        ignored_layers: list[str] | None = None,
    ) -> None:
        super().__init__()

        self.is_checkpoint_int8_serialized = is_checkpoint_int8_serialized

        if activation_scheme not in ACTIVATION_SCHEMES:
            raise ValueError(f"Unsupported activation scheme {activation_scheme}")
        self.activation_scheme = activation_scheme
        self.ignored_layers = ignored_layers or []

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "int8"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    def apply_vllm_mapper(self, hf_to_vllm_mapper: "WeightsMapper"):
        if self.ignored_layers is not None:
            self.ignored_layers = hf_to_vllm_mapper.apply_list(self.ignored_layers)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Int8Config":
        quant_method = cls.get_from_keys(config, ["quant_method"])
        is_checkpoint_int8_serialized = "int8" in quant_method
        activation_scheme = cls.get_from_keys(config, ["activation_scheme"])
        ignored_layers = cls.get_from_keys_or(config, ["ignored_layers"], None)

        if not ignored_layers:
            ignored_layers = cls.get_from_keys_or(config, ["modules_to_not_convert"], None)
        return cls(
            is_checkpoint_int8_serialized=is_checkpoint_int8_serialized,
            activation_scheme=activation_scheme,
            ignored_layers=ignored_layers,
        )

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional["QuantizeMethodBase"]:
        if current_omni_platform.is_npu():
            if isinstance(layer, LinearBase):
                if is_layer_skipped(
                    prefix=prefix,
                    ignored_layers=self.ignored_layers,
                    fused_mapping=self.packed_modules_mapping,
                ):
                    return UnquantizedLinearMethod()
                if not self.is_checkpoint_int8_serialized:
                    online_method = Int8OnlineLinearMethod(self)
                    return online_method
                else:
                    offline_method = Int8LinearMethod(self)
                    return offline_method
        else:
            logger.warning("The current platform is not supported.")
        return None


class Int8LinearMethod(LinearMethodBase):
    """
    Linear method for Int8
    Supports loading Int8 checkpoints with static weight scale and dynamic activation scale.

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: Int8Config):
        self.quant_config = quant_config
        self.out_dtype = torch.get_default_dtype()

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype

        weight = create_int8_weight_parameter(
            output_size_per_partition=output_size_per_partition,
            input_size_per_partition=input_size_per_partition,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        # WEIGHT OFFSET
        offset = create_int8_scale_parameter(
            ChannelQuantScaleParameter,
            output_partition_sizes,
            input_size_per_partition,
            None,
            weight_loader,
            params_dtype,
        )
        layer.register_parameter("weight_offset", offset)

        # WEIGHT SCALE
        scale = create_int8_scale_parameter(
            ChannelQuantScaleParameter,
            output_partition_sizes,
            input_size_per_partition,
            None,
            weight_loader,
            params_dtype,
        )
        layer.register_parameter("weight_scale", scale)

    def process_weights_after_loading(self, layer: Module) -> None:
        layer.weight.data = layer.weight.data.t().contiguous()
        layer.weight_scale.data = layer.weight_scale.data.squeeze()
        layer.weight_offset.data = layer.weight_offset.data.squeeze()

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        import torch_npu
        
        ori_shape = x.shape
        ori_dtype = x.dtype

        x = x.reshape(-1, ori_shape[-1])
        quantized_x, pertoken_scale = torch_npu.npu_dynamic_quant(x)

        output = torch_npu.npu_quant_matmul(
            quantized_x,
            layer.weight,
            layer.weight_scale,
            bias=bias,
            pertoken_scale=pertoken_scale,
            output_dtype=ori_dtype,
        )
        output = output.reshape(*ori_shape[:-1], -1)
        return output


class Int8OnlineLinearMethod(Int8LinearMethod):
    """
    Online version of Int8LinearMethod, loads the fp16/bf16 checkpoint
    and quantized the weights during loading.
    """

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

    def process_weights_after_loading(self, layer: Module) -> None:
        import torch_npu

        qweight, weight_scale = torch_npu.npu_dynamic_quant(layer.weight)

        layer.weight = None
        torch.npu.empty_cache()

        weight = qweight.t().contiguous()

        # Update layer with new values.
        replace_parameter(layer, "weight", weight)
        replace_parameter(layer, "weight_scale", weight_scale)


class DiffusionInt8Config(DiffusionQuantizationConfig):
    """
    Int8 quantization config optimized for diffusion transformers.

    Args:
        activation_scheme: Activation quantization scheme.
            - "dynamic": Per-token dynamic scaling (default, no calibration)
            Format: [block_n, block_k]. If None, uses per-tensor scaling.
        ignored_layers: List of layer name patterns to skip quantization.
    """
    quant_config_cls = Int8Config

    def __init__(
        self,
        activation_scheme: str = "dynamic",
        ignored_layers: list[str] | None = None,
    ):
        self.activation_scheme = activation_scheme
        self.ignored_layers = ignored_layers or []

        # Create underlying vLLM Int8 config
        self._vllm_config = Int8Config(
            is_checkpoint_int8_serialized=False,  # Online quantization
            activation_scheme=activation_scheme,
            ignored_layers=ignored_layers,
        )
