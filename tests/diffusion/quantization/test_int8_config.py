# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for Int8 quantization config."""

from unittest.mock import MagicMock, patch

import pytest
import torch
from pytest_mock import MockerFixture
from torch.nn import Module, Parameter
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod

from vllm_omni.diffusion.quantization import (
    get_diffusion_quant_config,
    get_vllm_quant_config_for_layers,
)


def test_int8_config_creation():
    """Test that Int8 config can be created."""
    config = get_diffusion_quant_config("int8")
    assert config is not None
    assert config.get_name() == "int8"


def test_vllm_config_extraction():
    """Test that vLLM config can be extracted from diffusion config."""
    diff_config = get_diffusion_quant_config("int8")
    vllm_config = get_vllm_quant_config_for_layers(diff_config)
    assert vllm_config is not None
    assert vllm_config.activation_scheme == "dynamic"


def test_none_quantization():
    """Test that None quantization returns None config."""
    config = get_diffusion_quant_config(None)
    assert config is None
    vllm_config = get_vllm_quant_config_for_layers(config)
    assert vllm_config is None


def test_invalid_quantization():
    """Test that invalid quantization method raises error."""
    with pytest.raises(ValueError, match="Unknown quantization method"):
        get_diffusion_quant_config("invalid_method")


def test_int8_config_with_custom_params():
    """Test Int8 config with custom parameters."""
    config = get_diffusion_quant_config(
        "int8",
        activation_scheme="dynamic",
        ignored_layers=["proj_out"],
    )
    assert config is not None
    assert config.activation_scheme == "dynamic"
    assert "proj_out" in config.ignored_layers


def test_supported_methods():
    """Test that supported methods list is correct."""
    from vllm_omni.diffusion.quantization import SUPPORTED_QUANTIZATION_METHODS

    assert "int8" in SUPPORTED_QUANTIZATION_METHODS


def test_quantization_integration():
    """Test end-to-end quantization flow through OmniDiffusionConfig."""
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    # Test with quantization string only
    config = OmniDiffusionConfig(model="test", quantization="int8")
    assert config.quantization_config is not None
    assert config.quantization_config.get_name() == "int8"

    # Test with quantization_config dict
    config2 = OmniDiffusionConfig(
        model="test",
        quantization_config={"method": "int8", "activation_scheme": "dynamic"},
    )
    assert config2.quantization_config is not None
    assert config2.quantization_config.get_name() == "int8"
    assert config2.quantization_config.activation_scheme == "dynamic"

    # Test that vLLM config can be extracted
    vllm_config = config.quantization_config.get_vllm_quant_config()
    assert vllm_config is not None


def test_quantization_dict_not_mutated():
    """Test that passing a dict to quantization_config doesn't mutate it."""
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    original_dict = {"method": "int8", "activation_scheme": "dynamic"}
    dict_copy = original_dict.copy()

    OmniDiffusionConfig(model="test", quantization_config=original_dict)

    # Original dict should be unchanged
    assert original_dict == dict_copy


def test_quantization_conflicting_methods_warning(caplog):
    """Test warning when quantization and quantization_config['method'] conflict."""
    import logging

    from vllm_omni.diffusion.data import OmniDiffusionConfig

    with caplog.at_level(logging.WARNING):
        config = OmniDiffusionConfig(
            model="test",
            quantization="int8",  # This should be overridden
            quantization_config={"method": "int8", "activation_scheme": "dynamic"},
        )
    # No warning when methods match
    assert config.quantization_config is not None


def test_get_quant_method(mocker: MockerFixture):
    """Test for get_quant_method method for GPU"""
    from vllm_omni.diffusion.quantization.int8 import Int8OnlineLinearMethod

    diff_config = get_diffusion_quant_config("int8")
    vllm_config = get_vllm_quant_config_for_layers(diff_config)

    def _fake_init(self, quant_config):
        pass

    layer = MagicMock(spec=LinearBase)
    mocker.patch.object(Int8OnlineLinearMethod, "__init__", _fake_init)

    prefix = "test_layer"

    # Mock the platform to be GPU
    with (patch("vllm_omni.platforms.current_omni_platform.is_cuda", return_value=True),
          patch("vllm_omni.platforms.current_omni_platform.is_npu", return_value=False)):
        method = vllm_config.get_quant_method(layer, prefix)
        assert isinstance(method, Int8OnlineLinearMethod)

    # Test skipping quantization for a layer
    vllm_config.ignored_layers = [prefix]
    method = vllm_config.get_quant_method(layer, prefix)
    assert isinstance(method, UnquantizedLinearMethod)


def test_get_npu_quant_method():
    """Test for get_quant_method method for NPU"""
    from vllm_omni.diffusion.quantization.int8 import NPUInt8OnlineLinearMethod

    diff_config = get_diffusion_quant_config("int8")
    vllm_config = get_vllm_quant_config_for_layers(diff_config)

    layer = MagicMock(spec=LinearBase)
    prefix = "test_layer"

    # Mock the platform to be NPU
    with (patch("vllm_omni.platforms.current_omni_platform.is_cuda", return_value=False),
          patch("vllm_omni.platforms.current_omni_platform.is_npu", return_value=True)):
        method = vllm_config.get_quant_method(layer, prefix)
        assert isinstance(method, NPUInt8OnlineLinearMethod)

    # Test skipping quantization for a layer
    vllm_config.ignored_layers = [prefix]
    method = vllm_config.get_quant_method(layer, prefix)
    assert isinstance(method, UnquantizedLinearMethod)


class TestInt8LinearMethod:
    @pytest.fixture
    def mock_quant_config(self, mocker):
        return mocker.Mock()

    @pytest.fixture
    def mock_kernel(self, mocker):
        kernel = mocker.Mock()
        kernel.process_weights_after_loading = mocker.Mock()
        kernel.apply_weights = mocker.Mock(return_value=torch.randn(1, 10))
        return kernel

    @pytest.fixture
    def patch_deps(self, mocker, mock_kernel):
        # mock init_int8_linear_kernel
        mocker.patch("vllm_omni.diffusion.quantization.int8.init_int8_linear_kernel", return_value=mock_kernel)
        return mock_kernel

    def test_init(self, patch_deps, mock_quant_config):
        # test for Int8LinearMethod init
        from vllm_omni.diffusion.quantization.int8 import Int8LinearMethod, init_int8_linear_kernel

        method = Int8LinearMethod(mock_quant_config)

        assert method.quant_config == mock_quant_config
        init_int8_linear_kernel.assert_called_once_with(
            is_channelwise=False, is_static_input_scheme=False, input_symmetric=True, module_name="Int8LinearMethod"
        )
        assert method.int8_linear == patch_deps

    def test_process_weights_after_loading(self, patch_deps, mock_quant_config):
        from vllm_omni.diffusion.quantization.int8 import Int8LinearMethod

        method = Int8LinearMethod(mock_quant_config)
        layer = Module()

        method.process_weights_after_loading(layer)
        patch_deps.process_weights_after_loading.assert_called_once_with(layer)

    def test_apply(self, patch_deps, mock_quant_config):
        from vllm_omni.diffusion.quantization.int8 import Int8LinearMethod

        method = Int8LinearMethod(mock_quant_config)
        layer = Module()
        x = torch.randn(1, 128)
        bias = torch.randn(128)

        output = method.apply(layer, x, bias)

        patch_deps.apply_weights.assert_called_once_with(layer, x, bias)
        assert isinstance(output, torch.Tensor)


class TestInt8OnlineLinearMethod:
    @pytest.fixture
    def mock_quant_config(self, mocker):
        return mocker.Mock()

    @pytest.fixture
    def mock_deps(self, mocker):
        # mock kernel
        mock_kernel = mocker.Mock()
        mock_kernel.layer_param_names = ("weight", "weight_scale", "input_scale", "input_zero_point", "azp_adj")
        mocker.patch("vllm_omni.diffusion.quantization.int8.init_int8_linear_kernel", return_value=mock_kernel)
        mocker.patch("vllm_omni.diffusion.quantization.int8.replace_parameter")

        # mock scaled_int8_quant return value
        mock_qweight = torch.ones((128, 64), dtype=torch.int8)
        mock_scale = torch.randn(128)
        mock_quant = mocker.patch(
            "vllm_omni.diffusion.quantization.int8.ops.scaled_int8_quant", return_value=(mock_qweight, mock_scale, None)
        )
        return {"kernel": mock_kernel, "quant": mock_quant, "mock_qweight": mock_qweight, "mock_scale": mock_scale}

    def test_process_weights_after_loading(self, mock_deps, mock_quant_config):
        from vllm_omni.diffusion.quantization.int8 import Int8OnlineLinearMethod

        method = Int8OnlineLinearMethod(mock_quant_config)
        layer = Module()
        layer.weight = Parameter(torch.randn(128, 64))
        method.process_weights_after_loading(layer)
        mock_deps["quant"].assert_called_once_with(layer.weight, scale=None)


class TestNPUInt8LinearMethod:
    qweight_mock = torch.randn((128,64)).to(dtype=torch.int8)
    scale_mock = torch.randn(128)
    out_mock = torch.randn((16,128))

    @pytest.fixture
    def mock_torch_npu(self, mocker):
        torch_npu = MagicMock()

        mocker.patch("vllm_omni.diffusion.quantization.int8.torch_npu",
                     return_value=torch_npu)
        mocker.patch("vllm_omni.diffusion.quantization.int8.torch_npu.npu_dynamic_quant",
                     return_value=(self.qweight_mock, self.scale_mock))
        mocker.patch("vllm_omni.diffusion.quantization.int8.torch_npu.npu_quant_matmul",
                     return_value=self.out_mock)
        return torch_npu

    @pytest.fixture
    def mock_quant_config(self, mocker):
        return mocker.Mock()

    @pytest.fixture
    def mock_layer(self, mocker):
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(self.qweight_mock, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(self.scale_mock, requires_grad=False)
        return layer

    def test_npu_int8_process_weights_after_loading(self, mock_layer, mock_quant_config, mock_torch_npu):
        from vllm_omni.diffusion.quantization.int8 import NPUInt8LinearMethod

        method = NPUInt8LinearMethod(mock_quant_config)
        ori_weight_shape = mock_layer.weight.shape

        method.process_weights_after_loading(mock_layer)

        assert mock_layer.weight.shape == ori_weight_shape[::-1]
        assert mock_layer.weight.is_contiguous()

    def test_npu_int8_apply(self, mock_layer, mock_quant_config, mock_torch_npu):
        from vllm_omni.diffusion.quantization.int8 import NPUInt8LinearMethod

        method = NPUInt8LinearMethod(mock_quant_config)
        x = torch.randn(1, 16, 64)

        output = method.apply(mock_layer, x)
        assert output.shape == (1, 16, 128)

    def test_npu_int8_online_process_weights(self, mock_layer, mock_quant_config, mock_torch_npu):
        from vllm_omni.diffusion.quantization.int8 import NPUInt8OnlineLinearMethod

        method = NPUInt8OnlineLinearMethod(mock_quant_config)
        method.process_weights_after_loading(mock_layer)

        assert mock_layer.weight.shape == (64, 128)
        assert torch.equal(mock_layer.weight_scale, self.scale_mock)
