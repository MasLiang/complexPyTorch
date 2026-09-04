"""BiReal-compatible complex residual models for San Francisco AIRSAR."""

import torch
import torch.nn.functional as F

from complexPyTorch.complexBinaryResNet import BinaryComplexResNet


STAGE_TO_PHASE = {
    "bireal_fp": 1,
    "bireal": 2,
    "bireal_lut": 3,
    "lut6_residual": 4,
    "lut6_lut4_residual": 5,
}

STAGE_TO_MODEL_PHASE = {
    stage: min(phase, 3) for stage, phase in STAGE_TO_PHASE.items()
}


def complex_adaptive_avg_pool2d(inp, output_size=1):
    """Adaptive average pooling that preserves the complex representation."""

    return torch.complex(
        F.adaptive_avg_pool2d(inp.real, output_size),
        F.adaptive_avg_pool2d(inp.imag, output_size),
    )


class SanFranciscoBiRealResNet(BinaryComplexResNet):
    """Complex BiReal network with pooling suitable for small PolSAR patches."""

    @staticmethod
    def _pad_for_downsampling(inp):
        pad_height = (-inp.size(-2)) % 4
        pad_width = (-inp.size(-1)) % 4
        if not pad_height and not pad_width:
            return inp
        return torch.complex(
            F.pad(inp.real, (0, pad_width, 0, pad_height), mode="replicate"),
            F.pad(inp.imag, (0, pad_width, 0, pad_height), mode="replicate"),
        )

    def forward_features(self, inp):
        if not torch.is_complex(inp):
            raise TypeError("SanFranciscoBiRealResNet expects complex input")

        inp = self._pad_for_downsampling(inp)
        output = self.bn1(self.conv1(inp))
        for block in self.stage2:
            output = block(output)
        for block in self.stage3:
            output = block(output)
        for block in self.stage4:
            output = block(output)
        return complex_adaptive_avg_pool2d(output, 1).flatten(1)

    def forward(self, inp):
        features = self.forward_features(inp)
        classifier_input = torch.cat((features.real, features.imag), dim=1)
        return self.fc(classifier_input)


def build_san_francisco_bireal_model(
    stage,
    in_channels=6,
    num_classes=5,
    start_filters=16,
    num_blocks=3,
    post_bn_mode="covariance",
    pair_lut_parameterization="categorical",
    shortcut_mode="fp",
):
    try:
        phase = STAGE_TO_MODEL_PHASE[stage]
    except KeyError as exc:
        raise ValueError("Unknown San Francisco BiReal stage: {}".format(stage)) from exc

    residual_stage = stage in ("lut6_residual", "lut6_lut4_residual")
    return SanFranciscoBiRealResNet(
        in_channels=in_channels,
        num_blocks=num_blocks,
        start_filters=start_filters,
        num_classes=num_classes,
        spectral_pool_scheme="none",
        per_channel=True,
        weight_grad_mode="ste",
        act_grad_mode="bireal",
        binary_stem=False,
        is_sar_input=True,
        is_binary=phase >= 2,
        phase=phase,
        post_bn_mode=post_bn_mode,
        phase3_operator="pair_lut4",
        pair_lut_parameterization=(
            "categorical_residual" if residual_stage else pair_lut_parameterization
        ),
        pair_lut_inputs=6 if residual_stage else 4,
        pair_lut_encoding="dominance" if residual_stage else "standard",
        dominance_grad_mode="stop",
        shortcut_mode=shortcut_mode,
    )
