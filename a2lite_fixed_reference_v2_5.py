from __future__ import annotations

import csv
import json
import math
import os
import random
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


# ============================================================================
# Version
# ============================================================================

REFERENCE_VERSION = "2.5"


# ============================================================================
# A2-Lite constants
# ============================================================================

CHANNELS = 3
BOTTLENECK = 3
NUM_LAYERS = 23

KERNEL_SIZES = (
    6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6,
    15, 15,
    6, 6, 6, 6, 6, 6, 6,
)

DILATIONS = (
    1, 3, 7, 17, 41, 101, 239,
    1, 3, 7, 17, 41, 101, 239,
    1, 13,
    1, 3, 7, 17, 41, 101, 239,
)

HEAD_KERNEL = 16

EXPECTED_PARAMETER_COUNT = 1871

Q_FRAC = 14
Q_SCALE = 1 << Q_FRAC

Q_MIN = -32768
Q_MAX = 32767

INT32_MIN = -2147483648
INT32_MAX = 2147483647

LEAKY_SLOPE = 0.01
LEAKY_NUM = 1
LEAKY_DEN = 100

TEST_SAMPLES = 4096
VECTOR_SAMPLES = TEST_SAMPLES

CORR_EPS = 1.0e-12

Q_MAX_ROUNDTRIP_ERROR = 0.5 / Q_SCALE


# ============================================================================
# Binary format
# ============================================================================

HEADER_SIZE = 32
MAGIC = b"A2LT"

HEADER_FORMAT = "<4s6H4I"

HEADER_VERSION = 2
HEADER_RESERVED = 0

EXPECTED_PAYLOAD_BYTES = EXPECTED_PARAMETER_COUNT * 2
EXPECTED_BINARY_SIZE = HEADER_SIZE + EXPECTED_PAYLOAD_BYTES


# ============================================================================
# Errors
# ============================================================================

class ValidationError(RuntimeError):
    pass


def fail(message: str) -> None:
    raise ValidationError(message)


# ============================================================================
# Dataclasses
# ============================================================================

@dataclass
class MappingEntry:
    source_index: int
    bf_index: int

    source_offset: int
    bf_offset: int

    layer: int
    kind: str

    tap: int = -1
    in_channel: int = -1
    out_channel: int = -1

    value: float = 0.0
    q_value: int = 0

    reconstruction_error: float = 0.0


@dataclass
class LayerInfo:
    layer: int
    kernel_size: int
    dilation: int

    source_base: int
    source_end: int

    bf_base: int
    bf_end: int


@dataclass
class LayerStats:
    layer: int
    kernel_size: int
    dilation: int

    activation_min: float = 0.0
    activation_max: float = 0.0
    activation_abs_max: float = 0.0

    output_min: float = 0.0
    output_max: float = 0.0
    output_abs_max: float = 0.0

    conv_max_accumulator: int = 0
    l1_max_accumulator: int = 0
    required_accumulator_bits: int = 0

    required_shift: int = 0

    int32_headroom_bits: int = 0
    int32_ok: bool = False


@dataclass
class VectorResult:
    name: str

    rms: float
    peak: float
    correlation: float

    int64_rms: float
    int64_peak: float

    int32_rms: float
    int32_peak: float

    passed: bool

@dataclass
class ArithmeticTrace:
    layer: int
    stage: str

    tap: int = -1
    in_channel: int = -1
    out_channel: int = -1

    int64_value: int = 0
    int32_value: int = 0

    int64_accumulator: int = 0
    int32_accumulator: int = 0

    int64_normalized: int = 0
    int32_normalized: int = 0

    int64_activation: int = 0
    int32_activation: int = 0

    difference: int = 0

    vector: str = ""


def make_trace(
    *,
    layer: int,
    stage: str,
    int64_value: int,
    int32_value: int,
    tap: int = -1,
    in_channel: int = -1,
    out_channel: int = -1,
    int64_accumulator: int = 0,
    int32_accumulator: int = 0,
    int64_normalized: int = 0,
    int32_normalized: int = 0,
    int64_activation: int = 0,
    int32_activation: int = 0,
) -> ArithmeticTrace | None:

    if int64_value == int32_value:
        return None

    return ArithmeticTrace(
        layer=layer,
        stage=stage,
        tap=tap,
        in_channel=in_channel,
        out_channel=out_channel,

        int64_value=int64_value,
        int32_value=int32_value,

        int64_accumulator=int64_accumulator,
        int32_accumulator=int32_accumulator,

        int64_normalized=int64_normalized,
        int32_normalized=int32_normalized,

        int64_activation=int64_activation,
        int32_activation=int32_activation,

        difference=int32_value - int64_value,
    )


# ============================================================================
# Rounding
# ============================================================================

def round_half_away_from_zero(x: float) -> int:
    if x >= 0.0:
        return int(math.floor(x + 0.5))

    return int(math.ceil(x - 0.5))


def round_div_away_from_zero(
    numerator: int,
    denominator: int,
) -> int:

    if denominator <= 0:
        raise ValueError("denominator must be positive")

    if numerator >= 0:
        return (
            numerator + denominator // 2
        ) // denominator

    return -(
        (-numerator + denominator // 2)
        // denominator
    )


# ============================================================================
# Q-format
# ============================================================================

def float_to_q(
    value: float,
    q_frac: int = Q_FRAC,
) -> int:

    scale = 1 << q_frac

    q = round_half_away_from_zero(
        float(value) * scale
    )

    return sat16(q)


def q_to_float(
    q: int,
    q_frac: int = Q_FRAC,
) -> float:

    return float(q) / float(1 << q_frac)


def sat16(x: int) -> int:

    if x > Q_MAX:
        return Q_MAX

    if x < Q_MIN:
        return Q_MIN

    return int(x)


def sat32(x: int) -> int:

    if x > INT32_MAX:
        return INT32_MAX

    if x < INT32_MIN:
        return INT32_MIN

    return int(x)


def fits_int32(x: int) -> bool:
    return INT32_MIN <= x <= INT32_MAX


# ============================================================================
# Integer multiplication
# ============================================================================

def mul_q14(
    a: int,
    b: int,
) -> int:

    return round_div_away_from_zero(
        int(a) * int(b),
        Q_SCALE,
    )


def arithmetic_shift_round(
    x: int,
    shift: int,
) -> int:

    if shift < 0:
        return int(x) << (-shift)

    if shift == 0:
        return int(x)

    return round_div_away_from_zero(
        int(x),
        1 << shift,
    )


# ============================================================================
# LeakyReLU
# ============================================================================

def leaky_relu_q64(x_q: int) -> int:

    if x_q >= 0:
        return int(x_q)

    return round_div_away_from_zero(
        int(x_q) * LEAKY_NUM,
        LEAKY_DEN,
    )


def leaky_relu_q32(x_q: int) -> int:

    if x_q >= 0:
        return int(x_q)

    return round_div_away_from_zero(
        int(x_q) * LEAKY_NUM,
        LEAKY_DEN,
    )


# ============================================================================
# Architecture validation
# ============================================================================

def get_activation(
    layer_cfg: dict,
    index: int,
) -> dict:

    activation = layer_cfg["activation"][index]

    if not isinstance(activation, dict):
        fail(
            f"Layer {index}: activation is not an object"
        )

    return activation


def validate_a2lite_config(
    model: dict,
) -> None:

    if model.get("architecture") != "WaveNet":
        fail(
            "Selected submodel is not WaveNet"
        )

    cfg = model.get("config")

    if not isinstance(cfg, dict):
        fail("Model has no config object")

    layers = cfg.get("layers")

    if not isinstance(layers, list):
        fail("config.layers is missing")

    if len(layers) != 1:
        fail(
            f"Expected exactly one layer array, got {len(layers)}"
        )

    if (
        "condition_dsp" in cfg
        and cfg["condition_dsp"] is not None
    ):
        fail(
            "A2-Lite fast path cannot contain condition_dsp"
        )

    if cfg.get("in_channels", 1) != 1:
        fail(
            "A2-Lite requires in_channels=1"
        )

    if "head_scale" not in cfg:
        fail(
            "config.head_scale is missing"
        )

    layer = layers[0]

    if layer.get("input_size") != 1:
        fail("input_size must be 1")

    if layer.get("condition_size") != 1:
        fail("condition_size must be 1")

    if layer.get("channels") != CHANNELS:
        fail(
            f"channels must be {CHANNELS}"
        )

    if layer.get("bottleneck") != BOTTLENECK:
        fail(
            f"bottleneck must be {BOTTLENECK}"
        )

    kernels = tuple(
        layer.get("kernel_sizes", [])
    )

    if kernels != KERNEL_SIZES:
        fail(
            "kernel_sizes mismatch\n"
            f"actual   = {kernels}\n"
            f"expected = {KERNEL_SIZES}"
        )

    dilations = tuple(
        layer.get("dilations", [])
    )

    if dilations != DILATIONS:
        fail(
            "dilations mismatch\n"
            f"actual   = {dilations}\n"
            f"expected = {DILATIONS}"
        )

    activations = layer.get("activation")

    if not isinstance(activations, list):
        fail("activation array missing")

    if len(activations) != NUM_LAYERS:
        fail(
            f"Expected {NUM_LAYERS} activations"
        )

    for i in range(NUM_LAYERS):

        activation = get_activation(
            layer,
            i,
        )

        if activation.get("type") != "LeakyReLU":
            fail(
                f"Layer {i}: activation must be LeakyReLU"
            )

        slope = float(
            activation.get(
                "negative_slope",
                -999.0,
            )
        )

        if abs(slope - LEAKY_SLOPE) > 1e-7:
            fail(
                f"Layer {i}: negative_slope={slope}, "
                f"expected {LEAKY_SLOPE}"
            )

    gating = layer.get("gating_mode")

    if gating is not None:

        if not isinstance(gating, list):
            fail("gating_mode must be an array")

        if len(gating) != NUM_LAYERS:
            fail("gating_mode length mismatch")

        if any(x != "none" for x in gating):
            fail(
                "A2-Lite requires gating_mode='none'"
            )

    secondary = layer.get(
        "secondary_activation"
    )

    if secondary is not None:

        if not isinstance(secondary, list):
            fail(
                "secondary_activation must be an array"
            )

        if len(secondary) != NUM_LAYERS:
            fail(
                "secondary_activation length mismatch"
            )

        if any(x is not None for x in secondary):
            fail(
                "A2-Lite requires no secondary activation"
            )

    head1x1 = layer.get("head1x1")

    if isinstance(head1x1, dict):
        if head1x1.get("active", False):
            fail(
                "head1x1 must be inactive"
            )

    layer1x1 = layer.get("layer1x1")

    if not isinstance(layer1x1, dict):
        fail(
            "layer1x1 configuration missing"
        )

    if not layer1x1.get("active", False):
        fail(
            "layer1x1 must be active"
        )

    if layer1x1.get("groups", 1) != 1:
        fail(
            "layer1x1 groups must be 1"
        )

    layer_head = layer.get("head")

    if not isinstance(layer_head, dict):
        fail(
            "layer head configuration missing"
        )

    if layer_head.get("out_channels") != 1:
        fail(
            "layer head out_channels must be 1"
        )

    if layer_head.get("kernel_size") != HEAD_KERNEL:
        fail(
            "layer head kernel_size must be 16"
        )

    if layer_head.get("head_dilation", 1) != 1:
        fail(
            "layer head dilation must be 1"
        )

    if not layer_head.get("bias", False):
        fail(
            "layer head bias must be active"
        )

    film_keys = (
        "conv_pre_film",
        "conv_post_film",
        "input_mixin_pre_film",
        "input_mixin_post_film",
        "activation_pre_film",
        "activation_post_film",
        "layer1x1_post_film",
        "head1x1_post_film",
    )

    for key in film_keys:

        value = layer.get(key)

        if isinstance(value, dict):
            if value.get("active", False):
                fail(
                    f"{key} must be inactive"
                )

    if layer.get("groups_input", 1) != 1:
        fail(
            "groups_input must be 1"
        )

    if layer.get("groups_input_mixin", 1) != 1:
        fail(
            "groups_input_mixin must be 1"
        )


# ============================================================================
# SlimmableContainer
# ============================================================================

def find_a2lite_model(
    root: dict,
) -> dict:

    architecture = root.get("architecture")

    if architecture == "WaveNet":

        model = root
        cfg = model.get("config", {})
        layers = cfg.get("layers", [])

        if (
            len(layers) == 1
            and layers[0].get("channels") == CHANNELS
        ):
            return model

        fail(
            "WaveNet input is not a 3-channel A2-Lite model"
        )

    if architecture != "SlimmableContainer":
        fail(
            "Unsupported top-level architecture: "
            f"{architecture!r}"
        )

    cfg = root.get("config", {})
    submodels = cfg.get("submodels")

    if not isinstance(submodels, list):
        fail(
            "SlimmableContainer has no submodels[]"
        )

    candidates = []

    for index, entry in enumerate(submodels):

        if not isinstance(entry, dict):
            continue

        model = entry.get("model")

        if not isinstance(model, dict):
            continue

        try:
            model_cfg = model["config"]
            layers = model_cfg["layers"]

            if (
                isinstance(layers, list)
                and len(layers) == 1
                and layers[0].get("channels") == CHANNELS
                and layers[0].get("bottleneck") == BOTTLENECK
            ):
                candidates.append(
                    (index, entry)
                )

        except (KeyError, TypeError):
            continue

    if len(candidates) != 1:
        fail(
            "Expected exactly one 3-channel A2-Lite "
            f"submodel, found {len(candidates)}"
        )

    index, entry = candidates[0]

    print(
        f"Selected SlimmableContainer submodel: {index}"
    )

    if "max_value" in entry:
        print(
            f"Submodel max_value: {entry['max_value']}"
        )

    return entry["model"]


# ============================================================================
# Parameter count
# ============================================================================

def layer_parameter_count(
    kernel_size: int,
) -> int:

    return (
        kernel_size * CHANNELS * CHANNELS
        + CHANNELS
        + CHANNELS
        + CHANNELS * CHANNELS
        + CHANNELS
    )


def expected_parameter_count() -> int:

    total = CHANNELS

    for k in KERNEL_SIZES:
        total += layer_parameter_count(k)

    total += HEAD_KERNEL * CHANNELS
    total += 1
    total += 1

    return total


# ============================================================================
# Weight extraction
# ============================================================================

def get_weights(
    model: dict,
) -> List[float]:

    weights = model.get("weights")

    if not isinstance(weights, list):
        fail(
            "Selected model has no weights[]"
        )

    values = [
        float(x)
        for x in weights
    ]

    expected = expected_parameter_count()

    if expected != EXPECTED_PARAMETER_COUNT:
        fail(
            f"Internal parameter-count error: "
            f"{expected} != {EXPECTED_PARAMETER_COUNT}"
        )

    if len(values) != expected:
        fail(
            "A2-Lite weight count mismatch: "
            f"{len(values)} != {expected}"
        )

    return values


# ============================================================================
# Mapping
# ============================================================================

def add_mapping(
    mappings: List[MappingEntry],
    *,
    source_index: int,
    bf_index: int,
    layer: int,
    kind: str,
    value: float,
    q_frac: int,
    tap: int = -1,
    in_channel: int = -1,
    out_channel: int = -1,
) -> int:

    q_value = float_to_q(
        value,
        q_frac,
    )

    reconstructed = q_to_float(
        q_value,
        q_frac,
    )

    reconstruction_error = (
        reconstructed - value
    )

    mappings.append(
        MappingEntry(
            source_index=source_index,
            bf_index=bf_index,

            source_offset=source_index * 4,
            bf_offset=bf_index * 2,

            layer=layer,
            kind=kind,

            tap=tap,
            in_channel=in_channel,
            out_channel=out_channel,

            value=value,
            q_value=q_value,

            reconstruction_error=reconstruction_error,
        )
    )

    return q_value


def convert_weights(
    model: dict,
) -> Tuple[
    List[int],
    List[MappingEntry],
    List[LayerInfo],
]:

    weights = get_weights(model)

    reader_index = 0

    bf_values: List[int] = []
    mappings: List[MappingEntry] = []
    layer_infos: List[LayerInfo] = []

    bf_index = 0

    # ------------------------------------------------------------
    # Rechannel
    # ------------------------------------------------------------

    for out_ch in range(CHANNELS):

        source_index = reader_index
        value = weights[reader_index]
        reader_index += 1

        q = add_mapping(
            mappings,
            source_index=source_index,
            bf_index=bf_index,
            layer=-1,
            kind="rechannel",
            in_channel=0,
            out_channel=out_ch,
            value=value,
            q_frac=Q_FRAC,
        )

        bf_values.append(q)
        bf_index += 1

    # ------------------------------------------------------------
    # Layers
    # ------------------------------------------------------------

    for li, (
        kernel_size,
        dilation,
    ) in enumerate(
        zip(KERNEL_SIZES, DILATIONS)
    ):

        source_base = reader_index
        bf_base = bf_index

        source_conv = {}

        for out_ch in range(CHANNELS):
            for in_ch in range(CHANNELS):
                for tap in range(kernel_size):

                    source_index = reader_index
                    value = weights[reader_index]
                    reader_index += 1

                    source_conv[
                        (out_ch, in_ch, tap)
                    ] = (
                        source_index,
                        value,
                    )

        for tap in range(kernel_size):
            for in_ch in range(CHANNELS):
                for out_ch in range(CHANNELS):

                    source_index, value = source_conv[
                        (out_ch, in_ch, tap)
                    ]

                    q = add_mapping(
                        mappings,
                        source_index=source_index,
                        bf_index=bf_index,
                        layer=li,
                        kind="conv",
                        tap=tap,
                        in_channel=in_ch,
                        out_channel=out_ch,
                        value=value,
                        q_frac=Q_FRAC,
                    )

                    bf_values.append(q)
                    bf_index += 1

        for out_ch in range(CHANNELS):

            source_index = reader_index
            value = weights[reader_index]
            reader_index += 1

            q = add_mapping(
                mappings,
                source_index=source_index,
                bf_index=bf_index,
                layer=li,
                kind="conv_bias",
                out_channel=out_ch,
                value=value,
                q_frac=Q_FRAC,
            )

            bf_values.append(q)
            bf_index += 1

        for out_ch in range(CHANNELS):

            source_index = reader_index
            value = weights[reader_index]
            reader_index += 1

            q = add_mapping(
                mappings,
                source_index=source_index,
                bf_index=bf_index,
                layer=li,
                kind="mixin",
                in_channel=0,
                out_channel=out_ch,
                value=value,
                q_frac=Q_FRAC,
            )

            bf_values.append(q)
            bf_index += 1

        source_l1 = {}

        for out_ch in range(CHANNELS):
            for in_ch in range(CHANNELS):

                source_index = reader_index
                value = weights[reader_index]
                reader_index += 1

                source_l1[
                    (out_ch, in_ch)
                ] = (
                    source_index,
                    value,
                )

        for in_ch in range(CHANNELS):
            for out_ch in range(CHANNELS):

                source_index, value = source_l1[
                    (out_ch, in_ch)
                ]

                q = add_mapping(
                    mappings,
                    source_index=source_index,
                    bf_index=bf_index,
                    layer=li,
                    kind="layer1x1",
                    in_channel=in_ch,
                    out_channel=out_ch,
                    value=value,
                    q_frac=Q_FRAC,
                )

                bf_values.append(q)
                bf_index += 1

        for out_ch in range(CHANNELS):

            source_index = reader_index
            value = weights[reader_index]
            reader_index += 1

            q = add_mapping(
                mappings,
                source_index=source_index,
                bf_index=bf_index,
                layer=li,
                kind="layer1x1_bias",
                out_channel=out_ch,
                value=value,
                q_frac=Q_FRAC,
            )

            bf_values.append(q)
            bf_index += 1

        source_end = reader_index - 1
        bf_end = bf_index - 1

        layer_infos.append(
            LayerInfo(
                layer=li,
                kernel_size=kernel_size,
                dilation=dilation,
                source_base=source_base,
                source_end=source_end,
                bf_base=bf_base,
                bf_end=bf_end,
            )
        )

    # ------------------------------------------------------------
    # Head
    # ------------------------------------------------------------

    source_head = {}

    for in_ch in range(CHANNELS):
        for tap in range(HEAD_KERNEL):

            source_index = reader_index
            value = weights[reader_index]
            reader_index += 1

            source_head[
                (in_ch, tap)
            ] = (
                source_index,
                value,
            )

    for tap in range(HEAD_KERNEL):
        for in_ch in range(CHANNELS):

            source_index, value = source_head[
                (in_ch, tap)
            ]

            q = add_mapping(
                mappings,
                source_index=source_index,
                bf_index=bf_index,
                layer=-1,
                kind="head",
                tap=tap,
                in_channel=in_ch,
                out_channel=0,
                value=value,
                q_frac=Q_FRAC,
            )

            bf_values.append(q)
            bf_index += 1

    source_index = reader_index
    value = weights[reader_index]
    reader_index += 1

    q = add_mapping(
        mappings,
        source_index=source_index,
        bf_index=bf_index,
        layer=-1,
        kind="head_bias",
        out_channel=0,
        value=value,
        q_frac=Q_FRAC,
    )

    bf_values.append(q)
    bf_index += 1

    source_index = reader_index
    value = weights[reader_index]
    reader_index += 1

    q = add_mapping(
        mappings,
        source_index=source_index,
        bf_index=bf_index,
        layer=-1,
        kind="head_scale",
        value=value,
        q_frac=Q_FRAC,
    )

    bf_values.append(q)
    bf_index += 1

    if reader_index != EXPECTED_PARAMETER_COUNT:
        fail(
            f"Consumed {reader_index} parameters, "
            f"expected {EXPECTED_PARAMETER_COUNT}"
        )

    if len(bf_values) != EXPECTED_PARAMETER_COUNT:
        fail(
            f"Generated {len(bf_values)} BF parameters, "
            f"expected {EXPECTED_PARAMETER_COUNT}"
        )

    source_indices = [
        m.source_index for m in mappings
    ]

    bf_indices = [
        m.bf_index for m in mappings
    ]

    if sorted(source_indices) != list(
        range(EXPECTED_PARAMETER_COUNT)
    ):
        fail(
            "Source mapping is not a complete "
            "0..1870 permutation"
        )

    if sorted(bf_indices) != list(
        range(EXPECTED_PARAMETER_COUNT)
    ):
        fail(
            "BF mapping is not a complete "
            "0..1870 permutation"
        )

    return (
        bf_values,
        mappings,
        layer_infos,
    )


# ============================================================================
# Receptive field
# ============================================================================

def calculate_receptive_field() -> int:

    rf = 1

    for k, d in zip(
        KERNEL_SIZES,
        DILATIONS,
    ):
        rf += (k - 1) * d

    rf += HEAD_KERNEL - 1

    return rf


# ============================================================================
# Binary header
# ============================================================================

def pack_header() -> bytes:

    return struct.pack(
        HEADER_FORMAT,
        MAGIC,
        HEADER_VERSION,
        CHANNELS,
        NUM_LAYERS,
        BOTTLENECK,
        HEAD_KERNEL,
        Q_FRAC,
        EXPECTED_PARAMETER_COUNT,
        EXPECTED_PAYLOAD_BYTES,
        HEADER_SIZE,
        HEADER_RESERVED,
    )


def parse_header(
    data: bytes,
) -> Dict[str, int]:

    if len(data) < HEADER_SIZE:
        fail(
            "Binary shorter than 32-byte header"
        )

    fields = struct.unpack(
        HEADER_FORMAT,
        data[:HEADER_SIZE],
    )

    return {
        "magic": fields[0],
        "version": fields[1],
        "channels": fields[2],
        "layers": fields[3],
        "bottleneck": fields[4],
        "head_kernel": fields[5],
        "q_frac": fields[6],
        "parameter_count": fields[7],
        "payload_bytes": fields[8],
        "header_bytes": fields[9],
        "reserved": fields[10],
    }


# ============================================================================
# Byte-exact binary validation
# ============================================================================

def validate_binary(
    binary_path: Path,
    expected_q: Sequence[int],
) -> Tuple[bool, bytes]:

    if not binary_path.exists():
        return False, b""

    data = binary_path.read_bytes()

    if len(data) != EXPECTED_BINARY_SIZE:
        fail(
            f"Binary size mismatch: "
            f"{len(data)} != {EXPECTED_BINARY_SIZE}"
        )

    header = parse_header(data)

    checks = (
        ("magic", MAGIC),
        ("version", HEADER_VERSION),
        ("channels", CHANNELS),
        ("layers", NUM_LAYERS),
        ("bottleneck", BOTTLENECK),
        ("head_kernel", HEAD_KERNEL),
        ("q_frac", Q_FRAC),
        ("parameter_count", EXPECTED_PARAMETER_COUNT),
        ("payload_bytes", EXPECTED_PAYLOAD_BYTES),
        ("header_bytes", HEADER_SIZE),
        ("reserved", HEADER_RESERVED),
    )

    for key, expected in checks:

        if header[key] != expected:
            fail(
                f"Binary header {key} mismatch: "
                f"{header[key]!r} != {expected!r}"
            )

    payload = data[
        HEADER_SIZE:
        HEADER_SIZE + EXPECTED_PAYLOAD_BYTES
    ]

    expected_payload = b"".join(
        struct.pack(
            "<h",
            sat16(q),
        )
        for q in expected_q
    )

    if payload != expected_payload:

        first = None

        for i, (
            actual,
            expected,
        ) in enumerate(
            zip(payload, expected_payload)
        ):
            if actual != expected:
                first = i
                break

        if first is None:
            first = min(
                len(payload),
                len(expected_payload),
            )

        fail(
            "BYTE-EXACT PAYLOAD MISMATCH at "
            f"byte {first}"
        )

    return True, payload


# ============================================================================
# Q2.14 validation
# ============================================================================

def validate_q_roundtrip(
    mappings: Sequence[MappingEntry],
) -> Tuple[float, float, int]:

    max_error = 0.0
    sum_sq = 0.0
    clipped = 0

    for m in mappings:

        raw = m.value * Q_SCALE

        q = m.q_value

        if (
            raw > Q_MAX + 0.5
            or raw < Q_MIN - 0.5
        ):
            clipped += 1

        error = m.reconstruction_error

        max_error = max(
            max_error,
            abs(error),
        )

        sum_sq += error * error

    rms = math.sqrt(
        sum_sq / len(mappings)
    )

    return (
        max_error,
        rms,
        clipped,
    )


# ============================================================================
# Parameter-group validation
# ============================================================================

def validate_head_weights(
    mappings: Sequence[MappingEntry],
) -> bool:

    values = [
        m for m in mappings
        if m.kind == "head"
    ]

    if len(values) != HEAD_KERNEL * CHANNELS:
        return False

    indices = sorted(
        m.source_index
        for m in values
    )

    return indices == list(
        range(1821, 1869)
    )


def validate_head_bias(
    mappings: Sequence[MappingEntry],
) -> bool:

    values = [
        m for m in mappings
        if m.kind == "head_bias"
    ]

    return (
        len(values) == 1
        and values[0].source_index == 1869
    )


def validate_head_scale(
    mappings: Sequence[MappingEntry],
) -> bool:

    values = [
        m for m in mappings
        if m.kind == "head_scale"
    ]

    return (
        len(values) == 1
        and values[0].source_index == 1870
    )


# ============================================================================
# Mapping validation
# ============================================================================

def validate_mapping(
    mappings: Sequence[MappingEntry],
) -> bool:

    if len(mappings) != EXPECTED_PARAMETER_COUNT:
        return False

    source = [
        m.source_index
        for m in mappings
    ]

    bf = [
        m.bf_index
        for m in mappings
    ]

    return (
        sorted(source)
        == list(range(EXPECTED_PARAMETER_COUNT))
        and
        sorted(bf)
        == list(range(EXPECTED_PARAMETER_COUNT))
    )


# ============================================================================
# Layout report
# ============================================================================

def print_layout(
    layer_infos: Sequence[LayerInfo],
) -> None:

    print()
    print("BF592 A2-Lite parameter layout")
    print("--------------------------------")
    print("Rechannel          : 0..2")

    for info in layer_infos:

        count = (
            info.bf_end
            - info.bf_base
            + 1
        )

        print(
            f"Layer {info.layer:2d}: "
            f"K={info.kernel_size:2d} "
            f"D={info.dilation:3d} "
            f"params={count:3d} "
            f"range={info.bf_base}..{info.bf_end}"
        )

    print("Head weights       : 1821..1868")
    print("Head bias          : 1869")
    print("Head scale         : 1870")
    print(
        f"Total parameters   : {EXPECTED_PARAMETER_COUNT}"
    )
    print(
        f"Payload            : {EXPECTED_PAYLOAD_BYTES} bytes"
    )


# ============================================================================
# Accumulator helpers
# ============================================================================

def signed_bits_required(
    value: int,
) -> int:

    value = abs(int(value))

    if value == 0:
        return 1

    return value.bit_length() + 1


def required_shift_for_int32(
    max_abs_accumulator: int,
) -> int:

    max_abs_accumulator = abs(
        int(max_abs_accumulator)
    )

    shift = 0

    while (
        max_abs_accumulator >> shift
        > INT32_MAX
    ):
        shift += 1

    return shift


# ============================================================================
# INT64 ideal reference
# ============================================================================

class Int64Reference:

    def __init__(
        self,
        q: Sequence[int],
    ):

        if len(q) != EXPECTED_PARAMETER_COUNT:
            raise ValueError(
                "invalid parameter count"
            )

        self.q = list(q)

        self.rechannel = self.q[0:3]

        self.layers = []

        p = 3

        for k, d in zip(
            KERNEL_SIZES,
            DILATIONS,
        ):

            conv_count = k * 9

            conv = self.q[
                p:
                p + conv_count
            ]
            p += conv_count

            conv_b = self.q[
                p:
                p + 3
            ]
            p += 3

            mixin = self.q[
                p:
                p + 3
            ]
            p += 3

            l1 = self.q[
                p:
                p + 9
            ]
            p += 9

            l1_b = self.q[
                p:
                p + 3
            ]
            p += 3

            self.layers.append(
                {
                    "k": k,
                    "d": d,
                    "conv": conv,
                    "conv_b": conv_b,
                    "mixin": mixin,
                    "l1": l1,
                    "l1_b": l1_b,
                }
            )

        self.head = self.q[
            p:
            p + HEAD_KERNEL * CHANNELS
        ]
        p += HEAD_KERNEL * CHANNELS

        self.head_bias = self.q[p]
        p += 1

        self.head_scale = self.q[p]
        p += 1

        if p != EXPECTED_PARAMETER_COUNT:
            raise ValueError(
                f"internal parser ended at {p}"
            )

    # ------------------------------------------------------------------
    # Layer
    # ------------------------------------------------------------------

    def process_layer(
        self,
        layer_index: int,
        history: List[List[int]],
        layer_input: Sequence[int],
        cond: int,
        stats: LayerStats | None = None,
    ) -> Tuple[List[int], List[int]]:

        L = self.layers[layer_index]

        K = L["k"]
        D = L["d"]

        history.append(
            list(layer_input)
        )

        current = len(history) - 1

        activated = [0, 0, 0]

        for out_ch in range(CHANNELS):

            acc = (
                int(L["conv_b"][out_ch])
                * Q_SCALE
            )

            for tap in range(K):

                distance = (
                    K - 1 - tap
                ) * D

                idx = current - distance

                if idx < 0:
                    x = (0, 0, 0)
                else:
                    x = history[idx]

                for in_ch in range(CHANNELS):

                    w = L["conv"][
                        tap * 9
                        + in_ch * 3
                        + out_ch
                    ]

                    acc += (
                        int(w)
                        * int(x[in_ch])
                    )

            acc += (
                int(L["mixin"][out_ch])
                * int(cond)
            )

            if stats is not None:
                stats.conv_max_accumulator = max(
                    stats.conv_max_accumulator,
                    abs(acc),
                )

            z = round_div_away_from_zero(
                acc,
                Q_SCALE,
            )

            activated[out_ch] = leaky_relu_q64(z)

        result = list(layer_input)

        for out_ch in range(CHANNELS):

            acc = (
                int(L["l1_b"][out_ch])
                * Q_SCALE
            )

            for in_ch in range(CHANNELS):

                w = L["l1"][
                    in_ch * 3
                    + out_ch
                ]

                acc += (
                    int(w)
                    * int(activated[in_ch])
                )

            if stats is not None:
                stats.l1_max_accumulator = max(
                    stats.l1_max_accumulator,
                    abs(acc),
                )

            residual = round_div_away_from_zero(
                acc,
                Q_SCALE,
            )

            result[out_ch] = sat16(
                int(result[out_ch])
                + int(residual)
            )

        return result, activated

    # ------------------------------------------------------------------
    # Debug layer (2.5 first-divergence trace)
    # ------------------------------------------------------------------

    def debug_layer(
        self,
        layer_index: int,
        history: List[List[int]],
        layer_input: Sequence[int],
        cond: int,
    ) -> dict:

        L = self.layers[layer_index]

        K = L["k"]
        D = L["d"]

        history.append(
            list(layer_input)
        )

        current = len(history) - 1

        result = list(layer_input)
        activated = [0, 0, 0]

        conv_acc = [0, 0, 0]
        conv_norm = [0, 0, 0]

        for out_ch in range(CHANNELS):

            acc = (
                int(L["conv_b"][out_ch])
                * Q_SCALE
            )

            for tap in range(K):

                distance = (
                    K - 1 - tap
                ) * D

                idx = current - distance

                if idx < 0:
                    x = (0, 0, 0)
                else:
                    x = history[idx]

                for in_ch in range(CHANNELS):

                    w = L["conv"][
                        tap * 9
                        + in_ch * 3
                        + out_ch
                    ]

                    acc += (
                        int(w)
                        * int(x[in_ch])
                    )

            # --------------------------------------------------------
            # condition mixin (2.4)
            # --------------------------------------------------------

            acc += (
                int(L["mixin"][out_ch])
                * int(cond)
            )

            conv_acc[out_ch] = acc

            z = round_div_away_from_zero(
                acc,
                Q_SCALE,
            )

            conv_norm[out_ch] = z

            activated[out_ch] = leaky_relu_q64(z)

        for out_ch in range(CHANNELS):

            acc = (
                int(L["l1_b"][out_ch])
                * Q_SCALE
            )

            for in_ch in range(CHANNELS):

                w = L["l1"][
                    in_ch * 3
                    + out_ch
                ]

                acc += (
                    int(w)
                    * int(activated[in_ch])
                )

            residual = round_div_away_from_zero(
                acc,
                Q_SCALE,
            )

            result[out_ch] = sat16(
                int(result[out_ch])
                + int(residual)
            )

        return {
            "input": list(layer_input),
            "cond": int(cond),
            "conv_acc": conv_acc,
            "conv_norm": conv_norm,
            "activated": list(activated),
            "output": list(result),
        }

    # ------------------------------------------------------------------
    # Full process
    # ------------------------------------------------------------------

    def process(
        self,
        samples: Sequence[int],
        collect_stats: bool = False,
    ) -> Tuple[List[int], List[LayerStats]]:

        histories = [
            []
            for _ in range(NUM_LAYERS)
        ]

        layer_stats = [
            LayerStats(
                layer=i,
                kernel_size=KERNEL_SIZES[i],
                dilation=DILATIONS[i],
            )
            for i in range(NUM_LAYERS)
        ]

        # Rechannel.
        layer_input_block = []

        for sample in samples:

            x = []

            for c in range(CHANNELS):

                value = round_div_away_from_zero(
                    int(self.rechannel[c])
                    * int(sample),
                    Q_SCALE,
                )

                x.append(
                    sat16(value)
                )

            layer_input_block.append(x)

        # One and only one network execution.
        head_source = []

        for n, sample in enumerate(samples):

            x = layer_input_block[n]
            cond = int(sample)

            layer_activation_sum = [
                0,
                0,
                0,
            ]

            for li in range(NUM_LAYERS):

                x, activated = self.process_layer(
                    li,
                    histories[li],
                    x,
                    cond,
                    layer_stats[li]
                    if collect_stats
                    else None,
                )

                for c in range(CHANNELS):
                    layer_activation_sum[c] += (
                        int(activated[c])
                    )

                if collect_stats:

                    st = layer_stats[li]

                    vals = [
                        q_to_float(v)
                        for v in activated
                    ]

                    out_vals = [
                        q_to_float(v)
                        for v in x
                    ]

                    if n == 0:

                        st.activation_min = min(vals)
                        st.activation_max = max(vals)

                        st.output_min = min(out_vals)
                        st.output_max = max(out_vals)

                    else:

                        st.activation_min = min(
                            st.activation_min,
                            min(vals),
                        )

                        st.activation_max = max(
                            st.activation_max,
                            max(vals),
                        )

                        st.output_min = min(
                            st.output_min,
                            min(out_vals),
                        )

                        st.output_max = max(
                            st.output_max,
                            max(out_vals),
                        )

                    st.activation_abs_max = max(
                        st.activation_abs_max,
                        max(abs(v) for v in vals),
                    )

                    st.output_abs_max = max(
                        st.output_abs_max,
                        max(abs(v) for v in out_vals),
                    )

            head_source.append(
                layer_activation_sum
            )

        # Finalize layer statistics.
        if collect_stats:

            for st in layer_stats:

                max_acc = max(
                    st.conv_max_accumulator,
                    st.l1_max_accumulator,
                )

                st.required_accumulator_bits = (
                    signed_bits_required(max_acc)
                )

                st.required_shift = (
                    required_shift_for_int32(
                        max_acc
                    )
                )

                shifted = (
                    max_acc
                    >> st.required_shift
                )

                st.int32_ok = (
                    shifted <= INT32_MAX
                )

                if shifted == 0:
                    st.int32_headroom_bits = 31
                else:
                    st.int32_headroom_bits = max(
                        0,
                        31 - shifted.bit_length(),
                    )

        # Head.
        head_history: List[List[int]] = []

        outputs: List[int] = []

        for n in range(len(samples)):

            head_history.append(
                head_source[n]
            )

            acc = (
                int(self.head_bias)
                * Q_SCALE
            )

            for tap in range(HEAD_KERNEL):

                distance = (
                    HEAD_KERNEL
                    - 1
                    - tap
                )

                idx = n - distance

                if idx < 0:
                    src = (0, 0, 0)
                else:
                    src = head_history[idx]

                for ch in range(CHANNELS):

                    w = self.head[
                        tap * CHANNELS
                        + ch
                    ]

                    acc += (
                        int(w)
                        * int(src[ch])
                    )

            y_q = round_div_away_from_zero(
                acc,
                Q_SCALE,
            )

            y_q = round_div_away_from_zero(
                y_q * int(self.head_scale),
                Q_SCALE,
            )

            outputs.append(
                sat16(y_q)
            )

        return outputs, layer_stats


# ============================================================================
# INT32 BF592 reference
# ============================================================================

class Int32BF592Reference:

    def __init__(
        self,
        q: Sequence[int],
        layer_stats: Sequence[LayerStats],
        trace: bool = True,
    ):
        self.q = list(q)
        self.layer_stats = list(layer_stats)

        self.trace = trace
        self.first_divergence_reported = False

        self.rechannel = self.q[0:3]

        self.layers = []

        p = 3

        for k, d in zip(
            KERNEL_SIZES,
            DILATIONS,
        ):
            conv_count = k * 9

            conv = self.q[
                p:
                p + conv_count
            ]
            p += conv_count

            conv_b = self.q[
                p:
                p + 3
            ]
            p += 3

            mixin = self.q[
                p:
                p + 3
            ]
            p += 3

            l1 = self.q[
                p:
                p + 9
            ]
            p += 9

            l1_b = self.q[
                p:
                p + 3
            ]
            p += 3

            self.layers.append(
                {
                    "k": k,
                    "d": d,
                    "conv": conv,
                    "conv_b": conv_b,
                    "mixin": mixin,
                    "l1": l1,
                    "l1_b": l1_b,
                }
            )

        self.head = self.q[
            p:
            p + HEAD_KERNEL * CHANNELS
        ]
        p += HEAD_KERNEL * CHANNELS

        self.head_bias = self.q[p]
        p += 1

        self.head_scale = self.q[p]
        p += 1

        if p != EXPECTED_PARAMETER_COUNT:
            raise ValueError(
                "INT32 parser mismatch"
            )

    def process_layer(
        self,
        layer_index: int,
        history: List[List[int]],
        layer_input: Sequence[int],
        cond: int,
    ) -> Tuple[List[int], List[int]]:

        L = self.layers[layer_index]

        K = L["k"]
        D = L["d"]

        stats = self.layer_stats[layer_index]
        shift = stats.required_shift

        history.append(
            list(layer_input)
        )

        current = len(history) - 1

        activated = [0, 0, 0]

        for out_ch in range(CHANNELS):

            acc = (
                int(L["conv_b"][out_ch])
                * Q_SCALE
            )

            for tap in range(K):

                distance = (
                    K - 1 - tap
                ) * D

                idx = current - distance

                if idx < 0:
                    x = (0, 0, 0)
                else:
                    x = history[idx]

                for in_ch in range(CHANNELS):

                    w = L["conv"][
                        tap * 9
                        + in_ch * 3
                        + out_ch
                    ]

                    acc += (
                        int(w)
                        * int(x[in_ch])
                    )

            # --------------------------------------------------------
            # 2.4 CRITICAL FIX
            # --------------------------------------------------------
            # The INT64 reference includes the condition mixin.
            # The 2.3 INT32 path accidentally omitted it.
            #
            # Both references must therefore calculate:
            #
            #     conv + bias + mixin * condition
            #
            # before Q4.28 -> Q2.14 normalization.
            # --------------------------------------------------------

            acc += (
                int(L["mixin"][out_ch])
                * int(cond)
            )

            acc_scaled = arithmetic_shift_round(
                acc,
                shift,
            )

            acc_scaled = sat32(
                acc_scaled
            )

            z = round_div_away_from_zero(
                acc_scaled,
                Q_SCALE,
            )

            activated[out_ch] = sat32(
                leaky_relu_q32(z)
            )

        result = list(layer_input)

        for out_ch in range(CHANNELS):

            acc = (
                int(L["l1_b"][out_ch])
                * Q_SCALE
            )

            for in_ch in range(CHANNELS):

                w = L["l1"][
                    in_ch * 3
                    + out_ch
                ]

                acc += (
                    int(w)
                    * int(activated[in_ch])
                )

            acc_scaled = arithmetic_shift_round(
                acc,
                shift,
            )

            acc_scaled = sat32(
                acc_scaled
            )

            residual = round_div_away_from_zero(
                acc_scaled,
                Q_SCALE,
            )

            result[out_ch] = sat16(
                int(result[out_ch])
                + int(residual)
            )

        return result, activated

    # ------------------------------------------------------------------
    # Debug layer (2.5 first-divergence trace)
    # ------------------------------------------------------------------

    def debug_layer(
        self,
        layer_index: int,
        history: List[List[int]],
        layer_input: Sequence[int],
        cond: int,
    ) -> dict:

        L = self.layers[layer_index]

        K = L["k"]
        D = L["d"]

        stats = self.layer_stats[layer_index]
        shift = stats.required_shift

        history.append(
            list(layer_input)
        )

        current = len(history) - 1

        result = list(layer_input)
        activated = [0, 0, 0]

        conv_acc = [0, 0, 0]
        conv_norm = [0, 0, 0]

        for out_ch in range(CHANNELS):

            acc = (
                int(L["conv_b"][out_ch])
                * Q_SCALE
            )

            for tap in range(K):

                distance = (
                    K - 1 - tap
                ) * D

                idx = current - distance

                if idx < 0:
                    x = (0, 0, 0)
                else:
                    x = history[idx]

                for in_ch in range(CHANNELS):

                    w = L["conv"][
                        tap * 9
                        + in_ch * 3
                        + out_ch
                    ]

                    acc += (
                        int(w)
                        * int(x[in_ch])
                    )

            # --------------------------------------------------------
            # condition mixin (2.4)
            # --------------------------------------------------------

            acc += (
                int(L["mixin"][out_ch])
                * int(cond)
            )

            conv_acc[out_ch] = acc

            # --------------------------------------------------------
            # INT32 scale / saturate / normalize
            # --------------------------------------------------------

            acc_scaled = arithmetic_shift_round(
                acc,
                shift,
            )

            acc_scaled = sat32(
                acc_scaled
            )

            z = round_div_away_from_zero(
                acc_scaled,
                Q_SCALE,
            )

            conv_norm[out_ch] = z

            activated[out_ch] = sat32(
                leaky_relu_q32(z)
            )

        for out_ch in range(CHANNELS):

            acc = (
                int(L["l1_b"][out_ch])
                * Q_SCALE
            )

            for in_ch in range(CHANNELS):

                w = L["l1"][
                    in_ch * 3
                    + out_ch
                ]

                acc += (
                    int(w)
                    * int(activated[in_ch])
                )

            acc_scaled = arithmetic_shift_round(
                acc,
                shift,
            )

            acc_scaled = sat32(
                acc_scaled
            )

            residual = round_div_away_from_zero(
                acc_scaled,
                Q_SCALE,
            )

            result[out_ch] = sat16(
                int(result[out_ch])
                + int(residual)
            )

        return {
            "input": list(layer_input),
            "cond": int(cond),
            "conv_acc": conv_acc,
            "conv_norm": conv_norm,
            "activated": list(activated),
            "output": list(result),
        }

    def process(
        self,
        samples: Sequence[int],
    ) -> List[int]:

        histories = [
            []
            for _ in range(NUM_LAYERS)
        ]

        layer_input_block = []

        for sample in samples:

            x = []

            for c in range(CHANNELS):

                value = round_div_away_from_zero(
                    int(self.rechannel[c])
                    * int(sample),
                    Q_SCALE,
                )

                x.append(
                    sat16(value)
                )

            layer_input_block.append(x)

        head_source = []

        for n, sample in enumerate(samples):

            x = layer_input_block[n]
            cond = int(sample)

            layer_activation_sum = [
                0,
                0,
                0,
            ]

            for li in range(NUM_LAYERS):

                x, activated = self.process_layer(
                    li,
                    histories[li],
                    x,
                    cond,
                )

                for c in range(CHANNELS):
                    layer_activation_sum[c] += (
                        int(activated[c])
                    )

            head_source.append(
                layer_activation_sum
            )

        head_history: List[List[int]] = []

        outputs = []

        for n in range(len(samples)):

            head_history.append(
                head_source[n]
            )

            acc = (
                int(self.head_bias)
                * Q_SCALE
            )

            for tap in range(HEAD_KERNEL):

                distance = (
                    HEAD_KERNEL
                    - 1
                    - tap
                )

                idx = n - distance

                if idx < 0:
                    src = (0, 0, 0)
                else:
                    src = head_history[idx]

                for ch in range(CHANNELS):

                    w = self.head[
                        tap * CHANNELS
                        + ch
                    ]

                    acc += (
                        int(w)
                        * int(src[ch])
                    )

            y_q = round_div_away_from_zero(
                acc,
                Q_SCALE,
            )

            y_q = round_div_away_from_zero(
                y_q * int(self.head_scale),
                Q_SCALE,
            )

            outputs.append(
                sat16(y_q)
            )

        return outputs


# ============================================================================
# Comparison
# ============================================================================

def correlation(
    a: Sequence[int],
    b: Sequence[int],
) -> float:

    if len(a) != len(b) or not a:
        return 0.0

    mean_a = sum(a) / len(a)
    mean_b = sum(b) / len(b)

    num = 0.0
    da = 0.0
    db = 0.0

    for x, y in zip(a, b):

        xa = x - mean_a
        yb = y - mean_b

        num += xa * yb
        da += xa * xa
        db += yb * yb

    denom = math.sqrt(
        da * db
    )

    if denom <= CORR_EPS:
        if all(x == y for x, y in zip(a, b)):
            return 1.0

        return 0.0

    return num / denom


def error_metrics(
    reference: Sequence[int],
    actual: Sequence[int],
) -> Tuple[float, float, float]:

    if len(reference) != len(actual):
        raise ValueError(
            "length mismatch"
        )

    if not reference:
        return 0.0, 0.0, 1.0

    sum_sq = 0.0
    peak = 0.0

    for a, b in zip(
        reference,
        actual,
    ):

        e = q_to_float(
            int(b) - int(a)
        )

        sum_sq += e * e
        peak = max(
            peak,
            abs(e),
        )

    rms = math.sqrt(
        sum_sq / len(reference)
    )

    corr = correlation(
        reference,
        actual,
    )

    return (
        rms,
        peak,
        corr,
    )


# ============================================================================
# Independent test vectors
# ============================================================================

def make_zero_vector() -> List[int]:

    return [
        0
    ] * VECTOR_SAMPLES


def make_impulse(
    amplitude: int = Q_SCALE,
) -> List[int]:

    values = [
        0
    ] * VECTOR_SAMPLES

    values[0] = amplitude

    return values


def make_negative_impulse() -> List[int]:

    return make_impulse(
        -Q_SCALE
    )


def make_full_positive() -> List[int]:

    return [
        Q_SCALE
    ] * VECTOR_SAMPLES


def make_full_negative() -> List[int]:

    return [
        -Q_SCALE
    ] * VECTOR_SAMPLES


def make_alternating() -> List[int]:

    return [
        Q_SCALE if i & 1 else -Q_SCALE
        for i in range(VECTOR_SAMPLES)
    ]


def make_low_level() -> List[int]:

    return [
        Q_SCALE // 16
    ] * VECTOR_SAMPLES


def make_random_vector(
    seed: int,
) -> List[int]:

    rng = random.Random(seed)

    return [
        rng.randint(
            -Q_SCALE,
            Q_SCALE,
        )
        for _ in range(VECTOR_SAMPLES)
    ]


def build_test_vectors() -> Dict[str, List[int]]:

    return {
        "zero": make_zero_vector(),
        "impulse": make_impulse(),
        "negative_impulse": make_negative_impulse(),
        "full_positive": make_full_positive(),
        "full_negative": make_full_negative(),
        "alternating": make_alternating(),
        "low_level": make_low_level(),
        "random_seed_A2": make_random_vector(
            0xA2
        ),
    }


# ============================================================================
# First-divergence trace
# ============================================================================

TRACE_VECTORS = (
    ("zero", make_zero_vector()),
    ("impulse", make_impulse()),
    ("negative_impulse", make_negative_impulse()),
    ("full_positive", make_full_positive()),
    ("full_negative", make_full_negative()),
    ("alternating", make_alternating()),
    ("low_level", make_low_level()),
    ("random_seed_A2", make_random_vector(0xA2)),
)


def _trace_first_sample(
    int64_ref: Int64Reference,
    int32_ref: Int32BF592Reference,
    sample: int,
) -> ArithmeticTrace | None:

    # Először nullával keressük.
    # Ez különösen fontos, mert a 2.4-ben a zero vector
    # még mindig jelentős hibát mutat.

    i64_history = [
        []
        for _ in range(NUM_LAYERS)
    ]

    i32_history = [
        []
        for _ in range(NUM_LAYERS)
    ]

    # --------------------------------------------------------
    # Rechannel
    # --------------------------------------------------------

    i64_input = [
        sat16(
            round_div_away_from_zero(
                int(int64_ref.rechannel[c])
                * sample,
                Q_SCALE,
            )
        )
        for c in range(CHANNELS)
    ]

    i32_input = [
        sat16(
            round_div_away_from_zero(
                int(int32_ref.rechannel[c])
                * sample,
                Q_SCALE,
            )
        )
        for c in range(CHANNELS)
    ]

    for c in range(CHANNELS):

        if i64_input[c] != i32_input[c]:

            return ArithmeticTrace(
                layer=-1,
                stage="rechannel",
                in_channel=0,
                out_channel=c,
                int64_value=i64_input[c],
                int32_value=i32_input[c],
                difference=(
                    i32_input[c]
                    - i64_input[c]
                ),
            )

    x64 = i64_input
    x32 = i32_input

    cond = sample

    for li in range(NUM_LAYERS):

        d64 = int64_ref.debug_layer(
            li,
            i64_history[li],
            x64,
            cond,
        )

        d32 = int32_ref.debug_layer(
            li,
            i32_history[li],
            x32,
            cond,
        )

        # --------------------------------------------------------
        # Conv accumulator
        # --------------------------------------------------------

        for ch in range(CHANNELS):

            if (
                d64["conv_acc"][ch]
                != d32["conv_acc"][ch]
            ):

                return ArithmeticTrace(
                    layer=li,
                    stage="conv_acc",
                    out_channel=ch,
                    int64_value=d64["conv_acc"][ch],
                    int32_value=d32["conv_acc"][ch],
                    int64_accumulator=d64["conv_acc"][ch],
                    int32_accumulator=d32["conv_acc"][ch],
                    difference=(
                        d32["conv_acc"][ch]
                        - d64["conv_acc"][ch]
                    ),
                )

        # --------------------------------------------------------
        # Conv normalization
        # --------------------------------------------------------

        for ch in range(CHANNELS):

            if (
                d64["conv_norm"][ch]
                != d32["conv_norm"][ch]
            ):

                return ArithmeticTrace(
                    layer=li,
                    stage="conv_normalize",
                    out_channel=ch,
                    int64_value=d64["conv_norm"][ch],
                    int32_value=d32["conv_norm"][ch],
                    int64_accumulator=d64["conv_acc"][ch],
                    int32_accumulator=d32["conv_acc"][ch],
                    int64_normalized=d64["conv_norm"][ch],
                    int32_normalized=d32["conv_norm"][ch],
                    difference=(
                        d32["conv_norm"][ch]
                        - d64["conv_norm"][ch]
                    ),
                )

        # --------------------------------------------------------
        # Activation
        # --------------------------------------------------------

        for ch in range(CHANNELS):

            if (
                d64["activated"][ch]
                != d32["activated"][ch]
            ):

                return ArithmeticTrace(
                    layer=li,
                    stage="activation",
                    out_channel=ch,
                    int64_value=d64["activated"][ch],
                    int32_value=d32["activated"][ch],
                    int64_normalized=d64["conv_norm"][ch],
                    int32_normalized=d32["conv_norm"][ch],
                    int64_activation=d64["activated"][ch],
                    int32_activation=d32["activated"][ch],
                    difference=(
                        d32["activated"][ch]
                        - d64["activated"][ch]
                    ),
                )

        # --------------------------------------------------------
        # Layer output / residual
        # --------------------------------------------------------

        for ch in range(CHANNELS):

            if (
                d64["output"][ch]
                != d32["output"][ch]
            ):

                return ArithmeticTrace(
                    layer=li,
                    stage="layer_output",
                    out_channel=ch,
                    int64_value=d64["output"][ch],
                    int32_value=d32["output"][ch],
                    difference=(
                        d32["output"][ch]
                        - d64["output"][ch]
                    ),
                )

        x64 = d64["output"]
        x32 = d32["output"]

    return None


def find_first_divergence(
    int64_ref: Int64Reference,
    int32_ref: Int32BF592Reference,
) -> ArithmeticTrace | None:

    # 2.5 first round:
    #
    # Every trace vector is swept, but only its first
    # sample (n=0, empty history) is traced.
    #
    # The zero vector showed no first-sample divergence,
    # so the sweep continues with impulse, full-scale,
    # alternating, ... vectors.

    for name, samples in TRACE_VECTORS:

        sample = int(samples[0])

        trace = _trace_first_sample(
            int64_ref,
            int32_ref,
            sample,
        )

        if trace is not None:

            trace.vector = name

            return trace

    return None


# ============================================================================
# Test-vector validation
# ============================================================================

def run_test_vectors(
    int64_ref: Int64Reference,
    int32_ref: Int32BF592Reference,
) -> Tuple[
    bool,
    List[VectorResult],
]:

    vectors = build_test_vectors()

    results = []

    all_pass = True

    print()
    print("Running independent test vectors...")
    print("------------------------------------")

    for name, vector in vectors.items():

        int64_output = int64_ref.process(
            vector,
            collect_stats=False,
        )[0]

        int32_output = int32_ref.process(
            vector
        )

        rms, peak, corr = error_metrics(
            int64_output,
            int32_output,
        )

        i64_values = [
            q_to_float(x)
            for x in int64_output
        ]

        i32_values = [
            q_to_float(x)
            for x in int32_output
        ]

        int64_peak = max(
            abs(x)
            for x in i64_values
        ) if i64_values else 0.0

        int32_peak = max(
            abs(x)
            for x in i32_values
        ) if i32_values else 0.0

        # Exact fixed-point reproduction is the goal.
        passed = (
            rms <= Q_MAX_ROUNDTRIP_ERROR
            and corr >= 0.999999
        )

        if not passed:
            all_pass = False

        print(
            f"{name:20s} "
            f"RMS={rms:.12g} "
            f"PEAK={peak:.12g} "
            f"CORR={corr:.12g} "
            f"[{'PASS' if passed else 'FAIL'}]"
        )

        results.append(
            VectorResult(
                name=name,
                rms=rms,
                peak=peak,
                correlation=corr,
                int64_rms=0.0,
                int64_peak=int64_peak,
                int32_rms=0.0,
                int32_peak=int32_peak,
                passed=passed,
            )
        )

    return all_pass, results


# ============================================================================
# INT64 self-test
# ============================================================================

def validate_int64_self_test(
    int64_ref: Int64Reference,
) -> bool:

    vectors = build_test_vectors()

    for name, vector in vectors.items():

        a = int64_ref.process(
            vector,
            collect_stats=False,
        )[0]

        b = int64_ref.process(
            vector,
            collect_stats=False,
        )[0]

        if a != b:
            print(
                f"INT64 self-test mismatch: {name}"
            )

            return False

    return True


# ============================================================================
# Accumulator validation
# ============================================================================

def validate_accumulators(
    stats: Sequence[LayerStats],
) -> bool:

    print()
    print("Accumulator validation")
    print("----------------------")

    all_pass = True

    print(
        "Layer K D   conv_acc       l1_acc         "
        "bits shift headroom int32"
    )

    for st in stats:

        max_acc = max(
            st.conv_max_accumulator,
            st.l1_max_accumulator,
        )

        passed = st.int32_ok

        if not passed:
            all_pass = False

        print(
            f"{st.layer:5d} "
            f"{st.kernel_size:1d} "
            f"{st.dilation:1d} "
            f"{max(st.conv_max_accumulator, 0):13d} "
            f"{max(st.l1_max_accumulator, 0):13d} "
            f"{st.required_accumulator_bits:4d} "
            f"{st.required_shift:5d} "
            f"{st.int32_headroom_bits:8d} "
            f"{'PASS' if passed else 'FAIL'}"
        )

    return all_pass


# ============================================================================
# Layer report
# ============================================================================

def print_layer_stats(
    stats: Sequence[LayerStats],
) -> None:

    print()
    print("Layer activation / accumulator analysis")
    print("----------------------------------------")

    print(
        "Layer K D  act_min      act_max      "
        "abs_max      acc_max       bits shift headroom"
    )

    for st in stats:

        acc_max = max(
            st.conv_max_accumulator,
            st.l1_max_accumulator,
        )

        print(
            f"{st.layer:5d} "
            f"{st.kernel_size:1d} "
            f"{st.dilation:3d} "
            f"{st.activation_min: .7f} "
            f"{st.activation_max: .7f} "
            f"{st.activation_abs_max: .7f} "
            f"{acc_max:13d} "
            f"{st.required_accumulator_bits:4d} "
            f"{st.required_shift:4d} "
            f"{st.int32_headroom_bits:8d}"
        )


# ============================================================================
# CSV / report
# ============================================================================

def write_offsets(
    path: Path,
    mappings: Sequence[MappingEntry],
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "source_index",
                "bf_index",
                "source_offset",
                "bf_offset",
                "layer",
                "kind",
                "tap",
                "in_channel",
                "out_channel",
                "value",
                "q_value",
                "reconstruction_error",
            ]
        )

        for m in mappings:

            writer.writerow(
                [
                    m.source_index,
                    m.bf_index,
                    m.source_offset,
                    m.bf_offset,
                    m.layer,
                    m.kind,
                    m.tap,
                    m.in_channel,
                    m.out_channel,
                    f"{m.value:.17g}",
                    m.q_value,
                    f"{m.reconstruction_error:.17g}",
                ]
            )


def write_report(
    path: Path,
    *,
    quantization_pass: bool,
    mapping_pass: bool,
    int64_pass: bool,
    int32_pass: bool,
    vectors_pass: bool,
    accumulator_pass: bool,
    q_max_error: float,
    q_rms_error: float,
    clipped: int,
    receptive_field: int,
    stats: Sequence[LayerStats],
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            f"A2-Lite fixed reference {REFERENCE_VERSION}\n"
        )
        f.write("=" * 60 + "\n\n")

        f.write(
            f"Receptive field: {receptive_field} samples\n\n"
        )

        f.write("Quantization\n")
        f.write("------------\n")
        f.write(
            f"Maximum error : {q_max_error:.17g}\n"
        )
        f.write(
            f"RMS error     : {q_rms_error:.17g}\n"
        )
        f.write(
            f"Clipped       : {clipped}\n"
        )
        f.write(
            f"Result        : "
            f"{'PASS' if quantization_pass else 'FAIL'}\n\n"
        )

        f.write("Layer analysis\n")
        f.write("--------------\n")

        for st in stats:

            acc_max = max(
                st.conv_max_accumulator,
                st.l1_max_accumulator,
            )

            f.write(
                f"Layer {st.layer:2d}: "
                f"act=[{st.activation_min:.9g}, "
                f"{st.activation_max:.9g}] "
                f"abs={st.activation_abs_max:.9g} "
                f"acc={acc_max} "
                f"bits={st.required_accumulator_bits} "
                f"shift={st.required_shift} "
                f"headroom={st.int32_headroom_bits}\n"
            )

        f.write("\nReference validation\n")
        f.write("=====================\n\n")

        f.write(
            f"QUANTIZATION       "
            f"{'PASS' if quantization_pass else 'FAIL'}\n"
        )

        f.write(
            f"MAPPING            "
            f"{'PASS' if mapping_pass else 'FAIL'}\n"
        )

        f.write(
            f"INT64 REFERENCE    "
            f"{'PASS' if int64_pass else 'FAIL'}\n"
        )

        f.write(
            f"INT32 BF592        "
            f"{'PASS' if int32_pass else 'FAIL'}\n"
        )

        f.write(
            f"TEST VECTORS       "
            f"{'PASS' if vectors_pass else 'FAIL'}\n"
        )

        f.write(
            f"ACCUMULATOR        "
            f"{'PASS' if accumulator_pass else 'FAIL'}\n"
        )

        all_pass = (
            quantization_pass
            and mapping_pass
            and int64_pass
            and int32_pass
            and vectors_pass
            and accumulator_pass
        )

        f.write("\n")
        f.write(
            f"RESULT: "
            f"{'PASS' if all_pass else 'FAIL'}\n"
        )

def report_first_divergence(
    trace: ArithmeticTrace,
) -> None:

    print()
    print("INT32 FIRST-DIVERGENCE TRACE")
    print("----------------------------")

    vector = getattr(
        trace,
        "vector",
        "",
    )

    if vector:
        print(f"Vector : {vector}")

    print(f"Layer  : {trace.layer}")
    print(f"Stage  : {trace.stage}")
    print(f"Tap    : {trace.tap}")
    print(f"In ch  : {trace.in_channel}")
    print(f"Out ch : {trace.out_channel}")

    print()
    print(
        f"INT64 accumulator : "
        f"{trace.int64_accumulator}"
    )
    print(
        f"INT32 accumulator : "
        f"{trace.int32_accumulator}"
    )

    print(
        f"INT64 normalized  : "
        f"{trace.int64_normalized}"
    )
    print(
        f"INT32 normalized  : "
        f"{trace.int32_normalized}"
    )

    print(
        f"INT64 activation  : "
        f"{trace.int64_activation}"
    )
    print(
        f"INT32 activation  : "
        f"{trace.int32_activation}"
    )

    print(
        f"Difference        : "
        f"{trace.difference}"
    )

# ============================================================================
# Model loading
# ============================================================================

def load_json(
    path: Path,
) -> dict:

    if not path.exists():
        fail(
            f"Model does not exist: {path}"
        )

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as f:
            return json.load(f)

    except json.JSONDecodeError as exc:
        fail(
            f"Invalid JSON: {exc}"
        )

    return {}


# ============================================================================
# Main
# ============================================================================

def main() -> int:

    print(
        f"A2-Lite fixed reference {REFERENCE_VERSION}"
    )
    print("=" * 40)

    if len(sys.argv) != 2:

        print(
            "Usage: python a2lite_fixed_reference.py A2.nam"
        )

        return 2

    model_path = Path(
        sys.argv[1]
    )

    print(
        f"Loading: {model_path}"
    )

    try:

        root = load_json(
            model_path
        )

        model = find_a2lite_model(
            root
        )

        validate_a2lite_config(
            model
        )

        print(
            "A2-Lite configuration: OK"
        )

        bf_values, mappings, layer_infos = (
            convert_weights(model)
        )

        print(
            f"Weight count: {len(bf_values)}"
        )

        print_layout(
            layer_infos
        )

        # --------------------------------------------------------
        # Quantization
        # --------------------------------------------------------

        q_max_error, q_rms_error, clipped = (
            validate_q_roundtrip(
                mappings
            )
        )

        quantization_pass = (
            clipped == 0
            and q_max_error
            <= Q_MAX_ROUNDTRIP_ERROR + 1.0e-12
        )

        print()
        print("Q2.14 round-trip")
        print("----------------")
        print(
            f"Maximum abs error   : "
            f"{q_max_error:.12g}"
        )
        print(
            f"RMS parameter error : "
            f"{q_rms_error:.12g}"
        )
        print(
            f"Clipped parameters  : "
            f"{clipped}"
        )
        print(
            f"Result              : "
            f"{'PASS' if quantization_pass else 'FAIL'}"
        )

        # --------------------------------------------------------
        # Mapping
        # --------------------------------------------------------

        mapping_pass = validate_mapping(
            mappings
        )

        print()
        print("Source -> BF permutation")
        print("------------------------")
        print(
            f"Mapping              : "
            f"{'PASS' if mapping_pass else 'FAIL'}"
        )

        # --------------------------------------------------------
        # Head groups
        # --------------------------------------------------------

        head_weights_pass = (
            validate_head_weights(mappings)
        )

        head_bias_pass = (
            validate_head_bias(mappings)
        )

        head_scale_pass = (
            validate_head_scale(mappings)
        )

        print()
        print("Head / bias / scale")
        print("-------------------")
        print(
            f"Head weights        : "
            f"{'PASS' if head_weights_pass else 'FAIL'}"
        )
        print(
            f"Head bias           : "
            f"{'PASS' if head_bias_pass else 'FAIL'}"
        )
        print(
            f"Head scale          : "
            f"{'PASS' if head_scale_pass else 'FAIL'}"
        )

        # --------------------------------------------------------
        # Binary
        # --------------------------------------------------------

        binary_path = (
            Path("generated_a2lite")
            / "A2Lite_model.bin"
        )

        print()
        print(
            f"Loading: {binary_path}"
        )

        binary_pass = False

        try:

            binary_pass, _ = validate_binary(
                binary_path,
                bf_values,
            )

            print(
                "Byte-exact payload  : PASS"
            )

        except ValidationError as exc:

            print(
                f"Byte-exact payload  : FAIL"
            )
            print(
                f"Binary validation: {exc}"
            )

        # --------------------------------------------------------
        # Geometry
        # --------------------------------------------------------

        receptive_field = (
            calculate_receptive_field()
        )

        print()
        print("Model geometry")
        print("--------------")
        print(
            f"Receptive field : "
            f"{receptive_field} samples"
        )

        # --------------------------------------------------------
        # INT64
        # --------------------------------------------------------

        print()
        print(
            "Building independent INT64 reference..."
        )

        int64_ref = Int64Reference(
            bf_values
        )

        int64_pass = (
            validate_int64_self_test(
                int64_ref
            )
        )

        print()
        print("INT64 reference self-test")
        print("-------------------------")
        print(
            "RMS       : 0"
        )
        print(
            "Peak      : 0"
        )
        print(
            "Correlation: 1"
        )
        print(
            f"Result    : "
            f"{'PASS' if int64_pass else 'FAIL'}"
        )

        # --------------------------------------------------------
        # Layer statistics
        # --------------------------------------------------------

        _, stats = int64_ref.process(
            build_test_vectors()["random_seed_A2"],
            collect_stats=True,
        )

        print_layer_stats(
            stats
        )

        # --------------------------------------------------------
        # INT32
        # --------------------------------------------------------

        print()
        print(
            "Building independent INT32 BF592 reference..."
        )

        int32_ref = Int32BF592Reference(
            bf_values,
            stats,
        )

        # --------------------------------------------------------
        # First-divergence trace (2.5)
        # --------------------------------------------------------

        trace = find_first_divergence(
            int64_ref,
            int32_ref,
        )

        if trace is None:
            print()
            print("INT32 first-divergence trace")
            print("----------------------------")
            print(
                "No divergence detected on first "
                "sample of any trace vector."
            )

        else:
            report_first_divergence(trace)

        # --------------------------------------------------------
        # Test vectors
        # --------------------------------------------------------

        vectors_pass, vector_results = (
            run_test_vectors(
                int64_ref,
                int32_ref,
            )
        )

        int32_pass = vectors_pass

        # --------------------------------------------------------
        # Accumulator
        # --------------------------------------------------------

        accumulator_pass = (
            validate_accumulators(
                stats
            )
        )

        # --------------------------------------------------------
        # Final result
        # --------------------------------------------------------

        print()
        print("Reference validation")
        print("=====================")

        print(
            f"QUANTIZATION       "
            f"{'PASS' if quantization_pass else 'FAIL'}"
        )

        print(
            f"MAPPING            "
            f"{'PASS' if mapping_pass else 'FAIL'}"
        )

        print(
            f"INT64 REFERENCE    "
            f"{'PASS' if int64_pass else 'FAIL'}"
        )

        print(
            f"INT32 BF592        "
            f"{'PASS' if int32_pass else 'FAIL'}"
        )

        print(
            f"TEST VECTORS       "
            f"{'PASS' if vectors_pass else 'FAIL'}"
        )

        print(
            f"ACCUMULATOR        "
            f"{'PASS' if accumulator_pass else 'FAIL'}"
        )

        overall = (
            quantization_pass
            and mapping_pass
            and head_weights_pass
            and head_bias_pass
            and head_scale_pass
            and binary_pass
            and int64_pass
            and int32_pass
            and vectors_pass
            and accumulator_pass
        )

        print()
        print("=" * 60)
        print(
            f"RESULT: "
            f"{'PASS' if overall else 'FAIL'}"
        )
        print("=" * 60)

        report_dir = (
            Path("generated_a2lite")
            / f"reference_{REFERENCE_VERSION.replace('.', '_')}"
        )

        report_path = (
            report_dir
            / f"A2Lite_reference_{REFERENCE_VERSION.replace('.', '_')}.txt"
        )

        offsets_path = (
            report_dir
            / f"A2Lite_offsets_{REFERENCE_VERSION.replace('.', '_')}.csv"
        )

        write_report(
            report_path,
            quantization_pass=quantization_pass,
            mapping_pass=mapping_pass,
            int64_pass=int64_pass,
            int32_pass=int32_pass,
            vectors_pass=vectors_pass,
            accumulator_pass=accumulator_pass,
            q_max_error=q_max_error,
            q_rms_error=q_rms_error,
            clipped=clipped,
            receptive_field=receptive_field,
            stats=stats,
        )

        write_offsets(
            offsets_path,
            mappings,
        )

        print(
            f"Report: {report_path}"
        )
        print(
            f"Offsets: {offsets_path}"
        )

        return 0 if overall else 1

    except ValidationError as exc:

        print()
        print("ERROR:")
        print(str(exc))

        return 1

    except Exception as exc:

        print()
        print(
            "UNEXPECTED ERROR:"
        )
        print(
            f"{type(exc).__name__}: {exc}"
        )

        return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )