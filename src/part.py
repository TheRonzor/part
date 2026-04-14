import os
import copy
import random
import colorsys
from datetime import datetime
import tempfile
from typing import Any, Optional, Sequence
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from IPython.display import display, clear_output
import ipywidgets as widgets
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_svg import FigureCanvasSVG

SettingsDict = dict[str, Any]
WidgetDict = dict[str, Any]
CurveArray = tuple[np.ndarray, np.ndarray]
SegmentList = list[CurveArray]
FigureAxes = tuple[Figure, Axes]

PNG_DPI=300

FAMILY_OPTIONS = [('Lissajous', 'lissajous'), ('Rose', 'rose'), ('Epitrochoid', 'epitrochoid'), ('Burst', 'burst')]
COLOR_MODE_OPTIONS = [('Palette', 'palette'), ('Colormap', 'colormap')]
BACKGROUND_MODE_OPTIONS = [('Solid', 'solid'), ('Linear', 'linear'), ('Radial', 'radial'), ('Mesh', 'mesh')]
PALETTE_DISTRIBUTION_OPTIONS = [('Even Segments', 'even_segments'), ('Weighted', 'weighted'), ('Gradient Blend', 'gradient_blend')]
DEFAULT_RANDOMIZATION_INTENSITY = 'moderate'
CURATED_COLORMAPS = ['winter', 'spring', 'summer', 'autumn', 'viridis', 'plasma', 'cividis', 'cool', 'twilight', 'turbo', 'hsv', 'jet']
EXPORT_FOLDER = os.path.abspath('.')
POINTS = 10_000
FIGSIZE = (7.5, 7.5)
DEFAULT_SLIDER_WIDTH = '310px'
DEFAULT_RANGE_SLIDER_WIDTH = '370px'
CONTROL_PANEL_WIDTH = '430px'

def now_timestamp_str() -> str:
    """Build a filesystem-safe timestamp string for artwork exports.

    Inputs:
        None: This function does not accept positional inputs.

Returns the current local time formatted as YYYYMMDD_HHMMSS."""
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp a numeric value to an inclusive range.

    Inputs:
        value: Numeric or generic value being processed by the helper.
        lo:    Lower clamp or perturbation bound.
        hi:    Upper clamp or perturbation bound.

Returns the bounded value."""
    return max(lo, min(hi, value))

def lerp(a: float, b: float, t: float) -> float:
    """Linearly interpolate between two numeric values.

    Inputs:
        a: Interpolation start value or first family parameter, depending on context.
        b: Interpolation end value or second family parameter, depending on context.
        t: Interpolation factor.

Returns the interpolated value."""
    return (1 - t) * a + t * b

def hex_to_rgb01(hex_color: str) -> np.ndarray:
    """Convert a hex color string into an RGB float vector in the 0-1 range.

    Inputs:
        hex_color: Color encoded as a hex string.

Returns a NumPy array containing normalized RGB channels."""
    return np.array(mcolors.to_rgb(hex_color), dtype=float)

def rgb01_to_hex(rgb: Sequence[float]) -> str:
    """Convert an RGB sequence in the 0-1 range into a hex color string.

    Inputs:
        rgb: RGB color channels expressed in the 0-1 range.

Returns the normalized color as a hex string."""
    rgb = np.clip(np.asarray(rgb, dtype=float), 0, 1)
    return mcolors.to_hex(rgb, keep_alpha=False)

def blend_rgb(c1: Sequence[float], c2: Sequence[float], t: float) -> np.ndarray:
    """Blend two RGB colors using a linear interpolation factor.

    Inputs:
        c1: First RGB color to blend.
        c2: Second RGB color to blend.
        t:  Interpolation factor.

Returns the blended RGB color as a NumPy array."""
    return (1 - t) * np.asarray(c1) + t * np.asarray(c2)

def adjust_color_brightness(hex_color: str, factor: float) -> str:
    """Adjust the value channel of a hex color in HSV space.

    Inputs:
        hex_color: Color encoded as a hex string.
        factor:    Brightness scaling factor.

Returns the brightness-adjusted color as a hex string."""
    rgb = hex_to_rgb01(hex_color)
    hsv = colorsys.rgb_to_hsv(*rgb)
    new_rgb = colorsys.hsv_to_rgb(hsv[0], hsv[1], clamp(hsv[2] * factor, 0, 1))
    return rgb01_to_hex(new_rgb)

def shift_color_hue(hex_color: str, shift: float) -> str:
    """Rotate the hue of a hex color in HSV space.

    Inputs:
        hex_color: Color encoded as a hex string.
        shift:     Hue shift amount expressed as a unit interval turn.

Returns the hue-shifted color as a hex string."""
    rgb = hex_to_rgb01(hex_color)
    hsv = colorsys.rgb_to_hsv(*rgb)
    new_rgb = colorsys.hsv_to_rgb((hsv[0] + shift) % 1.0, hsv[1], hsv[2])
    return rgb01_to_hex(new_rgb)

def random_hex_color(rng: np.random.Generator, low: float=0.2, high: float=1.0) -> str:
    """Generate a random hex color using bounded RGB channel values.

    Inputs:
        rng:  NumPy random generator used for deterministic perturbations.
        low:  Lower bound for sampled RGB channel values.
        high: Upper bound for sampled RGB channel values.

Returns the sampled color as a hex string."""
    rgb = rng.uniform(low, high, size=3)
    return rgb01_to_hex(rgb)


PALETTE_PRESETS = {'custom': [], 'nebula': ['#8fd3ff', '#bf8fff', '#ffffff'], 'aurora': ['#72f1b8', '#7ea8ff', '#d9a7ff'], 'ember': ['#ffb347', '#ff7043', '#fff1cc'], 'sunset_bloom': ['#ff8fa3', '#ffc6a5', '#ffe8d6'], 'ice_mono': ['#dff6ff', '#89c2ff', '#5a87ff'], 'noir_neon': ['#f5f5f5', '#ff00aa', '#101010'], 'gold_filament': ['#ffd166', '#fff1b6', '#8c6a00'], 'plasma': ['#00e5ff', '#7b61ff', '#ffffff']}
BACKGROUND_PRESETS = {'custom': {'mode': 'linear'}, 'deep_space': {'mode': 'linear', 'linear': {'top': '#120a2a', 'bottom': '#030308', 'angle': 90}}, 'purple_haze': {'mode': 'radial', 'radial': {'inner': '#31134b', 'outer': '#030308', 'center_x': 0.48, 'center_y': 0.56, 'radius_bias': 1.0}}, 'ember_fog': {'mode': 'radial', 'radial': {'inner': '#4b180f', 'outer': '#040202', 'center_x': 0.52, 'center_y': 0.45, 'radius_bias': 1.15}}, 'cyan_void': {'mode': 'linear', 'linear': {'top': '#07111d', 'bottom': '#020204', 'angle': 90}}, 'twilight_mesh': {'mode': 'mesh', 'mesh': {'tl': '#160a2f', 'tr': '#09203f', 'bl': '#030308', 'br': '#1e0d2b'}}, 'warm_smoke': {'mode': 'linear', 'linear': {'top': '#2b160f', 'bottom': '#060303', 'angle': 90}}, 'velvet_black': {'mode': 'radial', 'radial': {'inner': '#171721', 'outer': '#020202', 'center_x': 0.5, 'center_y': 0.5, 'radius_bias': 0.9}}, 'aurora_night': {'mode': 'linear', 'linear': {'top': '#0b1a20', 'bottom': '#130d29', 'angle': 90}}}
STYLE_PRESETS = {'custom': {}, 'fine_wire': {'line_width': 1.0, 'glow_layers': 4, 'glow_alpha': 0.025, 'glow_scale': 1.4, 'main_alpha': 0.98}, 'velvet_glow': {'line_width': 1.6, 'glow_layers': 8, 'glow_alpha': 0.055, 'glow_scale': 2.0, 'main_alpha': 0.95}, 'plasma_bloom': {'line_width': 1.2, 'glow_layers': 12, 'glow_alpha': 0.075, 'glow_scale': 2.6, 'main_alpha': 0.96}, 'poster_crisp': {'line_width': 1.9, 'glow_layers': 2, 'glow_alpha': 0.015, 'glow_scale': 1.15, 'main_alpha': 1.0}, 'ghost_trace': {'line_width': 0.9, 'glow_layers': 6, 'glow_alpha': 0.03, 'glow_scale': 1.7, 'main_alpha': 0.85}}
ARTWORK_PRESETS = {'custom': {}, 'nebula': {'curve': {'family': 'lissajous', 'turns': 3.0, 'asymmetry': 0.1, 'noise_strength': 0.002, 'params': {'a': 3, 'b': 2, 'delta': 1.8}}, 'style_preset': 'velvet_glow', 'palette_preset': 'nebula', 'background_preset': 'purple_haze', 'color': {'mode': 'palette', 'curve_color': '#9ed8ff'}, 'meta_seed': 7}, 'flower': {'curve': {'family': 'rose', 'turns': 2.0, 'asymmetry': 0.05, 'noise_strength': 0.001, 'params': {'k': 7, 'radial_wobble': 0.12, 'wobble_freq': 8}}, 'style_preset': 'velvet_glow', 'palette_preset': 'sunset_bloom', 'background_preset': 'ember_fog', 'color': {'mode': 'palette', 'curve_color': '#ffd7e6'}, 'meta_seed': 12}, 'electric': {'curve': {'family': 'burst', 'turns': 3.0, 'asymmetry': 0.22, 'noise_strength': 0.004, 'params': {'amp1': 0.45, 'amp2': 0.25, 'f1': 9, 'f2': 17, 'delta': 0.7}}, 'style_preset': 'plasma_bloom', 'palette_preset': 'plasma', 'background_preset': 'cyan_void', 'color': {'mode': 'palette'}, 'meta_seed': 21}, 'spiro': {'curve': {'family': 'epitrochoid', 'turns': 6.0, 'asymmetry': 0.03, 'noise_strength': 0.0, 'params': {'R': 5, 'r': 3, 'd': 5}}, 'style_preset': 'fine_wire', 'palette_preset': 'gold_filament', 'background_preset': 'warm_smoke', 'color': {'mode': 'palette', 'curve_color': '#fff1b6'}, 'meta_seed': 4}, 'ember_bloom': {'curve': {'family': 'rose', 'turns': 2.2, 'asymmetry': 0.08, 'noise_strength': 0.0015, 'params': {'k': 5, 'radial_wobble': 0.09, 'wobble_freq': 6}}, 'style_preset': 'velvet_glow', 'palette_preset': 'ember', 'background_preset': 'ember_fog', 'color': {'mode': 'palette'}, 'meta_seed': 17}, 'aurora_wire': {'curve': {'family': 'lissajous', 'turns': 2.5, 'asymmetry': 0.07, 'noise_strength': 0.001, 'params': {'a': 4, 'b': 7, 'delta': 0.9}}, 'style_preset': 'ghost_trace', 'palette_preset': 'aurora', 'background_preset': 'aurora_night', 'color': {'mode': 'palette'}, 'meta_seed': 8}, 'noir_neon': {'curve': {'family': 'epitrochoid', 'turns': 5.0, 'asymmetry': 0.02, 'noise_strength': 0.0, 'params': {'R': 7, 'r': 4, 'd': 5}}, 'style_preset': 'poster_crisp', 'palette_preset': 'noir_neon', 'background_preset': 'velvet_black', 'color': {'mode': 'palette', 'curve_color': '#f5f5f5'}, 'meta_seed': 30}, 'plasma_halo': {'curve': {'family': 'burst', 'turns': 4.0, 'asymmetry': 0.2, 'noise_strength': 0.006, 'params': {'amp1': 0.52, 'amp2': 0.18, 'f1': 11, 'f2': 19, 'delta': 0.35}}, 'style_preset': 'plasma_bloom', 'palette_preset': 'plasma', 'background_preset': 'deep_space', 'color': {'mode': 'palette'}, 'meta_seed': 26}}
FAMILY_PARAM_DEFAULTS = {'lissajous': {'a': 3.0, 'b': 2.0, 'delta': np.pi / 2}, 'rose': {'k': 5.0, 'radial_wobble': 0.12, 'wobble_freq': 6.0}, 'epitrochoid': {'R': 5.0, 'r': 3.0, 'd': 5.0}, 'burst': {'amp1': 0.45, 'amp2': 0.22, 'f1': 8.0, 'f2': 17.0, 'delta': 0.3}}
PARAM_RANGES = {'curve.turns': (0.5, 12.0), 'curve.asymmetry': (0.0, 0.5), 'curve.noise_strength': (0.0, 0.05), 'curve.zoom': (0.5, 2.0), 'style.line_width': (0.3, 5.0), 'style.glow_layers': (0, 20), 'style.glow_alpha': (0.0, 0.2), 'style.glow_scale': (1.0, 6.0), 'style.main_alpha': (0.1, 1.0), 'background.brightness': (0.5, 1.5), 'background.contrast': (0.5, 1.5), 'background.texture_strength': (0.0, 0.25), 'background.vignette_strength': (0.0, 0.8), 'lissajous.a': (1, 12), 'lissajous.b': (1, 12), 'lissajous.delta': (0, 2 * np.pi), 'rose.k': (1, 16), 'rose.radial_wobble': (0.0, 0.5), 'rose.wobble_freq': (1, 20), 'epitrochoid.R': (1, 12), 'epitrochoid.r': (1, 12), 'epitrochoid.d': (0, 12), 'burst.amp1': (0.0, 1.0), 'burst.amp2': (0.0, 1.0), 'burst.f1': (1, 30), 'burst.f2': (1, 30), 'burst.delta': (0, 2 * np.pi)}

def make_default_curve_params(family: str) -> dict[str, float]:
    """Create the default parameter dictionary for a specific curve family.

    Inputs:
        family: Normalized curve family identifier.

Returns a deep-copied family-parameter dictionary."""
    return copy.deepcopy(FAMILY_PARAM_DEFAULTS[family])

def make_default_settings() -> SettingsDict:
    """Create the canonical default settings dictionary for the full PART application state.

    Inputs:
        None: This function does not accept positional inputs.

Returns a fully populated settings dictionary."""
    return {'meta': {'artwork_preset': 'nebula', 'style_preset': 'velvet_glow', 'palette_preset': 'nebula', 'background_preset': 'purple_haze', 'seed': 7}, 'curve': {'family': 'lissajous', 'turns': 3.0, 'asymmetry': 0.1, 'noise_strength': 0.002, 'zoom': 1.0, 'params': make_default_curve_params('lissajous')}, 'style': {'glow_macro': 0.6, 'line_width': 1.6, 'glow_layers': 8, 'glow_alpha': 0.055, 'glow_scale': 2.0, 'main_alpha': 0.95, 'segment_blending': True}, 'color': {'mode': 'palette', 'curve_color': '#9ed8ff', 'palette_name': 'nebula', 'palette_colors': copy.deepcopy(PALETTE_PRESETS['nebula']), 'palette_size': 3, 'palette_distribution': 'even_segments', 'colormap': 'viridis', 'colormap_span': [0.1, 0.9], 'num_curve_colors': 3, 'curve_opacity': 0.95, 'color_shift': 0.0, 'manual_palette': False}, 'background': {'mode': 'radial', 'preset': 'purple_haze', 'solid_color': '#05060a', 'linear': {'top': '#120a2a', 'bottom': '#030308', 'angle': 90}, 'radial': {'inner': '#31134b', 'outer': '#030308', 'center_x': 0.48, 'center_y': 0.56, 'radius_bias': 1.0}, 'mesh': {'tl': '#160a2f', 'tr': '#09203f', 'bl': '#030308', 'br': '#1e0d2b'}, 'brightness': 1.0, 'contrast': 1.0, 'texture_strength': 0.0, 'vignette_strength': 0.12}}

def deep_copy_settings(settings: SettingsDict) -> SettingsDict:
    """Create a defensive deep copy of the settings dictionary.

    Inputs:
        settings: Application settings dictionary following the PART schema.

Returns a deep-copied settings dictionary."""
    return copy.deepcopy(settings)

def get_nested(settings: SettingsDict, path: str, default: Any=None) -> Any:
    """Read a nested settings value using a dot-delimited path.

    Inputs:
        settings: Application settings dictionary following the PART schema.
        path:     Dot-delimited path used to access nested dictionary values.
        default:  Fallback value returned when the path does not exist.

Returns the resolved value, or the fallback default when the path is missing."""
    cur = settings
    for part in path.split('.'):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur

def set_nested(settings: SettingsDict, path: str, value: float) -> None:
    """Write a nested settings value using a dot-delimited path.

    Inputs:
        settings: Application settings dictionary following the PART schema.
        path:     Dot-delimited path used to access nested dictionary values.
        value:    Value to write to the nested path.

Returns None after mutating the target settings dictionary in place."""
    parts = path.split('.')
    cur = settings
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value


def coerce_family_params(family: str, params: dict[str, Any]) -> dict[str, Any]:
    """Normalize and range-check the family-specific parameter dictionary for the requested curve family.

    Inputs:
        family: Normalized curve family identifier.
        params: Optional family-parameter dictionary to merge with defaults before validation.

Returns a validated family-parameter dictionary."""
    base = make_default_curve_params(family)
    base.update(params or {})
    if family == 'lissajous':
        base['a'] = clamp(float(base['a']), *PARAM_RANGES['lissajous.a'])
        base['b'] = clamp(float(base['b']), *PARAM_RANGES['lissajous.b'])
        base['delta'] = clamp(float(base['delta']), *PARAM_RANGES['lissajous.delta'])
    elif family == 'rose':
        base['k'] = clamp(float(base['k']), *PARAM_RANGES['rose.k'])
        base['radial_wobble'] = clamp(float(base['radial_wobble']), *PARAM_RANGES['rose.radial_wobble'])
        base['wobble_freq'] = clamp(float(base['wobble_freq']), *PARAM_RANGES['rose.wobble_freq'])
    elif family == 'epitrochoid':
        base['R'] = clamp(float(base['R']), *PARAM_RANGES['epitrochoid.R'])
        base['r'] = clamp(float(base['r']), *PARAM_RANGES['epitrochoid.r'])
        if abs(base['r']) < 1e-06:
            base['r'] = 1.0
        base['d'] = clamp(float(base['d']), *PARAM_RANGES['epitrochoid.d'])
    elif family == 'burst':
        base['amp1'] = clamp(float(base['amp1']), *PARAM_RANGES['burst.amp1'])
        base['amp2'] = clamp(float(base['amp2']), *PARAM_RANGES['burst.amp2'])
        base['f1'] = clamp(float(base['f1']), *PARAM_RANGES['burst.f1'])
        base['f2'] = clamp(float(base['f2']), *PARAM_RANGES['burst.f2'])
        base['delta'] = clamp(float(base['delta']), *PARAM_RANGES['burst.delta'])
    return base

def validate_color_settings(settings: SettingsDict) -> SettingsDict:
    """Normalize color-related settings so downstream rendering code can rely on valid palette and colormap values.

    Inputs:
        settings: Application settings dictionary following the PART schema.

Returns the mutated settings dictionary after color normalization."""
    color = settings['color']
    if color.get('mode') not in {v for _, v in COLOR_MODE_OPTIONS}:
        color['mode'] = 'palette'
    color.setdefault('curve_color', '#ffffff')
    color['curve_opacity'] = clamp(float(color['curve_opacity']), 0.1, 1.0)
    color['palette_size'] = int(clamp(int(color['palette_size']), 2, 6))
    color['num_curve_colors'] = int(clamp(int(color['num_curve_colors']), 1, 12))
    color['color_shift'] = clamp(float(color.get('color_shift', 0.0)), -0.5, 0.5)
    color['manual_palette'] = bool(color.get('manual_palette', False))
    if color.get('palette_distribution') not in {v for _, v in PALETTE_DISTRIBUTION_OPTIONS}:
        color['palette_distribution'] = 'even_segments'
    if color['palette_name'] not in PALETTE_PRESETS:
        color['palette_name'] = 'nebula'
    if not color['manual_palette']:
        preset = PALETTE_PRESETS.get(color['palette_name'], [])
        if preset:
            color['palette_colors'] = copy.deepcopy(preset)
            color['palette_size'] = int(clamp(len(color['palette_colors']), 2, 6))
    color['palette_colors'] = list(color.get('palette_colors', []))
    while len(color['palette_colors']) < color['palette_size']:
        color['palette_colors'].append('#ffffff')
    color['palette_colors'] = color['palette_colors'][:max(6, color['palette_size'])]
    span = color.get('colormap_span', [0.1, 0.9])
    if len(span) != 2:
        span = [0.1, 0.9]
    lo, hi = sorted([clamp(float(span[0]), 0.0, 1.0), clamp(float(span[1]), 0.0, 1.0)])
    if hi - lo < 0.05:
        hi = min(1.0, lo + 0.05)
    color['colormap_span'] = [lo, hi]
    if color['colormap'] not in CURATED_COLORMAPS:
        color['colormap'] = CURATED_COLORMAPS[0]
    return settings

def validate_style_settings(settings: SettingsDict) -> SettingsDict:
    """Normalize style-related settings so the macro and manual style controls stay in range.

    Inputs:
        settings: Application settings dictionary following the PART schema.

Returns the mutated settings dictionary after style normalization."""
    style = settings['style']
    style.pop('complexity_macro', None)
    style['glow_macro'] = clamp(float(style.get('glow_macro', 0.6)), 0.0, 1.0)
    style['line_width'] = clamp(float(style['line_width']), *PARAM_RANGES['style.line_width'])
    style['glow_layers'] = int(clamp(int(style['glow_layers']), *PARAM_RANGES['style.glow_layers']))
    style['glow_alpha'] = clamp(float(style['glow_alpha']), *PARAM_RANGES['style.glow_alpha'])
    style['glow_scale'] = clamp(float(style['glow_scale']), *PARAM_RANGES['style.glow_scale'])
    style['main_alpha'] = clamp(float(style['main_alpha']), *PARAM_RANGES['style.main_alpha'])
    style['segment_blending'] = bool(style.get('segment_blending', True))
    return settings

def validate_background_settings(settings: SettingsDict) -> SettingsDict:
    """Normalize background-related settings so all required background substructures and ranges are present.

    Inputs:
        settings: Application settings dictionary following the PART schema.

Returns the mutated settings dictionary after background normalization."""
    bg = settings['background']
    if bg['mode'] not in {v for _, v in BACKGROUND_MODE_OPTIONS}:
        bg['mode'] = 'linear'
    bg['brightness'] = clamp(float(bg['brightness']), *PARAM_RANGES['background.brightness'])
    bg['contrast'] = clamp(float(bg['contrast']), *PARAM_RANGES['background.contrast'])
    bg['texture_strength'] = clamp(float(bg['texture_strength']), *PARAM_RANGES['background.texture_strength'])
    bg['vignette_strength'] = clamp(float(bg['vignette_strength']), *PARAM_RANGES['background.vignette_strength'])
    bg.setdefault('linear', {'top': '#120a2a', 'bottom': '#030308', 'angle': 90})
    bg.setdefault('radial', {'inner': '#31134b', 'outer': '#030308', 'center_x': 0.5, 'center_y': 0.5, 'radius_bias': 1.0})
    bg.setdefault('mesh', {'tl': '#160a2f', 'tr': '#09203f', 'bl': '#030308', 'br': '#1e0d2b'})
    bg['radial']['center_x'] = clamp(float(bg['radial']['center_x']), 0.0, 1.0)
    bg['radial']['center_y'] = clamp(float(bg['radial']['center_y']), 0.0, 1.0)
    bg['radial']['radius_bias'] = clamp(float(bg['radial']['radius_bias']), 0.25, 2.0)
    bg['linear']['angle'] = clamp(float(bg['linear']['angle']), 0.0, 180.0)
    return settings

def apply_glow_macro(settings: SettingsDict) -> SettingsDict:
    """Resolve the high-level glow macro into concrete glow rendering fields.

    Inputs:
        settings: Application settings dictionary following the PART schema.

Returns the updated settings dictionary."""
    s = settings['style']
    g = clamp(float(s.get('glow_macro', 0.6)), 0.0, 1.0)
    s['glow_layers'] = int(round(lerp(0, 14, g)))
    s['glow_alpha'] = lerp(0.0, 0.085, g)
    s['glow_scale'] = lerp(1.0, 3.0, g)
    return settings

def infer_glow_macro(settings: SettingsDict) -> float:
    """Infer the glow macro from the concrete glow style fields."""
    style = settings['style']
    components = [
        clamp(style['glow_layers'] / 14.0, 0.0, 1.0),
        clamp(style['glow_alpha'] / 0.085, 0.0, 1.0),
        clamp((style['glow_scale'] - 1.0) / 2.0, 0.0, 1.0),
    ]
    return clamp(sum(components) / len(components), 0.0, 1.0)

def sync_macro_controls(settings: SettingsDict) -> SettingsDict:
    """Keep the remaining macro controls aligned with the concrete settings they summarize.

    Inputs:
        settings: Application settings dictionary following the PART schema.

Returns the updated settings dictionary."""
    settings['style']['glow_macro'] = infer_glow_macro(settings)
    settings['style'].pop('complexity_macro', None)
    return settings

def normalize_macro_controls(settings: SettingsDict) -> SettingsDict:
    """Synchronize macro controls from the lower-level settings they summarize.

    Inputs:
        settings: Application settings dictionary following the PART schema.

Returns the updated settings dictionary."""
    return sync_macro_controls(settings)

def validate_curve_settings(settings: SettingsDict) -> SettingsDict:
    """Normalize curve-related settings and coerce family-specific parameters into valid ranges.

    Inputs:
        settings: Application settings dictionary following the PART schema.

Returns the mutated settings dictionary after curve normalization."""
    curve = settings['curve']
    if curve['family'] not in {v for _, v in FAMILY_OPTIONS}:
        curve['family'] = 'lissajous'
    curve['turns'] = clamp(float(curve['turns']), *PARAM_RANGES['curve.turns'])
    curve.pop('points', None)
    curve['asymmetry'] = clamp(float(curve['asymmetry']), *PARAM_RANGES['curve.asymmetry'])
    curve['noise_strength'] = clamp(float(curve['noise_strength']), *PARAM_RANGES['curve.noise_strength'])
    curve['zoom'] = clamp(float(curve['zoom']), *PARAM_RANGES['curve.zoom'])
    curve['params'] = coerce_family_params(curve['family'], curve.get('params', {}))
    return settings

def validate_settings(settings: SettingsDict) -> SettingsDict:
    """Return a deep-copied, render-safe settings dictionary.

    Inputs:
        settings: Application settings dictionary following the PART schema.

Returns a validated settings dictionary suitable for rendering and UI sync."""
    settings = deep_copy_settings(settings)
    settings = validate_curve_settings(settings)
    settings = validate_color_settings(settings)
    settings = validate_background_settings(settings)
    settings = validate_style_settings(settings)
    settings = sync_macro_controls(settings)
    return settings


def make_theta(turns: float, points: int) -> np.ndarray:
    """Generate the parametric theta samples used by curve generators.

    Inputs:
        turns:  Number of parametric turns to sample.
        points: Number of sample points to generate along the curve.

Returns a NumPy array of sampled theta values."""
    return np.linspace(0, 2 * np.pi * turns, points)

def apply_asymmetry(x: np.ndarray, y: np.ndarray, theta: np.ndarray, amount: float) -> CurveArray:
    """Apply the deterministic asymmetry warp used to break perfect curve symmetry.

    Inputs:
        x:      Array of x coordinates.
        y:      Array of y coordinates.
        theta:  Array of parametric theta values.
        amount: Effect strength to apply.

Returns the transformed x and y coordinate arrays."""
    if amount <= 0:
        return (x, y)
    x = x + amount * np.sin(2.7 * theta + 0.3)
    y = y + 0.7 * amount * np.cos(3.9 * theta + 1.1)
    return (x, y)

def apply_noise(x: np.ndarray, y: np.ndarray, strength: float, seed: int) -> CurveArray:
    """Apply seeded positional noise to a generated curve.

    Inputs:
        x:        Array of x coordinates.
        y:        Array of y coordinates.
        strength: Effect strength for the current operation.
        seed:     Seed used for deterministic random behavior.

Returns the noise-perturbed x and y coordinate arrays."""
    if strength <= 0:
        return (x, y)
    rng = np.random.default_rng(seed)
    x = x + strength * rng.normal(size=len(x))
    y = y + strength * rng.normal(size=len(y))
    return (x, y)

def normalize_curve(x: np.ndarray, y: np.ndarray, family: str) -> CurveArray:
    """Scale a generated curve so it fits a normalized render space.

    Inputs:
        x:      Array of x coordinates.
        y:      Array of y coordinates.
        family: Normalized curve family identifier.

Returns normalized x and y coordinate arrays."""
    scale = max(np.max(np.abs(x)), np.max(np.abs(y)), 1e-09)
    if family == 'epitrochoid':
        return (x / scale, y / scale)
    return (x / scale, y / scale)

def apply_zoom(x: np.ndarray, y: np.ndarray, zoom: float) -> CurveArray:
    """Apply zoom scaling to a normalized curve.

    Inputs:
        x:    Array of x coordinates.
        y:    Array of y coordinates.
        zoom: Zoom factor applied to the normalized curve.

Returns zoom-adjusted x and y coordinate arrays."""
    z = max(zoom, 1e-06)
    return (x * z, y * z)

def generate_lissajous(params: dict[str, Any], turns: float, points: int, asymmetry: float, noise_strength: float, seed: int) -> CurveArray:
    """Generate a Lissajous curve from family-specific parameters and shared curve controls.

    Inputs:
        params:         Lissajous parameter dictionary containing a, b, and delta.
        turns:          Number of parametric turns to sample.
        points:         Number of sample points to generate along the curve.
        asymmetry:      Strength of the deterministic asymmetry warp.
        noise_strength: Strength of the seeded positional noise.
        seed:           Seed used for deterministic random behavior.

Returns x and y coordinate arrays for the generated curve."""
    theta = make_theta(turns, points)
    x = np.sin(params['a'] * theta + params['delta'])
    y = np.sin(params['b'] * theta)
    x, y = apply_asymmetry(x, y, theta, asymmetry)
    x, y = apply_noise(x, y, noise_strength, seed)
    return normalize_curve(x, y, 'lissajous')

def generate_rose(params: dict[str, Any], turns: float, points: int, asymmetry: float, noise_strength: float, seed: int) -> CurveArray:
    """Generate a rose curve from family-specific parameters and shared curve controls.

    Inputs:
        params:         Rose parameter dictionary containing k, radial_wobble, and wobble_freq.
        turns:          Number of parametric turns to sample.
        points:         Number of sample points to generate along the curve.
        asymmetry:      Strength of the deterministic asymmetry warp.
        noise_strength: Strength of the seeded positional noise.
        seed:           Seed used for deterministic random behavior.

Returns x and y coordinate arrays for the generated curve."""
    theta = make_theta(turns, points)
    r = np.cos(params['k'] * theta)
    r = r * (1 + params['radial_wobble'] * np.sin(params['wobble_freq'] * theta))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    x, y = apply_asymmetry(x, y, theta, asymmetry)
    x, y = apply_noise(x, y, noise_strength, seed)
    return normalize_curve(x, y, 'rose')

def generate_epitrochoid(params: dict[str, Any], turns: float, points: int, asymmetry: float, noise_strength: float, seed: int) -> CurveArray:
    """Generate an epitrochoid curve from family-specific parameters and shared curve controls.

    Inputs:
        params:         Epitrochoid parameter dictionary containing R, r, and d.
        turns:          Number of parametric turns to sample.
        points:         Number of sample points to generate along the curve.
        asymmetry:      Strength of the deterministic asymmetry warp.
        noise_strength: Strength of the seeded positional noise.
        seed:           Seed used for deterministic random behavior.

Returns x and y coordinate arrays for the generated curve."""
    theta = make_theta(turns, points)
    R, r, d = (params['R'], params['r'], params['d'])
    x = (R + r) * np.cos(theta) - d * np.cos((R + r) / r * theta)
    y = (R + r) * np.sin(theta) - d * np.sin((R + r) / r * theta)
    x, y = apply_asymmetry(x, y, theta, asymmetry)
    x, y = apply_noise(x, y, noise_strength, seed)
    return normalize_curve(x, y, 'epitrochoid')

def generate_burst(params: dict[str, Any], turns: float, points: int, asymmetry: float, noise_strength: float, seed: int) -> CurveArray:
    """Generate a burst curve from family-specific parameters and shared curve controls.

    Inputs:
        params:         Burst parameter dictionary containing amp1, amp2, f1, f2, and delta.
        turns:          Number of parametric turns to sample.
        points:         Number of sample points to generate along the curve.
        asymmetry:      Strength of the deterministic asymmetry warp.
        noise_strength: Strength of the seeded positional noise.
        seed:           Seed used for deterministic random behavior.

Returns x and y coordinate arrays for the generated curve."""
    theta = make_theta(turns, points)
    r = 1 + params['amp1'] * np.sin(params['f1'] * theta + params['delta']) + params['amp2'] * np.cos(params['f2'] * theta)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    x, y = apply_asymmetry(x, y, theta, asymmetry)
    x, y = apply_noise(x, y, noise_strength, seed)
    return normalize_curve(x, y, 'burst')

def generate_curve(settings: SettingsDict) -> CurveArray:
    """Dispatch to the active curve-family generator using the current settings dictionary.

    Inputs:
        settings: Application settings dictionary following the PART schema.

Returns x and y coordinate arrays for the active family."""
    settings = validate_settings(settings)
    curve = settings['curve']
    family = curve['family']
    p = curve['params']
    args = (p, curve['turns'], POINTS, curve['asymmetry'], curve['noise_strength'], settings['meta']['seed'])
    if family == 'lissajous':
        x, y = generate_lissajous(*args)
    elif family == 'rose':
        x, y = generate_rose(*args)
    elif family == 'epitrochoid':
        x, y = generate_epitrochoid(*args)
    elif family == 'burst':
        x, y = generate_burst(*args)
    else:
        raise ValueError(f'Unknown family: {family}')
    return apply_zoom(x, y, curve['zoom'])


def apply_background_brightness_contrast(img: np.ndarray, brightness: float, contrast: float) -> np.ndarray:
    """Apply brightness and contrast adjustments to a background image array.

    Inputs:
        img:        Background image array in RGB float format.
        brightness: Brightness multiplier applied to the background image.
        contrast:   Contrast multiplier applied to the background image.

Returns the adjusted RGB image array."""
    out = img * brightness
    out = (out - 0.5) * contrast + 0.5
    return np.clip(out, 0, 1)

def apply_texture(img: np.ndarray, strength: float, seed: int) -> np.ndarray:
    """Apply seeded texture noise to a background image array.

    Inputs:
        img:      Background image array in RGB float format.
        strength: Effect strength for the current operation.
        seed:     Seed used for deterministic random behavior.

Returns the textured RGB image array."""
    if strength <= 0:
        return img
    rng = np.random.default_rng(seed + 101)
    noise = rng.normal(0, strength * 0.18, size=img.shape)
    return np.clip(img + noise, 0, 1)

def apply_vignette(img: np.ndarray, strength: float) -> np.ndarray:
    """Apply a radial vignette falloff to a background image array.

    Inputs:
        img:      Background image array in RGB float format.
        strength: Effect strength for the current operation.

Returns the vignette-adjusted RGB image array."""
    if strength <= 0:
        return img
    h, w, _ = img.shape
    y = np.linspace(-1, 1, h)[:, None]
    x = np.linspace(-1, 1, w)[None, :]
    radius = np.sqrt(x ** 2 + y ** 2)
    vignette = 1 - strength * np.clip(radius, 0, 1)
    return np.clip(img * vignette[..., None], 0, 1)

def make_solid_background_array(settings: SettingsDict, width: int=700, height: int=700) -> np.ndarray:
    """Build a solid-color RGB background array.

    Inputs:
        settings: Application settings dictionary following the PART schema.
        width:    Output image width in pixels.
        height:   Output image height in pixels.

Returns an RGB image array for the solid background."""
    color = hex_to_rgb01(settings['background']['solid_color'])
    img = np.zeros((height, width, 3), dtype=float)
    img[:] = color
    return img

def make_linear_gradient_array(settings: SettingsDict, width: int=700, height: int=700) -> np.ndarray:
    """Build a linear background gradient whose labels match the rendered preview.

    Inputs:
        settings: Application settings dictionary following the PART schema.
        width:    Output image width in pixels.
        height:   Output image height in pixels.

Returns an RGB image array for the linear gradient background."""
    bg = settings['background']['linear']
    top_color = hex_to_rgb01(bg['top'])
    bottom_color = hex_to_rgb01(bg['bottom'])
    angle = np.deg2rad(bg['angle'])
    yy, xx = np.mgrid[0:height, 0:width]
    xn = xx / max(1, width - 1) * 2 - 1
    yn = yy / max(1, height - 1) * 2 - 1
    projection = np.cos(angle) * xn + np.sin(angle) * yn
    t = (projection - projection.min()) / max(1e-09, projection.max() - projection.min())
    img = (1 - t[..., None]) * bottom_color + t[..., None] * top_color
    return img

def make_radial_gradient_array(settings: SettingsDict, width: int=700, height: int=700) -> np.ndarray:
    """Build a radial background gradient in the renderer's displayed coordinate system.

    Inputs:
        settings: Application settings dictionary following the PART schema.
        width:    Output image width in pixels.
        height:   Output image height in pixels.

Returns an RGB image array for the radial gradient background."""
    bg = settings['background']['radial']
    c1 = hex_to_rgb01(bg['inner'])
    c2 = hex_to_rgb01(bg['outer'])
    yy, xx = np.mgrid[0:height, 0:width]
    xn = xx / max(1, width - 1)
    yn = yy / max(1, height - 1)
    radius = np.sqrt((xn - bg['center_x']) ** 2 + (yn - bg['center_y']) ** 2)
    radius = np.clip(radius * bg['radius_bias'] / max(radius.max(), 1e-09), 0, 1)
    img = (1 - radius[..., None]) * c1 + radius[..., None] * c2
    return img

def make_mesh_gradient_array(settings: SettingsDict, width: int=700, height: int=700) -> np.ndarray:
    """Build a four-corner mesh gradient whose corner labels match the preview.

    Inputs:
        settings: Application settings dictionary following the PART schema.
        width:    Output image width in pixels.
        height:   Output image height in pixels.

Returns an RGB image array for the mesh gradient background."""
    bg = settings['background']['mesh']
    tl = hex_to_rgb01(bg['tl'])
    tr = hex_to_rgb01(bg['tr'])
    bl = hex_to_rgb01(bg['bl'])
    br = hex_to_rgb01(bg['br'])
    yy, xx = np.mgrid[0:height, 0:width]
    tx = xx / max(1, width - 1)
    ty = yy / max(1, height - 1)
    top = (1 - tx[..., None]) * tl + tx[..., None] * tr
    bottom = (1 - tx[..., None]) * bl + tx[..., None] * br
    img = (1 - ty[..., None]) * bottom + ty[..., None] * top
    return img

def make_background_array(settings: SettingsDict, width: int=700, height: int=700) -> np.ndarray:
    """Build the active background image array and apply post-processing adjustments.

    Inputs:
        settings: Application settings dictionary following the PART schema.
        width:    Output image width in pixels.
        height:   Output image height in pixels.

Returns an RGB image array for the current background mode."""
    settings = validate_settings(settings)
    mode = settings['background']['mode']
    if mode == 'solid':
        img = make_solid_background_array(settings, width, height)
    elif mode == 'linear':
        img = make_linear_gradient_array(settings, width, height)
    elif mode == 'radial':
        img = make_radial_gradient_array(settings, width, height)
    elif mode == 'mesh':
        img = make_mesh_gradient_array(settings, width, height)
    else:
        img = make_solid_background_array(settings, width, height)
    img = apply_background_brightness_contrast(img, settings['background']['brightness'], settings['background']['contrast'])
    img = apply_texture(img, settings['background']['texture_strength'], settings['meta']['seed'])
    img = apply_vignette(img, settings['background']['vignette_strength'])
    return img

def render_background(ax: Axes, settings: SettingsDict) -> None:
    """Render the current background image beneath the curve artwork.

    Inputs:
        ax:       Matplotlib axes used for drawing the artwork.
        settings: Application settings dictionary following the PART schema.

Returns None after drawing the background onto the provided axes."""
    bg = make_background_array(settings)
    ax.imshow(bg, extent=[-1.8, 1.8, -1.8, 1.8], origin='lower', interpolation='bicubic')

def interpolate_color_sequence(colors: Sequence[str], n: int) -> list[str]:
    """Expand a color sequence into n smoothly interpolated colors.

    Inputs:
        colors: Ordered source color sequence.
        n:      Requested number of output colors.

Returns a list of hex colors spanning the input sequence."""
    colors = list(colors)
    n = max(1, int(n))
    if not colors:
        colors = copy.deepcopy(PALETTE_PRESETS['nebula'])
    if len(colors) == 1 or n == 1:
        return [colors[0]] * n
    anchors = np.linspace(0.0, 1.0, len(colors))
    xs = np.linspace(0.0, 1.0, n)
    palette_rgb = [hex_to_rgb01(c) for c in colors]
    out: list[str] = []
    for x in xs:
        idx = int(np.searchsorted(anchors, x, side='right') - 1)
        idx = int(clamp(idx, 0, len(colors) - 2))
        left = anchors[idx]
        right = anchors[idx + 1]
        t = 0.0 if right <= left else (x - left) / (right - left)
        out.append(rgb01_to_hex(blend_rgb(palette_rgb[idx], palette_rgb[idx + 1], t)))
    return out

def apply_color_shift_to_list(colors: Sequence[str], shift: float) -> list[str]:
    """Apply a global hue shift to a sequence of hex colors.

    Inputs:
        colors: Sequence of hex colors.
        shift:  Hue shift expressed as a unit-interval turn.

Returns the shifted list of hex colors."""
    shift = clamp(float(shift), -0.5, 0.5)
    if abs(shift) < 1e-09:
        return list(colors)
    return [shift_color_hue(c, shift) for c in colors]

def get_palette_colors(settings: SettingsDict, n: int) -> list[str]:
    """Resolve the active palette into a list of curve colors.

    Inputs:
        settings: Application settings dictionary following the PART schema.
        n:        Requested number of output colors.

Returns a list of hex colors."""
    color = settings['color']
    palette = color['palette_colors'][:max(2, color['palette_size'])]
    if len(palette) == 0:
        palette = copy.deepcopy(PALETTE_PRESETS['nebula'])
    distribution = color.get('palette_distribution', 'even_segments')
    if distribution == 'gradient_blend':
        return interpolate_color_sequence(palette, n)
    if distribution == 'weighted':
        weights = np.linspace(len(palette), 1, len(palette), dtype=float)
        cumulative = np.cumsum(weights) / weights.sum()
        xs = (np.arange(max(1, n)) + 0.5) / max(1, n)
        idxs = np.searchsorted(cumulative, xs, side='left')
        return [palette[min(int(idx), len(palette) - 1)] for idx in idxs]
    if n <= 1:
        return [palette[0]]
    idxs = np.floor(np.linspace(0, len(palette), n, endpoint=False)).astype(int)
    return [palette[min(int(idx), len(palette) - 1)] for idx in idxs]

def get_colormap_colors(settings: SettingsDict, n: int) -> list[str]:
    """Sample a curated matplotlib colormap into a list of curve colors.

    Inputs:
        settings: Application settings dictionary following the PART schema.
        n:        Requested number of output colors.

Returns a list of hex colors."""
    color = settings['color']
    cmap = plt.get_cmap(color['colormap'])
    lo, hi = color['colormap_span']
    xs = np.linspace(lo, hi, n)
    return [rgb01_to_hex(cmap(v)[:3]) for v in xs]

def get_curve_colors(settings: SettingsDict, n_segments: int) -> list[str]:
    """Resolve the active color strategy into the colors used to draw the curve.

    Inputs:
        settings:   Application settings dictionary following the PART schema.
        n_segments: Number of colored curve segments to generate.

Returns a list of hex colors."""
    color = settings['color']
    mode = color['mode']
    if mode == 'palette':
        base_colors = get_palette_colors(settings, n_segments)
    else:
        base_colors = get_colormap_colors(settings, n_segments)
    return apply_color_shift_to_list(base_colors, color.get('color_shift', 0.0))

def segment_curve(x: np.ndarray, y: np.ndarray, n_segments: int) -> SegmentList:
    """Split a curve into contiguous x/y segments for per-segment coloring.

    Inputs:
        x:          Array of x coordinates.
        y:          Array of y coordinates.
        n_segments: Number of colored curve segments to generate.

Returns a list of x/y segment tuples."""
    n_segments = max(1, int(n_segments))
    if n_segments == 1:
        return [(x, y)]
    segments = []
    idxs = np.linspace(0, len(x) - 1, n_segments + 1, dtype=int)
    for i in range(n_segments):
        s = idxs[i]
        e = max(idxs[i + 1], s + 2)
        segments.append((x[s:e], y[s:e]))
    return segments

def build_line_segments(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Convert x/y curve coordinates into matplotlib line-collection segments.

    Inputs:
        x: Array of x coordinates.
        y: Array of y coordinates.

Returns an array shaped for LineCollection."""
    points = np.column_stack([x, y])
    if len(points) < 2:
        return np.empty((0, 2, 2), dtype=float)
    return np.stack([points[:-1], points[1:]], axis=1)

def plot_segmented_curve(ax: Axes, x: np.ndarray, y: np.ndarray, colors: Sequence[str] | None, settings: SettingsDict, *, draw_glow: bool=True, draw_main: bool=True) -> None:
    """Plot segmented multicolor curves, optionally drawing glow, main stroke, or both.

    Inputs:
        ax:        Matplotlib axes used for drawing the artwork.
        x:         Array of x coordinates.
        y:         Array of y coordinates.
        colors:    Sequence of per-segment curve colors.
        settings:  Application settings dictionary following the PART schema.
        draw_glow: Whether to draw the glow pass.
        draw_main: Whether to draw the main stroke pass.

Returns None after drawing the requested curve passes."""
    if not colors:
        colors = [settings['color']['curve_color']]
    style = settings['style']
    main_alpha = min(settings['color']['curve_opacity'], style['main_alpha'])
    if len(colors) > 1 and style.get('segment_blending', False):
        line_segments = build_line_segments(x, y)
        if len(line_segments) == 0:
            return
        blended_colors = interpolate_color_sequence(colors, len(line_segments))
        if draw_glow:
            for i in range(style['glow_layers'], 0, -1):
                rgba = [mcolors.to_rgba(c, alpha=style['glow_alpha']) for c in blended_colors]
                ax.add_collection(LineCollection(line_segments, colors=rgba, linewidths=style['line_width'] + i * style['glow_scale']))
        if draw_main:
            rgba = [mcolors.to_rgba(c, alpha=main_alpha) for c in blended_colors]
            ax.add_collection(LineCollection(line_segments, colors=rgba, linewidths=style['line_width']))
        return
    segments = segment_curve(x, y, len(colors))
    for (xs, ys), c in zip(segments, colors):
        if draw_glow:
            for i in range(style['glow_layers'], 0, -1):
                ax.plot(xs, ys, color=c, lw=style['line_width'] + i * style['glow_scale'], alpha=style['glow_alpha'])
        if draw_main:
            ax.plot(xs, ys, color=c, lw=style['line_width'], alpha=main_alpha)

def plot_glow_curve(ax: Axes, x: np.ndarray, y: np.ndarray, settings: SettingsDict, colors: Sequence[str] | None=None) -> None:
    """Plot only the glow layers for the current curve.

    Inputs:
        ax:       Matplotlib axes used for drawing the artwork.
        x:        Array of x coordinates.
        y:        Array of y coordinates.
        settings: Application settings dictionary following the PART schema.
        colors:   Optional sequence of curve colors to use for segmented glow rendering.

Returns None after drawing the glow pass."""
    if colors is None or len(set(colors)) == 1:
        c = colors[0] if colors else settings['color']['curve_color']
        for i in range(settings['style']['glow_layers'], 0, -1):
            ax.plot(x, y, color=c, lw=settings['style']['line_width'] + i * settings['style']['glow_scale'], alpha=settings['style']['glow_alpha'])
    else:
        plot_segmented_curve(ax, x, y, colors, settings, draw_glow=True, draw_main=False)

def plot_main_curve(ax: Axes, x: np.ndarray, y: np.ndarray, settings: SettingsDict, colors: Sequence[str] | None=None) -> None:
    """Plot only the main visible curve stroke.

    Inputs:
        ax:       Matplotlib axes used for drawing the artwork.
        x:        Array of x coordinates.
        y:        Array of y coordinates.
        settings: Application settings dictionary following the PART schema.
        colors:   Optional sequence of curve colors to use for segmented main-stroke rendering.

Returns None after drawing the main stroke."""
    if colors is None or len(set(colors)) == 1:
        c = colors[0] if colors else settings['color']['curve_color']
        ax.plot(x, y, color=c, lw=settings['style']['line_width'], alpha=min(settings['style']['main_alpha'], settings['color']['curve_opacity']))
    else:
        plot_segmented_curve(ax, x, y, colors, settings, draw_glow=False, draw_main=True)

def setup_figure(figsize: tuple[float, float]=(8, 8)) -> FigureAxes:
    """Create a matplotlib figure and axes configured for square artwork rendering.

    Inputs:
        figsize: Requested figure size in inches as a width-height tuple.

Returns the created figure and axes."""
    fig, ax = plt.subplots(figsize=figsize)
    return (fig, ax)

def apply_plot_framing(ax: Axes, x: np.ndarray, y: np.ndarray, settings: SettingsDict) -> None:
    """Apply final axis limits, aspect ratio, and axis visibility for the rendered artwork.

    Inputs:
        ax:       Matplotlib axes used for drawing the artwork.
        x:        Array of x coordinates.
        y:        Array of y coordinates.
        settings: Application settings dictionary following the PART schema.

Returns None after configuring the axes."""
    ax.set_aspect('equal')
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.8, 1.8)
    ax.axis('off')

def render_artwork(settings: SettingsDict, ax: Axes | None=None, figsize: tuple[float, float]=(8, 8)) -> FigureAxes:
    """Render a complete artwork preview from validated settings.

    Inputs:
        settings: Application settings dictionary following the PART schema.
        ax:       Optional axes to reuse for rendering instead of creating a new figure.
        figsize:  Requested figure size in inches as a width-height tuple.

Returns the figure and axes containing the rendered artwork."""
    settings = validate_settings(settings)
    x, y = generate_curve(settings)
    colors = get_curve_colors(settings, max(1, settings['color']['num_curve_colors']))
    own_fig = False
    if ax is None:
        fig, ax = setup_figure(figsize=figsize)
        own_fig = True
    else:
        fig = ax.figure
    render_background(ax, settings)
    plot_glow_curve(ax, x, y, settings, colors)
    plot_main_curve(ax, x, y, settings, colors)
    apply_plot_framing(ax, x, y, settings)
    return (fig, ax)


def apply_style_preset(settings: SettingsDict, preset_name: str) -> SettingsDict:
    """Apply a named style preset to the current settings.

    Inputs:
        settings:    Application settings dictionary following the PART schema.
        preset_name: Name of the preset to apply.

Returns a validated settings dictionary."""
    settings = deep_copy_settings(settings)
    if preset_name not in STYLE_PRESETS:
        return validate_settings(settings)
    preset = STYLE_PRESETS[preset_name]
    settings['meta']['artwork_preset'] = 'custom'
    settings['meta']['style_preset'] = preset_name
    for k, v in preset.items():
        settings['style'][k] = copy.deepcopy(v)
    return validate_settings(settings)

def apply_palette_preset(settings: SettingsDict, preset_name: str) -> SettingsDict:
    """Apply a named palette preset to the current settings.

    Inputs:
        settings:    Application settings dictionary following the PART schema.
        preset_name: Name of the preset to apply.

Returns a validated settings dictionary."""
    settings = deep_copy_settings(settings)
    settings['meta']['artwork_preset'] = 'custom'
    if preset_name not in PALETTE_PRESETS or preset_name == 'custom':
        settings['meta']['palette_preset'] = 'custom'
        return validate_settings(settings)
    settings['meta']['palette_preset'] = preset_name
    settings['color']['mode'] = 'palette'
    settings['color']['palette_name'] = preset_name
    settings['color']['palette_colors'] = copy.deepcopy(PALETTE_PRESETS[preset_name])
    settings['color']['palette_size'] = min(6, len(settings['color']['palette_colors']))
    settings['color']['manual_palette'] = False
    return validate_settings(settings)

def apply_background_preset(settings: SettingsDict, preset_name: str) -> SettingsDict:
    """Apply a named background preset and validate the resulting background state.

    Inputs:
        settings:    Application settings dictionary following the PART schema.
        preset_name: Name of the preset to apply.

Returns a validated settings dictionary."""
    settings = deep_copy_settings(settings)
    settings['meta']['artwork_preset'] = 'custom'
    if preset_name not in BACKGROUND_PRESETS or preset_name == 'custom':
        settings['meta']['background_preset'] = 'custom'
        settings['background']['preset'] = 'custom'
        return validate_settings(settings)
    settings['meta']['background_preset'] = preset_name
    preset = BACKGROUND_PRESETS[preset_name]
    settings['background']['preset'] = preset_name
    settings['background']['mode'] = preset['mode']
    if 'linear' in preset:
        settings['background']['linear'].update(copy.deepcopy(preset['linear']))
    if 'radial' in preset:
        settings['background']['radial'].update(copy.deepcopy(preset['radial']))
    if 'mesh' in preset:
        settings['background']['mesh'].update(copy.deepcopy(preset['mesh']))
    return validate_settings(settings)

def apply_artwork_preset(settings: SettingsDict, preset_name: str) -> SettingsDict:
    """Apply a full artwork preset spanning curve, style, color, and background settings.

    Inputs:
        settings:    Application settings dictionary following the PART schema.
        preset_name: Name of the preset to apply.

Returns a validated settings dictionary."""
    settings = deep_copy_settings(settings)
    if preset_name not in ARTWORK_PRESETS or preset_name == 'custom':
        settings['meta']['artwork_preset'] = 'custom'
        return validate_settings(settings)
    preset = ARTWORK_PRESETS[preset_name]
    settings['meta']['artwork_preset'] = preset_name
    settings['curve']['family'] = preset['curve']['family']
    settings['curve']['params'] = make_default_curve_params(settings['curve']['family'])
    settings['curve']['params'].update(copy.deepcopy(preset['curve']['params']))
    settings['curve']['turns'] = preset['curve']['turns']
    settings['curve']['asymmetry'] = preset['curve']['asymmetry']
    settings['curve']['noise_strength'] = preset['curve']['noise_strength']
    settings['meta']['seed'] = preset.get('meta_seed', settings['meta']['seed'])
    settings = apply_style_preset(settings, preset['style_preset'])
    settings = apply_palette_preset(settings, preset['palette_preset'])
    settings = apply_background_preset(settings, preset['background_preset'])
    settings['meta']['artwork_preset'] = preset_name
    settings['color'].update(copy.deepcopy(preset.get('color', {})))
    return validate_settings(settings)

NON_COMPOSITION_WIDGET_KEYS = {'seed'}
CURVE_WIDGET_KEYS = {'family', 'turns', 'asymmetry', 'noise_strength', 'zoom', 'liss_a', 'liss_b', 'liss_delta', 'rose_k', 'rose_radial_wobble', 'rose_wobble_freq', 'epi_R', 'epi_r', 'epi_d', 'burst_amp1', 'burst_amp2', 'burst_f1', 'burst_f2', 'burst_delta'}
STYLE_WIDGET_KEYS = {'glow_macro', 'line_width', 'glow_layers', 'glow_alpha', 'glow_scale', 'main_alpha', 'segment_blending'}
PALETTE_WIDGET_KEYS = {'color_mode', 'palette_name', 'palette_manual_enabled', 'palette_size', 'palette_distribution', 'colormap', 'colormap_span', 'curve_opacity', 'num_curve_colors', 'color_shift'} | {f'palette_color_{i}' for i in range(1, 7)}
BACKGROUND_WIDGET_KEYS = {'background_mode', 'bg_solid_color', 'bg_linear_top', 'bg_linear_bottom', 'bg_linear_angle', 'bg_radial_inner', 'bg_radial_outer', 'bg_radial_center_x', 'bg_radial_center_y', 'bg_radial_radius_bias', 'bg_mesh_tl', 'bg_mesh_tr', 'bg_mesh_bl', 'bg_mesh_br', 'bg_brightness', 'bg_contrast', 'bg_texture_strength', 'bg_vignette_strength'}

def get_widget_key_from_change(ui: WidgetDict, change: Any=None) -> Optional[str]:
    """Resolve the stable widget key for an ipywidgets change payload."""
    owner = change.get('owner') if isinstance(change, dict) else getattr(change, 'owner', None)
    if owner is None:
        return None
    for key, widget in ui.items():
        if widget is owner:
            return key
    return None

def apply_macro_change_if_needed(settings: SettingsDict, changed_key: Optional[str]) -> SettingsDict:
    """Apply macro writers only for direct macro-control edits."""
    settings = deep_copy_settings(settings)
    if changed_key == 'glow_macro':
        settings = apply_glow_macro(settings)
    return settings

def mark_presets_custom_if_needed(settings: SettingsDict, changed_keys: Optional[Sequence[str]]=None) -> SettingsDict:
    """Clear only the preset labels affected by the supplied manual-edit widget keys.

    Inputs:
        settings:     Application settings dictionary following the PART schema.
        changed_keys: Optional widget-key sequence describing which controls changed.

Returns the updated settings dictionary."""
    settings = deep_copy_settings(settings)
    if changed_keys is None:
        settings['meta']['artwork_preset'] = 'custom'
        settings['meta']['style_preset'] = 'custom'
        settings['meta']['palette_preset'] = 'custom'
        settings['meta']['background_preset'] = 'custom'
        settings['background']['preset'] = 'custom'
        return settings
    affected: set[str] = set()
    for key in changed_keys:
        if not key or key in NON_COMPOSITION_WIDGET_KEYS:
            continue
        if key in CURVE_WIDGET_KEYS | STYLE_WIDGET_KEYS | PALETTE_WIDGET_KEYS | BACKGROUND_WIDGET_KEYS:
            affected.add('artwork')
        if key in STYLE_WIDGET_KEYS:
            affected.add('style')
        if key in PALETTE_WIDGET_KEYS:
            affected.add('palette')
        if key in BACKGROUND_WIDGET_KEYS:
            affected.add('background')
    if 'artwork' in affected:
        settings['meta']['artwork_preset'] = 'custom'
    if 'style' in affected:
        settings['meta']['style_preset'] = 'custom'
    if 'palette' in affected:
        settings['meta']['palette_preset'] = 'custom'
    if 'background' in affected:
        settings['meta']['background_preset'] = 'custom'
        settings['background']['preset'] = 'custom'
    return settings

def perturb_value(rng: np.random.Generator, value: float, lo: float, hi: float, intensity: str, integer: bool=False) -> float | int:
    """Perturb a numeric value within bounded limits according to the selected randomization intensity.

    Inputs:
        rng:       NumPy random generator used for deterministic perturbations.
        value:     Current numeric value to perturb.
        lo:        Lower clamp or perturbation bound.
        hi:        Upper clamp or perturbation bound.
        intensity: Randomization intensity level.
        integer:   Whether the perturbed value should be rounded to an integer.

Returns the perturbed numeric value."""
    pct = {'subtle': 0.1, 'moderate': 0.28, 'wild': 0.65}[intensity]
    span = hi - lo
    if intensity == 'wild':
        out = rng.uniform(lo, hi)
    else:
        out = value + rng.normal(0, pct * span)
    out = clamp(out, lo, hi)
    return int(round(out)) if integer else float(out)

def choose_related_palette(current_palette_name: str, intensity: str, rng: np.random.Generator) -> str:
    """Choose a palette preset that is compatible with the requested randomization intensity.

    Inputs:
        current_palette_name: Current palette preset name.
        intensity:            Randomization intensity level.
        rng:                  NumPy random generator used for deterministic perturbations.

Returns the selected palette preset name."""
    names = [k for k in PALETTE_PRESETS.keys() if k != 'custom']
    if current_palette_name in names and intensity == 'subtle':
        return current_palette_name
    return rng.choice(names)

def choose_related_background(current_bg_name: str, intensity: str, rng: np.random.Generator) -> str:
    """Choose a background preset that is compatible with the requested randomization intensity.

    Inputs:
        current_bg_name: Current background preset name.
        intensity:       Randomization intensity level.
        rng:             NumPy random generator used for deterministic perturbations.

Returns the selected background preset name."""
    names = [k for k in BACKGROUND_PRESETS.keys() if k != 'custom']
    if current_bg_name in names and intensity == 'subtle':
        return current_bg_name
    return rng.choice(names)

def randomize_lissajous_params(params: dict[str, Any], intensity: str, rng: np.random.Generator) -> dict[str, float]:
    """Randomize family-specific parameters for a Lissajous curve.

    Inputs:
        params:    Current Lissajous parameter dictionary.
        intensity: Randomization intensity level.
        rng:       NumPy random generator used for deterministic perturbations.

Returns a randomized Lissajous parameter dictionary."""
    curated = [(2, 3), (3, 4), (3, 5), (4, 7), (5, 8), (7, 5), (8, 3)]
    if intensity == 'subtle':
        a, b = (params['a'], params['b'])
    else:
        a, b = curated[rng.integers(0, len(curated))]
    return {'a': perturb_value(rng, a, *PARAM_RANGES['lissajous.a'], intensity, integer=True), 'b': perturb_value(rng, b, *PARAM_RANGES['lissajous.b'], intensity, integer=True), 'delta': perturb_value(rng, params['delta'], *PARAM_RANGES['lissajous.delta'], intensity, integer=False)}

def randomize_rose_params(params: dict[str, Any], intensity: str, rng: np.random.Generator) -> dict[str, float]:
    """Randomize family-specific parameters for a rose curve.

    Inputs:
        params:    Current rose-parameter dictionary.
        intensity: Randomization intensity level.
        rng:       NumPy random generator used for deterministic perturbations.

Returns a randomized rose-parameter dictionary."""
    return {'k': perturb_value(rng, params['k'], *PARAM_RANGES['rose.k'], intensity, integer=True), 'radial_wobble': perturb_value(rng, min(params['radial_wobble'], 0.25), *PARAM_RANGES['rose.radial_wobble'], intensity, integer=False), 'wobble_freq': perturb_value(rng, params['wobble_freq'], *PARAM_RANGES['rose.wobble_freq'], intensity, integer=True)}

def randomize_epitrochoid_params(params: dict[str, Any], intensity: str, rng: np.random.Generator) -> dict[str, float]:
    """Randomize family-specific parameters for an epitrochoid curve.

    Inputs:
        params:    Current epitrochoid-parameter dictionary.
        intensity: Randomization intensity level.
        rng:       NumPy random generator used for deterministic perturbations.

Returns a randomized epitrochoid-parameter dictionary."""
    ratios = [(5, 3), (7, 4), (8, 3), (9, 5), (6, 2)]
    if intensity == 'wild':
        R = rng.integers(3, 12)
        r = rng.integers(1, max(2, R))
    else:
        R, r = ratios[rng.integers(0, len(ratios))]
    d = perturb_value(rng, params['d'], *PARAM_RANGES['epitrochoid.d'], intensity, integer=False)
    return {'R': float(R), 'r': float(r), 'd': d}

def randomize_burst_params(params: dict[str, Any], intensity: str, rng: np.random.Generator) -> dict[str, float]:
    """Randomize family-specific parameters for a burst curve.

    Inputs:
        params:    Current burst-parameter dictionary.
        intensity: Randomization intensity level.
        rng:       NumPy random generator used for deterministic perturbations.

Returns a randomized burst-parameter dictionary."""
    f1 = perturb_value(rng, params['f1'], *PARAM_RANGES['burst.f1'], intensity, integer=True)
    f2 = perturb_value(rng, params['f2'], *PARAM_RANGES['burst.f2'], intensity, integer=True)
    if f1 == f2:
        f2 = clamp(f2 + 2, *PARAM_RANGES['burst.f2'])
    return {'amp1': perturb_value(rng, max(params['amp1'], params['amp2']), *PARAM_RANGES['burst.amp1'], intensity, integer=False), 'amp2': perturb_value(rng, min(params['amp2'], params['amp1']), *PARAM_RANGES['burst.amp2'], intensity, integer=False), 'f1': float(f1), 'f2': float(f2), 'delta': perturb_value(rng, params['delta'], *PARAM_RANGES['burst.delta'], intensity, integer=False)}

def resolve_randomization_intensity(settings: SettingsDict, intensity: str | None=None) -> str:
    """Resolve the randomization intensity while ignoring removed UI-only state.

    Inputs:
        settings:  Application settings dictionary following the PART schema.
        intensity: Optional explicit randomization intensity.

Returns the selected randomization intensity."""
    if intensity in {'subtle', 'moderate', 'wild'}:
        return intensity
    legacy_intensity = get_nested(settings, 'randomization.intensity')
    if legacy_intensity in {'subtle', 'moderate', 'wild'}:
        return legacy_intensity
    return DEFAULT_RANDOMIZATION_INTENSITY

def randomize_style(settings: SettingsDict, intensity: str=None, rng: np.random.Generator=None, *, glow_only: bool=False) -> SettingsDict:
    """Randomize style settings, optionally restricting changes to glow-specific fields only.

    Inputs:
        settings:  Application settings dictionary following the PART schema.
        intensity: Randomization intensity level.
        rng:       NumPy random generator used for deterministic perturbations.
        glow_only: Whether to randomize only glow-related style controls.

Returns a validated settings dictionary."""
    settings = deep_copy_settings(settings)
    rng = np.random.default_rng() if rng is None else rng
    intensity = resolve_randomization_intensity(settings, intensity)
    settings['meta']['artwork_preset'] = 'custom'
    settings['meta']['style_preset'] = 'custom'
    if not glow_only:
        settings['style']['line_width'] = perturb_value(rng, settings['style']['line_width'], *PARAM_RANGES['style.line_width'], intensity)
        settings['style']['main_alpha'] = perturb_value(rng, settings['style']['main_alpha'], *PARAM_RANGES['style.main_alpha'], intensity)
    settings['style']['glow_layers'] = perturb_value(rng, settings['style']['glow_layers'], *PARAM_RANGES['style.glow_layers'], intensity, integer=True)
    settings['style']['glow_alpha'] = perturb_value(rng, settings['style']['glow_alpha'], *PARAM_RANGES['style.glow_alpha'], intensity)
    settings['style']['glow_scale'] = perturb_value(rng, settings['style']['glow_scale'], *PARAM_RANGES['style.glow_scale'], intensity)
    return validate_settings(settings)

def randomize_palette(settings: SettingsDict, intensity: str=None, rng: np.random.Generator=None) -> SettingsDict:
    """Randomize palette-related color settings without mutating unrelated groups.

    Inputs:
        settings:  Application settings dictionary following the PART schema.
        intensity: Randomization intensity level.
        rng:       NumPy random generator used for deterministic perturbations.

Returns a validated settings dictionary."""
    settings = deep_copy_settings(settings)
    rng = np.random.default_rng() if rng is None else rng
    intensity = resolve_randomization_intensity(settings, intensity)
    settings['meta']['artwork_preset'] = 'custom'
    settings['meta']['palette_preset'] = 'custom'
    palette_name = choose_related_palette(settings['color']['palette_name'], intensity, rng)
    if intensity == 'wild' and rng.random() < 0.25:
        settings['color']['mode'] = 'colormap'
        settings['color']['colormap'] = rng.choice(CURATED_COLORMAPS)
    else:
        if settings['color']['mode'] not in {v for _, v in COLOR_MODE_OPTIONS}:
            settings['color']['mode'] = 'palette'
        settings['color']['palette_name'] = palette_name
        settings['color']['palette_colors'] = copy.deepcopy(PALETTE_PRESETS[palette_name])
        settings['color']['palette_size'] = int(clamp(len(settings['color']['palette_colors']) + rng.integers(-1, 2), 2, 6))
        settings['color']['num_curve_colors'] = int(clamp(settings['color']['num_curve_colors'] + rng.integers(-1, 3), 1, 12))
        distribution_choices = [value for _, value in PALETTE_DISTRIBUTION_OPTIONS]
        change_distribution = {'subtle': 0.2, 'moderate': 0.45, 'wild': 0.7}[intensity]
        if rng.random() < change_distribution:
            settings['color']['palette_distribution'] = rng.choice(distribution_choices)
    shift_delta = {'subtle': 0.04, 'moderate': 0.1, 'wild': 0.22}[intensity]
    settings['color']['color_shift'] = clamp(settings['color'].get('color_shift', 0.0) + rng.uniform(-shift_delta, shift_delta), -0.5, 0.5)
    return validate_settings(settings)

def randomize_background(settings: SettingsDict, intensity: str=None, rng: np.random.Generator=None) -> SettingsDict:
    """Randomize background settings and clear preset labels after post-preset perturbations.

    Inputs:
        settings:  Application settings dictionary following the PART schema.
        intensity: Randomization intensity level.
        rng:       NumPy random generator used for deterministic perturbations.

Returns a validated settings dictionary."""
    settings = deep_copy_settings(settings)
    rng = np.random.default_rng() if rng is None else rng
    intensity = resolve_randomization_intensity(settings, intensity)
    preset_name = choose_related_background(settings['background']['preset'], intensity, rng)
    settings = apply_background_preset(settings, preset_name)
    if intensity != 'subtle' and rng.random() < 0.25:
        settings['background']['mode'] = rng.choice(['solid', 'linear', 'radial', 'mesh'])
    settings['background']['brightness'] = perturb_value(rng, settings['background']['brightness'], *PARAM_RANGES['background.brightness'], intensity)
    settings['background']['contrast'] = perturb_value(rng, settings['background']['contrast'], *PARAM_RANGES['background.contrast'], intensity)
    settings['background']['texture_strength'] = perturb_value(rng, settings['background']['texture_strength'], *PARAM_RANGES['background.texture_strength'], intensity)
    settings['background']['vignette_strength'] = perturb_value(rng, settings['background']['vignette_strength'], *PARAM_RANGES['background.vignette_strength'], intensity)
    settings['meta']['artwork_preset'] = 'custom'
    settings['meta']['background_preset'] = 'custom'
    settings['background']['preset'] = 'custom'
    return validate_settings(settings)

def randomize_curve(settings: SettingsDict, intensity: str=None, keep_family: bool=False, rng: np.random.Generator=None, *, include_shared_curve_controls: bool=True) -> SettingsDict:
    """Randomize curve settings, with an option to restrict changes to family-specific parameters only.

    Inputs:
        settings:                      Application settings dictionary following the PART schema.
        intensity:                     Randomization intensity level.
        keep_family:                   Whether to preserve the current family while randomizing.
        rng:                           NumPy random generator used for deterministic perturbations.
        include_shared_curve_controls: Whether to randomize shared curve controls in addition to family-specific parameters.

Returns a validated settings dictionary."""
    settings = deep_copy_settings(settings)
    rng = np.random.default_rng() if rng is None else rng
    intensity = resolve_randomization_intensity(settings, intensity)
    if not keep_family:
        family_change_probability = {'subtle': 0.2, 'moderate': 0.5, 'wild': 1.0}.get(intensity, 0.5)
        if rng.random() < family_change_probability:
            choices = [v for _, v in FAMILY_OPTIONS if v != settings['curve']['family']]
            if choices:
                settings['curve']['family'] = rng.choice(choices)
                settings['curve']['params'] = make_default_curve_params(settings['curve']['family'])
    fam = settings['curve']['family']
    if fam == 'lissajous':
        settings['curve']['params'] = randomize_lissajous_params(settings['curve']['params'], intensity, rng)
    elif fam == 'rose':
        settings['curve']['params'] = randomize_rose_params(settings['curve']['params'], intensity, rng)
    elif fam == 'epitrochoid':
        settings['curve']['params'] = randomize_epitrochoid_params(settings['curve']['params'], intensity, rng)
    elif fam == 'burst':
        settings['curve']['params'] = randomize_burst_params(settings['curve']['params'], intensity, rng)
    if include_shared_curve_controls:
        settings['curve']['turns'] = perturb_value(rng, settings['curve']['turns'], *PARAM_RANGES['curve.turns'], intensity)
        settings['curve']['asymmetry'] = perturb_value(rng, settings['curve']['asymmetry'], *PARAM_RANGES['curve.asymmetry'], intensity)
        settings['curve']['noise_strength'] = perturb_value(rng, settings['curve']['noise_strength'], *PARAM_RANGES['curve.noise_strength'], intensity)
        settings['curve']['zoom'] = perturb_value(rng, settings['curve']['zoom'], *PARAM_RANGES['curve.zoom'], intensity)
    settings['meta']['artwork_preset'] = 'custom'
    return validate_settings(settings)

def shuffle_seed(settings: SettingsDict, rng: np.random.Generator=None) -> SettingsDict:
    """Assign a new variation seed.

    Inputs:
        settings: Application settings dictionary following the PART schema.
        rng:      NumPy random generator used for deterministic perturbations.

Returns a settings dictionary with an updated seed."""
    settings = deep_copy_settings(settings)
    rng = np.random.default_rng() if rng is None else rng
    settings['meta']['seed'] = int(rng.integers(0, 10000))
    return settings

def randomize_settings(settings: SettingsDict, scope: Any=None, intensity: str=None) -> SettingsDict:
    """Randomize settings according to the selected scope while keeping scope boundaries consistent.

    Inputs:
        settings:  Application settings dictionary following the PART schema.
        scope:     Value used by this function.
        intensity: Randomization intensity level.

Returns a validated settings dictionary."""
    settings = deep_copy_settings(settings)
    scope = scope or 'all'
    intensity = resolve_randomization_intensity(settings, intensity)
    rng = np.random.default_rng(settings['meta']['seed'] + 777)
    if scope == 'seed_only':
        settings = shuffle_seed(settings, rng)
        return validate_settings(settings)
    if scope == 'all':
        settings = shuffle_seed(settings, rng)
        settings = randomize_curve(settings, intensity=intensity, keep_family=False, rng=rng)
        settings = randomize_style(settings, intensity=intensity, rng=rng)
        settings = randomize_palette(settings, intensity=intensity, rng=rng)
        settings = randomize_background(settings, intensity=intensity, rng=rng)
    elif scope == 'curve':
        settings = randomize_curve(settings, intensity=intensity, keep_family=False, rng=rng)
    elif scope == 'current_family':
        settings = randomize_curve(settings, intensity=intensity, keep_family=True, rng=rng)
    elif scope == 'curve_params':
        settings = randomize_curve(settings, intensity=intensity, keep_family=True, rng=rng, include_shared_curve_controls=False)
    elif scope == 'style':
        settings = randomize_style(settings, intensity=intensity, rng=rng)
    elif scope == 'palette':
        settings = randomize_palette(settings, intensity=intensity, rng=rng)
    elif scope == 'background':
        settings = randomize_background(settings, intensity=intensity, rng=rng)
    elif scope == 'glow_only':
        settings = randomize_style(settings, intensity=intensity, rng=rng, glow_only=True)
    return validate_settings(settings)


def build_export_filename(settings: SettingsDict, ext: str='png') -> str:
    """Build the export filename for the current artwork settings.

    Inputs:
        settings: Application settings dictionary following the PART schema.
        ext:      File extension to use for the generated filename.

Returns the filesystem path for the export target."""
    family = settings['curve']['family']
    preset = settings['meta']['artwork_preset']
    return os.path.join(EXPORT_FOLDER, f'parametric_art_{family}_{preset}_{now_timestamp_str()}.{ext}')

def _save_rendered_figure_atomic(settings: SettingsDict, ext: str, folder: str | None=None) -> str:
    """Render artwork to disk atomically so failed exports do not leave partial files behind."""
    path = build_export_filename(settings, ext)
    if folder:
        path = os.path.join(folder, os.path.basename(path))
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix='part_export_', suffix=f'.{ext}', dir=os.path.dirname(path) or '.')
    os.close(fd)
    fig = None
    try:
        fig, _ = render_artwork(settings, figsize=(8, 8))
        if ext == 'png':
            FigureCanvasAgg(fig)
            fig.canvas.draw()
            fig.savefig(tmp_path, format='png', dpi=PNG_DPI, bbox_inches='tight', pad_inches=0)
        elif ext == 'svg':
            FigureCanvasSVG(fig)
            fig.savefig(tmp_path, format='svg', bbox_inches='tight', pad_inches=0)
        else:
            raise ValueError(f'Unsupported export format: {ext}')
        os.replace(tmp_path, path)
        return path
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    finally:
        if fig is not None:
            plt.close(fig)

def save_current_png(settings: SettingsDict, folder: str | None=None) -> str:
    """Render and save the current artwork as a PNG file.

    Inputs:
        settings: Application settings dictionary following the PART schema.
        folder:   Optional folder override for the saved export.

Returns the saved PNG path."""
    return _save_rendered_figure_atomic(settings, 'png', folder=folder)

def save_current_svg(settings: SettingsDict, folder: str | None=None) -> str:
    """Render and save the current artwork as an SVG file.

    Inputs:
        settings: Application settings dictionary following the PART schema.
        folder:   Optional folder override for the saved export.

Returns the saved SVG path."""
    return _save_rendered_figure_atomic(settings, 'svg', folder=folder)

def _dropdown(options: Sequence[tuple[str, str]], value: float, description: str, width: str='240px') -> Any:
    """Create a standard dropdown widget with the notebook's shared layout conventions.

    Inputs:
        options:     Dropdown option pairs in display-label/value form.
        value:       Numeric or generic value being processed by the helper.
        description: Human-readable widget label.
        width:       CSS width string used for the widget layout.

Returns the configured dropdown widget."""
    return widgets.Dropdown(options=options, value=value, description=description, layout=widgets.Layout(width=width))

def _slider(value: float, minv: float, maxv: float, step: float, description: str, width: str=DEFAULT_SLIDER_WIDTH, readout_format: str='.2f') -> Any:
    """Create a standard floating-point slider widget with the notebook's shared layout conventions.

    Inputs:
        value:          Numeric or generic value being processed by the helper.
        minv:           Minimum allowed slider value.
        maxv:           Maximum allowed slider value.
        step:           Slider increment.
        description:    Human-readable widget label.
        width:          CSS width string used for the widget layout.
        readout_format: Numeric display format used by the slider readout.

Returns the configured float slider widget."""
    return widgets.FloatSlider(value=value, min=minv, max=maxv, step=step, description=description, readout_format=readout_format, continuous_update=False, layout=widgets.Layout(width=width))

def _int_slider(value: float, minv: float, maxv: float, step: float, description: str, width: str=DEFAULT_SLIDER_WIDTH) -> Any:
    """Create a standard integer slider widget with the notebook's shared layout conventions.

    Inputs:
        value:       Numeric or generic value being processed by the helper.
        minv:        Minimum allowed slider value.
        maxv:        Maximum allowed slider value.
        step:        Slider increment.
        description: Human-readable widget label.
        width:       CSS width string used for the widget layout.

Returns the configured integer slider widget."""
    return widgets.IntSlider(value=value, min=minv, max=maxv, step=step, description=description, continuous_update=False, layout=widgets.Layout(width=width))

def build_ui_widgets() -> WidgetDict:
    """Create and return the full widget dictionary used by the notebook UI.

    Inputs:
        None: This function does not accept positional inputs.

Returns a dictionary of initialized widgets."""
    s = validate_settings(make_default_settings())
    ui = {}
    ui['family'] = _dropdown(FAMILY_OPTIONS, s['curve']['family'], 'Family:')
    ui['artwork_preset'] = _dropdown([(k.replace('_', ' ').title(), k) for k in ARTWORK_PRESETS.keys()], s['meta']['artwork_preset'], 'Preset:')
    ui['style_preset'] = _dropdown([(k.replace('_', ' ').title(), k) for k in STYLE_PRESETS.keys()], s['meta']['style_preset'], 'Style:')
    ui['palette_preset'] = _dropdown([(k.replace('_', ' ').title(), k) for k in PALETTE_PRESETS.keys()], s['meta']['palette_preset'], 'Palette:')
    ui['background_preset'] = _dropdown([(k.replace('_', ' ').title(), k) for k in BACKGROUND_PRESETS.keys()], s['meta']['background_preset'], 'Background:')
    ui['seed'] = _int_slider(s['meta']['seed'], 0, 9999, 1, 'Seed:')
    ui['glow_macro'] = _slider(s['style']['glow_macro'], 0, 1, 0.01, 'Glow:')
    ui['btn_randomize_all'] = widgets.Button(description='Randomize All', button_style='primary')
    ui['btn_randomize_curve'] = widgets.Button(description='Randomize Curve')
    ui['btn_randomize_colors'] = widgets.Button(description='Randomize Colors')
    ui['btn_randomize_background'] = widgets.Button(description='Randomize BG')
    ui['btn_save_png'] = widgets.Button(description='Save PNG', button_style='success')
    ui['btn_save_svg'] = widgets.Button(description='Save SVG')
    ui['turns'] = _slider(s['curve']['turns'], 0.5, 12, 0.1, 'Cycles:')
    ui['asymmetry'] = _slider(s['curve']['asymmetry'], 0, 0.5, 0.005, 'Asymmetry:')
    ui['noise_strength'] = _slider(s['curve']['noise_strength'], 0, 0.05, 0.0005, 'Noise:')
    ui['zoom'] = _slider(s['curve']['zoom'], 0.5, 2.0, 0.01, 'Zoom:')
    ui['liss_a'] = _slider(s['curve']['params']['a'], 1, 12, 1, 'A Freq:', readout_format='.0f')
    ui['liss_b'] = _slider(s['curve']['params']['b'], 1, 12, 1, 'B Freq:', readout_format='.0f')
    ui['liss_delta'] = _slider(s['curve']['params']['delta'], 0, 2 * np.pi, 0.01, 'Phase:')
    ui['rose_k'] = _slider(5, 1, 16, 1, 'Petal K:', readout_format='.0f')
    ui['rose_radial_wobble'] = _slider(0.12, 0, 0.5, 0.005, 'Wobble:')
    ui['rose_wobble_freq'] = _slider(6, 1, 20, 1, 'Wobble F:', readout_format='.0f')
    ui['epi_R'] = _slider(5, 1, 12, 1, 'Outer R:', readout_format='.0f')
    ui['epi_r'] = _slider(3, 1, 12, 1, 'Inner r:', readout_format='.0f')
    ui['epi_d'] = _slider(5, 0, 12, 0.1, 'Offset d:')
    ui['burst_amp1'] = _slider(0.45, 0, 1, 0.01, 'Amp 1:')
    ui['burst_amp2'] = _slider(0.22, 0, 1, 0.01, 'Amp 2:')
    ui['burst_f1'] = _slider(8, 1, 30, 1, 'Freq 1:', readout_format='.0f')
    ui['burst_f2'] = _slider(17, 1, 30, 1, 'Freq 2:', readout_format='.0f')
    ui['burst_delta'] = _slider(0.3, 0, 2 * np.pi, 0.01, 'Phase:')
    ui['color_mode'] = widgets.ToggleButtons(options=COLOR_MODE_OPTIONS, value=s['color']['mode'], description='Mode:')
    ui['palette_name'] = _dropdown([(k.replace('_', ' ').title(), k) for k in PALETTE_PRESETS.keys() if k != 'custom'], s['color']['palette_name'], 'Palette:')
    ui['palette_manual_enabled'] = widgets.Checkbox(value=bool(s['color'].get('manual_palette', False)), description='Manual palette')
    ui['palette_size'] = _int_slider(s['color']['palette_size'], 2, 6, 1, 'Palette Size:')
    ui['palette_distribution'] = _dropdown(PALETTE_DISTRIBUTION_OPTIONS, s['color'].get('palette_distribution', 'even_segments'), 'Distribution:')
    for i in range(1, 7):
        default_color = s['color']['palette_colors'][min(i - 1, len(s['color']['palette_colors']) - 1)]
        ui[f'palette_color_{i}'] = widgets.ColorPicker(value=default_color, description=f'Color {i}:')
    ui['colormap'] = _dropdown([(name, name) for name in CURATED_COLORMAPS], s['color']['colormap'], 'Colormap:')
    ui['colormap_span'] = widgets.FloatRangeSlider(value=tuple(s['color']['colormap_span']), min=0, max=1, step=0.01, description='CM Span:', continuous_update=False, layout=widgets.Layout(width=DEFAULT_RANGE_SLIDER_WIDTH))
    ui['curve_opacity'] = _slider(s['color']['curve_opacity'], 0.1, 1.0, 0.01, 'Opacity:')
    ui['num_curve_colors'] = _int_slider(s['color']['num_curve_colors'], 1, 12, 1, 'Curve Colors:')
    ui['color_shift'] = _slider(s['color'].get('color_shift', 0.0), -0.5, 0.5, 0.01, 'Color Shift:')
    ui['background_mode'] = widgets.ToggleButtons(options=BACKGROUND_MODE_OPTIONS, value=s['background']['mode'], description='BG Mode:')
    ui['bg_solid_color'] = widgets.ColorPicker(value=s['background']['solid_color'], description='Solid:')
    ui['bg_linear_top'] = widgets.ColorPicker(value=s['background']['linear']['top'], description='Top:')
    ui['bg_linear_bottom'] = widgets.ColorPicker(value=s['background']['linear']['bottom'], description='Bottom:')
    ui['bg_linear_angle'] = _slider(s['background']['linear']['angle'], 0, 180, 1, 'Angle:', readout_format='.0f')
    ui['bg_radial_inner'] = widgets.ColorPicker(value=s['background']['radial']['inner'], description='Inner:')
    ui['bg_radial_outer'] = widgets.ColorPicker(value=s['background']['radial']['outer'], description='Outer:')
    ui['bg_radial_center_x'] = _slider(s['background']['radial']['center_x'], 0, 1, 0.01, 'Center X:')
    ui['bg_radial_center_y'] = _slider(s['background']['radial']['center_y'], 0, 1, 0.01, 'Center Y:')
    ui['bg_radial_radius_bias'] = _slider(s['background']['radial']['radius_bias'], 0.25, 2.0, 0.01, 'Radius Bias:')
    ui['bg_mesh_tl'] = widgets.ColorPicker(value=s['background']['mesh']['tl'], description='Top Left:')
    ui['bg_mesh_tr'] = widgets.ColorPicker(value=s['background']['mesh']['tr'], description='Top Right:')
    ui['bg_mesh_bl'] = widgets.ColorPicker(value=s['background']['mesh']['bl'], description='Bottom Left:')
    ui['bg_mesh_br'] = widgets.ColorPicker(value=s['background']['mesh']['br'], description='Bottom Right:')
    ui['bg_brightness'] = _slider(s['background']['brightness'], 0.5, 1.5, 0.01, 'Brightness:')
    ui['bg_contrast'] = _slider(s['background']['contrast'], 0.5, 1.5, 0.01, 'Contrast:')
    ui['bg_texture_strength'] = _slider(s['background']['texture_strength'], 0, 0.25, 0.005, 'Texture:')
    ui['bg_vignette_strength'] = _slider(s['background']['vignette_strength'], 0, 0.8, 0.01, 'Vignette:')
    ui['line_width'] = _slider(s['style']['line_width'], 0.3, 5.0, 0.05, 'Line Width:')
    ui['glow_layers'] = _int_slider(s['style']['glow_layers'], 0, 20, 1, 'Glow Layers:')
    ui['glow_alpha'] = _slider(s['style']['glow_alpha'], 0.0, 0.2, 0.0025, 'Glow Alpha:')
    ui['glow_scale'] = _slider(s['style']['glow_scale'], 1.0, 6.0, 0.05, 'Glow Scale:')
    ui['main_alpha'] = _slider(s['style']['main_alpha'], 0.1, 1.0, 0.01, 'Main Alpha:')
    ui['segment_blending'] = widgets.Checkbox(value=s['style']['segment_blending'], description='Mystery Switch')
    return ui

def build_quick_create_row(ui: WidgetDict) -> Any:
    """Assemble the always-visible quick-create control row.

    Inputs:
        ui: Dictionary of notebook widgets keyed by stable control names.

Returns the widget container for the quick-create controls."""
    dropdown_row = widgets.HBox(
        [ui['artwork_preset'], ui['family'], ui['palette_preset'], ui['style_preset'], ui['background_preset']],
        layout=widgets.Layout(
            flex_flow='row wrap',
            align_items='center',
            justify_content='flex-start',
            width='auto',
            min_width='0',
            align_self='stretch'
        )
    )
    shared_curve_row = widgets.HBox(
        [ui['zoom'], ui['turns'], ui['noise_strength'], ui['asymmetry']],
        layout=widgets.Layout(
            flex_flow='row wrap',
            align_items='center',
            justify_content='center',
            width='auto',
            min_width='0',
            align_self='stretch'
        )
    )
    randomize_row = widgets.HBox(
        [ui['btn_randomize_all'], ui['btn_randomize_curve'], ui['btn_randomize_colors'], ui['btn_randomize_background']],
        layout=widgets.Layout(
            flex_flow='row wrap',
            align_items='center',
            justify_content='flex-start',
            width='auto',
            min_width='0',
            align_self='stretch'
        )
    )
    dropdown_wrap = widgets.Box(
        [dropdown_row],
        layout=widgets.Layout(width='auto', min_width='0', align_self='stretch', padding='0 0 8px 0', border_bottom='1px solid #ececec')
    )
    shared_curve_wrap = widgets.Box(
        [shared_curve_row],
        layout=widgets.Layout(width='auto', min_width='0', align_self='stretch', padding='8px 0', border_bottom='1px solid #ececec')
    )
    randomize_wrap = widgets.Box(
        [randomize_row],
        layout=widgets.Layout(width='auto', min_width='0', align_self='stretch', padding='8px 0 0 0')
    )
    return widgets.VBox(
        [dropdown_wrap, shared_curve_wrap, randomize_wrap],
        layout=widgets.Layout(
            width='auto',
            min_width='0',
            align_self='stretch',
            padding='10px 12px',
            border='1px solid #d9d9d9',
            margin='0 0 10px 0'
        )
    )

def build_curve_panel(ui: WidgetDict) -> Any:
    """Assemble the advanced curve-detail panel.

    Inputs:
        ui: Dictionary of notebook widgets keyed by stable control names.

Returns the widget container for the curve panel."""
    liss = widgets.VBox([ui['liss_a'], ui['liss_b'], ui['liss_delta']])
    rose = widgets.VBox([ui['rose_k'], ui['rose_radial_wobble'], ui['rose_wobble_freq']])
    epi = widgets.VBox([ui['epi_R'], ui['epi_r'], ui['epi_d']])
    burst = widgets.VBox([ui['burst_amp1'], ui['burst_amp2'], ui['burst_f1'], ui['burst_f2'], ui['burst_delta']])
    ui['_curve_family_boxes'] = {'lissajous': liss, 'rose': rose, 'epitrochoid': epi, 'burst': burst}
    return widgets.VBox([liss, rose, epi, burst])

def build_color_panel(ui: WidgetDict) -> Any:
    """Assemble the advanced color and palette panel.

    Inputs:
        ui: Dictionary of notebook widgets keyed by stable control names.

Returns the widget container for the color panel."""
    manual = widgets.VBox([ui['palette_manual_enabled'], ui['palette_size']] + [ui[f'palette_color_{i}'] for i in range(1, 7)])
    colormap = widgets.VBox([ui['colormap'], ui['colormap_span']])
    palette = widgets.VBox([ui['palette_name'], ui['palette_distribution'], manual])
    ui['_color_mode_boxes'] = {'palette': palette, 'colormap': colormap}
    return widgets.VBox([ui['color_mode'], ui['_color_mode_boxes']['palette'], ui['_color_mode_boxes']['colormap'], ui['curve_opacity'], ui['num_curve_colors'], ui['color_shift']])

def build_background_panel(ui: WidgetDict) -> Any:
    """Assemble the advanced background panel.

    Inputs:
        ui: Dictionary of notebook widgets keyed by stable control names.

Returns the widget container for the background panel."""
    solid = widgets.VBox([ui['bg_solid_color']])
    linear = widgets.VBox([ui['bg_linear_top'], ui['bg_linear_bottom'], ui['bg_linear_angle']])
    radial = widgets.VBox([ui['bg_radial_inner'], ui['bg_radial_outer'], ui['bg_radial_center_x'], ui['bg_radial_center_y'], ui['bg_radial_radius_bias']])
    mesh = widgets.VBox([ui['bg_mesh_tl'], ui['bg_mesh_tr'], ui['bg_mesh_bl'], ui['bg_mesh_br']])
    ui['_background_mode_boxes'] = {'solid': solid, 'linear': linear, 'radial': radial, 'mesh': mesh}
    return widgets.VBox([ui['background_mode'], solid, linear, radial, mesh, ui['bg_brightness'], ui['bg_contrast'], ui['bg_texture_strength'], ui['bg_vignette_strength']])

def build_style_panel(ui: WidgetDict) -> Any:
    """Assemble the advanced style panel.

    Inputs:
        ui: Dictionary of notebook widgets keyed by stable control names.

Returns the widget container for the style panel."""
    return widgets.VBox([ui['line_width'], ui['glow_layers'], ui['glow_alpha'], ui['glow_scale'], ui['main_alpha'], ui['segment_blending']])


def refresh_visibility(ui: WidgetDict, settings: SettingsDict) -> None:
    """Show and hide widgets so the visible controls match the active settings.

    Inputs:
        ui:       Dictionary of notebook widgets keyed by stable control names.
        settings: Application settings dictionary following the PART schema.

Returns None after updating widget visibility."""
    family = settings['curve']['family']
    for key, box in ui['_curve_family_boxes'].items():
        box.layout.display = '' if key == family else 'none'
    mode = settings['color']['mode']
    for key, box in ui['_color_mode_boxes'].items():
        box.layout.display = '' if key == mode else 'none'
    manual = settings['color'].get('manual_palette', False) and mode == 'palette'
    for i in range(1, 7):
        ui[f'palette_color_{i}'].layout.display = '' if manual and i <= settings['color']['palette_size'] else 'none'
    ui['palette_size'].layout.display = '' if manual and mode == 'palette' else 'none'
    bg_mode = settings['background']['mode']
    for key, box in ui['_background_mode_boxes'].items():
        box.layout.display = '' if key == bg_mode else 'none'


def read_all_widgets_into_settings(ui: WidgetDict, settings: SettingsDict) -> SettingsDict:
    """Read the current widget values into a copied settings dictionary.

    Inputs:
        ui:       Dictionary of notebook widgets keyed by stable control names.
        settings: Application settings dictionary following the PART schema.

Returns a validated settings dictionary reflecting the current widget state."""
    settings = deep_copy_settings(settings)
    settings['curve']['family'] = ui['family'].value
    settings['meta']['artwork_preset'] = ui['artwork_preset'].value
    settings['meta']['style_preset'] = ui['style_preset'].value
    settings['meta']['palette_preset'] = ui['palette_preset'].value
    settings['meta']['background_preset'] = ui['background_preset'].value
    settings['meta']['seed'] = int(ui['seed'].value)
    settings['style']['glow_macro'] = float(ui['glow_macro'].value)
    settings['curve']['turns'] = float(ui['turns'].value)
    settings['curve']['asymmetry'] = float(ui['asymmetry'].value)
    settings['curve']['noise_strength'] = float(ui['noise_strength'].value)
    settings['curve']['zoom'] = float(ui['zoom'].value)
    fam = settings['curve']['family']
    if fam == 'lissajous':
        settings['curve']['params'] = {'a': float(ui['liss_a'].value), 'b': float(ui['liss_b'].value), 'delta': float(ui['liss_delta'].value)}
    elif fam == 'rose':
        settings['curve']['params'] = {'k': float(ui['rose_k'].value), 'radial_wobble': float(ui['rose_radial_wobble'].value), 'wobble_freq': float(ui['rose_wobble_freq'].value)}
    elif fam == 'epitrochoid':
        settings['curve']['params'] = {'R': float(ui['epi_R'].value), 'r': float(ui['epi_r'].value), 'd': float(ui['epi_d'].value)}
    elif fam == 'burst':
        settings['curve']['params'] = {'amp1': float(ui['burst_amp1'].value), 'amp2': float(ui['burst_amp2'].value), 'f1': float(ui['burst_f1'].value), 'f2': float(ui['burst_f2'].value), 'delta': float(ui['burst_delta'].value)}
    settings['color']['mode'] = ui['color_mode'].value
    settings['color']['palette_name'] = ui['palette_name'].value
    settings['color']['manual_palette'] = bool(ui['palette_manual_enabled'].value)
    settings['color']['palette_size'] = int(ui['palette_size'].value)
    settings['color']['palette_distribution'] = ui['palette_distribution'].value
    settings['color']['palette_colors'] = [ui[f'palette_color_{i}'].value for i in range(1, settings['color']['palette_size'] + 1)]
    settings['color']['colormap'] = ui['colormap'].value
    settings['color']['colormap_span'] = list(ui['colormap_span'].value)
    settings['color']['curve_opacity'] = float(ui['curve_opacity'].value)
    settings['color']['num_curve_colors'] = int(ui['num_curve_colors'].value)
    settings['color']['color_shift'] = float(ui['color_shift'].value)
    settings['background']['mode'] = ui['background_mode'].value
    settings['background']['solid_color'] = ui['bg_solid_color'].value
    settings['background']['linear']['top'] = ui['bg_linear_top'].value
    settings['background']['linear']['bottom'] = ui['bg_linear_bottom'].value
    settings['background']['linear']['angle'] = float(ui['bg_linear_angle'].value)
    settings['background']['radial']['inner'] = ui['bg_radial_inner'].value
    settings['background']['radial']['outer'] = ui['bg_radial_outer'].value
    settings['background']['radial']['center_x'] = float(ui['bg_radial_center_x'].value)
    settings['background']['radial']['center_y'] = float(ui['bg_radial_center_y'].value)
    settings['background']['radial']['radius_bias'] = float(ui['bg_radial_radius_bias'].value)
    settings['background']['mesh']['tl'] = ui['bg_mesh_tl'].value
    settings['background']['mesh']['tr'] = ui['bg_mesh_tr'].value
    settings['background']['mesh']['bl'] = ui['bg_mesh_bl'].value
    settings['background']['mesh']['br'] = ui['bg_mesh_br'].value
    settings['background']['brightness'] = float(ui['bg_brightness'].value)
    settings['background']['contrast'] = float(ui['bg_contrast'].value)
    settings['background']['texture_strength'] = float(ui['bg_texture_strength'].value)
    settings['background']['vignette_strength'] = float(ui['bg_vignette_strength'].value)
    settings['style']['line_width'] = float(ui['line_width'].value)
    settings['style']['glow_layers'] = int(ui['glow_layers'].value)
    settings['style']['glow_alpha'] = float(ui['glow_alpha'].value)
    settings['style']['glow_scale'] = float(ui['glow_scale'].value)
    settings['style']['main_alpha'] = float(ui['main_alpha'].value)
    settings['style']['segment_blending'] = bool(ui['segment_blending'].value)
    return validate_settings(settings)

def write_all_settings_to_widgets(settings: SettingsDict, ui: WidgetDict) -> None:
    """Write the current settings dictionary back into every bound widget.

    Inputs:
        settings: Application settings dictionary following the PART schema.
        ui:       Dictionary of notebook widgets keyed by stable control names.

Returns None after updating widget values."""
    settings = validate_settings(settings)
    ui['family'].value = settings['curve']['family']
    ui['artwork_preset'].value = settings['meta']['artwork_preset']
    ui['style_preset'].value = settings['meta']['style_preset']
    ui['palette_preset'].value = settings['meta']['palette_preset']
    ui['background_preset'].value = settings['meta']['background_preset']
    ui['seed'].value = int(settings['meta']['seed'])
    ui['glow_macro'].value = float(settings['style']['glow_macro'])
    ui['turns'].value = float(settings['curve']['turns'])
    ui['asymmetry'].value = float(settings['curve']['asymmetry'])
    ui['noise_strength'].value = float(settings['curve']['noise_strength'])
    ui['zoom'].value = float(settings['curve']['zoom'])
    fam = settings['curve']['family']
    params = settings['curve']['params']
    if fam == 'lissajous':
        ui['liss_a'].value = float(params['a'])
        ui['liss_b'].value = float(params['b'])
        ui['liss_delta'].value = float(params['delta'])
    elif fam == 'rose':
        ui['rose_k'].value = float(params['k'])
        ui['rose_radial_wobble'].value = float(params['radial_wobble'])
        ui['rose_wobble_freq'].value = float(params['wobble_freq'])
    elif fam == 'epitrochoid':
        ui['epi_R'].value = float(params['R'])
        ui['epi_r'].value = float(params['r'])
        ui['epi_d'].value = float(params['d'])
    elif fam == 'burst':
        ui['burst_amp1'].value = float(params['amp1'])
        ui['burst_amp2'].value = float(params['amp2'])
        ui['burst_f1'].value = float(params['f1'])
        ui['burst_f2'].value = float(params['f2'])
        ui['burst_delta'].value = float(params['delta'])
    ui['color_mode'].value = settings['color']['mode']
    if settings['color']['palette_name'] in [opt[1] for opt in ui['palette_name'].options]:
        ui['palette_name'].value = settings['color']['palette_name']
    ui['palette_manual_enabled'].value = bool(settings['color']['manual_palette'])
    ui['palette_size'].value = int(settings['color']['palette_size'])
    ui['palette_distribution'].value = settings['color'].get('palette_distribution', 'even_segments')
    for i in range(1, 7):
        color_list = settings['color']['palette_colors']
        value = color_list[min(i - 1, len(color_list) - 1)] if color_list else '#ffffff'
        ui[f'palette_color_{i}'].value = value
    ui['colormap'].value = settings['color']['colormap']
    ui['colormap_span'].value = tuple(settings['color']['colormap_span'])
    ui['curve_opacity'].value = float(settings['color']['curve_opacity'])
    ui['num_curve_colors'].value = int(settings['color']['num_curve_colors'])
    ui['color_shift'].value = float(settings['color'].get('color_shift', 0.0))
    ui['background_mode'].value = settings['background']['mode']
    ui['bg_solid_color'].value = settings['background']['solid_color']
    ui['bg_linear_top'].value = settings['background']['linear']['top']
    ui['bg_linear_bottom'].value = settings['background']['linear']['bottom']
    ui['bg_linear_angle'].value = float(settings['background']['linear']['angle'])
    ui['bg_radial_inner'].value = settings['background']['radial']['inner']
    ui['bg_radial_outer'].value = settings['background']['radial']['outer']
    ui['bg_radial_center_x'].value = float(settings['background']['radial']['center_x'])
    ui['bg_radial_center_y'].value = float(settings['background']['radial']['center_y'])
    ui['bg_radial_radius_bias'].value = float(settings['background']['radial']['radius_bias'])
    ui['bg_mesh_tl'].value = settings['background']['mesh']['tl']
    ui['bg_mesh_tr'].value = settings['background']['mesh']['tr']
    ui['bg_mesh_bl'].value = settings['background']['mesh']['bl']
    ui['bg_mesh_br'].value = settings['background']['mesh']['br']
    ui['bg_brightness'].value = float(settings['background']['brightness'])
    ui['bg_contrast'].value = float(settings['background']['contrast'])
    ui['bg_texture_strength'].value = float(settings['background']['texture_strength'])
    ui['bg_vignette_strength'].value = float(settings['background']['vignette_strength'])
    ui['line_width'].value = float(settings['style']['line_width'])
    ui['glow_layers'].value = int(settings['style']['glow_layers'])
    ui['glow_alpha'].value = float(settings['style']['glow_alpha'])
    ui['glow_scale'].value = float(settings['style']['glow_scale'])
    ui['main_alpha'].value = float(settings['style']['main_alpha'])
    ui['segment_blending'].value = bool(settings['style']['segment_blending'])
STATE = {'settings': None, 'ui': None, 'preview_output': None, 'status_html': None, 'is_updating_widgets': False}

def set_status(message: str) -> None:
    """Update the bottom status message when the status widget is present.

    Inputs:
        message: Status text to display in the footer message area.

Returns None after updating the visible status widget when available."""
    status_html = STATE.get('status_html')
    if status_html is None:
        return None
    safe = str(message)
    status_html.value = (
        "<div style='box-sizing:border-box;padding:8px 12px;color:#444;"
        "border-top:1px solid #e6e6e6;background:#fafafa;overflow-wrap:anywhere'>"
        f"{safe}</div>"
    )
    return None

def rerender_preview() -> None:
    """Render the current artwork preview into the notebook output area.

    Inputs:
        None: This function does not accept positional inputs.

Returns None after refreshing the preview."""
    settings = validate_settings(STATE['settings'])
    STATE['settings'] = settings
    refresh_visibility(STATE['ui'], settings)
    with STATE['preview_output']:
        clear_output(wait=True)
        fig, ax = render_artwork(settings, figsize=FIGSIZE)
        display(fig)
        plt.close(fig)

def _apply_settings_and_sync(settings: SettingsDict, message: str='') -> None:
    """Replace the active settings state, synchronize widgets, and rerender the preview.

    Inputs:
        settings: Application settings dictionary following the PART schema.
        message:  Status message to surface in the notebook UI.

Returns None after syncing state and UI."""
    STATE['settings'] = validate_settings(settings)
    STATE['is_updating_widgets'] = True
    try:
        write_all_settings_to_widgets(STATE['settings'], STATE['ui'])
    finally:
        STATE['is_updating_widgets'] = False
    if message:
        set_status(message)
    rerender_preview()

def on_any_value_change(change: Any=None) -> None:
    """Handle generic control changes by reading widget state, applying macro writes, and clearing only affected preset labels.

    Inputs:
        change: Widget observer payload for the triggering change event.

Returns None after processing the change event."""
    if STATE['is_updating_widgets']:
        return
    changed_key = get_widget_key_from_change(STATE['ui'], change)
    settings = read_all_widgets_into_settings(STATE['ui'], STATE['settings'])
    if changed_key == 'glow_macro':
        settings['style']['glow_macro'] = float(STATE['ui']['glow_macro'].value)
    settings = apply_macro_change_if_needed(settings, changed_key)
    settings = mark_presets_custom_if_needed(settings, [changed_key] if changed_key else [])
    _apply_settings_and_sync(settings)

def on_family_changed(change: Any) -> None:
    """Handle family dropdown changes by resetting family-specific parameters and resynchronizing the UI.

    Inputs:
        change: Widget observer payload for the triggering change event.

Returns None after processing the change event."""
    if STATE['is_updating_widgets']:
        return
    settings = read_all_widgets_into_settings(STATE['ui'], STATE['settings'])
    settings['curve']['params'] = make_default_curve_params(settings['curve']['family'])
    settings = mark_presets_custom_if_needed(settings, ['family'])
    _apply_settings_and_sync(settings, f"Family set to {settings['curve']['family']}.")

def on_artwork_preset_changed(change: Any) -> None:
    """Handle artwork preset changes by applying the selected preset and syncing the UI.

    Inputs:
        change: Widget observer payload for the triggering change event.

Returns None after processing the change event."""
    if STATE['is_updating_widgets']:
        return
    settings = apply_artwork_preset(STATE['settings'], STATE['ui']['artwork_preset'].value)
    _apply_settings_and_sync(settings, f"Applied artwork preset: {(STATE['ui']['artwork_preset'].label if hasattr(STATE['ui']['artwork_preset'], 'label') else settings['meta']['artwork_preset'])}")

def on_style_preset_changed(change: Any) -> None:
    """Handle style preset changes by applying the selected preset and syncing the UI.

    Inputs:
        change: Widget observer payload for the triggering change event.

Returns None after processing the change event."""
    if STATE['is_updating_widgets']:
        return
    settings = read_all_widgets_into_settings(STATE['ui'], STATE['settings'])
    settings = apply_style_preset(settings, STATE['ui']['style_preset'].value)
    _apply_settings_and_sync(settings, f"Applied style preset: {settings['meta']['style_preset']}")

def on_palette_preset_changed(change: Any) -> None:
    """Handle palette preset changes by applying the selected preset and syncing the UI.

    Inputs:
        change: Widget observer payload for the triggering change event.

Returns None after processing the change event."""
    if STATE['is_updating_widgets']:
        return
    settings = read_all_widgets_into_settings(STATE['ui'], STATE['settings'])
    settings = apply_palette_preset(settings, STATE['ui']['palette_preset'].value)
    _apply_settings_and_sync(settings, f"Applied palette preset: {settings['meta']['palette_preset']}")

def on_background_preset_changed(change: Any) -> None:
    """Handle background preset changes by applying the selected preset and syncing the UI.

    Inputs:
        change: Widget observer payload for the triggering change event.

Returns None after processing the change event."""
    if STATE['is_updating_widgets']:
        return
    settings = read_all_widgets_into_settings(STATE['ui'], STATE['settings'])
    settings = apply_background_preset(settings, STATE['ui']['background_preset'].value)
    _apply_settings_and_sync(settings, f"Applied background preset: {settings['meta']['background_preset']}")

def on_randomize_all_clicked(button: Any) -> None:
    """Handle the Randomize All button by randomizing every unlocked settings group.

    Inputs:
        button: Button widget that triggered the callback.

Returns None after processing the button click."""
    settings = read_all_widgets_into_settings(STATE['ui'], STATE['settings'])
    settings = randomize_settings(settings, scope='all', intensity=DEFAULT_RANDOMIZATION_INTENSITY)
    _apply_settings_and_sync(settings, 'Randomized full composition.')

def on_randomize_curve_clicked(button: Any) -> None:
    """Handle the Randomize Curve button by randomizing curve-related settings.

    Inputs:
        button: Button widget that triggered the callback.

Returns None after processing the button click."""
    settings = read_all_widgets_into_settings(STATE['ui'], STATE['settings'])
    settings = randomize_curve(settings, intensity=DEFAULT_RANDOMIZATION_INTENSITY, keep_family=False)
    settings['meta']['artwork_preset'] = 'custom'
    _apply_settings_and_sync(settings, 'Randomized curve settings.')

def on_randomize_colors_clicked(button: Any) -> None:
    """Handle the Randomize Colors button by randomizing palette-related settings.

    Inputs:
        button: Button widget that triggered the callback.

Returns None after processing the button click."""
    settings = read_all_widgets_into_settings(STATE['ui'], STATE['settings'])
    settings = randomize_palette(settings, intensity=DEFAULT_RANDOMIZATION_INTENSITY)
    _apply_settings_and_sync(settings, 'Randomized colors.')

def on_randomize_background_clicked(button: Any) -> None:
    """Handle the Randomize Background button by randomizing background settings.

    Inputs:
        button: Button widget that triggered the callback.

Returns None after processing the button click."""
    settings = read_all_widgets_into_settings(STATE['ui'], STATE['settings'])
    settings = randomize_background(settings, intensity=DEFAULT_RANDOMIZATION_INTENSITY)
    _apply_settings_and_sync(settings, 'Randomized background.')

def on_save_png_clicked(button: Any) -> None:
    """Handle the Save PNG button by exporting the current artwork and surfacing the saved path.

    Inputs:
        button: Button widget that triggered the callback.

Returns None after processing the button click."""
    path = save_current_png(STATE['settings'])
    set_status(f'Saved PNG to: {path}')

def on_save_svg_clicked(button: Any) -> None:
    """Handle the Save SVG button by exporting the current artwork and surfacing the saved path.

    Inputs:
        button: Button widget that triggered the callback.

Returns None after processing the button click."""
    path = save_current_svg(STATE['settings'])
    set_status(f'Saved SVG to: {path}')

def connect_widget_events(ui: WidgetDict) -> None:
    """Bind all widget observers and button callbacks to their event handlers.

    Inputs:
        ui: Dictionary of notebook widgets keyed by stable control names.

Returns None after wiring the widget events."""
    ui['family'].observe(on_family_changed, names='value')
    ui['artwork_preset'].observe(on_artwork_preset_changed, names='value')
    ui['style_preset'].observe(on_style_preset_changed, names='value')
    ui['palette_preset'].observe(on_palette_preset_changed, names='value')
    ui['background_preset'].observe(on_background_preset_changed, names='value')
    watch_keys = ['seed', 'glow_macro', 'turns', 'asymmetry', 'noise_strength', 'zoom', 'liss_a', 'liss_b', 'liss_delta', 'rose_k', 'rose_radial_wobble', 'rose_wobble_freq', 'epi_R', 'epi_r', 'epi_d', 'burst_amp1', 'burst_amp2', 'burst_f1', 'burst_f2', 'burst_delta', 'color_mode', 'palette_name', 'palette_manual_enabled', 'palette_size', 'palette_distribution', 'colormap', 'colormap_span', 'curve_opacity', 'num_curve_colors', 'color_shift', 'background_mode', 'bg_solid_color', 'bg_linear_top', 'bg_linear_bottom', 'bg_linear_angle', 'bg_radial_inner', 'bg_radial_outer', 'bg_radial_center_x', 'bg_radial_center_y', 'bg_radial_radius_bias', 'bg_mesh_tl', 'bg_mesh_tr', 'bg_mesh_bl', 'bg_mesh_br', 'bg_brightness', 'bg_contrast', 'bg_texture_strength', 'bg_vignette_strength', 'line_width', 'glow_layers', 'glow_alpha', 'glow_scale', 'main_alpha', 'segment_blending'] + [f'palette_color_{i}' for i in range(1, 7)]
    for key in watch_keys:
        ui[key].observe(on_any_value_change, names='value')
    ui['btn_randomize_all'].on_click(on_randomize_all_clicked)
    ui['btn_randomize_curve'].on_click(on_randomize_curve_clicked)
    ui['btn_randomize_colors'].on_click(on_randomize_colors_clicked)
    ui['btn_randomize_background'].on_click(on_randomize_background_clicked)
    ui['btn_save_png'].on_click(on_save_png_clicked)
    ui['btn_save_svg'].on_click(on_save_svg_clicked)

def build_and_display_app() -> None:
    """Create the full notebook application, display it, and render the initial preview.

    Inputs:
        None: This function does not accept positional inputs.

Returns None after building and displaying the app."""
    STATE['settings'] = validate_settings(make_default_settings())
    STATE['ui'] = build_ui_widgets()
    STATE['preview_output'] = widgets.Output(layout=widgets.Layout(width='100%'))
    STATE['status_html'] = widgets.HTML(
        value='',
        layout=widgets.Layout(width='auto', min_width='0', align_self='stretch', margin='10px 0 0 0')
    )
    curve_panel = build_curve_panel(STATE['ui'])
    color_panel = build_color_panel(STATE['ui'])
    background_panel = build_background_panel(STATE['ui'])
    style_panel = build_style_panel(STATE['ui'])
    accordion = widgets.Accordion(children=[curve_panel, color_panel, background_panel, style_panel], layout=widgets.Layout(width='auto', align_self='stretch'))
    accordion.set_title(0, 'Curve Details')
    accordion.set_title(1, 'Color & Palette')
    accordion.set_title(2, 'Background')
    accordion.set_title(3, 'Render Style')
    STATE['is_updating_widgets'] = True
    try:
        write_all_settings_to_widgets(STATE['settings'], STATE['ui'])
    finally:
        STATE['is_updating_widgets'] = False
    connect_widget_events(STATE['ui'])
    refresh_visibility(STATE['ui'], STATE['settings'])

    header = widgets.HTML(
        "<div style='text-align:center;margin:0 0 10px 0'><h2 style='margin:0'>Parametric Art Studio</h2></div>",
        layout=widgets.Layout(width='auto', min_width='0', align_self='stretch')
    )
    quick_create = build_quick_create_row(STATE['ui'])

    preview_panel = widgets.Box(
        [STATE['preview_output']],
        layout=widgets.Layout(
            width='auto',
            flex='0 0 auto',
            align_items='flex-start',
            justify_content='flex-start',
            padding='0 12px 0 0'
        )
    )

    controls_column = widgets.VBox(
        [
            STATE['ui']['glow_macro'],
            STATE['ui']['seed'],
            widgets.HBox(
                [STATE['ui']['btn_save_png'], STATE['ui']['btn_save_svg']],
                layout=widgets.Layout(width='auto', justify_content='center', align_items='center', min_width='0', align_self='stretch')
            ),
            accordion,
        ],
        layout=widgets.Layout(
            width=CONTROL_PANEL_WIDTH,
            min_width=CONTROL_PANEL_WIDTH,
            max_width=CONTROL_PANEL_WIDTH,
            padding='0',
            align_items='stretch'
        )
    )

    controls_panel = widgets.Box(
        [controls_column],
        layout=widgets.Layout(
            width='auto',
            flex='1 1 auto',
            min_width='0',
            padding='0 0 0 12px',
            justify_content='flex-start',
            align_items='flex-start',
            overflow_x='hidden'
        )
    )

    body = widgets.HBox(
        [preview_panel, controls_panel],
        layout=widgets.Layout(width='auto', min_width='0', align_self='stretch', align_items='flex-start', justify_content='flex-start', overflow_x='hidden')
    )

    layout = widgets.VBox(
        [header, quick_create, body, STATE['status_html']],
        layout=widgets.Layout(width='auto', min_width='0', align_self='stretch', overflow_x='hidden')
    )
    display(layout)
    rerender_preview()
    set_status('Ready.')


def run():
    build_and_display_app()