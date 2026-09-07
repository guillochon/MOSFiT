"""Decode SESN SEDONA velocity and composition profiles from latent params.

The ``sesn_sedona`` model stores latent coefficients (``eta_vel``, ``eta_he``,
…) plus mass / velocity scales. These Dash + Riem autoencoders map those
latents back to radial profiles for posterior visualization.

Weight files live next to the SESN SEDONA SED emulator weights (or under
``$MOSFIT_EMULATOR_DATA/sesn_sedona``):

* ``velocity_profile.pt`` — velocity (Dash)
* ``helium_profile.pt`` — He
* ``nickel_profile.pt`` — Ni
* ``opacity_profile.pt`` — bulk opacity
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError as exc:
    raise ImportError(
        'The SESN SEDONA emulator requires PyTorch. Install the optional '
        '`sedona` extra with `uv sync --extra sedona` or '
        "`pip install 'mosfit[sedona]'`."
    ) from exc


from mosfit.emulators import emulator_weights_dir

_WEIGHTS = emulator_weights_dir("sesn_sedona")

_REQUIRED = (
    "velocity_profile.pt",
    "helium_profile.pt",
    "nickel_profile.pt",
    "opacity_profile.pt",
)


class DashAutoencoder(nn.Module):
    """Velocity-profile autoencoder."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 16)
        self.fc3 = nn.Linear(16, 1)
        self.fc4 = nn.Linear(1, 16)
        self.fc5 = nn.Linear(16, 64)
        self.fc6 = nn.Linear(64, 128)
        self.fc7 = nn.Linear(128, 512)
        self.fc8 = nn.Linear(512, 256)
        self.double()

    def decoder(self, x):
        for fc in (self.fc4, self.fc5, self.fc6, self.fc7, self.fc8):
            x = torch.sigmoid(fc(x))
        return x


class Autoencoder_2_4(nn.Module):
    """Helium (2.4) mass-profile autoencoder."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 1)
        self.fc5 = nn.Linear(1, 128)
        self.fc6 = nn.Linear(128, 128)
        self.fc7 = nn.Linear(128, 512)
        self.fc8 = nn.Linear(512, 256)
        self.double()

    def decoder(self, x):
        for fc in (self.fc5, self.fc6, self.fc7, self.fc8):
            x = torch.sigmoid(fc(x))
        return x


class Autoencoder_28_56(nn.Module):
    """Nickel (28.56) mass-profile autoencoder."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 1)
        self.fc6 = nn.Linear(1, 128)
        self.fc7 = nn.Linear(128, 128)
        self.fc8 = nn.Linear(128, 128)
        self.fc9 = nn.Linear(128, 512)
        self.fc10 = nn.Linear(512, 256)
        self.double()

    def decoder(self, x):
        for fc in (self.fc6, self.fc7, self.fc8, self.fc9, self.fc10):
            x = torch.sigmoid(fc(x))
        return x


class OpacityAutoencoder(nn.Module):
    """Bulk-opacity mass-profile autoencoder."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 16)
        self.fc3 = nn.Linear(16, 1)
        self.fc4 = nn.Linear(1, 16)
        self.fc5 = nn.Linear(16, 64)
        self.fc6 = nn.Linear(64, 128)
        self.fc7 = nn.Linear(128, 512)
        self.fc8 = nn.Linear(512, 256)
        self.double()

    def decoder(self, x):
        for fc in (self.fc4, self.fc5, self.fc6, self.fc7, self.fc8):
            x = torch.sigmoid(fc(x))
        return x


_ELEMENT_MODELS = {
    "2_4": (Autoencoder_2_4, "helium_profile.pt"),
    "28_56": (Autoencoder_28_56, "nickel_profile.pt"),
    "opacity": (OpacityAutoencoder, "opacity_profile.pt"),
}

# Cache loaded decoders so notebook loops do not re-read .pt files.
_MODEL_CACHE = {}


def weights_dir(path=None) -> Path:
    """Return the directory that should hold Dash / Riem ``.pt`` files."""
    if path is not None:
        return Path(path).expanduser().resolve()
    return Path(_WEIGHTS)


def missing_weight_files(path=None):
    """List required weight basenames that are not present."""
    root = weights_dir(path)
    return [name for name in _REQUIRED if not (root / name).is_file()]


def _load_model(file_loc, model):
    model.load_state_dict(
        torch.load(file_loc, map_location=torch.device("cpu"))
    )
    model.eval()
    return model


def _cached_model(key, path, cls):
    cache_key = (str(path), key)
    if cache_key not in _MODEL_CACHE:
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing SESN SEDONA profile weight '{path.name}' in {path.parent}. "
                f"Expected files: {', '.join(_REQUIRED)}"
            )
        _MODEL_CACHE[cache_key] = _load_model(str(path), cls())
    return _MODEL_CACHE[cache_key]


def remove_drops(dist):
    """Monotonize the velocity grid (linear-interp over any decreasing runs)."""
    dist = np.asarray(dist, dtype=float).copy()
    i = 0
    while i < len(dist) - 1:
        if dist[i + 1] < dist[i]:
            j = i + 1
            while j < len(dist) and dist[j] < dist[i]:
                j += 1
            if j < len(dist):
                step = j - i
                for k in range(1, step):
                    dist[i + k] = dist[i] + (dist[j] - dist[i]) * (k / step)
            i = j
        else:
            i += 1
    return dist


def get_profile(
    latent_var,
    model,
    total_mass=np.nan,
    max_val=np.nan,
    min_val=np.nan,
):
    """Decode a 1-D profile from a scalar latent.

    Pass either ``total_mass`` (composition / opacity, normalize to that sum)
    or both ``max_val`` and ``min_val`` (velocity: ``min + dist * max``).
    """
    dist = (
        model.decoder(
            torch.from_numpy(np.asarray([latent_var], dtype=float))
        )
        .detach()
        .numpy()
    )
    dist = np.asarray(dist, dtype=float).reshape(-1)
    neg = dist < 0
    if np.any(neg):
        pos = dist[dist > 0]
        fill = float(pos.min()) if len(pos) else 0.0
        dist = dist.copy()
        dist[neg] = fill
    if not np.isnan(total_mass) and (
        np.isnan(max_val) and np.isnan(min_val)
    ):
        dist = dist * (total_mass / np.sum(dist))
    elif np.isnan(total_mass) and (
        not np.isnan(max_val) and not np.isnan(min_val)
    ):
        dist = dist * max_val + min_val
    else:
        raise ValueError(
            "get_profile: pass either total_mass or (max_val and min_val)"
        )
    return dist


def get_vel_profile(D, max_vel, min_vel, weights_path=None):
    """Velocity profile from Dash latent ``D`` and ``del_vel`` / ``min_vel``."""
    root = weights_dir(weights_path)
    model = _cached_model("dash", root / "velocity_profile.pt", DashAutoencoder)
    return remove_drops(
        get_profile(
            latent_var=D, max_val=max_vel, min_val=min_vel, model=model
        )
    )


def get_element_profile(
    R,
    element,
    total_mass=np.nan,
    max_val=np.nan,
    min_val=np.nan,
    weights_path=None,
):
    """Composition / opacity profile from Riem latent ``R``.

    ``element`` is one of ``'2_4'`` (He), ``'28_56'`` (Ni), ``'opacity'``.
    """
    if element not in _ELEMENT_MODELS:
        raise KeyError(
            f"Unknown element '{element}'; expected one of "
            f"{sorted(_ELEMENT_MODELS)}"
        )
    cls, filename = _ELEMENT_MODELS[element]
    root = weights_dir(weights_path)
    model = _cached_model(element, root / filename, cls)
    return get_profile(
        latent_var=R,
        total_mass=total_mass,
        max_val=max_val,
        min_val=min_val,
        model=model,
    )
