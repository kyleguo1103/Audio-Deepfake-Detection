"""Pre-declared waveform augmentation families: codec, noise, channel.

Every random choice comes from an RNG seeded by (global_seed, sample_id, family),
so the same clip always receives the same augmentation and the parameters are
logged next to the embedding. Real and fake clips follow identical rules.
"""

from __future__ import annotations

import subprocess
from typing import Any

import numpy as np
from scipy.signal import butter, fftconvolve, resample_poly, sosfilt, lfilter

from pilot_common import stable_seed

SR = 16000
FAMILIES = ("codec", "noise", "channel")


def _match_rms(new: np.ndarray, reference: np.ndarray) -> np.ndarray:
    ref = float(np.sqrt(np.mean(reference.astype(np.float64) ** 2)) + 1e-12)
    cur = float(np.sqrt(np.mean(new.astype(np.float64) ** 2)) + 1e-12)
    out = new * (ref / cur)
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 0.999:
        out = out * (0.999 / peak)
    return out.astype(np.float32)


# ------------------------------------------------------------------ codec

def run_codec(signal: np.ndarray, ffmpeg: str, encoder: str, fmt: str, kbps: int) -> np.ndarray:
    pcm = signal.astype(np.float32).tobytes()
    encode = subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "pipe:0",
         "-c:a", encoder, "-b:a", f"{kbps}k", "-f", fmt, "pipe:1"],
        input=pcm, capture_output=True, check=True,
    )
    decode = subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", "pipe:0", "-f", "f32le", "-ar", str(SR), "-ac", "1", "pipe:1"],
        input=encode.stdout, capture_output=True, check=True,
    )
    out = np.frombuffer(decode.stdout, dtype=np.float32).copy()
    if out.size == 0:
        raise RuntimeError(f"codec {encoder}@{kbps}k returned no audio")
    out = out[_codec_delay(signal, out):]  # remove encoder priming delay so the copy stays time-aligned
    return out[: len(signal)] if len(out) >= len(signal) else np.pad(out, (0, len(signal) - len(out)))


def _codec_delay(reference: np.ndarray, decoded: np.ndarray, max_lag: int = 4096) -> int:
    """Lag (0..max_lag samples) maximising cross-correlation between decoded and reference audio."""
    n = min(len(reference), len(decoded) - max_lag, 4 * SR)
    if n <= 256:
        return 0
    ref = reference[:n].astype(np.float64)
    seg = decoded[: n + max_lag].astype(np.float64)
    size = 1 << int(np.ceil(np.log2(len(seg) + n)))
    corr = np.fft.irfft(np.fft.rfft(seg, size) * np.conj(np.fft.rfft(ref, size)), size)[: max_lag + 1]
    return int(np.argmax(corr))


# ------------------------------------------------------------------ noise

def coloured_noise(n: int, kind: str, rng: np.random.Generator) -> np.ndarray:
    white = rng.standard_normal(n)
    if kind == "white":
        return white
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, d=1.0 / SR)
    freqs[0] = freqs[1] if len(freqs) > 1 else 1.0
    exponent = 0.5 if kind == "pink" else 1.0  # amplitude ~ f^-0.5 (pink) or f^-1 (brown)
    shaped = np.fft.irfft(spectrum / freqs ** exponent, n=n)
    return shaped / (shaped.std() + 1e-12)


def add_noise(signal: np.ndarray, kind: str, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    noise = coloured_noise(len(signal), kind, rng)
    p_signal = float(np.mean(signal.astype(np.float64) ** 2)) + 1e-12
    p_noise = float(np.mean(noise ** 2)) + 1e-12
    scale = np.sqrt(p_signal / (p_noise * 10 ** (snr_db / 10.0)))
    return (signal + scale * noise).astype(np.float32)


# ------------------------------------------------------------------ channel

def peaking_biquad(f0: float, gain_db: float, q: float) -> tuple[np.ndarray, np.ndarray]:
    a_gain = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * f0 / SR
    alpha = np.sin(w0) / (2 * q)
    b = np.array([1 + alpha * a_gain, -2 * np.cos(w0), 1 - alpha * a_gain])
    a = np.array([1 + alpha / a_gain, -2 * np.cos(w0), 1 - alpha / a_gain])
    return b / a[0], a / a[0]


def channel_effect(signal: np.ndarray, subtype: str, cfg: dict[str, Any], rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
    params: dict[str, Any] = {"subtype": subtype}
    x = signal.astype(np.float64)
    if subtype == "telephone_300_3400_8k":
        sos = butter(4, [300, 3400], btype="bandpass", fs=SR, output="sos")
        y = sosfilt(sos, x)
        y = resample_poly(resample_poly(y, 1, 2), 2, 1)[: len(x)]
    elif subtype == "lowpass":
        lo, hi = cfg["lowpass_cutoff_hz_range"]
        cutoff = float(rng.uniform(lo, hi))
        params["cutoff_hz"] = round(cutoff, 1)
        y = sosfilt(butter(6, cutoff, btype="lowpass", fs=SR, output="sos"), x)
    elif subtype == "parametric_eq":
        y = x
        bands = []
        lo_f, hi_f = cfg["eq_center_hz_range"]
        for _ in range(int(cfg["eq_bands"])):
            f0 = float(np.exp(rng.uniform(np.log(lo_f), np.log(hi_f))))
            gain = float(rng.uniform(*cfg["eq_gain_db_range"]))
            q = float(rng.uniform(*cfg["eq_q_range"]))
            b, a = peaking_biquad(f0, gain, q)
            y = lfilter(b, a, y)
            bands.append({"f0": round(f0, 1), "gain_db": round(gain, 2), "q": round(q, 2)})
        params["bands"] = bands
    elif subtype == "synthetic_reverb":
        rt60 = float(rng.uniform(*cfg["reverb_rt60_s_range"]))
        wet = float(rng.uniform(*cfg["reverb_wet_range"]))
        length = max(1, int(rt60 * SR))
        t = np.arange(length) / SR
        rir = rng.standard_normal(length) * np.exp(-6.9078 * t / rt60)
        rir[0] = 1.0
        rir /= np.sqrt(np.sum(rir ** 2))
        reverberant = fftconvolve(x, rir)[: len(x)]
        y = (1 - wet) * x + wet * reverberant
        params.update({"rt60_s": round(rt60, 3), "wet": round(wet, 3)})
    else:
        raise ValueError(f"unknown channel subtype {subtype}")
    return np.asarray(y, dtype=np.float32), params


# ------------------------------------------------------------------ dispatcher

def augment(signal: np.ndarray, family: str, sample_id: str, cfg: dict[str, Any], ffmpeg: str,
            available_encoders: set[str]) -> tuple[np.ndarray, dict[str, Any]]:
    aug_cfg = cfg["augmentation"]
    rng = np.random.default_rng(stable_seed(aug_cfg["global_seed"], sample_id, family))
    fam_cfg = aug_cfg["families"][family]
    params: dict[str, Any] = {"family": family}
    if family == "codec":
        options = [o for o in fam_cfg["options"] if o["encoder"] in available_encoders]
        if not options:
            raise RuntimeError("no configured codec encoder is available in ffmpeg")
        option = options[int(rng.integers(len(options)))]
        kbps = int(option["bitrates_kbps"][int(rng.integers(len(option["bitrates_kbps"])))])
        out = run_codec(signal, ffmpeg, option["encoder"], option["format"], kbps)
        params.update({"encoder": option["encoder"], "kbps": kbps})
    elif family == "noise":
        kind = fam_cfg["types"][int(rng.integers(len(fam_cfg["types"])))]
        snr = float(rng.uniform(*fam_cfg["snr_db_range"]))
        out = add_noise(signal, kind, snr, rng)
        params.update({"noise": kind, "snr_db": round(snr, 2)})
    elif family == "channel":
        subtype = fam_cfg["subtypes"][int(rng.integers(len(fam_cfg["subtypes"])))]
        out, extra = channel_effect(signal, subtype, fam_cfg, rng)
        params.update(extra)
    else:
        raise ValueError(f"unknown family {family}")
    return _match_rms(out[: len(signal)], signal), params


def mix_family(sample_id: str, cfg: dict[str, Any]) -> str:
    """Family used for this clip in the size-matched 'mix' condition (fixed before any result)."""
    rng = np.random.default_rng(stable_seed(cfg["augmentation"]["global_seed"], sample_id, "mix"))
    return FAMILIES[int(rng.integers(len(FAMILIES)))]
