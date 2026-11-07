"""Was this picture made by a camera or by a model?

Declared evidence first: Content Credentials (C2PA manifests, which Adobe,
OpenAI, Microsoft and others sign into generated files), the IPTC digital
source type, generator names in metadata, and the generation recipe that
Stable Diffusion front-ends write into PNG text chunks (prompt, seed, model).
Then weak statistical signals that are reported as exactly that: the spectral
fingerprint of upsampling networks, an unnaturally clean sensor-noise floor,
and generator-typical pixel sizes. Nothing here says "AI-generated" unless
the file itself does; the vision analyst's opinion is labelled AI-inferred.
"""

import asyncio
import json
import re
from typing import Any

import cv2
import numpy as np

from earwitness import photo
from earwitness.modules.base import CaseContext
from earwitness.modules.base import Reporter
from earwitness.modules.base import Skip

_GENERATOR_WORDS = [
    (re.compile(r"midjourney", re.I), "Midjourney"),
    (re.compile(r"dall[·\-\. ]?e", re.I), "DALL·E"),
    (re.compile(r"chatgpt|openai", re.I), "OpenAI"),
    (
        re.compile(r"stable[\s-]?diffusion|sd[\s-]?xl|sdxl|stability\.?ai", re.I),
        "Stable Diffusion",
    ),
    (
        re.compile(r"automatic1111|a1111|comfyui|invokeai|fooocus|forge", re.I),
        "a Stable Diffusion front-end",
    ),
    (re.compile(r"novelai", re.I), "NovelAI"),
    (re.compile(r"adobe firefly|firefly", re.I), "Adobe Firefly"),
    (
        re.compile(r"\bimagen\b|gemini|google ai|synthid", re.I),
        "Google Imagen / Gemini",
    ),
    (re.compile(r"flux\.?1|black forest labs", re.I), "FLUX"),
    (re.compile(r"leonardo\.ai", re.I), "Leonardo.Ai"),
    (re.compile(r"ideogram", re.I), "Ideogram"),
    (
        re.compile(r"bing image creator|microsoft designer|designer\.microsoft", re.I),
        "Microsoft Designer / Bing Image Creator",
    ),
    (re.compile(r"\bgrok\b|xai", re.I), "Grok"),
    (re.compile(r"meta ai|imagined with ai|emu", re.I), "Meta AI"),
    (
        re.compile(
            r"recraft|krea|playground\.?ai|nightcafe|dreamstudio|runway|pika", re.I
        ),
        "an AI image service",
    ),
    (
        re.compile(
            r"trainedAlgorithmicMedia|compositeWithTrainedAlgorithmicMedia", re.I
        ),
        "declared algorithmic media (IPTC)",
    ),
]
_SD_PARAMS = re.compile(
    r"Steps:\s*(\d+).*?Sampler:\s*([^,]+).*?CFG scale:\s*([\d.]+).*?Seed:\s*(\d+).*?Size:\s*(\d+x\d+)",
    re.S,
)
_SD_MODEL = re.compile(r"(?<!\w)Model:\s*([^,\n]+)")
_SD_MODEL_HASH = re.compile(r"Model hash:\s*([0-9a-fA-F]+)")
_SIZES = {
    256,
    320,
    384,
    448,
    512,
    576,
    640,
    704,
    768,
    832,
    896,
    960,
    1024,
    1088,
    1152,
    1216,
    1280,
    1344,
    1408,
    1472,
    1536,
    1792,
    2048,
}


def _c2pa_strings(blob: bytes) -> dict[str, Any]:
    """Readable facts from a JUMBF/C2PA box without a full CBOR parser."""
    text = blob.decode("latin-1", "replace")
    out: dict[str, Any] = {}
    generator = re.search(
        r"claim_generator[^A-Za-z0-9]{1,12}([A-Za-z0-9 ._/()\-]{3,80})", text
    )

    if generator:
        out["claim_generator"] = generator.group(1).strip()

    info = re.search(
        r"claim_generator_info.{0,40}?name[^A-Za-z0-9]{1,12}([A-Za-z0-9 ._/()\-]{3,80})",
        text,
        re.S,
    )

    if info:
        out["generator_name"] = info.group(1).strip()

    actions = sorted(
        set(
            re.findall(
                r"c2pa\.(created|edited|placed|resized|cropped|color_adjustments|filtered|opened|drawing|converted|produced|published|removed|repackaged|transcoded|unknown)",
                text,
            )
        )
    )

    if actions:
        out["actions"] = actions

    source = re.findall(r"digitalsourcetype/([A-Za-z]+)", text)

    if source:
        out["digital_source_type"] = sorted(set(source))

    agents = sorted(
        set(
            re.findall(
                r"softwareAgent[^A-Za-z0-9]{1,12}([A-Za-z0-9 ._/()\-]{3,60})", text
            )
        )
    )

    if agents:
        out["software_agents"] = agents[:5]

    issuer = re.search(r"(?:O=|CN=)([A-Za-z0-9 ,._\-]{3,60})", text)

    if issuer:
        out["certificate"] = issuer.group(1).strip()

    out["bytes"] = len(blob)

    return out


def _spectral_signals(gray: np.ndarray) -> dict[str, Any]:
    """Azimuthally averaged power spectrum: upsampling networks leave peaks near f = 1/2 and 1/4."""
    side = 512
    patch = cv2.resize(gray, (side, side), interpolation=cv2.INTER_AREA).astype(
        np.float64
    )
    patch -= patch.mean()
    window = np.hanning(side)
    spectrum = (
        np.abs(np.fft.fftshift(np.fft.fft2(patch * window[:, None] * window[None, :])))
        ** 2
    )
    centre = side // 2
    yy, xx = np.mgrid[:side, :side]
    radius = np.hypot(yy - centre, xx - centre).astype(int)
    profile = np.bincount(radius.ravel(), weights=spectrum.ravel()) / (
        np.bincount(radius.ravel()) + 1e-9
    )
    profile = profile[1:centre]
    log = np.log10(profile + 1e-12)
    freqs = np.arange(1, centre) / side
    # A natural image decays roughly as a power law; fit it on 0.05..0.3 and
    # look at how the top band (0.35..0.5) departs and whether it has peaks.
    fit_band = (freqs > 0.05) & (freqs < 0.3)
    slope, intercept = np.polyfit(np.log10(freqs[fit_band]), log[fit_band], 1)
    predicted = slope * np.log10(freqs) + intercept
    high = freqs > 0.35
    excess = float((log[high] - predicted[high]).mean())
    residual = log - predicted
    smooth = cv2.GaussianBlur(
        residual.reshape(-1, 1).astype(np.float32), (1, 0), 3
    ).ravel()
    peaks = residual - smooth
    peak_z = float(peaks[high].max() / (peaks[fit_band].std() + 1e-9))

    return {
        "slope": float(slope),
        "high_frequency_excess_db": excess * 10,
        "peak_z": peak_z,
        "profile_db": (10 * log).tolist(),
        "freqs": freqs.tolist(),
    }


def _noise_floor(gray: np.ndarray) -> float:
    """Robust standard deviation of the high-pass residual in flat areas."""
    g = gray.astype(np.float32)
    residual = g - cv2.GaussianBlur(g, (0, 0), 1.0)
    texture = cv2.GaussianBlur(
        np.abs(cv2.Laplacian(cv2.GaussianBlur(g, (0, 0), 2.0), cv2.CV_32F)), (0, 0), 4
    )
    flat = texture < np.percentile(texture, 30)

    if flat.sum() < 1000:
        return float(np.median(np.abs(residual)) * 1.4826)

    return float(np.median(np.abs(residual[flat])) * 1.4826)


class Synthetic:
    name = "synthetic"
    title = "AI GENERATION SIGNALS"
    description = "Content Credentials (C2PA), IPTC digital source type, generator names and Stable Diffusion recipes in metadata; spectral, noise-floor and size signals, honestly weighted."
    depends = ["provenance", "jpeg"]

    async def run(self, ctx: CaseContext, out: Reporter) -> None:
        image = ctx.image()
        info = ctx.image_info()

        if image is None or info is None:
            raise Skip("not a decodable image")

        provenance = await ctx.wait_for("provenance") or {}
        loaded = photo.load(ctx.path)

        with open(ctx.path, "rb") as handle:
            data = handle.read()

        jpeg = photo.jpeg_anatomy(data) if data[:2] == b"\xff\xd8" else None
        png = photo.png_anatomy(data) if data[:8] == b"\x89PNG\r\n\x1a\n" else None
        declared: list[str] = []
        signals: list[str] = []
        published: dict[str, Any] = {
            "declared": [],
            "signals": [],
            "recipe": None,
            "c2pa": None,
        }

        # ---- Content Credentials / C2PA ---------------------------------------------------------------
        out.progress(0.1, "looking for Content Credentials")
        c2pa_blobs: list[bytes] = []

        if jpeg:
            for segment in jpeg.segments:
                if "JUMBF" in (segment.get("app") or "") or "C2PA" in (
                    segment.get("app") or ""
                ):
                    c2pa_blobs.append(
                        data[segment["offset"] : segment["offset"] + segment["length"]]
                    )

        if png:
            for chunk in png.chunks:
                if chunk["type"] == "caBX":
                    c2pa_blobs.append(
                        data[
                            chunk["offset"] + 8 : chunk["offset"] + 8 + chunk["length"]
                        ]
                    )

        if not c2pa_blobs and b"c2pa" in data[:4_000_000]:
            position = data.find(b"c2pa")
            c2pa_blobs.append(data[max(0, position - 64) : position + 200_000])

        if c2pa_blobs:
            facts = _c2pa_strings(b"".join(c2pa_blobs))
            published["c2pa"] = facts
            generated = any(
                "trainedAlgorithmicMedia" in s
                for s in facts.get("digital_source_type", [])
            ) or "c2pa.created" in str(facts.get("actions"))
            maker = (
                facts.get("generator_name")
                or facts.get("claim_generator")
                or "unknown software"
            )
            out.artifact(
                "c2pa",
                "json",
                "Content Credentials (C2PA manifest, readable parts)",
                data=facts,
            )

            if generated or any(p.search(maker) for p, _ in _GENERATOR_WORDS):
                declared.append(f"Content Credentials from {maker}")
                out.finding(
                    "critical",
                    f"Content Credentials: made with {maker}",
                    "The file carries a signed C2PA manifest, the industry's provenance label. "
                    + (
                        f"It declares the source type {', '.join(facts['digital_source_type'])}: the picture was produced by a trained model. "
                        if facts.get("digital_source_type")
                        else ""
                    )
                    + (
                        f"Recorded actions: {', '.join(facts['actions'])}. "
                        if facts.get("actions")
                        else ""
                    )
                    + (
                        f"Signed under “{facts['certificate']}”. "
                        if facts.get("certificate")
                        else ""
                    )
                    + "This is the generator's own statement, which is the strongest evidence a file can carry.",
                    tags=["synthetic", "c2pa", "declared", "editor"],
                    data=facts,
                )
            else:
                out.finding(
                    "notice",
                    f"Content Credentials present (signed by {maker})",
                    "A C2PA manifest records the file's history"
                    + (
                        f": {', '.join(facts['actions'])}"
                        if facts.get("actions")
                        else ""
                    )
                    + ". It does not declare the image as generated.",
                    tags=["c2pa", "history"],
                    data=facts,
                )

        # ---- generator names and recipes in metadata ------------------------------------------------------
        out.progress(0.3, "generator names in metadata")
        texts: list[str] = []
        exif = photo.parse_exif(loaded.exif) if loaded else None

        if exif:
            texts += [
                f"{k}: {v}"
                for ifd in exif.ifds.values()
                for k, v in ifd.items()
                if isinstance(v, str)
            ]

        if loaded:
            texts.append(loaded.xmp or "")
            texts += [f"{k}: {v}" for k, v in loaded.text.items()]

        if jpeg:
            texts += jpeg.comments

        joined = "\n".join(texts)
        named: list[str] = []

        for pattern, label in _GENERATOR_WORDS:
            if pattern.search(joined) and label not in named:
                named.append(label)

        recipe_text = (loaded.text.get("parameters") if loaded else None) or (
            png.text.get("parameters") if png else None
        )
        recipe: dict[str, Any] | None = None

        if recipe_text:
            prompt, _, rest = recipe_text.partition("\nNegative prompt:")
            negative, _, settings = (
                rest.partition("\n") if rest else ("", "", recipe_text.split("\n")[-1])
            )

            if not rest:
                prompt, _, settings = recipe_text.rpartition("\n")

            found = _SD_PARAMS.search(settings) or _SD_PARAMS.search(recipe_text)
            recipe = {
                "prompt": prompt.strip()[:1500],
                "negative_prompt": negative.strip()[:800],
                "settings": settings.strip()[:800],
            }

            if found:
                recipe |= {
                    "steps": int(found.group(1)),
                    "sampler": found.group(2).strip(),
                    "cfg_scale": float(found.group(3)),
                    "seed": int(found.group(4)),
                    "size": found.group(5),
                    "model": (
                        _SD_MODEL.search(recipe_text).group(1).strip()
                        if _SD_MODEL.search(recipe_text)
                        else None
                    ),
                    "model_hash": (
                        _SD_MODEL_HASH.search(recipe_text).group(1)
                        if _SD_MODEL_HASH.search(recipe_text)
                        else None
                    ),
                }

            published["recipe"] = recipe
            declared.append("Stable Diffusion recipe")
            out.artifact(
                "recipe", "json", "Generation recipe stored in the file", data=recipe
            )
            out.finding(
                "critical",
                f"Generated by Stable Diffusion: prompt “{recipe['prompt'][:70]}{'…' if len(recipe['prompt']) > 70 else ''}”",
                "The PNG carries the generator's own recipe in a text chunk: the prompt, "
                + (
                    f"seed {recipe['seed']}, {recipe['steps']} steps with {recipe['sampler']}, CFG {recipe['cfg_scale']}"
                    + (f", model {recipe['model']}" if recipe.get("model") else "")
                    if found
                    else "sampler settings"
                )
                + ". With these the picture can be regenerated exactly; nothing a camera could write.",
                tags=["synthetic", "declared", "generator", "editor"],
                data=recipe,
            )

        comfy = (
            (loaded.text.get("workflow") or loaded.text.get("prompt"))
            if loaded
            else None
        )

        if comfy and not recipe_text:
            try:
                parsed = json.loads(comfy)
                nodes = parsed if isinstance(parsed, dict) else {}
                prompts = [
                    v.get("inputs", {}).get("text")
                    for v in (nodes.values() if isinstance(nodes, dict) else [])
                    if isinstance(v, dict)
                    and isinstance(v.get("inputs", {}).get("text"), str)
                ]
                models = [
                    v.get("inputs", {}).get("ckpt_name")
                    for v in (nodes.values() if isinstance(nodes, dict) else [])
                    if isinstance(v, dict) and v.get("inputs", {}).get("ckpt_name")
                ]
            except ValueError:
                prompts, models = [], []

            declared.append("ComfyUI workflow")
            out.artifact(
                "recipe",
                "json",
                "ComfyUI workflow stored in the file",
                data={
                    "prompts": prompts[:6],
                    "models": models[:4],
                    "characters": len(comfy),
                },
            )
            out.finding(
                "critical",
                "Generated with ComfyUI"
                + (f": “{prompts[0][:70]}”" if prompts else ""),
                f"A ComfyUI workflow of {len(comfy):,} characters is embedded in the PNG"
                + (f", naming the model {models[0]}" if models else "")
                + ". It is the full graph that produced the picture.",
                tags=["synthetic", "declared", "generator", "editor"],
                data={"prompts": prompts[:6], "models": models[:4]},
            )

        plain_named = [
            n
            for n in named
            if not n.startswith("declared algorithmic") and not (recipe or comfy)
        ]

        if plain_named:
            evidence = next(
                (
                    line
                    for line in joined.split("\n")
                    if re.search(re.escape(plain_named[0].split()[0]), line, re.I)
                ),
                "",
            )
            declared += plain_named
            out.finding(
                "critical",
                f"Generator named in metadata: {', '.join(plain_named[:3])}",
                f"The file's metadata names {' and '.join(plain_named[:3])}. Generators and their export tools write their name; a camera never would. Evidence: “{evidence[:140]}”.",
                tags=["synthetic", "declared", "generator"],
                data={"generators": plain_named, "evidence": evidence[:300]},
            )

        for label in named:
            if label.startswith("declared algorithmic"):
                declared.append("IPTC digital source type")
                out.finding(
                    "critical",
                    "Declared as algorithmic media (IPTC)",
                    "The IPTC/XMP DigitalSourceType field says the picture was created or composited by a trained algorithm. Meta, Google and news agencies write this label.",
                    tags=["synthetic", "declared"],
                    data={"source": "IPTC DigitalSourceType"},
                )

        if provenance.get("generators"):
            for g in provenance["generators"]:
                if g not in declared:
                    declared.append(g)

        # ---- statistical signals (weak, reported as such) --------------------------------------------------
        out.progress(0.55, "statistical signals")
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        spectral = await asyncio.to_thread(_spectral_signals, gray)
        floor = await asyncio.to_thread(_noise_floor, gray)
        camera_named = bool(provenance.get("make") or provenance.get("model"))
        screenshot = bool(provenance.get("screenshot"))
        out.artifact(
            "spectrum", "series", "Radial power spectrum (dB) against spatial frequency",
            data={"x": spectral["freqs"], "series": [{"name": "power (dB)", "y": spectral["profile_db"]}], "x_unit": "cycles/pixel", "y_unit": "dB"},
        )  # fmt: skip
        published["spectral"] = {
            k: v for k, v in spectral.items() if k not in ("profile_db", "freqs")
        }
        published["noise_floor"] = floor

        if spectral["peak_z"] > 6 and not screenshot:
            signals.append(
                f"a spectral peak {spectral['peak_z']:.0f} σ above the power-law trend near the top of the spectrum (upsampling-network fingerprint)"
            )

        if spectral["high_frequency_excess_db"] > 6 and not screenshot:
            signals.append(
                f"{spectral['high_frequency_excess_db']:.0f} dB more fine-detail energy than a camera's optics pass"
            )

        if floor < 0.6 and not camera_named and not screenshot and info.format != "PNG":
            signals.append(
                f"an almost noiseless image (grain {floor:.2f} levels) with no camera named"
            )
        elif floor < 0.35 and not camera_named and not screenshot:
            signals.append(
                f"an almost noiseless image (grain {floor:.2f} levels) with no camera named"
            )

        w, h = info.stored_width, info.stored_height

        if (w in _SIZES and h in _SIZES) and not camera_named and not screenshot:
            signals.append(
                f"generator-typical pixel size {w}×{h} (a multiple of 64 on both sides) with no camera named"
            )

        if (
            jpeg is None
            and not camera_named
            and not screenshot
            and info.format == "PNG"
            and not provenance.get("exif_present")
        ):
            signals.append(
                "a PNG photograph without any camera metadata (cameras write JPEG/HEIC with EXIF; generators and screenshots write PNG)"
            )

        published["declared"] = declared
        published["signals"] = signals

        if declared:
            pass  # the declared findings above carry the weight
        elif len(signals) >= 2:
            out.finding(
                "warning",
                f"{len(signals)} weak signs of a synthetic image",
                "Nothing in the file declares a generator, but "
                + "; ".join(signals)
                + ". Each of these also occurs in heavily processed, upscaled or stripped real photos, "
                "so together they are grounds for suspicion, not a verdict. The AI eyewitness, if enabled, gives an opinion labelled as such.",
                tags=["synthetic", "signals"],
                data={
                    "signals": signals,
                    "noise_floor": floor,
                    "spectral": published["spectral"],
                },
            )
        elif signals:
            out.finding(
                "notice",
                "One weak sign of a synthetic image",
                signals[0][0].upper()
                + signals[0][1:]
                + ". On its own this proves nothing; real photos that were upscaled, denoised or stripped of metadata look the same.",
                tags=["synthetic", "signals"],
                data={
                    "signals": signals,
                    "noise_floor": floor,
                    "spectral": published["spectral"],
                },
            )
        else:
            out.finding(
                "info",
                "No signs of AI generation",
                f"No Content Credentials or generator names, no generation recipe, a natural spectrum (slope {spectral['slope']:.2f}) and a sensor-like noise floor ({floor:.2f} levels). "
                "Absence of signals is not proof of a camera: a generated image re-photographed or re-saved through a phone loses them.",
                tags=["synthetic"],
                data={"noise_floor": floor, "spectral": published["spectral"]},
            )

        if declared and camera_named:
            out.finding(
                "warning",
                "Camera metadata on a declared AI image",
                f"The file names a camera ({provenance.get('make') or ''} {provenance.get('model') or ''}) and also carries {declared[0]}. "
                "Either the generator kept a source photo's labels (image-to-image editing) or the camera tags were added to make it look real.",
                tags=["synthetic", "inconsistency"],
                data={
                    "declared": declared,
                    "make": provenance.get("make"),
                    "model": provenance.get("model"),
                },
            )

        ctx.publish(self.name, published)
        out.progress(1.0)
