"""Video -> MiniMax H3 prompt through an OpenAI-compatible vision server (vLLM, SGLang, llama.cpp, LM Studio...).

The clip goes to the server as a small H.264 video (or as JPEG frames) with instructions for H3's text-to-video
layout: integrated_multimodal_description ([Shot N] per continuous take) + overall_soundscape + non_diegetic_music.
The API key is read from an environment variable at run time, so it never lands in a workflow or in output metadata.
TLS is always verified (system trust store, or a CA file you name).
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import ssl
import urllib.error
import urllib.request

FORMATS = ["H3 text-to-video (shots + soundscape)", "plain description"]
SECTIONS = ("integrated_multimodal_description:", "overall_soundscape:", "non_diegetic_music:")

H3_INSTRUCTIONS = """You write prompts for MiniMax H3, a model that generates video with synchronized audio from text.
You are shown an existing clip, {duration:.1f} seconds long. Write the H3 prompt that would generate this exact clip.

Output exactly these three sections, in this order, and nothing else:

integrated_multimodal_description:
[Shot 1] From 0.0 s to X s, ...
[Shot 2] From X s to Y s, ... (only when the clip really cuts to a new camera take)

overall_soundscape:
...

non_diegetic_music:
...

Rules for each shot:
- Open with the time span in seconds, then the framing and lens feel (extreme close-up, medium shot, wide shot; shallow or deep focus).
- Describe every visible subject precisely so it can be redrawn: species or apparent age and build, hair, skin, clothing with colours and materials, distinctive marks, and where it sits in the frame.
- Describe the setting, the light (direction, quality, time of day) and the colour palette.
- Describe the action as an ordered physical chain with timing: what moves, how fast, and what it affects.
- Name exactly one camera behaviour per shot: static, slow push-in, pan left, handheld follow, and so on.
- Name the visual style: photoreal live action, film look, 3D animation, anime, and so on.
- Describe only what is visible. Invent no story, names, dialogue or text; quote on-screen text exactly.

overall_soundscape: you cannot hear the clip. State plainly the diegetic sounds that the visible actions and the place would make (footsteps, wind, fabric, room tone, voices without inventing words), each tied to the action that makes it.
non_diegetic_music: write N/A unless the clip is plainly a music performance or music video.

State everything as fact: no hedging words such as "likely", "perhaps", "appears to" or "seems".
Present tense, plain English, no markdown, no lists, no preamble, no closing remarks."""

PLAIN_INSTRUCTIONS = """Describe this {duration:.1f}-second clip as one detailed paragraph that a text-to-video model could use to
regenerate it: subjects and their exact appearance, setting, light and colour, the action in order, the camera
behaviour and the visual style. Only what is visible; no preamble, no markdown."""


def instructions(fmt, duration, extra=""):
    text = (H3_INSTRUCTIONS if fmt == FORMATS[0] else PLAIN_INSTRUCTIONS).format(duration=duration)
    if extra and extra.strip():
        text += "\n\nAdditional instructions from the user (they override the rules above where they conflict):\n" + extra.strip()
    return text


def clean_response(text, fmt):
    """Drop reasoning blocks, code fences and any preamble before the first H3 section."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"^.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)   # unterminated opening tag
    text = re.sub(r"```[a-zA-Z]*\n?", "", text).replace("```", "")
    text = text.replace("**", "").strip()
    if fmt == FORMATS[0]:
        start = text.lower().find(SECTIONS[0])
        if start > 0:
            text = text[start:]
        text = re.sub(r"(?im)^\s*#+\s*", "", text)
    return text.strip()


def missing_sections(text):
    low = text.lower()
    return [s.rstrip(":") for s in SECTIONS if s not in low]


# ------------------------------------------------------------------ media

def encode_video(frames, fps, sample_fps=4.0, max_side=448, crf=28):
    """frames: float tensor [N, H, W, 3] in 0..1 at `fps` -> H.264 MP4 bytes at `sample_fps`, longest side max_side."""
    import av
    import numpy as np
    import torch
    import torch.nn.functional as F

    count = frames.shape[0]
    take = max(1, int(round(count / fps * sample_fps)))
    index = [min(count - 1, int(round(i * fps / sample_fps))) for i in range(take)]
    h, w = frames.shape[1], frames.shape[2]
    scale = min(1.0, max_side / max(h, w))
    oh, ow = max(2, int(h * scale) // 2 * 2), max(2, int(w * scale) // 2 * 2)
    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=max(1, int(round(sample_fps))))
        stream.width, stream.height, stream.pix_fmt = ow, oh, "yuv420p"
        stream.options = {"crf": str(crf), "preset": "veryfast"}
        for i in index:
            frame = frames[i:i + 1, :, :, :3].movedim(-1, 1).float()
            if (oh, ow) != (h, w):
                frame = F.interpolate(frame, size=(oh, ow), mode="bilinear", antialias=True, align_corners=False)
            rgb = (frame[0].movedim(0, -1).clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
            for packet in stream.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb), format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return buffer.getvalue(), len(index), (ow, oh)


def encode_frames(frames, fps, count=8, max_side=672, quality=85):
    """Evenly spaced JPEG frames (bytes) with their timestamps in seconds."""
    from PIL import Image
    n = frames.shape[0]
    picks = sorted({min(n - 1, round(i * (n - 1) / max(1, count - 1))) for i in range(count)})
    out = []
    for i in picks:
        image = Image.fromarray((frames[i, :, :, :3].clamp(0, 1) * 255).round().byte().cpu().numpy())
        image.thumbnail((max_side, max_side))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        out.append((buffer.getvalue(), i / fps))
    return out


# ------------------------------------------------------------------ server

def _context(ca_file):
    return ssl.create_default_context(cafile=ca_file) if ca_file else ssl.create_default_context()


def _request(url, key, body=None, ca_file="", timeout=300):
    headers = {"Authorization": "Bearer " + key} if key else {}
    data = None
    if body is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, context=_context(ca_file), timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as e:
        detail = e.read()[:500].decode(errors="replace")
        raise RuntimeError(f"{url}: HTTP {e.code} {detail}") from None
    except urllib.error.URLError as e:
        raise RuntimeError(f"{url}: {e.reason} (is the server up? for a self-signed certificate, trust its CA in "
                           "Windows or set ca_file)") from None


def _windows_saved_env(name):
    """The variable as saved in Windows (user, then system). A process started by a launcher that was already running
    when the variable was set does not inherit it, so ComfyUI may lack it even after a restart."""
    if os.name != "nt":
        return ""
    import winreg
    for root, path in ((winreg.HKEY_CURRENT_USER, r"Environment"),
                       (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment")):
        try:
            with winreg.OpenKey(root, path) as key:
                value, kind = winreg.QueryValueEx(key, name)
        except OSError:
            continue
        if value:
            return os.path.expandvars(value) if kind == winreg.REG_EXPAND_SZ else str(value)
    return ""


def api_key(env_name):
    name = (env_name or "").strip()
    if not name:
        return ""
    key = os.environ.get(name, "") or _windows_saved_env(name)
    if not key:
        raise RuntimeError(f"environment variable {name} is not set: not in the ComfyUI process and not saved in "
                           "Windows (user or system). Set it in Windows' environment variables (or with setx).")
    return key


def first_model(base_url, key, ca_file=""):
    models = _request(base_url.rstrip("/") + "/models", key, ca_file=ca_file, timeout=30).get("data", [])
    if not models:
        raise RuntimeError(f"{base_url}/models lists no models")
    return models[0]["id"]


def generate(base_url, key, model, text, media, max_tokens=1500, temperature=0.3, seed=0, thinking=False,
             ca_file=""):
    """media: ('video', mp4 bytes) or ('frames', [(jpeg bytes, seconds), ...]). Returns (content, response json)."""
    content = []
    if media[0] == "video":
        content.append({"type": "video_url",
                        "video_url": {"url": "data:video/mp4;base64," + base64.b64encode(media[1]).decode()}})
    else:
        for jpeg, seconds in media[1]:
            content.append({"type": "text", "text": f"Frame at {seconds:.1f} s:"})
            content.append({"type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}})
    content.append({"type": "text", "text": text})
    body = {"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": int(max_tokens),
            "temperature": float(temperature), "chat_template_kwargs": {"enable_thinking": bool(thinking)}}
    if seed:
        body["seed"] = int(seed)
    response = _request(base_url.rstrip("/") + "/chat/completions", key, body, ca_file=ca_file)
    message = response["choices"][0]["message"]
    return message.get("content") or "", response
