"""Open-vocabulary naming with a small vision-language model from Hugging Face.

YOLO only knows the 80 COCO classes. When a hold completes, the region the user is pointing at is
cropped and SmolVLM2 is asked to name the main object in it. This also covers objects YOLO
can't detect at all (pens, notebooks, chargers): then the crop is a square region just ahead of
the fingertip.

Inference takes ~1 s on an M-series GPU, so it runs on a worker thread and the video keeps
going; for recorded clips `wait=True` makes it synchronous and therefore reproducible.

Head motion blurs many frames, so the pipeline keeps the sharpest crop seen during the hold
(variance of the Laplacian) and sends that one rather than the last frame. The crop is square
for context, so pixels outside the object's segmentation outline are faded (a visual prompt).
"""

import queue
import threading
from dataclasses import dataclass

import cv2
import numpy as np

DEFAULT_MODEL = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
PROMPT = "What is the main object in this image? Answer with just its name in two to four words."
# Used when the user points at something in their other hand: otherwise the model tends to
# describe the hands ("finger pointing") instead of the object.
HELD_PROMPT = "What object is the hand holding? Answer with just its name in two to four words."
# Asked as a follow-up when the answer is a sign, so station names and notices get read out.
SIGN_PROMPT = "What does the sign say? Answer with only the words written on it."
SIGN_WORDS = ("sign", "signboard", "board", "poster", "banner", "plaque", "notice")


def is_sign(answer: str) -> bool:
    return any(w in SIGN_WORDS for w in answer.lower().split())


def sign_text(text: str, answer: str, max_len: int = 40) -> str | None:
    """Keep the read-out text only if it is short and actually adds something."""
    text = text.strip().strip('"').strip()
    if not text or len(text) > max_len or text.lower() in answer.lower() or text.lower() in ("none", "no text", "nothing"):
        return None
    return text


def object_crop(box: np.ndarray, frame_size: tuple[int, int], pad: float = 0.15, min_px: int = 160):
    """Square-ish crop around a detection box, padded so the model sees some context."""
    w, h = frame_size
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * (1 + 2 * pad)
    side = max(side, min_px)
    return _clip_square(cx, cy, side, w, h)


def region_crop(tip: np.ndarray, direction: np.ndarray, frame_size: tuple[int, int],
                ahead: float = 0.12, size: float = 0.28):
    """Square region just ahead of the fingertip, for objects the detector didn't find."""
    w, h = frame_size
    diag = float(np.hypot(w, h))
    cx, cy = tip + direction * ahead * diag
    return _clip_square(cx, cy, size * diag, w, h)


def centered_crop(center: np.ndarray, frame_size: tuple[int, int], size: float = 0.2):
    """Square region around a point, e.g. where the pointing ray meets the other hand."""
    w, h = frame_size
    return _clip_square(center[0], center[1], size * float(np.hypot(w, h)), w, h)


def _clip_square(cx, cy, side, w, h):
    side = min(side, w, h)
    x1 = int(np.clip(cx - side / 2, 0, w - side))
    y1 = int(np.clip(cy - side / 2, 0, h - side))
    return np.array([x1, y1, x1 + int(side), y1 + int(side)], dtype=np.float32)


def sharpness(image_bgr: np.ndarray, size: int = 160) -> float:
    """Variance of the Laplacian on a downscaled grey crop: low means motion blur."""
    h, w = image_bgr.shape[:2]
    scale = size / max(h, w)
    small = cv2.resize(image_bgr, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def highlight(crop_bgr: np.ndarray, contour: np.ndarray | None, offset, fade: float = 0.7) -> np.ndarray:
    """Visual prompt: fade everything outside the object's outline towards grey, so the model
    knows which object in a cluttered crop is meant."""
    if contour is None:
        return crop_bgr
    mask = np.zeros(crop_bgr.shape[:2], np.uint8)
    cv2.fillPoly(mask, [(contour - np.asarray(offset, np.float32)).astype(np.int32).reshape(-1, 1, 2)], 255)
    pad = max(3, int(0.02 * max(mask.shape)))
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1)))
    a = (mask.astype(np.float32) / 255.0)[..., None]
    faded = crop_bgr.astype(np.float32) * (1 - fade) + 128 * fade
    return (crop_bgr * a + faded * (1 - a)).astype(np.uint8)


def clean_answer(text: str) -> str:
    text = text.strip().split("\n")[0].strip().rstrip(".").strip()
    for prefix in ("The main object in this image is", "The main object is", "This is", "It is", "A ", "An "):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
    return text.lower()


def answer_score(step_logits, tokens, tokenizer) -> float:
    """Geometric-mean probability of the generated tokens, ignoring end-of-answer/special tokens."""
    import torch

    logps = []
    special = set(tokenizer.all_special_ids)
    for logits, tok in zip(step_logits, tokens.tolist()):
        if tok in special:
            break
        logps.append(torch.log_softmax(logits[0].float(), dim=-1)[tok].item())
    return float(np.exp(np.mean(logps))) if logps else 0.0


@dataclass
class Request:
    key: object
    image: np.ndarray  # BGR crop
    prompt: str = PROMPT


class Describer:
    def __init__(self, model_id: str = DEFAULT_MODEL, device: str | None = None, max_new_tokens: int = 12):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._torch = torch
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device != "cpu" else torch.float32
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype).to(self.device).eval()
        self.max_new_tokens = max_new_tokens
        self._requests: queue.Queue[Request | None] = queue.Queue()
        self._results: queue.Queue[tuple[object, str]] = queue.Queue()
        self._thread = threading.Thread(target=self._work, daemon=True)
        self._thread.start()

    def describe(self, image_bgr: np.ndarray, prompt: str = PROMPT, read_signs: bool = True) -> tuple[str, float]:
        """Returns (answer, score). The score is the model's own confidence: the geometric mean of
        the probabilities of the answer tokens it generated (1.0 = completely sure)."""
        from PIL import Image

        image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=text, images=[image], return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                                      output_scores=True, return_dict_in_generate=True)
        new = out.sequences[:, inputs["input_ids"].shape[1]:]
        answer = clean_answer(self.processor.batch_decode(new, skip_special_tokens=True)[0])
        score = answer_score(out.scores, new[0], self.processor.tokenizer)
        if read_signs and is_sign(answer):
            raw = self._generate(image, SIGN_PROMPT)
            text = sign_text(raw.split("\n")[0], answer)
            if text:
                answer = f'{answer} that says "{text}"'
        return answer, score

    def _generate(self, image, prompt: str) -> str:
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=text, images=[image], return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=16, do_sample=False)
        return self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()

    def submit(self, key, image_bgr: np.ndarray, wait: bool = False, prompt: str = PROMPT):
        """Queue a crop. With wait=True, describe it now and return the answer."""
        if wait:
            return self.describe(image_bgr, prompt)
        self._requests.put(Request(key, image_bgr.copy(), prompt))
        return None

    def poll(self) -> list[tuple[object, str]]:
        done = []
        while not self._results.empty():
            done.append(self._results.get_nowait())
        return done

    def _work(self):
        while True:
            req = self._requests.get()
            if req is None:
                return
            try:
                self._results.put((req.key, self.describe(req.image, req.prompt)))
            except Exception as exc:  # keep the video running if the model fails
                self._results.put((req.key, (f"(error: {exc})", 0.0)))

    def close(self):
        self._requests.put(None)
