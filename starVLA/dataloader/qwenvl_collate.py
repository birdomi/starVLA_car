"""Collate that runs the Qwen-VL processor inside DataLoader workers.

By default the HF processor (image patching + chat-template tokenization)
runs on the main process inside `build_qwenvl_inputs`, serialized with the
GPU work (~0.5s per 128-sample batch). Doing it in the workers overlaps it
with training. The batch produced here is a dict:
    {"qwen_inputs": {input_ids, attention_mask, pixel_values, image_grid_thw, ...} (CPU tensors),
     "examples":    the original list of example dicts (action/state/lang/image kept)}
Message construction mirrors QWen3_5 `build_qwenvl_inputs` exactly.
"""


class ExampleList:
    """Opaque holder so DataLoader pin_memory leaves the raw examples
    (PIL images / numpy arrays) alone and pins only the tensor inputs."""

    __slots__ = ("items",)

    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)


def build_image_content(images, frame_tags=None):
    """[{"type":"image"}...] with an optional text tag in front of every image
    when the sample has more than one frame (tags identify past vs current)."""
    content = []
    use_tags = bool(frame_tags) and len(images) > 1
    for i, img in enumerate(images):
        if use_tags and i < len(frame_tags) and frame_tags[i]:
            content.append({"type": "text", "text": str(frame_tags[i])})
        content.append({"type": "image", "image": img})
    return content


class QwenVLPreprocessCollate:
    def __init__(self, processor, cot_prompt: str | None = None, frame_tags=None):
        self.processor = processor
        self.cot_prompt = cot_prompt
        # Optional text tag placed before each image when a sample carries
        # several frames (e.g. ["Previous frame, 0.5 seconds ago:", "Current frame:"]).
        self.frame_tags = list(frame_tags) if frame_tags else None

    def __call__(self, batch):
        messages = []
        for ex in batch:
            content = build_image_content(ex["image"], self.frame_tags)
            prompt = self.cot_prompt.replace("{instruction}", ex["lang"]) if self.cot_prompt else ex["lang"]
            content.append({"type": "text", "text": prompt})
            messages.append([{"role": "user", "content": content}])
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, padding=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )
        return {"qwen_inputs": dict(inputs), "examples": ExampleList(batch)}
