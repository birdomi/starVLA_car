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


class QwenVLPreprocessCollate:
    def __init__(self, processor, cot_prompt: str | None = None):
        self.processor = processor
        self.cot_prompt = cot_prompt

    def __call__(self, batch):
        messages = []
        for ex in batch:
            content = [{"type": "image", "image": img} for img in ex["image"]]
            prompt = self.cot_prompt.replace("{instruction}", ex["lang"]) if self.cot_prompt else ex["lang"]
            content.append({"type": "text", "text": prompt})
            messages.append([{"role": "user", "content": content}])
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, padding=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )
        return {"qwen_inputs": dict(inputs), "examples": ExampleList(batch)}
