from typing import Optional, Any

import torch

from transformers.utils import is_accelerate_available, is_bitsandbytes_available
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    pipeline,
)

from peft import PeftModel

ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task, paired with an input that provides "
    "further context. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
)


def load_adapted_hf_generation_pipeline(
    base_model_name,
    lora_model_name,
    temperature: float = 0.7,
    top_p: float = 1.,
    max_tokens: int = 512,
    batch_size: int = 16,
    device: str = "cuda",
    load_in_8bit: bool = True,
    generation_kwargs: Optional[dict] = None,
):
    """
    Load a huggingface model & adapt with PEFT.
    Borrowed from https://github.com/tloen/alpaca-lora/blob/main/generate.py
    """

    if device == "cuda":
        if not is_accelerate_available():
            raise ValueError("Install `accelerate`")
    if load_in_8bit and not is_bitsandbytes_available():
            raise ValueError("Install `bitsandbytes`")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    task = "text-generation"

    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            load_in_8bit=load_in_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_model_name,
            torch_dtype=torch.float16,
        )
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_model_name,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_model_name,
            device_map={"": device},
        )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_in_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()

    generation_kwargs = generation_kwargs if generation_kwargs is not None else {}
    config = GenerationConfig(
        do_sample=True,
        temperature=temperature,
        max_new_tokens=max_tokens,
        top_p=top_p,
        **generation_kwargs,
    )
    pipe = pipeline(
        task,
        model=model,
        tokenizer=tokenizer,
        batch_size=16, # TODO: make a parameter
        generation_config=config,
        framework="pt",
    )

    return pipe

if __name__ == '__main__':
    pipe = load_adapted_hf_generation_pipeline(
    base_model_name="openlm-research/open_llama_3b_v2",
    lora_model_name="/home/brick2/models/國中生物大雜燴黑板講解-30-45-60_epochs4_batch32"
    )
    prompt_template = ALPACA_TEMPLATE.format(
        instruction="你現在是一個國中生物老師，請解釋青梅菌是什麼。",
        input="青梅菌是什麼？"
    )
    print(pipe(prompt_template)[0]['generated_text'])

