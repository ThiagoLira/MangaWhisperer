from parler_tts import ParlerTTSForConditionalGeneration
import soundfile as sf
from rubyinserter import add_ruby
from typing import List
import numpy.typing as npt
import torch
from transformers import AutoTokenizer

def tts_from_text(text, description_character) -> npt.NDArray:
    """
    Generate a wav transcription from an input text 
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = ParlerTTSForConditionalGeneration.from_pretrained("2121-8/japanese-parler-tts-mini").to(device)
    prompt_tokenizer = AutoTokenizer.from_pretrained("2121-8/japanese-parler-tts-mini", subfolder="prompt_tokenizer")
    description_tokenizer = AutoTokenizer.from_pretrained("2121-8/japanese-parler-tts-mini", subfolder="description_tokenizer")

    prompt = text


    prompt = add_ruby(prompt)
    input_ids = description_tokenizer(description_character, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = prompt_tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
    audio_arr = generation.cpu().numpy().squeeze()
    #sf.write(f"parler_tts_japanese_out_{n}.wav", audio_arr, model.config.sampling_rate)
    return audio_arr
