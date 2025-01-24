from utils import read_image, parse_transcript
from transformers import AutoModel, AutoTokenizer
import torch
from PIL import Image
from typing import List


OCR_MODEL = AutoModel.from_pretrained(
    'openbmb/MiniCPM-o-2_6',
    trust_remote_code=True,
    attn_implementation='sdpa', # sdpa or flash_attention_2
    torch_dtype=torch.bfloat16,
    init_vision=True,
    init_audio=False,
    init_tts=False,
    load_in_8bit=True
)

OCR_TOKENIZER = AutoTokenizer.from_pretrained('openbmb/MiniCPM-o-2_6', trust_remote_code=True)

def ocr_from_image(image) -> List[str]:
    """
    Generate an OCR from a cropped manga panel, return list of detected texts.
    """
    PROMPT_ = """
    Please transcribe the Japanese text from the speech bubbles in the provided image. Ensure that you extract the complete text from each bubble, maintaining the order of appearance. Provide the transcription in the following structured format:

    [1]: FIRST SPEECH BUBBLE TEXT
    [2]: SECOND SPEECH BUBBLE TEXT
    [n]: ...

    Important Notes:

        Only transcribe the text exactly as it appears in the speech bubbles.
        Do not add any additional explanations, translations, or comments.
        Ensure the transcription is accurate and complete.
    """
    msgs = [{'role': 'user', 'content': [image, PROMPT_]}]
    res = OCR_MODEL.chat(
        image=None,
        msgs=msgs,
        tokenizer=OCR_TOKENIZER
    )
    transcripted_lines = parse_transcript(res) 

    return transcripted_lines
