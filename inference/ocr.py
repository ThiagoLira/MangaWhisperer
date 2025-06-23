from utils import read_image, parse_transcript, parse_character_bank
import torch
from PIL import Image
from typing import List
import base64
import io

PROMPT_OCR = """
    Please transcribe the Japanese text from the speech bubbles in the provided image. 
    Ensure that you extract the complete text from each bubble, maintaining the order of appearance. 
    Provide the transcription in the following structured format using placeholders for each bubble.
    THE TEXT FROM THE SAME BUBBLE SHOULD BE 

    [1]: FIRST SPEECH BUBBLE TEXT
    [2]: SECOND SPEECH BUBBLE TEXT
    [n]: ...

    Important Notes:

        Only transcribe the text exactly as it appears in the speech bubbles.
        Do not add any additional explanations, translations, or comments.
        Ensure the transcription is accurate and complete.
"""

PROMPT_FILL_CHARACTER_BANK = """
You will receive a manga page, your task is to create a CHARACTER BANK of all the characters you see:

Current character bank:


{CURRENT_CHARACTER_BANK}


Add textual descriptions of the characters from this page that are not yet on the characters bank, no need for names, just the descriptions.
You wil also image the voices of the characters on the end of the description. RETURN JUST THE NEW ENTRIES


Example of character bank:

1: This character is tall and slender, and uses a straw hat, he is energetic and child-like, his voice is high pitched like and naturally high.
2: This character is a humanoid reindeer with a funny hat. His voice is high pitched and he speaks in a serious tone, as if afraid to speak.
...

"""

PROMPT_DETECT_CHARACTER_SPEAKING = """
You will receive an image and a transcripted line being spoken in a manga page by some character, 
and also a list of all the characters in this manga, by their descriptions. 

Your task is to return the KEY of the correct character in the character bank that is speaking this line.

LINE:
{LINE}

CHARACTER BANK:
{CURRENT_CHARACTER_BANK}

Answer ONLY WITH THE KEY OF THE CORRECT CHARACTER AND NOTHING ELSE.
IF YOU ARE UNSURE JUST RETURN 0
"""

class BaseProvider:
    """Base class for OCR providers."""

    def call_image_model(self, image, prompt: str):
        raise NotImplementedError


class LlamacppProvider(BaseProvider):
    def __init__(self):
        from transformers import AutoModel, AutoTokenizer

        self.OCR_MODEL = AutoModel.from_pretrained(
            'openbmb/MiniCPM-o-2_6',
            trust_remote_code=True,
            attn_implementation='sdpa',  # sdpa or flash_attention_2
            torch_dtype=torch.bfloat16,
            init_vision=True,
            init_audio=False,
            init_tts=False,
            load_in_8bit=True,
        )
        self.OCR_TOKENIZER = AutoTokenizer.from_pretrained(
            'openbmb/MiniCPM-o-2_6', trust_remote_code=True
        )

    def call_image_model(self, image, prompt: str):
        msgs = [{'role': 'user', 'content': [image, prompt]}]
        res = self.OCR_MODEL.chat(
            max_new_tokens=100,
            image=None,
            msgs=msgs,
            tokenizer=self.OCR_TOKENIZER,
        )
        return res


class OpenAIProvider(BaseProvider):
    def __init__(self):
        from openai import OpenAI
        from dotenv import load_dotenv
        import os

        load_dotenv()
        if not os.getenv('OPENAI_API_KEY'):
            raise ValueError(
                "OPENAI_API_KEY not found! Make sure you have a .env file with the correct key."
            )
        self.API_CLIENT = OpenAI()

    def call_image_model(self, image, prompt: str):
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_bytes = buffered.getvalue()
        base64_image = base64.b64encode(img_bytes).decode("utf-8")
        data_uri = f"data:image/jpeg;base64,{base64_image}"

        messages_content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_uri}},
        ]

        response = self.API_CLIENT.chat.completions.create(
            model="gpt-4.1-mini-2025-04-14",
            messages=[{"role": "user", "content": messages_content}],
            max_tokens=100,
        )
        return response.choices[0].message.content


class OCR:
    def __init__(self, provider_name: str = "openai"):
        provider_name = provider_name.lower()
        if provider_name == "llamacpp":
            self.provider = LlamacppProvider()
        elif provider_name == "openai":
            self.provider = OpenAIProvider()
        else:
            raise ValueError(f"Unknown provider: {provider_name}")

        self.character_bank = {}

    def ocr_from_image(self, image):
        return parse_transcript(
            self.provider.call_image_model(image, PROMPT_OCR)
        )

    def update_character_bank(self, image):
        raw_output = self.provider.call_image_model(
            image,
            PROMPT_FILL_CHARACTER_BANK.format(
                CURRENT_CHARACTER_BANK=str(self.character_bank)
            ),
        )
        self.character_bank.update(parse_character_bank(raw_output))

    def get_description_of_character_speaking(self, image, line):
        """Determine which character is speaking a line."""
        raw_output = self.provider.call_image_model(
            image,
            PROMPT_DETECT_CHARACTER_SPEAKING.format(
                LINE=line,
                CURRENT_CHARACTER_BANK=str(self.character_bank),
            ),
        )
        try:
            desc = self.character_bank[int(raw_output)]
        except Exception:
            desc = next(
                iter(self.character_bank.values()),
                "A male speaker with a high pitched voice, he speaks loudly and with energy.",
            )
        return desc
