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

class ImageModel:
    def __init__(self, is_local=False):
        if is_local:
            from transformers import AutoModel, AutoTokenizer
            self.OCR_MODEL = AutoModel.from_pretrained(
                    'openbmb/MiniCPM-o-2_6',
                    trust_remote_code=True,
                    attn_implementation='sdpa', # sdpa or flash_attention_2
                    torch_dtype=torch.bfloat16,
                    init_vision=True,
                    init_audio=False,
                    init_tts=False,
                    load_in_8bit=True)

            self.OCR_TOKENIZER = AutoTokenizer.from_pretrained('openbmb/MiniCPM-o-2_6', trust_remote_code=True)
        else:
            from openai import OpenAI
            from dotenv import load_dotenv
            import os
            # Load environment variables from .env file
            load_dotenv()

            # Ensure the key is loaded
            if not os.getenv('OPENAI_API_KEY'):
                raise ValueError("OPENAI_API_KEY not found! Make sure you have a .env file with the correct key.")

            self.API_CLIENT = OpenAI()
 
        self.is_local = is_local
        self.character_bank = {}

    def call_api_image_model(self, image, prompt: str):
        # Convert the PIL image to JPEG bytes, then encode as a Base64 string.
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")  # You can also use PNG if preferred.
        img_bytes = buffered.getvalue()
        base64_image = base64.b64encode(img_bytes).decode("utf-8")
        data_uri = f"data:image/jpeg;base64,{base64_image}"

        # Construct the message as a list containing both text and image elements.
        messages_content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_uri}},
        ]

        # Call the GPT‑4o API.
        response = self.API_CLIENT.chat.completions.create(
            model="gpt-4o-mini",  # Use the appropriate model identifier.
            messages=[
                {"role": "user", "content": messages_content}
            ],
            max_tokens=100
        )
        # Parse and return the transcription using your helper.
        return response.choices[0].message.content

    def call_local_image_model(self, image, prompt):
        msgs = [{'role': 'user', 'content': [image, prompt]}]
        res = self.OCR_MODEL.chat(
            max_new_tokens=100,
            image=None,
            msgs=msgs,
            tokenizer=self.OCR_TOKENIZER
        )
        return res

    def ocr_from_image(self,image):
        if self.is_local:
            return parse_transcript(self.call_local_image_model(image, PROMPT_OCR))
        else:
            return parse_transcript(self.call_api_image_model(image,PROMPT_OCR))

    def update_character_bank(self, image):
        if self.is_local:
            raw_output = self.call_local_image_model(image,
                                                          PROMPT_FILL_CHARACTER_BANK.format(CURRENT_CHARACTER_BANK=str(self.character_bank)))
        else:
            raw_output = self.call_api_image_model(image,
                                                          PROMPT_FILL_CHARACTER_BANK.format(CURRENT_CHARACTER_BANK=str(self.character_bank)))

        self.character_bank.update(parse_character_bank(raw_output))

    def get_description_of_character_speaking(self, image, line):
        """
        Given image and line, use the character bank and determine who is speaking on this current line with the image model
        """
        if self.is_local:
            raw_output = self.call_local_image_model(image, PROMPT_DETECT_CHARACTER_SPEAKING.format(LINE=line,
                                                                                                    CURRENT_CHARACTER_BANK=str(self.character_bank)))
        else:
            raw_output = self.call_api_image_model(image,
                                                          PROMPT_DETECT_CHARACTER_SPEAKING.format(LINE=line,
                                                                                                  CURRENT_CHARACTER_BANK=str(self.character_bank)))
        try: 
            desc = self.character_bank[int(raw_output)]
        except: 
            # return default voice if empty
            desc = next(iter(self.character_bank.values()), "A male speaker with a high pitched voice, he speaks loudly and with energy.")

        return desc
