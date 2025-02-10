from utils import read_image, parse_transcript
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

```
{CURRENT_CHARACTER_BANK}
```

Add textual descriptions of the characters from this page that are not yet on the characters bank, no need for names, just the descriptions.
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
 
        self.ocr_from_image = self.ocr_from_image_local if is_local else self.ocr_from_image_api

    def ocr_from_image_local(self, image) -> List[str]:
        """
        Generate an OCR from a cropped manga panel, return list of detected texts.
        """

        msgs = [{'role': 'user', 'content': [image, PROMPT_OCR]}]
        res = self.OCR_MODEL.chat(
            max_new_tokens=100,
            image=None,
            msgs=msgs,
            tokenizer=OCR_TOKENIZER
        )
        transcripted_lines = parse_transcript(res) 
        return transcripted_lines

    def ocr_from_image_api(self, image: Image.Image) -> List[str]:
        """
        Generate an OCR transcription from a cropped manga panel using the GPT‑4o API.
        The function returns a list of detected texts structured as:
            [1]: FIRST SPEECH BUBBLE TEXT
            [2]: SECOND SPEECH BUBBLE TEXT
            ...
        """


        # Convert the PIL image to JPEG bytes, then encode as a Base64 string.
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")  # You can also use PNG if preferred.
        img_bytes = buffered.getvalue()
        base64_image = base64.b64encode(img_bytes).decode("utf-8")
        data_uri = f"data:image/jpeg;base64,{base64_image}"
        
        # Construct the message as a list containing both text and image elements.
        messages_content = [
            {"type": "text", "text": PROMPT_OCR},
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
        transcripted_lines = parse_transcript(response.choices[0].message.content)
        return transcripted_lines
