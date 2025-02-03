from utils import read_image, parse_transcript
import torch
from PIL import Image
from typing import List
import base64
import io

# IF LOCAL
OCR_MODEL = None
OCR_TOKENIZER = None

# IF API
API_CLIENT = None

def ocr_from_image_local(image) -> List[str]:
    """
    Generate an OCR from a cropped manga panel, return list of detected texts.
    """

    global OCR_MODEL,OCR_TOKENIZER

    if OCR_MODEL is None:
        from transformers import AutoModel, AutoTokenizer

        OCR_MODEL = AutoModel.from_pretrained(
                'openbmb/MiniCPM-o-2_6',
                trust_remote_code=True,
                attn_implementation='sdpa', # sdpa or flash_attention_2
                torch_dtype=torch.bfloat16,
                init_vision=True,
                init_audio=False,
                init_tts=False,
                load_in_8bit=True)

        OCR_TOKENIZER = AutoTokenizer.from_pretrained('openbmb/MiniCPM-o-2_6', trust_remote_code=True)

    PROMPT_ = """
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
    msgs = [{'role': 'user', 'content': [image, PROMPT_]}]
    res = OCR_MODEL.chat(
        max_new_tokens=100,
        image=None,
        msgs=msgs,
        tokenizer=OCR_TOKENIZER
    )
    transcripted_lines = parse_transcript(res) 
    return transcripted_lines

def ocr_from_image_api(image: Image.Image) -> List[str]:
    """
    Generate an OCR transcription from a cropped manga panel using the GPT‑4o API.
    The function returns a list of detected texts structured as:
        [1]: FIRST SPEECH BUBBLE TEXT
        [2]: SECOND SPEECH BUBBLE TEXT
        ...
    """
    PROMPT_ = """
    Please transcribe the Japanese text from the speech bubbles in the provided image.
    Ensure that you extract the complete text from each bubble, maintaining the order of appearance.
    Provide the transcription in the following structured format using placeholders for each bubble.
    THE TEXT FROM THE SAME BUBBLE SHOULD BE ALL TRANSCRIBED TOGETHER

    [1]: FIRST SPEECH BUBBLE TEXT
    [2]: SECOND SPEECH BUBBLE TEXT
    [n]: ...

    Important Notes:
        Only transcribe the text exactly as it appears in the speech bubbles.
        Do not add any additional explanations, translations, or comments.
        Ensure the transcription is accurate and complete.
    """
    global API_CLIENT
    if API_CLIENT is None:
        from openai import OpenAI
        from dotenv import load_dotenv
        import os
        # Load environment variables from .env file
        load_dotenv()

        # Ensure the key is loaded
        if not os.getenv('OPENAI_API_KEY'):
            raise ValueError("OPENAI_API_KEY not found! Make sure you have a .env file with the correct key.")

        API_CLIENT = OpenAI()

    # Convert the PIL image to JPEG bytes, then encode as a Base64 string.
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")  # You can also use PNG if preferred.
    img_bytes = buffered.getvalue()
    base64_image = base64.b64encode(img_bytes).decode("utf-8")
    data_uri = f"data:image/jpeg;base64,{base64_image}"
    
    # Construct the message as a list containing both text and image elements.
    messages_content = [
        {"type": "text", "text": PROMPT_},
        {"type": "image_url", "image_url": {"url": data_uri}},
    ]
    
    # Call the GPT‑4o API.
    response = API_CLIENT.chat.completions.create(
        model="gpt-4o-mini",  # Use the appropriate model identifier.
        messages=[
            {"role": "user", "content": messages_content}
        ],
        max_tokens=100
    )
    # Parse and return the transcription using your helper.
    transcripted_lines = parse_transcript(response.choices[0].message.content)
    return transcripted_lines
