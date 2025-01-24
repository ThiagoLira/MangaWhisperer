import os
import argparse
from utils import read_image, parse_transcript
import numpy as np
from transformers import AutoModel, AutoTokenizer
import torch
from PIL import Image
from parler_tts import ParlerTTSForConditionalGeneration
import soundfile as sf
from rubyinserter import add_ruby
from typing import List
import numpy.typing as npt

SECTIONING_MODEL = AutoModel.from_pretrained("ragavsachdeva/magiv2", trust_remote_code=True).cuda().eval()

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


def tts_from_text(text) -> npt.NDArray:
    """
    Generate a wav transcription from an input text 
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = ParlerTTSForConditionalGeneration.from_pretrained("2121-8/japanese-parler-tts-mini").to(device)
    prompt_tokenizer = AutoTokenizer.from_pretrained("2121-8/japanese-parler-tts-mini", subfolder="prompt_tokenizer")
    description_tokenizer = AutoTokenizer.from_pretrained("2121-8/japanese-parler-tts-mini", subfolder="description_tokenizer")

    prompt = text
    description = "A female speaker with a slightly high-pitched voice delivers her words at a moderate speed with a quite monotone tone in a confined environment, resulting in a quite clear audio recording."


    prompt = add_ruby(prompt)
    input_ids = description_tokenizer(description, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = prompt_tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
    audio_arr = generation.cpu().numpy().squeeze()
    #sf.write(f"parler_tts_japanese_out_{n}.wav", audio_arr, model.config.sampling_rate)
    return audio_arr



def main():
    """
    Main function to process a manga page image.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process a manga page image.")
    parser.add_argument("manga_page_path", type=str, help="Path to the manga page image")

    # Parse arguments
    args = parser.parse_args()
    manga_page_path = args.manga_page_path

    # Verify the file exists
    if not os.path.isfile(manga_page_path):
        print(f"Error: File not found at path '{manga_page_path}'")
        exit(1)

    # Try to open the image
    try:
        manga_page = read_image(manga_page_path)
        print(f"Successfully loaded manga page: {manga_page_path}")
    except Exception as e:
        print(f"Error: Failed to open image. Details: {e}")
        exit(1)

    # Process the image
    process_image(manga_page)

def process_image(image):
    with torch.no_grad():
        per_page_results = SECTIONING_MODEL.predict_detections_and_associations([image])

    bounding_boxes = per_page_results[0]['panels']

    # Convert the NumPy array to a PIL Image
    image = Image.fromarray(image)

    # Iterate through the bounding boxes and save each cropped region
    for i, (x_min, y_min, x_max, y_max) in enumerate(bounding_boxes):
        # Crop the image using the bounding box
        cropped_image = image.crop((x_min, y_min, x_max, y_max)).convert('RGB')
        transcripted_lines = ocr_from_image(cropped_image)
        for l in [transcripted_lines[1]]:
            print(l)
            tts_from_text(l)

if __name__ == "__main__":
    main()

