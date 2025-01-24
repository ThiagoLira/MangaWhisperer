import os
import argparse
from utils import read_image, parse_transcript
import numpy as np
from transformers import AutoModel, AutoTokenizer
import torch
from PIL import Image
from inference.ocr import ocr_from_image
from inference.tts import tts_from_text

SECTIONING_MODEL = AutoModel.from_pretrained("ragavsachdeva/magiv2", trust_remote_code=True).cuda().eval()



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

