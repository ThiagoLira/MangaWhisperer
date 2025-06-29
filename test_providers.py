from PIL import Image, ImageDraw, ImageFont
import argparse
import os



def create_sample_image() -> Image.Image:
    """Create a simple sample image with Japanese text."""
    img = Image.new("RGB", (200, 100), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    text = "こんにちは"
    # Attempt to load a default font that supports Japanese characters
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 24)
    except Exception:
        font = None
    draw.text((10, 40), text, fill=(0, 0, 0), font=font)
    return img


def test_ocr_providers(img: Image.Image, providers=None) -> None:
    from inference.ocr import OCR
    providers = providers or ["openai", "llamacpp"]
    for provider in providers:
        try:
            print(f"Testing OCR provider: {provider}")
            ocr = OCR(provider_name=provider)
            lines = ocr.ocr_from_image(img)
            print(f"Output ({provider}):", lines)
        except Exception as e:
            print(f"Failed to use OCR provider '{provider}': {e}")


def test_tts_providers(text: str, providers=None) -> None:
    from inference.tts import TTS
    from utils import convert_np_array_to_wav
    providers = providers or ["parler", "kokoro"]
    for provider in providers:
        try:
            print(f"Testing TTS provider: {provider}")
            tts = TTS(provider=provider)
            audio = tts.tts_from_text(text)
            wav_path = f"sample_files/{provider}_sample.wav"
            convert_np_array_to_wav(audio, path_to_file=wav_path)
            print(f"Output ({provider}) saved to {wav_path}")
        except Exception as e:
            print(f"Failed to use TTS provider '{provider}': {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test OCR and TTS providers")
    parser.add_argument(
        "provider",
        choices=["openai", "llamacpp", "parler", "kokoro"],
        help="Name of the provider to test",
    )
    args = parser.parse_args()

    os.makedirs("sample_files", exist_ok=True)
    img = create_sample_image()
    img.save("sample_files/sample_image.png")

    if args.provider in ["openai", "llamacpp"]:
        test_ocr_providers(img, providers=[args.provider])
    else:
        test_tts_providers("こんにちは、世界", providers=[args.provider])


if __name__ == "__main__":
    main()
