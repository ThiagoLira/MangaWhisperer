from PIL import Image, ImageDraw, ImageFont

from inference.ocr import OCR
from inference.tts import TTS
from utils import convert_np_array_to_wav


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


def test_ocr_providers(img: Image.Image) -> None:
    for provider in ["openai", "llamacpp"]:
        try:
            print(f"Testing OCR provider: {provider}")
            ocr = OCR(provider_name=provider)
            lines = ocr.ocr_from_image(img)
            print(f"Output ({provider}):", lines)
        except Exception as e:
            print(f"Failed to use OCR provider '{provider}': {e}")


def test_tts_providers(text: str) -> None:
    for provider in ["parler", "kokoro"]:
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
    img = create_sample_image()
    img.save("sample_files/sample_image.png")
    test_ocr_providers(img)
    test_tts_providers("こんにちは、世界")


if __name__ == "__main__":
    main()
