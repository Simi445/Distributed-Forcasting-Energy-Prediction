from pathlib import Path

from pptx import Presentation


def shape_text(shape):
    if not getattr(shape, "has_text_frame", False):
        return ""
    return "\n".join(
        paragraph.text for paragraph in shape.text_frame.paragraphs if paragraph.text
    ).strip()


def inspect(path: Path):
    prs = Presentation(path)
    print(f"\n=== {path.name} ===")
    print(f"Slides: {len(prs.slides)}")
    print(f"Size: {prs.slide_width} x {prs.slide_height}")
    for idx, slide in enumerate(prs.slides, start=1):
        texts = []
        for shape in slide.shapes:
            text = shape_text(shape)
            if text:
                texts.append(text.replace("\n", " | "))
        print(f"\nSlide {idx}: layout={slide.slide_layout.name!r}")
        for text in texts:
            print(f"  - {text[:500]}")


def main():
    root = Path(__file__).resolve().parents[1]
    inspect(root / "Template Licenta 7.pptx")
    inspect(root / "Prezentare-PrataMirela 7.pptx")


if __name__ == "__main__":
    main()
