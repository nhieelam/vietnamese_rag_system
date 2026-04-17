from pdf2image import convert_from_path
from pathlib import Path
import tempfile

def pdf_to_images(pdf_path, dpi=200, poppler_path=None):
    args = {'dpi': dpi}
    if poppler_path:
        args['poppler_path'] = poppler_path
    return convert_from_path(str(pdf_path), **args)

def pdf_to_image_files(pdf_path, out_dir=None, dpi=200, poppler_path=None):
    out_dir = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="pdf_images_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    images = pdf_to_images(pdf_path, dpi=dpi, poppler_path=poppler_path)
    paths = []
    for i, img in enumerate(images, start=1):
        p = out_dir / f"page_{i}.png"
        img.save(p)
        paths.append(str(p))
    return paths

