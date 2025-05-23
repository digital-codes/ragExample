from docling.document_converter import DocumentConverter
import json

USE_OCR = True

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TesseractCliOcrOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

import sys
if len(sys.argv) > 1:
    source = sys.argv[1]  # document per local path or URL
else:
    source = "https://arxiv.org/pdf/2408.09869"  # document per local path or URL

# Set lang=["auto"] with a tesseract OCR engine: TesseractOcrOptions, TesseractCliOcrOptions
# ocr_options = TesseractOcrOptions(lang=["auto"])
ocr_options = TesseractCliOcrOptions(lang=["auto"])

IMAGE_RESOLUTION_SCALE = 2.0

if USE_OCR:
    pipeline_options = PdfPipelineOptions(
        do_ocr=True,
        force_full_page_ocr=True,
        ocr_options=ocr_options
    )
else:
    pipeline_options = PdfPipelineOptions()

# defaults
pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
pipeline_options.generate_page_images = True
pipeline_options.generate_picture_images = True


converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options,
        )
    }
)

result = converter.convert(source)

# filename from source
source_f = source.split("/")[-1]
if source_f.lower().endswith(".pdf"):
    outraw = source_f.replace(".pdf", "_raw.json")
    outtext = source_f.replace(".pdf", "_out.json")
    outmd = source_f.replace(".pdf", "_out.md")
else:
    outraw = source_f + "_raw.json"
    outtext = source_f + "_out.json"
    outmd = source_f + "_out.md"

with open(outmd, "w") as f:
    f.write(result.document.export_to_markdown(image_mode="embedded"))


with open(outraw, "w") as f:
    json.dump(
        result.document.export_to_dict(), f, indent=4
    )  # output: {"title": "Docling Technical Report[...]"}

# simplified
doc = {
    "metadata": {
        "title": "Some title... ",
        "source": source,
    },
    "content": result.document.export_to_text(),
}

with open(outtext, "w") as f:
    json.dump(doc, f)
