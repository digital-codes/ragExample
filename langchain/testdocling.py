from docling.document_converter import DocumentConverter
import json

USE_OCR = True

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TesseractCliOcrOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption


source = "https://arxiv.org/pdf/2408.09869"  # document per local path or URL

# Set lang=["auto"] with a tesseract OCR engine: TesseractOcrOptions, TesseractCliOcrOptions
# ocr_options = TesseractOcrOptions(lang=["auto"])
ocr_options = TesseractCliOcrOptions(lang=["auto"])

pipeline_options = PdfPipelineOptions(
    do_ocr=True, force_full_page_ocr=True, ocr_options=ocr_options
)
if USE_OCR:
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            )
        }
    )
else:
    converter = DocumentConverter()

result = converter.convert(source)
print(
    result.document.export_to_markdown()
)  # output: "## Docling Technical Report[...]"

with open("docling_test_raw.json", "w") as f:
    json.dump(
        result.document.export_to_dict(), f, indent=4
    )  # output: {"title": "Docling Technical Report[...]"}

doc = {
    "metadata": {
        "title": "Docling Technical Report[...]",
        "source": "https://arxiv.org/pdf/2408.09869",
    },
    "content": result.document.export_to_text(),
}

with open("docling_test.json", "w") as f:
    json.dump(doc, f)
