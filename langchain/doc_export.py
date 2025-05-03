import os
import pandas as pd
from pathlib import Path
import sys
import time
from docling.datamodel import document
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem, ProvenanceItem

if len(sys.argv) > 1:
    input_file = sys.argv[1]
    outdir = input_file.replace(".json", "_extract")
else:
    input_file = "docling_test_raw.json"
    outdir = "scratch"

doc =  document.DoclingDocument.load_from_json(input_file)
# results in document already 

IMAGE_RESOLUTION_SCALE = 2.0


start_time = time.time()

output_dir = Path(outdir)
output_dir.mkdir(parents=True, exist_ok=True)

# Export tables
for table_ix, table in enumerate(doc.tables):
    table_df: pd.DataFrame = table.export_to_dataframe()
    print(f"## Table {table_ix}")
    print(table_df.to_markdown())

    # Save the table as csv
    element_csv_filename = output_dir / f"{input_file}-table-{table_ix + 1}.csv"
    print(f"Saving CSV table to {element_csv_filename}")
    table_df.to_csv(element_csv_filename)

    # Save the table as html
    element_html_filename = output_dir / f"{input_file}-table-{table_ix + 1}.html"
    print(f"Saving HTML table to {element_html_filename}")
    with element_html_filename.open("w") as fp:
        fp.write(table.export_to_html(doc=doc))

# Save page images
for page_no, page in doc.pages.items():
    page_no = page.page_no
    page_image_filename = output_dir / f"{input_file}-{page_no}.png"
    if page.image is not None:
        with page_image_filename.open("wb") as fp:
            page.image.pil_image.save(fp, format="PNG")

# Save images of figures and tables
table_counter = 0
picture_counter = 0
for element, _level in doc.iterate_items():
    if isinstance(element, TableItem):
        img = element.get_image(doc)
        if img == None:
            print("Invalid table image on page",element.prov[0].page_no)
            continue            
        table_counter += 1
        element_image_filename = (
            output_dir / f"{input_file}-table-{table_counter}.png"
        )
        with element_image_filename.open("wb") as fp:
            img.save(fp, "PNG")

    if isinstance(element, PictureItem):
        img = element.get_image(doc)
        if img == None:
            print("Invaild picture on page",element.prov[0].page_no)
            continue            
        picture_counter += 1
        element_image_filename = (
            output_dir / f"{input_file}-picture-{picture_counter}.png"
        )
        with element_image_filename.open("wb") as fp:
            img.save(fp, "PNG")

