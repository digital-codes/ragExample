#!/bin/bash
if [ $# -lt 1 ]; then
    echo "Usage: $0 FILE" >&2
    exit 2
fi

file="$1"

if [ ! -e "$file" ]; then
    echo "Error: '$file' does not exist." >&2
    exit 3
fi

if [ ! -r "$file" ]; then
    echo "Error: '$file' is not readable." >&2
    exit 4
fi

# docling itslef could run with multiple files. however, pycontabo seems to have
# problems with that. use bash loop in that case

# proceed with processing "$file"
/opt/pyenvs/rag/bin/docling --verbose -vv --pipeline standard --to md --image-export-mode referenced --tables --ocr-engine tesseract --table-mode accurate --num-threads 8 --output ./ko_extracted/ "$file" 
