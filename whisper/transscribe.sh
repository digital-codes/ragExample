#!/bin/bash

echo "Make sure to have whisper gpu path set ..."
# CONFIGURATION
MODEL_PATH="/opt/llama/whisper/models/ggml-medium.bin"
LANGUAGE="auto"
DEVICE="GPU"
#AUDIO_DIR="./audio"
AUDIO_DIR="./audio"
OUTPUT_DIR="./transcripts"

# Create output dir if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Loop over all supported audio files in the directory
for file in "$AUDIO_DIR"/*.{wav,flac,mp3,m4a}; do
    [ -e "$file" ] || continue  # Skip if no match

    filename=$(basename "$file")
    name="${filename%.*}"
    output_file="$OUTPUT_DIR/${name}.txt"

    echo "▶️ Transcribing: $filename"

    whisper-cli \
        -m "$MODEL_PATH" \
        -f "$file" \
        -l "$LANGUAGE" \
        -otxt \
        --output-file "$output_file"

    echo "✅ Saved transcript to: $output_file"
    echo
done

echo "🟢 All files processed."
