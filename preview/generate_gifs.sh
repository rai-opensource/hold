#!/usr/bin/env bash

set -e

INPUT_DIR="./mp4"
OUTPUT_DIR="./gif"

mkdir -p "$OUTPUT_DIR"

for video in "$INPUT_DIR"/*.mp4; do
    [ -e "$video" ] || continue

    filename=$(basename "$video")
    name="${filename%.*}"

    echo "Processing $filename..."

    # Get duration in seconds (float)
    duration=$(ffprobe -v error \
        -show_entries format=duration \
        -of default=noprint_wrappers=1:nokey=1 \
        "$video")

    # Decide start time
    start=0
    if (( $(echo "$duration > 14" | bc -l) )); then
        start=4
    fi

    palette="/tmp/${name}_palette.png"
    gif="$OUTPUT_DIR/${name}.gif"

    # Generate palette
    ffmpeg -y \
        -ss "$start" -t 10 -i "$video" \
        -vf "fps=15,scale=240:240:flags=lanczos,palettegen" \
        "$palette"

    # Generate GIF using palette
    ffmpeg -y \
        -ss "$start" -t 10 -i "$video" -i "$palette" \
        -lavfi "fps=15,scale=240:240:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3" \
        "$gif"

    rm -f "$palette"

    echo "Created $gif"
done

echo "Done."