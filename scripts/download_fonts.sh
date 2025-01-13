#!/bin/bash

################################################################################
# Script to Download Free Fonts from Google Fonts Repository Using curl
################################################################################

# Exit immediately if a command exits with a non-zero status
set -e

# Directory to save downloaded fonts
FONT_DIR="fonts"
mkdir -p "$FONT_DIR"

# Array of font URLs from Google Fonts GitHub repository
FONT_URLS=(
    "https://github.com/google/fonts/raw/main/apache/roboto/Roboto-Bold.ttf"
    "https://github.com/google/fonts/raw/main/apache/lato/Lato-Regular.ttf"
    "https://github.com/google/fonts/raw/main/apache/lato/Lato-Bold.ttf"
    "https://github.com/google/fonts/raw/main/ofl/opensans/OpenSans-Regular.ttf"
    "https://github.com/google/fonts/raw/main/ofl/opensans/OpenSans-Bold.ttf"
    "https://github.com/google/fonts/raw/main/ofl/montserrat/Montserrat-Regular.ttf"
    "https://github.com/google/fonts/raw/main/ofl/montserrat/Montserrat-Bold.ttf"
)

# Function to download a single font using curl
download_font() {
    local url=$1
    local output_dir=$2

    # Extract the filename from the URL
    local filename=$(basename "$url")

    echo "Downloading $filename from $url"

    # Use curl to download the font with error handling
    curl -fLo "$output_dir/$filename" --create-dirs "$url" \
        && echo "Successfully downloaded $filename" \
        || { echo "Failed to download $filename"; }
}

# Download each font in the FONT_URLS array
for url in "${FONT_URLS[@]}"; do
    download_font "$url" "$FONT_DIR"
done

echo "All font download attempts completed. Check the '$FONT_DIR/' directory."
