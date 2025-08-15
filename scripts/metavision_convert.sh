#!/bin/bash

ORDNER="${1:-.}"

for datei in "$ORDNER"/*; do
    if [ -f "$datei" ]; then
        echo "Verarbeite $datei"
        metavision_file_to_hdf5 -i "$datei"
    fi
done