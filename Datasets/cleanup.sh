#!/bin/bash

# Folder where you want to delete files
folder="."

# Find and delete files matching the pattern
echo "Deleting files starting with 'iris_cont' in $folder..."
rm -f "$folder/iris_cont"*

echo "Done!"
