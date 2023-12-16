#!/usr/bin/env bash
#
# Resize all the images in a specified folder with ImageMagick. Put the
# resized images in another specified folder.
#
# Of course, this script is an overkill. Just typing the loop on the command
# line would be enough...
#
# Daniel Weibel <danielmweibel@gmail.com>, 27 Dec. 2014
#------------------------------------------------------------------------------#

set -e

# Default values
resize=30%
ext=png

usage() {
  echo "USAGE"
  echo "  $(basename $0) -i dir -o dir [-r resizing] [-e extension]"
  echo
  echo "DESCRIPTION"
  echo "  Resize all images in a directory. Requires ImageMagick."
  echo 
  echo "OPTIONS                                             [DEFAULT]"
  echo "  -i: Directory containing images to resize"
  echo "  -o: Output directory"
  echo "  -r: Resize specification (ImageMagick syntax)     [50%]"
  echo "  -e: File extension of images in input directory   [jpg]"
}
[[ "$1" = -h ]] && { usage; exit; }

[[ $(which convert) = "" ]] && \
  { echo "Error: ImageMagick not installed"; exit 1; }

while getopts ":i:o:r:e:" opt; do
  case $opt in
    i) idir=${OPTARG%/}; ;;  # Remove trailing slash if present
    o) odir=${OPTARG%/}; ;;  # Remove trailing slash if present
    r) resize=$OPTARG;   ;;
    e) ext=$OPTARG;      ;;
    \?) echo "Error: invalid option: -$OPTARG";             exit 1 ;;
    :)  echo "Error: option -$OPTARG requires an argument"; exit 1 ;;
  esac
done

[[ -z "$idir" ]] && { echo "Error: Must specify input directory";  exit 1; }
[[ -z "$odir" ]] && { echo "Error: Must specify output directory"; exit 1; }

# nullglob: if *.jpg doesn't match any files, glob is replaced by "" not "*.jpg"
# nullglob should always be unset, so we activate it only in a subshell ($())
[[ $(shopt -s nullglob; echo "$idir"/*.$ext) = "" ]] \
  && { echo "Error: No files with extension $ext in directory $idir"; exit 1; }

for image in "$idir"/*.$ext; do
  echo -n "Resizing ${image##*/}... "
  # ImageMagick call
  convert "$image"     \
    -colorspace RGB  \
    -resize $resize  \
    -colorspace sRGB \
    "$odir"/$(basename "$image")
  # 1. Convert colorspace to linear (remove Gamma correction)
  # 2. Resize image
  # 3. Convert colorspace back to non-linear (add Gamma correction)
  # Note: don't use -quality, because an appropriate quality (compression) is
  # determined automatically for every image.
  echo Done
done