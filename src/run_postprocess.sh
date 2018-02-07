#!/bin/bash

docker run --rm --env-file .env -v $(pwd):/output_dir -w /output_dir tonychangcsp/keras:gdal sh -c './mapbox_postprocess.sh'
