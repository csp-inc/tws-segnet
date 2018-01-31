docker build . -t py-stretch-gdal:simple
docker run \
    -v $(pwd):/content \
    -w /content -it --rm \
    -u $(id -u):$(id -g) \
    -e STRIDEX=256 \
    -e STRIDEY=256 \
    -e WINDOWSIZEX=256 \
    -e WINDOWSIZEY=256 \
    py-stretch-gdal:simple \
    /bin/bash
