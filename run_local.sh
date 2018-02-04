nvidia-docker run -it --rm -v $(pwd):/contents -w /contents \
    -v /home/tony/repos/skynet-data/data:/data -p 8888:8888 tonychangcsp/keras:gdal 
