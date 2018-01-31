nvidia-docker run -it --rm -v /home/tony/repos/CSP/tws-segnet:/contents -v /home/tony/repos/skynet-data/data:/data -w /contents -p 8787:8888 tonychangcsp/keras:gdal 
