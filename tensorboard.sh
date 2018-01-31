LOGNAME=$1
if [ -z "$1" ]
then
    LOGNAME='.'
fi
nvidia-docker run -it --rm -v /home/tony/repos/CSP/tws-segnet:/contents -w /contents -p 6006:6006 tonychangcsp/keras:gdal tensorboard --logdir="/contents/logs/$LOGNAME" --port=6006
