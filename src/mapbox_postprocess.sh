#!/bin/bash
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
      
#compress the tiffs for mapbox
ls *.tif | xargs -n 1 -I {} sh -c 'rio calc "(asarray (take a 1) (take a 2) (take a 3))" --co compress=lzw --co tiled=true --co blockxsize=256 --co blockysize=256 --name a={} ./mapbox/{}' && \
lls *.tif >> ./mapbox/tif_list.log
#ls *.tif | xargs -n 1 -I {} sh -c 'rio calc "(asarray (take a 1) (take a 2) (take a 3))" --co compress=lzw --co tiled=true --co blockxsize=256 --co blockysize=256 --name a={} ./mapbox/{} && rio edit-info --nodata 0 ./mapbox/{}' && \

#upload to mapbox
LIST=./mapbox/tif_list.log
USERNAME=tonychang
while read GTIFF;
do 
    TILENAME=$(echo $(basename $GTIFF .tif))
    mapbox --access-token $MAPBOX_TOKEN upload $USERNAME.$TILENAME \
            --name $TILENAME './mapbox/'$GTIFF & wait
done < "$LIST"

