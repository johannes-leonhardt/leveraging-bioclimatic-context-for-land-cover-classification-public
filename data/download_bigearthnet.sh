wget -r https://bigearth.net/downloads/BigEarthNet-S2-v1.0.tar.gz
wget https://bigearth.net/static/documents/patches_with_seasonal_snow.csv
wget https://bigearth.net/static/documents/patches_with_cloud_and_shadow.csv
tar -xf bigearth.net/downloads/BigEarthNet-S2-v1.0.tar.gz -C .
mv BigEarthNet-v1.0 bigearthnet
rm -r bigearth.net
wget -O bigearthnet.gpkg https://uni-bonn.sciebo.de/s/HP92XuNgM4lbu9u/download