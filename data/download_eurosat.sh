wget -r https://zenodo.org/record/7711810/files/EuroSAT_MS.zip?download=1
unzip zenodo.org/record/7711810/files/EuroSAT_MS.zip?download=1 -d .
mkdir eurosat
mv EuroSAT_MS eurosat/geotiff
rm -r zenodo.org/
wget -O eurosat.gpkg https://uni-bonn.sciebo.de/s/Et1g52C2scbiO6E/download
