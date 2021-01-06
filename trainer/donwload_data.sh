wget https://object.pouta.csc.fi/Tatoeba-Challenge/eng-spa.tar
tar -xvf eng-spa.tar
mv data/eng-spa/ .

cd data/eng-spa/
rm -d data

cd eng-spa/
gzip -d train.*