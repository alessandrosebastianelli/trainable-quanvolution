rm EuroSAT.zip
rm -r 2750
rm -r dataset

wget https://madm.dfki.de/files/sentinel/EuroSAT.zip --no-check-certificate || curl -O https://madm.dfki.de/files/sentinel/EuroSAT.zip

unzip EuroSAT.zip
mkdir dataset

mkdir dataset/validation
mv 2750 EuroSAT
mv EuroSAT dataset/
cd dataset
mv EuroSAT training
cd ..

python scripts/split_dataset.py
