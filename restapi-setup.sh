#!/bin/bash
git clone -b master --single-branch --depth 1 https://github.com/thaiminhpv/Doctor-Cyclop-Hackathon-2021
mv ~/Doctor-Cyclop-Hackathon-2021 ~/Doctors-Cyclop

chmod 777 ~/Doctors-Cyclop/Trainer/install-dependency.sh
~/Doctors-Cyclop/Trainer/install-dependency.sh
pip3 -q install -U flask_ngrok

cp -rf /content/drive/MyDrive/output/trained-model-for-inference.zip ~/trained-model-for-inference.zip
unzip -d ~/Doctors-Cyclop/resources/models ~/trained-model-for-inference.zip

python3 ~/Doctors-Cyclop/Server/main.py

