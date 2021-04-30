#!/bin/bash
is_seg=false

git clone -b master --single-branch --depth 1 https://github.com/thaiminhpv/Doctor-Cyclop-Hackathon-2021
mv ~/Doctor-Cyclop-Hackathon-2021 ~/Doctors-Cyclop

if [ "$is_seg" = true ] ; then
  read -rsp 'kaggle.json: ' kaggle_json_file
  echo
  pip3 install -q --upgrade --force-reinstall --no-deps kaggle
  mkdir -p /root/.kaggle/
  echo "$kaggle_json_file" >/root/.kaggle/kaggle.json
  chmod 600 /root/.kaggle/kaggle.json

  cd ~/Doctors-Cyclop/resources/input || exit
  kaggle competitions download -c ranzcr-clip-catheter-line-classification &&
    unzip -qd ranzcr-clip-catheter-line-classification ranzcr-clip-catheter-line-classification.zip &&
    rm -f ranzcr-clip-catheter-line-classification.zip
else
  cp -rf /content/drive/MyDrive/output/resized_images.zip ~/Doctors-Cyclop/resources/input/resized_images.zip &&
  cd ~/Doctors-Cyclop/resources/input/ && unzip -q resized_images.zip && rm resized_images.zip && mv resized_images train_images || exit

  cp -rf /content/drive/MyDrive/output/pred_masks.zip ~/pred_masks.zip
  unzip -qod ~/Doctors-Cyclop/resources/input/pred_masks ~/pred_masks.zip || exit
  rm ~/pred_masks.zip
fi

chmod 777 ~/Doctors-Cyclop/Trainer/install-dependency.sh
~/Doctors-Cyclop/Trainer/install-dependency.sh

# mount to Drive
ln -s /models/ ~/Doctors-Cyclop/resources/models
ln -s /logs/ ~/Doctors-Cyclop/resources/logs

if [ "$is_seg" = true ] ; then
  # Seg
  time (
    python3 ~/Doctors-Cyclop/Trainer/segmentation/main.py 2> >(tee -a /logs/error.log >&2) &&
    cd resources && zip -qr /pred_masks/pred_masks_out.zip mask_unet++_6epo
  )
  echo 'Done Seg!'
else
  # Cls
  time (
    python3 ~/Doctors-Cyclop/Trainer/classification/main.py 2> >(tee -a /logs/error.log >&2)
  )
  echo 'Done Cls!'
fi

