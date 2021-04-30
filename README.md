# Doctor's Cyclop Model

---

This is the *Model Server* part of the project **Doctor's Cyclop**, and works with [Web App](https://github.com/DAN3002/Doctors-Cyclop-Webapp)

# Introduction

This is a Model implementation integrated with Web App, the dataset is got from the [RANZCR CLiP - Catheter and Line Position Challenge](https://www.kaggle.com/c/ranzcr-clip-catheter-line-classification)

# How to Build and Run

## 1. Mount folders at Google Drive

- Create shortcut (mount) to this [public folder](https://drive.google.com/drive/folders/1XxKDVzCms_O6UVG2zNKvmwV0rNtTmtIq?usp=sharing), add at root (`/`):

After done that you should have this at your Google Drive's root:

![root google drive structure](readme-assets/images/Google-Drive-Structure.png)

## 2. SSH to server

- run [this notebook](https://colab.research.google.com/drive/1L-ibyztYYcM0rmuXkPihN8LHP0TxkRi4?usp=sharing) with GPU enabled to create a Google Colab virtual machine
- Copy file [setup.sh](setup.sh) or [restapi-setup.sh](restapi-setup.sh) to the newly created virtual machine (by `scp`)

## 3. Run

- ssh to the created machine (password is `haha`), then run **either**:
  - [setup.sh](setup.sh) for **Training**
  - [restapi-setup.sh](restapi-setup.sh) for **start Online Prediction Server**
- You can change `is_seg` to `true` in [setup.sh](setup.sh) to train segmentation instead *(the default is `false`, which is training classification)*

**When training:** Logs and checkpoints are automatically mounted at `%Your-Google-Drive-Root%/log/`

**When start Online Prediction Server:** RestAPI is callable at `/`, you can `tmux` to this machine to get public ip address