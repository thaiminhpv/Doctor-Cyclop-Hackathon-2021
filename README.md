# Doctor's Cyclop Model

**Doctor's Cyclop** is one project of the **Phoenix team** in **FPT Edu Hackathon 2021**. This repository contains code of the model implementation. For the Web App, view 
[Doctors-Cyclop-Webapp](https://github.com/DAN3002/Doctors-Cyclop-Webapp)

## Introduction

This is a Model implementation integrated with Web App, the dataset is got from the [RANZCR CLiP - Catheter and Line Position Challenge](https://www.kaggle.com/c/ranzcr-clip-catheter-line-classification)

## How to Build and Run

### 1. Mount folders at Google Drive

- Create shortcut (mount) to this [public folder](https://drive.google.com/drive/folders/1XxKDVzCms_O6UVG2zNKvmwV0rNtTmtIq?usp=sharing), add at your Google Drive's root (`/`):

After this you should now have this at your Google Drive's root:

![root google drive structure](readme-assets/images/Google-Drive-Structure.png)

### 2. SSH to server

- Run [this notebook](https://colab.research.google.com/drive/1L-ibyztYYcM0rmuXkPihN8LHP0TxkRi4?usp=sharing) with GPU enabled to create a Google Colab virtual machine
- Copy file [setup.sh](setup.sh) or [restapi-setup.sh](restapi-setup.sh) to the newly created virtual machine (by `scp`)

### 3. Run

- `ssh` to the created machine *(default password is `haha`)*, then run **either**:
  - [restapi-setup.sh](restapi-setup.sh) to start **Online Prediction Server** as a RestAPI service
  - [setup.sh](setup.sh) for **Training**:
    - The default is training *Classification*
    - To train *Segmentation* instead, change `is_seg` to `true` in [setup.sh](setup.sh)

### 4. Note:
- **When training:** Logs and checkpoints are automatically mounted at `%Your-Google-Drive-Root%/log/`

- **When start Online Prediction Server as a service:** RestAPI is callable at `/`, you can `tmux` to this remote machine to get public API endpoint

![Online Prediction](readme-assets/images/RestAPI-Online-Prediction.png)
