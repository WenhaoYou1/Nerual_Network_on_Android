# Deployment of Frequency Regularization Models on Android Devices

In this repository, we introduce two methods to deploy Frequency Regularization Models on Android devices: set up Linux environment and develop an android application.

## Introduction

### Repository Structure

```
.
├── README.md
├── future_work
│   └── future_plan.pdf
├── class_presentation
│   └── DeploymentOnMobile.pptx
├── proposal_and_literature_review
│   └── ...
├── report
│   ├── IEEE_format
│   ├── spring_conference_format
│   └── raw_data
│   │   ├── 1st_data
│   │   └── ...
└── source_code_frequency_regularization
    └── ...
```

## Getting Started

### Set Up Linux Environment on Android Devices

1. Start an Android emulator (e.g. [Genymotion](https://dl.genymotion.com/releases/genymotion-3.5.1/genymotion-3.5.1.dmg)) and choose Android vesion == 12.0.x.

2. Install [Termux](https://github.com/termux/termux-app/releases/download/v0.118.0/termux-app_v0.118.0+github-debug_universal.apk) to deploy the Linux environment. **DONOT** download Termux on Google Play.

3. Update termux, install wget, proot, and git.

   ```sh
   apt-get update && apt-get upgrade -y
   apt-get install wget -y
   apt-get install proot -y
   apt-get install git -y
   ```

4. Download a script to deploy ubutu in termux.

   ```sh
   git clone https://github.com/MFDGaming/ubuntu-in-termux.git
   ```

5. Move to script folder and give its execution permission.

   ```sh
   cd ubuntu-in-termux
   chmod +x ubuntu.sh
   ```

6. Run the script and start ubuntu environment.

   ```sh
   ./ubuntu.sh -y
   ./startubuntu.sh
   ```

7. Run the system update command to refresh the APT repositories cache. This is necessary because many times after installing a fresh minimal Debian or Ubuntu Linux, it won’t recognize any packages to install. It is because there is no list of packages in the cache, that the system can identify to install.

   ```sh
   apt update && apt upgrade
   ```

8. Install sudo on ubuntu server.

   ```sh
   apt install sudo
   ```

9. Install Python and python-pip tool to deploy our Frequency Regulazation package quickly. Use `python --version` to check its version should be larger or same to 3.10.12.

   ```sh
   sudo apt install python3
   sudo apt-get -y install python3-pip
   ```

10. Install Frequency Regulazation package. You will see `Successfully installed frereg-0.1.0` while deploying it well.
    ```sh
    pip3 install frereg
    ```

### Run Android Application on Android Studio

1. Download an Android emulator (e.g. [Android Studio Hedgehog](https://developer.android.com/studio)) and choose version == 2023.1.1.x.

2. Download our source code.

3. Make sure the project structure is correct and rebuild the gradle file.

   ```
   Android Gradle Plugin Version: 7.1.0
   Gradle Version: 7.2
   ```

4. Run the source code and select the image you want to implement segmentation.

## Current and Future Plans

| Milestones                                           | Status       |
| ---------------------------------------------------- | ------------ |
| Package Up Frequency Regularization                  | ✔️ Completed |
| Implement Linux Envrionment                          | ✔️ Completed |
| Implement Python Library on Android                  | ✔️ Completed |
| Develop Android Application with FR                  | ✔️ Completed |
| Develop Advanced User Interfaces and Functionalities | 🔜 Upcoming  |
| Expand More Models not only U-Net                    | 🔜 Upcoming  |

## Acknowledgement

(Wenhao You and Leo Chang contributed equally to this work.)

## Citation

If you find our deployment of Frequency Regularization technique on Android Devices or utilize it in your research, we kindly encourage you to cite our paper:

```bibtex
@ARTICLE{fr_android,
  author={You, Wenhao and Chang, Leo and Dong, Guanfang, and Basu, Anup},
  title={Deployment of Frequency Regularization on Android Devices},
  year={2023},
}
```
