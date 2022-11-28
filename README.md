# Breast Cancer Segmentation

Lesion segmentation for breast cancer diagnosis.

## CMPE 258 Project Submission Checklist

- [] 4\. Ten minutes presentation and program demo:

    - [ ] (4.1) PPT (up to 5-7 slides) for 5-7 minutes presentation;
    - [ ] (4.2) Demo, 1 minute;
    - [ ] (4.3) Code walk-through for 1-2 minutes;
    - [ ] (4.4) Q&A, 1 minute.

- [ ] 5\. Save up to 20 ~ 50 seconds demo video into a file for submission.
- [ ] 6. Submit:

    - [ ] a. Executive Summary;
    - [ ] b. PPT;
    - [ ] c. Your saved video clip;
    - [ ] d. The program package (source code and all relevant files and folders);
    - [ ] e. A readme file. Be sure detailed adequate information is provided for testing and verification purpose.

- [ ] 7. Put all the above files into one file and zip it.
- [ ] 8. Use the following file naming convention for your zip file:

    - [ ] firstNamePerson1_firstNamePerson2_FirstNamePerson3_FirstNamePerson4_CoordinatorSID(last-4-digits)_cmpe258_team.zip.
    - [ ] Ex: `Yoonjun_Choi_Omkar_Suryakant_Naik_Archil_Beridze_James_Medel_6649_cmpe258_team.zip`

- [ ] Submit it to the class canvas.


## Outline

## Software Dependencies

- Rasa 3.2.4 (Rasa Chatbot Server)
- Rasa SDK 3.2.2 (Rasa Actions Server)
- TensorFlow-GPU 2.7.1 (Needed for Rasa Actions Server)
- Torch 1.13.0 (Needed for Rasa Actions Server)
- OpenCV-Python 4.6.0.66 (Needed for Rasa Actions Server)
- Scikit-Learn 0.24.2
- Seaborn 0.11.2
- Matplotlib 3.3.4
- Jupyter 1.0.0

**conda environment yml files**

TODO: Update environment yml files since I am missing some packages

Create a `environment_rasa_tf_cpu.yml` conda environment:

~~~bash
conda env create -f environment_rasa_tf_cpu.yml
~~~

Create a `environment_rasa_tf_gpu.yml` conda environment:

~~~bash
conda env create -f environment_rasa_tf_gpu.yml
~~~

1\. Clone this repo:

~~~bash
git clone https://github.com/AI-Medical-Robotics/Breast-Cancer-Segmentation.git
~~~

2\. Go to breast cancer segmentation project:

~~~bash
cd path/to/Breast-Cancer-Segmentation
~~~

## Setup Software Dev Environment

1\. Build rasa chatbot docker container:

~~~bash
cd Breast-Cancer-Segmentation
docker build -t rasa_3.1.0_rasa_sdk_3.1.1:dev .
~~~

## How to Run Demo

## Docker Compose to Auto Launch Rasa-Run Server & Actions Server

TODO: Create docker-compose.yml file

## Auto Launch Rasa-Run Server & Actions Server

TODO: Create a shell script to auto launch both rasa servers in new terminals

### Run Rasa Breast Cancer Diagnosis in Docker

1\. Launch rasa chatbot container:

~~~bash
docker run --name rasa-chatbot --gpus all -it --privileged -v C:\Users\JamesMedel\GitHub\Breast-Cancer-Segmentation\diagnosis_va:/app rasa_3.1.0_rasa_sdk_1.1.1:dev
~~~

2\. Train the Rasa chatbot:

~~~bash
cd app/
rasa train --domain domain.yml --data data --out models
~~~

3\. Now you're inside the container, run rasa chatbot:

~~~bash
cd /app
rasa shell
~~~

### Run Rasa Breast Cancer Diagnosis Natively

If you just cloned this repo, on a fresh environment, you will need to train Rasa chatbot:

~~~bash
cd diagnosis_va/
rasa train --domain domain.yml --data data --out models
~~~

Jump into Rasa shell

~~~bash
cd diagnosis_va/
rasa shell
~~~


