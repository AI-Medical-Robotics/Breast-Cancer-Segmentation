# Breast Cancer Segmentation

Lesion segmentation for breast cancer diagnosis.

## Outline

## Software Dependencies

- Rasa 3.1.0 (Rasa Chatbot Server)
- Rasa 3.1.1 SDK (Rasa Actions Server)
- TensorFlow-GPU 2.7.1 (Needed for Rasa Actions Server)
- OpenCV-Python 4.6.0.66 (Needed for Rasa Actions Server)
- Scikit-Learn 0.24.2
- Seaborn 0.11.2
- Matplotlib 3.3.4
- Pandas 1.3.4
- Jupyter 1.0.0
- Ngrok version latest

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

2\. Now you're inside the container, run rasa chatbot:

~~~bash
cd /app
rasa shell
~~~

### Run Rasa Breast Cancer Diagnosis Natively

If you just cloned this repo, on a fresh environment, you will need to train Rasa chatbot:

~~~bash
cd diagnosis_va/
rasa train --domain domain.yl --data data --out models
~~~

Jump into Rasa shell

~~~bash
cd diagnosis_va/
rasa shell
~~~


