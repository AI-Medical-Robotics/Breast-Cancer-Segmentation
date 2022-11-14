# Breast Cancer Segmentation

Lesion segmentation for breast cancer diagnosis.

## Outline

## Software Dependencies

- Rasa 3.1.0 (Rasa Chatbot Server)
- Rasa 3.1.1 SDK (Rasa Actions Server)
- TensorFlow-GPU 2.7.3 (Needed for Rasa Actions Server)
- OpenCV-Python 4.5.5.64 (Needed for Rasa Actions Server)
- Scikit-Learn 0.24.2
- Seaborn 0.11.2
- Matplotlib 3.5.1
- Pandas 1.3.4
- Jupyter 1.0.0
- Ngrok version latest

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

1\. Launch rasa chatbot container:

~~~bash
docker run --name rasa-chatbot --gpus all -it --privileged -v C:\Users\JamesMedel\GitHub\Breast-Cancer-Segmentation\diagnosis_va:/app rasa_3.1.0_rasa_sdk_1.1.1:dev
~~~

2\. Now you're inside the container, run rasa chatbot:

~~~bash
cd /app
rasa shell
~~~



