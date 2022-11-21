FROM continuumio/anaconda3:latest

# conda install: cudnn opencv cudatoolkit==10.1.243

# RUN apt-get -y update
# RUN apt-get -y install ffmpeg libsm6 libxext6
# RUN conda install -y jupyter
# rasa also installs matplotlib tensorflow==2.7.1
# RUN pip install rasa==3.1.0 rasa-sdk==3.1.1 scikit-learn scrapy scrapydo lxml tqdm absl-py easydict pillow opencv-python

COPY environments/attent_unet_env/gpu/environment_rasa_tf_gpu.yml .
RUN conda env create -f environment_rasa_tf_gpu.yml

EXPOSE 5005
ENTRYPOINT ["/bin/bash"]