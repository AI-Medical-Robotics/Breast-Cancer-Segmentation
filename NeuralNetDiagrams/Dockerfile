# Jupyter Deep Learning Notebook
FROM continuumio/anaconda3:latest

RUN apt-get -y update

RUN wget https://github.com/jgm/pandoc/releases/download/2.17.1.1/pandoc-2.17.1.1-1-amd64.deb
RUN dpkg -i pandoc-2.17.1.1-1-amd64.deb
RUN apt-get install -y texlive-latex-base texlive-xetex texlive-fonts-recommended texlive-fonts-extra texlive-plain-generic texlive-latex-extra

RUN mkdir /sjsu

# Go into working dir /sjsu
WORKDIR /sjsu

# copy dev host cmpe258_proj/ folder to working dir in container
COPY . ./cmpe258_proj

# start jupyter notebook
CMD ["/bin/bash"]