# Generate Neural Network Diagrams

Build the docker image:

~~~bash
cd ${HOME}/Breast-Cancer-Segmentation/NeuralNetDiagrams
docker build -t cmpe258_proj:dev .
~~~

Put your python code in `pyexamples/` folder, then run the following commands:

~~~bash
# Clone repo
git clone git@github.com:HarisIqbal88/PlotNeuralNet.git

PATH_TO_NNP_DIR="${HOME}/Breast-Cancer-Segmentation/NeuralNetDiagrams/PlotNeuralNet"
cp $PATH_TO_NNP_DIR/pyexamples/msgrap.py .
~~~

Launch the docker container from `cmpe258_proj:dev` docker image with volume mount to our NeuralNetDiagrams:

~~~bash
cd $PATH_TO_NNP_DIR
docker run --name plotnn --privileged -it -v $PWD:/sjsu/PlotNeuralNet cmpe257_ml:dev

# assuming you just cloned the repo
cd PlotNeuralNet/pyexamples

# run an example: shell script runs your python script, test_simple
# then generates a neural network graph as pdf
bash ../tikzmake.sh test_simple

# ex: copy over your py nn diagram code to pyexamples/

bash ../tikzmake.sh msgrap

# You should see a msgrap.pdf and msgrap.tex be generated
~~~