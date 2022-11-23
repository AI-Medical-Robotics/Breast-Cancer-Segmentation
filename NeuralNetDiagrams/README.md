# Generate Neural Network Diagrams

Clone repo: `git clone git@github.com:HarisIqbal88/PlotNeuralNet.git`

Put your python code in `pyexamples/` folder, then run the following commands:

~~~bash
PATH_TO_NNP_DIR="${HOME}/Breast-Cancer-Segmentation/NeuralNetDiagrams"
# assuming you just cloned the repo
cd PlotNeuralNet/pyexampless

# run an example: shell script runs your python script, test_simple
# then generates a neural network graph as pdf
bash ../tikzmake.sh test_simple

# ex: copy over your py nn diagram code to pyexamples/
cp $PATH_TO_NNP_DIR/msgrap.py .
bash ../tikzmake.sh msgrap

# You should see a msgrap.pdf and msgrap.tex be generated
~~~