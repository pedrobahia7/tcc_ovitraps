import subprocess


notebook = 'NeuralNetworks_1.ipynb'

for i in range(10):
    subprocess.run(['jupyter', 'nbconvert', '--to', 'notebook', '--execute', notebook])
