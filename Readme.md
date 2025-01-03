# Deepmind Reinforcement Learning re-implementation
## Code base structure based on [Plastic](https://github.com/dojeon-ai/PLASTIC) and [Simba](https://github.com/SonyResearch/simba) (Nice codebases)

## Requirements
We assume you have access to a GPU that can run CUDA 11.1 and CUDNN 8. 
Then, the simplest way to install all required dependencies is to create an anaconda environment by running

```
conda env create -f requirements.yaml
pip install hydra-core --upgrade
pip install opencv-python
```

After the instalation ends you can activate your environment with
```
conda activate atari
```

## Installing Atari environment

### Download Rom dataset
```
python
import urllib.request
urllib.request.urlretrieve('http://www.atarimania.com/roms/Roms.rar','Roms.rar')
```

### Connect Rom dataset to atari_py library
```
apt-get install unrar
unrar x Roms.rar
mkdir rars
mv HC\ ROMS rars
mv ROMS rars
python -m atari_py.import_roms rars
``` 

## Instructions

To run a single run, use the `run.py` script
```
python run.py 
```

To run the Atari-100k benchmark (26 games with 5 random sees), use `run_parallel.py` script
```
python run_parallel.py
```



