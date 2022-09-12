docker run \
--gpus "device=0"\
-it\
-v /home/cosimorulli/CompressMLP-Mixer\
cosimorulli/prova-env:v1 \
python trainer.py --help


docker run --gpus '"device:0"' -v /home/francomarianardini/CompressMLP-Mixer:/code cosimorulli/comp_mlp:0.1 bash train.sh


docker build -t cosimorulli/comp_mlp:0.1 -< DOCKERFILE