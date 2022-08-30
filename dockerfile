FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

RUN \
  DEBIAN_FRONTEND=noninteractive apt-get update && \ 
  DEBIAN_FRONTEND=noninteractive apt-get install -y openssh-server && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y vim python3 python3-pip && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y tmux && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y net-tools && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y lsof && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y libgl1-mesa-glx
   
RUN \
  pip3 install --upgrade pip && \
  pip3 install setuptools>=40.3.0 && \
  pip3 install -U scipy scikit-learn && \
  pip3 install yacs && \
  pip3 install pytorch_toolbelt && \
  pip3 install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio===0.10.1 -f https://download.pytorch.org/whl/torch_stable.html && \
  pip3 install torchsummary && \
  pip3 install matplotlib==3.3.4 && \
  pip3 install numpy==1.17.5 && \
  pip3 install tqdm==4.57.0 && \
  pip3 install imageio==2.9.0 && \ 
  pip3 install pandas==0.24.2 && \
  pip3 install Pillow==8.2.0 && \
  pip3 install scikit-image && \
  pip3 install tensorboardX==2.2 && \ 
  pip3 install tensorboard && \
  pip3 install tensorwatch==0.9.1 
  
