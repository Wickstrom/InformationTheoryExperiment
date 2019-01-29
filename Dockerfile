FROM pytorch/pytorch:1.0-cuda10.0-cudnn7-runtime

RUN conda install  \
	scikit-learn

WORKDIR /root

ENTRYPOINT []
CMD bash


