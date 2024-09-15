#FROM python:3.9-slim
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /input /output/images/automated-petct-lesion-segmentation \
    && chown -R algorithm:algorithm /opt/algorithm /input /output

USER algorithm

WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"

RUN python -m pip install --user -U pip

COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
COPY --chown=algorithm:algorithm monai_unet.py /opt/algorithm/
RUN python -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
RUN python -m pip install --user -r requirements.txt

COPY --chown=algorithm:algorithm fold0_model_ep=0284.pth /opt/algorithm/
COPY --chown=algorithm:algorithm fold1_model_ep=0300.pth /opt/algorithm/
COPY --chown=algorithm:algorithm fold2_model_ep=0368.pth /opt/algorithm/
COPY --chown=algorithm:algorithm fold3_model_ep=0252.pth /opt/algorithm/
COPY --chown=algorithm:algorithm fold4_model_ep=0312.pth /opt/algorithm/

COPY --chown=algorithm:algorithm process.py /opt/algorithm/

ENTRYPOINT python -m process $0 $@
