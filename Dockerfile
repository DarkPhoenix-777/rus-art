FROM python:3.10
WORKDIR /app

COPY . /app

VOLUME /app/data
RUN pip3 install -r requirements.txt

RUN python3 -c "import timm; timm.create_model('eva02_base_patch14_448.mim_in22k_ft_in1k', pretrained=True, num_classes=0)"


RUN chmod +x /app/make_submission.py
