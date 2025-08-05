# 1. Setup and Installation

## Create virtual environment

```shell
python3 -m venv .venv
```

## Activate virtual environment:

```shell
source .venv/bin/activate
```

## Install the project dependencies

```shell
pip install -r requirements.txt
```

## Download spacy en core web sm
```shell
python -m spacy download en_core_web_sm
```

## Create util folders

In the root of the project, create the folders `cleaned_texts` and  `pdfs`


## Download the file to analyze

Download the file [PND_2025-2030_v250226_14.pdf](https://drive.google.com/file/d/1O20jR5Bdkof1lZuXCuutjvaUlnDOuG70/view?usp=sharing) and save it into the `data/input/` directory.