# Setup and Installation

## Pre-requisities
- [Python](https://www.python.org/downloads/) is installed
- [Virtual enviroment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) for python is installed

## 1. Create virtual environment

```shell
python3 -m venv .venv
```

## 2. Activate virtual environment:

**For linux and Mac:**

```shell
source .venv/bin/activate
```

**For Windows:**
The correct command to activate a virtual environment on Windows depends on the terminal you're using.

For Command Prompt (CMD): Use the .bat file.

```shell
.venv\Scripts\activate.bat
```
For PowerShell: Use the .ps1 file.

```shell
.venv\Scripts\Activate.ps1
```

## 3. Install the project dependencies

```shell
pip install -r requirements.txt
```

## 4. Download spacy en core web sm

```shell
python -m spacy download en_core_web_sm
```

## 5. Create util folders

In the root of the project, create the folders `cleaned_texts` and  `pdfs`


## 6. Download the file to analyze

Download the file [PND_2025-2030_v250226_14.pdf](https://drive.google.com/file/d/1O20jR5Bdkof1lZuXCuutjvaUlnDOuG70/view?usp=sharing) and save it into the `data/input/` directory.