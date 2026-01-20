# Face Similarity (dlib) — projeto de estudo em Python

Este repositório é um **projeto pessoal/estudo** feito para aprender Python e visão computacional: ele compara **similaridade entre rostos** usando **embeddings** gerados pelo modelo de reconhecimento facial do **dlib** (ResNet).

> Importante: o resultado exibido como “score 0–100” é **heurístico** (um indicador visual). Não é uma probabilidade real nem uma “prova” definitiva.

---

## Por que esse projeto existe?

Eu queria praticar:
- leitura de imagens e pipeline simples de CV (OpenCV)
- detecção de rosto + landmarks (dlib)
- extração de embeddings e cálculo de distância (NumPy)
- organização de um script de linha de comando com argumentos

---

## O que ele faz

- Compara **duas imagens** (1 rosto por imagem) e imprime:
  - distância L2 entre embeddings
  - um “score” 0–100 baseado em threshold (heurístico)

- Compara **duas pastas de imagens** (todas as combinações) e imprime:
  - distância média
  - score médio

---

## Estrutura do projeto

```
.
├─ main.py
├─ requirements.txt
├─ models/                 # coloque aqui os modelos .dat do dlib (não versionar)
└─ examples/
   ├─ person_a/
   └─ person_b/
```

---

## Como usar

### 1) Requisitos
- Python 3.10+ (recomendado)
- Dependências do `requirements.txt`

Instalação:
```bash
pip install -r requirements.txt
```

> Observação: `dlib` pode exigir ferramentas de build em alguns ambientes (especialmente no Windows).
> Se der erro, uma abordagem comum é usar um ambiente com wheels já disponíveis, ou instalar via conda.
> Este projeto é um estudo e não tenta “resolver instalação em todo cenário”.

---

### 2) Baixar os modelos do dlib

Você precisa colocar estes arquivos na pasta `models/`:

- `shape_predictor_68_face_landmarks.dat`
- `dlib_face_recognition_resnet_model_v1.dat`

Depois disso, o script já encontra automaticamente por padrão.

---

### 3) Comparar duas imagens

```bash
python main.py image --img1 examples/person_a/a1.jpg --img2 examples/person_b/b1.jpg
```

Saída esperada (exemplo):
- Distância (L2): 0.52
- Score (heurístico): 13.33

---

### 4) Comparar duas pastas

```bash
python main.py folders --person_a examples/person_a --person_b examples/person_b
```

---

## Como interpretar os resultados (sem “vender milagre”)

- O dlib gera um vetor (embedding) que representa o rosto.
- A **distância L2** entre embeddings é uma medida de diferença:
  - **menor distância** → rostos mais parecidos segundo o modelo
  - **maior distância** → rostos menos parecidos

O “score 0–100” deste projeto é apenas:
```
score = max(0, 1 - distancia/threshold) * 100
```

Ou seja: é um **mapeamento** para facilitar leitura humana, mas não é ciência exata.

---

## Limitações conhecidas

- Sensível a iluminação, ângulo, expressão, qualidade da imagem.
- Este script exige **apenas 1 rosto por imagem** (por simplicidade).
- O “score” não é probabilidade; é só um indicador.

---

## Segurança e privacidade

Este repositório é educacional. Evite usar imagens de terceiros sem permissão.  
Não use este projeto para identificação, vigilância ou qualquer uso inadequado.

---

## English summary 

A small study project in Python that compares face embeddings using dlib + OpenCV.  
It prints the embedding L2 distance and a heuristic 0–100 score (not a real probability).
