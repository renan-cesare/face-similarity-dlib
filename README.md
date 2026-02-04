# Face Similarity Embeddings (dlib + OpenCV)

Projeto de estudo em Python para comparar similaridade entre rostos usando embeddings (dlib + OpenCV), com execução via linha de comando para comparação entre duas imagens ou entre duas pastas de imagens.

> English (short): Study project in Python to compare face similarity using dlib embeddings (ResNet) and OpenCV, with CLI commands for image-to-image and folder-to-folder comparison.

---

## Principais recursos

* Extração de embeddings faciais com dlib (ResNet)
* Detecção de rosto e alinhamento por landmarks (shape predictor)
* Comparação por **distância L2** entre embeddings
* Modo de execução via CLI:

  * `image`: compara 2 imagens (1 rosto por imagem)
  * `folders`: compara duas pastas (todas as combinações) e retorna média
* Score heurístico 0–100 baseado em threshold configurável (não é probabilidade)

---

## Contexto

Em cenários de visão computacional, embeddings faciais são utilizados para representar rostos como vetores numéricos, permitindo medir “proximidade” entre duas imagens por meio de uma métrica (ex.: distância L2).

Este repositório foi criado como **projeto de estudo**, com foco em:

* aprender o pipeline (detecção → landmarks → alinhamento → embedding)
* testar comparação entre imagens e conjuntos simples
* praticar organização mínima de projeto e execução via CLI

---

## Aviso importante (uso autorizado)

Este repositório é apresentado como exemplo técnico/portfólio.

* **Não utilize** este projeto para fins de vigilância, identificação indevida ou qualquer uso que viole privacidade
* Use apenas **imagens e ambientes autorizados**
* Evite utilizar dados pessoais reais — biometria é dado sensível (LGPD)
* Este projeto não foi desenhado para produção (robustez, governança e controles são limitados)

---

## Estrutura do projeto

```
.
├─ examples/
│  └─ .gitkeep
├─ models/
│  └─ .gitkeep
├─ main.py
├─ requirements.txt
├─ LICENSE
└─ README.md
```

---

## Requisitos

* Python 3.10+
* dlib (pode exigir dependências nativas/compilação dependendo do ambiente)
* OpenCV
* Modelos do dlib (arquivos `.dat`) **baixados localmente**:

  * `shape_predictor_68_face_landmarks.dat`
  * `dlib_face_recognition_resnet_model_v1.dat`

> Observação: este projeto espera esses arquivos na pasta `models/` por padrão.

---

## Instalação

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Configuração (modelos)

Coloque os arquivos `.dat` dentro de `models/`:

* `models/shape_predictor_68_face_landmarks.dat`
* `models/dlib_face_recognition_resnet_model_v1.dat`

Por padrão, o projeto já aponta para esses caminhos. Se quiser customizar, use os flags:

* `--predictor`
* `--encoder`

---

## Execução

### Comparar duas imagens

```bash
python main.py image --img1 "caminho/para/img1.jpg" --img2 "caminho/para/img2.jpg"
```

### Comparar duas pastas

```bash
python main.py folders --person_a "caminho/para/pasta_a" --person_b "caminho/para/pasta_b"
```

### Ajustar threshold (score heurístico)

```bash
python main.py image --img1 "img1.jpg" --img2 "img2.jpg" --threshold 0.6
```

O processo:

* carrega os modelos do dlib
* gera embeddings das imagens
* calcula a distância L2 entre vetores
* imprime distância e score (0–100, heurístico)

---

## Saídas geradas

* Saída no console com:

  * distância L2
  * score heurístico (0–100)

Este projeto não gera arquivos automaticamente.

---

## Sanitização de dados

Este repositório não contém dados reais.

* imagens de teste devem permanecer fora do Git (recomendado)
* arquivos de modelo `.dat` devem permanecer fora do versionamento
* evite commitar qualquer amostra com dados pessoais reais

---

## Licença

MIT
