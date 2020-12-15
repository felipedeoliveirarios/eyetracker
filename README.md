# Rastreador de Foco Visual em Python
Este repositório contém uma implementação do trabalho *Gaze determination via images of irises*[[1]](#1), por J.-G. Wang e E. Sung, publicado em janeiro de 2001, usando OpenCV, dlib e python.

A idéia consiste em segmentar rostos à partir da imagem, olhos à partir do rosto, e as íris à partir dos olhos. Em seguida, a íris é modelada como uma elipse. Esta elipse é assumida como sendo a projeção de um disco plano rotacionado, onde o ângulo de rotação do disco corresponde ao ângulo de rotação do olho.

## Segmentação
A segmentação do rosto é feita usando o detector de rostos incluso no dlib, obtido através da chamada `get_frontal_face_detector()` e usado através da chamada como método do objeto retornado. O objeto detecta rostos através de um *sliding window classificator*.

A segmentação dos olhos é realizada através de um objeto `shape_predictor`, que distribui vértices por todo o rosto, seguindo a implementação do artigo *One Millisecond Face Alignment with an Ensemble of Regression Trees*[[2]](#2), por Vahid Kazemi e Josephine Sullivan. Os vértices que compõem o contorno dos olhos são então encaixados em retângulos, contendo as imagens dos olhos. Também, usando o formato sugerido dos olhos gerado pelos vértices distribuídos, é criada uma máscara que reduz a área de trabalho do próximo passo da segmentação. Os vértices distribuídos seguem o seguinte diagrama:

![Diagrama de vértices distribuídos sobre a face.](https://github.com/felipedeoliveirarios/eyetracker/blob/main/facial_landmarks.jpg)

A segmentação das íris é feita através da equalização do histograma da imagem, seguido por uma binarização de otsu e operações morfológicas de abertura e fechamento, e, por fim, inversão de cores. Em seguida, é encontrada a maior área em branco restante na imagem (que assume-se que será a íris) e as demais áreas serão excluídas usando uma máscara que contém apenas a maior área detectada.

O artigo de J.-G. Wang et. al. especifica também o uso de um operador linear de borda vertical 3x3, que deixaria apenas as bordas laterais da íris, que seriam posteriormente utilizadas para determinar a elipse correspondente. Este componente foi implementado, mas não é utilizado na iteração atual, dado que o método de detecção da elipse se baseia no contorno geral da íris segmentada. Reconhece-se essa divergência como a provável causa da baixa precisão final do resultado.

## Assunções sobre os ângulos
A rotação dos globos oculares dentro das órbitas se dá nos eixos x e y; porém, no modelo utilizado nesta implementação, as rotações consideradas são sobre os eixos y e z, chamadas respectivamente de *slant* e *tilt*, onde o *slant* é aplicado sempre antes do *tilt*. Isso é feito porque o método que calcula as elipses retorna também a rotação da elipse, que equivale ao *tilt*, e o *slant* pode ser calculado como sendo arccos(min(R, r)/max(R, r)), onde, caso R seja o eixo vertical da elipse, o *tilt* deve ser aumentado em 90°. O diagrama a seguir ilustra o uso de *tilt* e *slant* sobre um círculo, e as elipses geradas como resultado da projeção.
![Diagrama de Tilt e Slant](https://github.com/felipedeoliveirarios/eyetracker/blob/main/slant_tilt_diagram.png)

## Requisitos
Embora boa parte dos requisitos estejam listados no arquivo `requirements.txt`, os requisitos do `dlib` não estarão na lista. Os requisitos não-listados são:
- Python 3
- Boost
- Boost.Python
- CMake
- X11/XQuartz

Todos podem ser instalados usando a seguinte sequência (para Ubuntu e similares):
```bash
sudo apt-get install python3
sudo apt-get install build-essential cmake
sudo apt-get install libgtk-3-dev
sudo apt-get install libboost-all-dev
```
Para guias de instalação para outros sistemas operacionais, clique [aqui](https://www.pyimagesearch.com/2018/01/22/install-dlib-easy-complete-guide/). Após clonar o repositório, é possível instalar os pacotes necessários usando o pip com o comando ```pip install -r requirements.txt```. A instalação do dlib pode demorar um pouco. Talvez seja conveniente fazer a instalação dos pacotes pip em um ambiente virtual.

## Executando o Projeto
Para executar o projeto, com as dependências devidamente instaladas, use o comando ```python main.py <caminho_da_imagem>```. O projeto inclui imagens de teste na pasta test.

## Referências
<a id="1">[1]</a> 
WANG, Jian-Gang; SUNG, Eric.
**Gaze determination via images of irises**.
Image and Vision Computing 19, 2001. p. 891-911

<a id="2">[2]</a> 
KAZEMI, Vahid; SULLIVAN, Josephine. 
**One Millisecond Face Alignment with an Ensemble of Regression Trees**. 
Proceedings of the IEEE conference on computer vision and pattern recognition, 2014. p. 1867-1874.
