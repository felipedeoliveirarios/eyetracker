# Rastreador de Foco Visual em Python

Este repositório contém uma implementação do trabalho "Gaze determination via images of irises", por J.-G. Wang e E. Sung, publicado em janeiro de 2001, usando OpenCV, dlib e python.

A idéia consiste em segmentar rostos à partir da imagem, olhos à partir do rosto, e as íris à partir dos olhos. Em seguida, a íris é modelada como uma elipse. Esta elipse é assumida como sendo a projeção de um disco plano rotacionado, onde o ângulo de rotação do disco corresponde ao ângulo de rotação do olho.

