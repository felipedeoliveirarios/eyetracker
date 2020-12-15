'''
Implementação de um eyetracker, baseado em cv2 e dlib.

O detector de faces baseado em redes neurais incluso no dlib é utilizado para ,
a princípio, encontrar o rosto, e, em seguida, localizar os olhos. A partir daí,
utiliza-se o método proposto por J.-G. Wang e E. Sung em "Gaze determination 
via images of irises" (Jan. 2001)

O método proposto consiste em encontrar as bordas das íris na imagem e fazer 
uso de um algoritmo de ellipse-fitting para encontrar uma elipse que corresponda
ao formato aparente da íris. Em seguida, a elipse é assumida como um círculo 
inclinado, e calculando-se essa inclinação, obtém-se uma boa aproximação do
ângulo de rotação dos olhos. Extendendo-se as normais dos planos relativos aos
círculos, obtém-se, no ponto de interseção das normais, uma estimativa do foco
visual. 

Por Felipe Rios
'''

import cv2
import dlib
import numpy as np
import argparse
import os.path
import pyautogui

right_eye_landmarks = [36, 37, 38, 39, 40, 41]
left_eye_landmarks = [42, 43, 44, 45, 46, 47]

def shape_to_np(shape, dtype="int"):
	'''Converte um conjunto de dlib.points em uma lista de tuplas.'''
	coords = np.zeros((68, 2), dtype=dtype)
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	return coords

def fit_inside_box(points, extra = 0):
	'''
	Encontra um retângulo que contenha um conjunto de vértices.

	Parâmetros:
		points (list): lista de vértices contidos no retângulo
		extra (int): tamanho extra dos limites da caixa

	Retorno:
		tupla<<int, int>, <int, int>>: uma tupla contendo duas outras tuplas, com os vértices dos cantos superior esquerdo e inferior direito.
	'''
	max_x, max_y = points[0]
	min_x, min_y = points[0]

	for point in points:
		if point[0] > max_x:
			max_x = point[0]
		if point[1] > max_y:
			max_y = point[1]
		if point[0] < min_x:
			min_x = point[0]
		if point[1] < min_y:
			min_y = point[1]

		max_x += extra
		min_x -= extra
		max_y += extra
		min_y -= extra

	return (int(min_x), int(min_y)), (int(max_x), int(max_y))

def filter_eye_image(image, mask=None):
	'''
	Aplica filtros sobre a imagem de um olho de forma a segmentar apenas a íris.

	Parâmetros:
		image (np.array): array contendo os pixels da imagem
		mask (np.array): array contendo a máscara da região de interesse

	Retorno:
		np.array: Imagem filtrada, contendo apenas a região de interesse.
	'''

	# Obtém as dimensões da imagem.
	w, h, _ = image.shape

	# Define as matrizes a serem utilizadas nas operações morfológicas.
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
	kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
	kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
	vertical_operator = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

	# Converte a imagem p/ escala de cinza e equaliza o histograma.
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = cv2.equalizeHist(image)

	# Binariza a imagem usando o limite de Otsu.
	_, image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)

	# Aplica operações morfológicas para reduzir a quantidade de informação
	image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
	image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel3)
	image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
	image = cv2.bitwise_not(image)
	image = cv2.bitwise_and(image, image, mask=cv2.erode(mask, kernel2))
	image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

	# Encontra os contornos de todas as áreas brancas.
	contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	# Calcula as áreas de todos os contornos.
	areas = [cv2.contourArea(c) for c in contours]
	# Escolhe o maior contorno
	max_index = np.argmax(areas)

	# Cria uma máscara que mantém apenas o maior contorno
	mask_hole = contours[max_index]
	mask = np.zeros((w, h), np.uint8)
	mask = cv2.fillConvexPoly(mask, mask_hole, 255)

	# Aplica a máscara e mantém apenas a maior área branca na imagem (a íris)
	image = cv2.bitwise_and(image, image, mask=mask)

	'''
	# Separa as duas bordas laterais da íris.
	image_left = cv2.filter2D(image, -1, -vertical_operator)
	image_right = cv2.filter2D(image, -1, vertical_operator)
	image = image_left + image_right
	'''

	return image

def find_ellipses(contours):
	'''
	Dado um conjunto de contornos, encontra elipses que se aproximam do formato.

	Parâmetros:
		contours (np.array<np.array>): os contornos a serem analisados.

	Retorno:
		lista<cv.rotatedRect>: Lista contendo os dados das elipses.
	'''
	ellipses = []
	if len(contours) != 0:
		for cont in contours:
			if len(cont) < 5:
				break
			elps = cv2.fitEllipse(cont)
			ellipses.append(elps)
	return ellipses

def auto_resize(image):
	'''
	Reduz automaticamente o tamanho de uma imagem para facilitar a visualização.

	Parâmetros:
		image (np.array): Imagem a ser ajustada

	Retorno:
		np.array: Imagem ajustada.
	'''
	# Obtém as dimensões da tela.
	s_width, s_height = pyautogui.size()

	# Ajusta as dimensões para comparação e cálculo.
	s_width *= 0.8
	s_height *= 0.8

	# Obtém as dimensões da imagem.
	i_height, i_width = image.shape[0], image.shape[1]

	# Redimensiona a imagem como necessário.
	if i_width > s_width or i_height > s_height:
		rate = 1
		if i_width > s_width:
			rate = s_width/i_width
		else:
			rate = s_height/i_height

		new_width = int(i_width * rate)
		new_height = int(i_height * rate)

		image = cv2.resize(image, (new_width, new_height))

	return image

def run(imagePath, show=False, output=None):
	'''
	Segmenta as íris da imagem e tenta aplicar o método proposto por J.-G. Wang
	et. al. para determinar o ângulo de rotação dos olhos.

	Parâmetros:
		imagePath (string): caminho válido para uma imagem
		show (bool): flag de exibição dos resultados
		output (string): local de salvamento da saída
	'''
	# Carrega a imagem
	img = cv2.imread(imagePath)

	# Testa se foi possível carregar a imagem
	if img.size == 0:
		print('Image file not found.')
		return

	# Cria cópias da imagem para visualização.
	original = img.copy()
	preview = img.copy()

	# Converte a imagem para escala de cinza.
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# Carrega o detector de rostos do dlib.
	detector = dlib.get_frontal_face_detector()
	# Detecta rostos individuais na imagem.
	rects = detector(gray, 1)
	print("{} faces detected.".format(len(rects)))

	# Carrega o modelo pré-treinado de predição de formas.
	predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

	'''
	O modelo tentará alocar vértices sobre os rostos encontrados na imagem. Para
	informações sobre as localizações dos vértices quanto ao rosto, vide o
	arquivo facial_landmarks.jpg
	'''

	for (i, rect) in enumerate(rects):

		# Separa os cantos do retângulo contendo o rosto para fácil acesso.
		rect_tl_corner = (rect.left(), rect.top())
		rect_br_corner = (rect.right(), rect.bottom())
		# Desenha o retângulo contendo o rosto em azul (cor BGR).
		cv2.rectangle(preview, rect_tl_corner, rect_br_corner, (255, 0, 0), 2)

		# Gera previsões com o modelo, encontrando pontos específicos no rosto.
		shape = predictor(gray, rect)
		# Converte os pontos encontrados em arrays numpy.
		shape = shape_to_np(shape)

		# Cria listas para armazenar os pontos relativos a ambos os olhos.
		left_eye = []
		right_eye = []

		# Itera sobre os vértices encontrados pelo modelo.
		for j, (x, y) in enumerate(shape):
			# Desenha todos os vértices no rosto.
			cv2.circle(preview, (x, y), 4, (0, 0, 255), -1)

			# Adiciona os vértices relativos aos olhos nas respectivas listas.
			if j in left_eye_landmarks:
				left_eye.append([x, y])

			elif j in right_eye_landmarks:
				right_eye.append([x, y])

		# Encontra os retângulos externos que contém os olhos.
		lb_start, lb_end = fit_inside_box(left_eye)
		rb_start, rb_end = fit_inside_box(right_eye)

		# Desenha os retângulos externos dos olhos na imagem.
		cv2.rectangle(preview, lb_start, lb_end, (0, 255, 0), 2)
		cv2.rectangle(preview, rb_start, rb_end, (0, 255, 0), 2)

		# Cria uma máscara que contém branco apenas nos olhos.
		mask = np.zeros(img.shape[:2], np.uint8)
		mask = cv2.fillConvexPoly(mask, np.array(left_eye), 255)
		mask = cv2.fillConvexPoly(mask, np.array(right_eye), 255)
		mask = cv2.dilate(mask, np.ones((9, 9), np.uint8))

		# Segmenta as imagens dos olhos.
		left_eye_img = img[lb_start[1]:lb_end[1], lb_start[0]:lb_end[0]]
		right_eye_img = img[rb_start[1]:rb_end[1], rb_start[0]:rb_end[0]]

		# Segmenta as máscaras dos olhos.
		left_eye_mask = mask[lb_start[1]:lb_end[1], lb_start[0]:lb_end[0]]
		right_eye_mask = mask[rb_start[1]:rb_end[1], rb_start[0]:rb_end[0]]

		# Aplica uma série de filtros às imagens dos olhos.
		left_eye_img = filter_eye_image(left_eye_img, left_eye_mask)
		right_eye_img = filter_eye_image(right_eye_img, right_eye_mask)

		# Procura os contornos das íris nas imagens dos olhos.
		left_eye_contours, _ = cv2.findContours(left_eye_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		right_eye_contours, _ = cv2.findContours(right_eye_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		# Tenta encontrar elipses entre os contornos dos olhos.
		'''
		Este método é diferente do proposto por J.-G. Wang el. al., e essa é a
		provável causa da falta de acurácia da aplicação.
		'''
		left_eye_ellipses = find_ellipses(left_eye_contours)
		right_eye_ellipses = find_ellipses(right_eye_contours)

		# Troca o esquema de cores das imagens dos olhos de volta para colorida.
		left_eye_img = cv2.cvtColor(left_eye_img, cv2.COLOR_GRAY2BGR)
		right_eye_img = cv2.cvtColor(right_eye_img, cv2.COLOR_GRAY2BGR)

		# Desenha os contornos nas imagens dos olhos.
		cv2.drawContours(left_eye_img, left_eye_contours, -1, (0, 0, 255), 1)
		cv2.drawContours(right_eye_img, right_eye_contours, -1, (0, 0, 255), 1)

		# Desenha as elipses nas imagens dos olhos e do rosto.
		for ellipse in left_eye_ellipses:
			cv2.ellipse(left_eye_img, ellipse, (0, 255, 0), 1)
			cv2.ellipse(original[lb_start[1]:lb_end[1], lb_start[0]:lb_end[0]], ellipse, (0, 255, 0), 1)

		for ellipse in right_eye_ellipses:
			cv2.ellipse(right_eye_img, ellipse, (0, 255, 0), 1)
			cv2.ellipse(original[rb_start[1]:rb_end[1], rb_start[0]:rb_end[0]], ellipse, (0, 255, 0), 1)

		# Separa as elipses das íris
		left_eye_ellipse = left_eye_ellipses[0]
		right_eye_ellipse = right_eye_ellipses[0]

		'''
		A rotação do olho é modelada em apenas dois ângulos: tilt, sobre o eixo
		z e slant sobre o eixo y, onde o slant é aplicado depois do tilt. Para
		uma visualização melhor, abrir o arquivo slant_tilt_diagram.png
		'''

		# Obtém o ângulo de tilt da elipse do olho esquerdo.
		left_eye_tilt = left_eye_ellipse[2]

		# Obtém o ângulo de slant da elipse do olho esquerdo.
		left_eye_slant = 0
		left_eye_ellipse_w = left_eye_ellipse[1][0]
		left_eye_ellipse_h = left_eye_ellipse[1][1]

		if left_eye_ellipse_w < left_eye_ellipse_h:
			cos_value = left_eye_ellipse_w/left_eye_ellipse_h
			left_eye_slant = np.rad2deg(np.arccos(cos_value))
			
		elif left_eye_ellipse_h < left_eye_ellipse_w:
			left_eye_tilt += 90
			cos_value = left_eye_ellipse_h/left_eye_ellipse_w
			left_eye_slant = np.rad2deg(np.arccos(cos_value))

		# O else definiria o slant como 0, mas já é o valor padrão.

		# Obtém o ângulo de tilt da elipse do olho direito.
		right_eye_tilt = right_eye_ellipse[2]

		# Obtém o ângulo de slant da elipse do olho direito.
		right_eye_slant = 0
		right_eye_ellipse_w = right_eye_ellipse[1][0]
		right_eye_ellipse_h = right_eye_ellipse[1][1]

		if right_eye_ellipse_w < right_eye_ellipse_h:
			cos_value = right_eye_ellipse_w / right_eye_ellipse_h
			right_eye_slant = np.rad2deg(np.arccos(cos_value))
			
		elif right_eye_ellipse_h < right_eye_ellipse_w:
			right_eye_tilt += 90
			cos_value = right_eye_ellipse_h / right_eye_ellipse_w
			right_eye_slant = np.rad2deg(np.arccos(cos_value))

		# O else definiria o slant como 0, mas já é o valor padrão.

		# Exibe os dados obtidos para o rosto atual.
		print('\nFace {}:'.format(i))
		print('\tLeft Eye:\ttilt: [{tilt: .2f}°;\tslant: {slant: .2f}°]'.format(
			tilt=left_eye_tilt, slant=left_eye_slant))
		print('\tRight Eye:\ttilt: [{tilt: .2f}°;\tslant: {slant: .2f}°]'.format(
			tilt=right_eye_tilt, slant=right_eye_slant))

		# Exibe os resultados parciais e o resultado final.
		if show:
			preview = auto_resize(preview)
			cv2.imshow("preview", preview)

			cv2.imshow("Left Eye", left_eye_img)

			cv2.imshow("Right Eye", right_eye_img)

			original = auto_resize(original)
			cv2.imshow("Result", original)
			cv2.waitKey()
			cv2.destroyAllWindows()

		if output:
			cv2.imwrite(output, original)

def main():
	description = 'Searches for irises and tries to find its rotation angles.'
	parser = argparse.ArgumentParser(description = description)
	parser.add_argument('image', help='The path to an image to be tested.')
	parser.add_argument('-o', '--output', help='Path to an output file.', default=None)
	parser.add_argument('--show', help='Flag to show the images at the end.', action='store_true')
	args = vars(parser.parse_args())

	if not os.path.exists(args['image']):
		parser.error("The file %s does not exist!" % args['image'])
		return

	run(args['image'], args['show'], args['output'])

if __name__ == "__main__":
	main()