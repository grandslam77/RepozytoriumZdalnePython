
import numpy as np															# biblioteka do tworzenia zmiennych wielowymiarowych
import pygame																# biblioteka do wyświetlania obrazków
import random																# biblioteka do generowania wartości losowych
import sys
from itertools import chain
from sklearn.cluster import KMeans											# biblioteka do grupowania (kolorów w przestrzeni RGB)
import copy																	# biblioteka do głębokiego kopiowania zmiennych

pygame.init()																# inicjowanie pygame. Musi zawsze wystąpić
plik = 'C:/Users/plakopec/OneDrive/profilowe_3.jpg'							# zaczytanie pliku do zmiennej 'plik'
#"plik = 'C:/Users/grandslam/OneDrive/profilowe_s1.jpg'
def getPixelArray(filename):												# funkcja konwerująca obrazek w trójwymiarową tablicę kolorów RGB dla każdego piksela
	image = pygame.image.load(plik)
	return pygame.surfarray.array3d(image)

pixels = getPixelArray(plik)												# wywołanie powyższej funkcji na zaczytanym obrazku
game_width = pixels.shape[0]												# okreslenie rozmiarów obrazka na podstawie zwymiarowania zmiennej 'pixels'
game_height = pixels.shape[1]


def prawd_krzyzowania(kl):													# ustawienie prawdopodobieństwa krzyżowania
	dee=copy.deepcopy(kl)
	if dee<800:
		prawdopodobienstwo_krzyzowania=1.6
	else:
		prawdopodobienstwo_krzyzowania=0.4
	return prawdopodobienstwo_krzyzowania

swiezy_material=0.05														# określenie jaką cześć populacji w kazdym pokoleniu maja stanowić osobniki wygenerowane losowo

wielkosc_populacji = 101													# wielkość populacji
licznosc_grupy_turniejowej=int(0.03*wielkosc_populacji)						# określenie nacisku selektywnego, im liczniejsza grupa turniejowa tym większy nacisk selekcyjny
liczba_najlepszych_elitism=1												# ile najlepszych rozwiązań z poprzedniego pokolenia ma przechodzić do następnego pokolenia
liczba_klastrow=7															# teoretyczna liczba głównych kolorów z jakich powinien składać się obrazek

def prawdopod_mutacji(di):													# funkcja określająca ile genów(pikseli) w każdym chromosomie(obrazku) będzie podlegało mutacji w danym pokoleniu
	see = copy.deepcopy(di)
	if see < 800:
		prawdopodobienstwo_mutacji = random.uniform(1/(game_height*game_width),200/(game_height*game_width))			#losowo - od 1 do 200 pikseli
	elif 800<see<1200:
		prawdopodobienstwo_mutacji = random.uniform(1 / (game_height * game_width), 30 / (game_height * game_width))
	else:
		prawdopodobienstwo_mutacji = random.uniform(1 / (game_height * game_width), 5 / (game_height * game_width))
	return prawdopodobienstwo_mutacji

def ilosc_punktow_wymiany_fun():											# funkcja określająca ile genów w z drugiego chromosomu będzie podlegało wymianie w procesie krzyżowania
    ilosc_punktow_wymiany=int(random.uniform(0.001*game_height*game_width,0.2*game_height*game_width))
    return ilosc_punktow_wymiany


rozPixels = pixels[0:pixels.shape[1],0:pixels.shape[1],0:3]					# rozbijanie zmiennej trójwymiarowej na listę
rozPixels = rozPixels.tolist()
rozPixels = list(chain.from_iterable(rozPixels))
RGBwzorzec = np.array(rozPixels)											# określenie wzorca, do którego będa porównywane rozwiązania

kmeans = KMeans(n_clusters=liczba_klastrow, random_state=1).fit(RGBwzorzec)	# określenie grup kolorów
centra = kmeans.cluster_centers_.astype(int)								# wyznaczanie centroid grup kolorów, będą stanowić kolory z których odzyskiwany będzie obrazek zaczytany do zmiennej 'plik'
centra=centra.reshape(liczba_klastrow,3)									# zmiana wymiaru listy przechowującej dane o centroidach
gameDisplay = pygame.display.set_mode((game_width, game_height))			# wyznaczanie obszaru rysowania
kolory = np.asarray([[int(random.uniform(0,255)),int(random.uniform(0,255)),int(random.uniform(0,255))] for elem in range(game_width*game_height)])	# generowanie pikseli w losowych kolorach do obszaru rysowania
kolory = kolory.reshape((game_height, game_width, 3))						# zmiana wymiaru zmiennej kolory
pygame.surfarray.blit_array(gameDisplay, kolory)							# przypiecie pikseli do obszaru rysowania pygame
pygame.display.flip()														# wyświetlenie pikseli w obszarze rysowania

class PierwszaGeneracja:													# klasa(szablon) osobnika pierwotnego
	def __init__(self):
		self.chromosom = np.asarray([[random.choice(centra)] for elem in range(game_height * game_width)]).reshape(game_height * game_width,3)	# piksele losowo generowane z centroid
		self.dopasowanie=-1*abs(sum(sum(abs(RGBwzorzec-self.chromosom))))	# funkcja obliczająca dopasowanie do wzorca (bezwzględną odległość kolorów od obrazka docelowego)

class NowaGeneracja():														# klasa (szablon) nowych pokoleń
	def __init__(self,chromosom,dopasowanie,nr_generacji):
		self.chromosom=chromosom
		self.dopasowanie=dopasowanie
		self.nr_generacji=nr_generacji

pierwszaGen=[PierwszaGeneracja() for elem in range(0,wielkosc_populacji)]			# generowanie populacji pierwotnej
lego=sorted(pierwszaGen, key=lambda pierwsza: pierwsza.dopasowanie,reverse=True)	# sortowanie populacji pierwotnej po atrybucie 'dopasowanie'

set1 = [i for i in range(0, game_width*game_height)]								# zbiór liczb określających liczbę genów
set2 = [i for i in range(0, wielkosc_populacji)]									# zbiór liczb okreśłających wielkość populacji

def lista_przedzialow_do_krzyzowania(lista_losowania,liczba_przedzialow):			# funkcja określająca, które geny pierwszego chromosomu krzyżują się z genami drugiego chromosomu
	lista=np.array(sorted(random.sample(lista_losowania, liczba_przedzialow)))
	return lista

def nowyChromosom(material_x,material_y,przedzialy):								# funkcja reprodukująca nowego osobnika na podstawie krzyżowania dwóch osobników
	material_z = copy.deepcopy(material_x)
	newOsobnik = copy.deepcopy(material_z)
	material_k = copy.deepcopy(material_y)
	for punkty in przedzialy:
		newOsobnik[punkty],material_k[punkty]=material_k[punkty],newOsobnik[punkty]	# wszczepienie pikseli z osobnika  k w osobnika z (a może odwrtonie :-))
	return newOsobnik

def selekcja_turniejowa(lista_obiektow,licznosc_grupy_turniejowej):					# selekcja osobników, wybór najbardziej dopasowanych do populacji potomnej
	grupa_turniejowa=random.sample(lista_obiektow,licznosc_grupy_turniejowej)		# przechodzą ci, którzy mają największe dopasowanie
	grupa_turniejowa_sort=sorted(grupa_turniejowa, key=lambda pierwsza: pierwsza.dopasowanie)
	zwyciezca=grupa_turniejowa_sort[len(grupa_turniejowa_sort)-1]
	return zwyciezca

def mutacja(chromosom,punkty_mutacji):												# mutacja
	kopia_pierwotnego_DNA=copy.deepcopy(chromosom)									# kopiowanie chromosomu
	for ge, geny in enumerate(punkty_mutacji):										# pętla wykonywana tyle razy ile genów ma zostac zmutowanych
		wybor = random.choice(centra)												# losowy wybór koloru ze zbioru centroid
		#while np.all(wybor == kopia_pierwotnego_DNA[geny]):							# jeśli taki sam jak kolor już obecny, losuj dalej
		#	wybor = random.choice(centra)
		kopia_pierwotnego_DNA[geny] = wybor											# podmiana koloru pixela na nowy
	zmutowany_chromosom=kopia_pierwotnego_DNA										# zwrot zmutowanego chromosomu
	return zmutowany_chromosom

def mutacja2(chromosom,punkty_mutacji):												# inna koncepcja mutacji, na razie nie działa
	kopia_pierwotnego_DNA=copy.deepcopy(chromosom)
	kopia_pomocnicza_DNA=copy.deepcopy(chromosom)

	for ge, geny in enumerate(punkty_mutacji):

		wybor = random.choice(centra)
		while np.all(wybor == kopia_pierwotnego_DNA[geny]):
			wybor = random.choice(centra)

		kopia_pomocnicza_DNA[geny] = wybor
		pomocnicz_dost=-1 * abs(sum(sum(abs(RGBwzorzec - kopia_pomocnicza_DNA))))
		pierwotne_dost=-1 * abs(sum(sum(abs(RGBwzorzec - kopia_pierwotnego_DNA))))
		if pomocnicz_dost>pierwotne_dost:
			kopia_pierwotnego_DNA[geny] = wybor

	zmutowany_chromosom=kopia_pierwotnego_DNA
	return zmutowany_chromosom

for z in range (1,100000):															# główna pętla algorytmu
	print('pokolenie nr:',z)

	# budowanie populacji przejściowej
	# selekcja
	populacja_przejsciowa=[]
	populacja_potomna = []
	populacja_przedpotomna = []

	for s in range(0,int(wielkosc_populacji*(1-swiezy_material))):					# realizowanie selekcji, cześc miejsca w populacji pozostawiono dla osobników losowych
		nowy_os_przejsciowy=selekcja_turniejowa(lego,licznosc_grupy_turniejowej)
		populacja_przejsciowa.append(nowy_os_przejsciowy)

	for i in range(1,int(wielkosc_populacji*swiezy_material)):						# dołączanie osobników losowych (bardzo ważne)
		populacja_przejsciowa.append(PierwszaGeneracja())


	############################ ######################## budowanie populacji potomnej #############################################################
	####### krzyżowanie ######## ###################################################################################################################
	############################ ###################################################################################################################

	if liczba_najlepszych_elitism != 0:												# elityzm zakodować tak aby następne pokolenie mogło uwzględniać większą ilość jednostek elitarnych
		populacja_przejsciowa.append(lego[0])

	for i,osobniki in enumerate(populacja_przejsciowa):

		if prawd_krzyzowania(z)/2>random.random():									# jeśli wylosuje się liczbę mniejszą niż prawdopodobieństwo krzyżowania to krzyżuj

			para = random.sample(populacja_przejsciowa, 2)							# losowanie osobników do krzyżowania
			#print(para,'para')
			x = para[0].chromosom
			y = para[1].chromosom

			przedzialy = lista_przedzialow_do_krzyzowania(set1, ilosc_punktow_wymiany_fun())
			noweGeny = nowyChromosom(x, y, przedzialy)									# wysyłanie chromosomów i danych o punktach krzyżowania do funkcji realizującej krzyżowanie
			dopas = -1 * abs(sum(sum(abs(RGBwzorzec - noweGeny))))						# wyliczenie dopasowania dla nowego osobnika
			populacja_przedpotomna.append(NowaGeneracja(noweGeny, dopas,'krzyżowanie'))	# dodanie nowego kompletnego osobnika do populacji przedpotomnej

		else:

			populacja_przedpotomna.append(populacja_przejsciowa[i])						# jeśli nie ma krzyżowania dodaj niezmienionego osobnika z populacji bazowej

	populacja_przedpotomna=sorted(populacja_przedpotomna,key=lambda druga: druga.dopasowanie,reverse=True)	# posortuj populację przedpotomną po wartości atrybutu dopasowanie

	############ mutacja ###########################################################################################################

	if prawdopod_mutacji(z)>0:
		for osobniaczki in range(2,len(populacja_przedpotomna)):
			losowy = random.sample(set1, int(prawdopod_mutacji(z) * game_width * game_height))
			zmutowaneX=mutacja(populacja_przedpotomna[osobniaczki].chromosom,losowy)
			dopas = -1 * abs(sum(sum(abs(RGBwzorzec - zmutowaneX))))
			populacja_potomna.append(NowaGeneracja(zmutowaneX,dopas,z))

	################################################################################################################################





	populacja_potomna = sorted(populacja_potomna, key=lambda pierwsza: pierwsza.dopasowanie, reverse=True)
	#print(len(populacja_potomna))
	lego=sorted(lego,key=lambda druga: druga.dopasowanie,reverse=True)
	print(round(populacja_potomna[0].dopasowanie/1000,1),'najlepsze dopasowanie z populacji potomnej')
	print(round(lego[0].dopasowanie/1000,1), 'najlepsze dopasowanie z populacji bazowej')

	duza_populacja=populacja_potomna+lego
	duza_populacja=sorted(duza_populacja,key=lambda druga: druga.dopasowanie,reverse=True)
	populacja_potomna=duza_populacja[0:wielkosc_populacji]

	lego=populacja_potomna
	ostatni_obiekt=lego[0]
	ostatni = lego[0].chromosom
	ostatnie=lego[0].dopasowanie

	b = ostatni.reshape((game_height, game_width, 3))
	pygame.surfarray.blit_array(gameDisplay, b)
	pygame.event.pump()
	pygame.display.flip()



gameExit = False
while not gameExit:
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			pygame.quit()
			quit()

