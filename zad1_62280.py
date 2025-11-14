import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from io import BytesIO
from PIL import Image

URL_OBRAZU = "https://upload.wikimedia.org/wikipedia/commons/a/a4/A_strangely_created_lemon.jpg"

def wczytaj_zdalny_obraz(url):
    try:
        with urllib.request.urlopen(url) as odpowiedz:
            dane_obrazu = odpowiedz.read()

        obraz_pil = Image.open(BytesIO(dane_obrazu))
        obraz_np = np.array(obraz_pil)

        if obraz_np.ndim == 3:
             obraz_cv = cv2.cvtColor(obraz_np, cv2.COLOR_RGB2BGR)
        else:
             obraz_cv = obraz_np

        return obraz_cv
    except Exception as e:
        print(f"Błąd: {e}")
        return None

def wyswietl_obraz(obraz, tytul="Obraz", konwersja_koloru=True):
    if obraz is not None:
        plt.figure(figsize=(6, 6))
        if len(obraz.shape) == 3 and konwersja_koloru:
            obraz_wyswietlany = cv2.cvtColor(obraz, cv2.COLOR_BGR2RGB)
            plt.imshow(obraz_wyswietlany)
        else:
            plt.imshow(obraz, cmap='gray')
        plt.title(tytul)
        plt.axis('off')
        plt.show()


obraz_oryginalny = wczytaj_zdalny_obraz(URL_OBRAZU)

if obraz_oryginalny is None:
    exit()

wyswietl_obraz(obraz_oryginalny, "Oryginal")

procent_skali = 50
szerokosc = int(obraz_oryginalny.shape[1] * procent_skali / 100)
wysokosc = int(obraz_oryginalny.shape[0] * procent_skali / 100)
wymiary = (szerokosc, wysokosc)

obraz_zmniejszony = cv2.resize(obraz_oryginalny, wymiary, interpolation=cv2.INTER_AREA)

if len(obraz_zmniejszony.shape) == 3:
    obraz_szary = cv2.cvtColor(obraz_zmniejszony, cv2.COLOR_BGR2GRAY)
else:
    obraz_szary = obraz_zmniejszony

obraz_obrocony = cv2.rotate(obraz_szary, cv2.ROTATE_90_CLOCKWISE)

wyswietl_obraz(obraz_obrocony, "Wynikowy", konwersja_koloru=False)

print("Macierz wynikowego obrazu (fragment):")
print(obraz_obrocony[:5, :5])