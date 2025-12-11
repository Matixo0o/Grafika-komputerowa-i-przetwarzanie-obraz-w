import requests
from PIL import Image
import numpy as np
from io import BytesIO

IMAGE_URL = 'https://upload.wikimedia.org/wikipedia/commons/5/5a/Mountains_in_snow%2C_Mountain_lake%2C_Chola_Valley%2C_Nepal%2C_Himalayas.jpg'

def load_remote_image(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        return image
    except requests.exceptions.RequestException as e:
        print(f"Blad: {e}")
        if 'response' in locals() and response is not None:
             print(f"Status HTTP: {response.status_code}")
        return None
    except Exception as e:
        print(f"Blad przetwarzania: {e}")
        return None

def display_histogram_data(img_np):
    img_gray = np.dot(img_np[...,:3], [0.2989, 0.5870, 0.1140])
    hist, bins = np.histogram(img_gray.ravel(), bins=256, range=[0, 256])
    
    print("\n--- Analiza Wstepna Histogramu (Intensywnosc) ---")
    print("Wartosc | Liczba Pikseli")
    print("-------------------------")
    
    intensities = [0, 1, 50, 127, 200, 254, 255]
    for i in intensities:
        print(f"{i:7} | {hist[i]:15}")

    colors = ('czerwony', 'zielony', 'niebieski')
    
    for i, color in enumerate(colors):
        channel_data = img_np[:, :, i].ravel()
        mean = np.mean(channel_data)
        print(f"Srednia intensywnosc kanalu {color}: {mean:.2f}")

    return img_gray.ravel()

def assess_quality(hist_flat):
    H, _ = np.histogram(hist_flat, bins=256, range=[0, 256])
    N = len(hist_flat)
    
    non_zero_indices = np.where(H[1:-1] > 0)[0] + 1
    if non_zero_indices.size == 0:
         I_min, I_max = 0, 255
    else:
        I_min = non_zero_indices.min()
        I_max = non_zero_indices.max()
    
    RR = (I_max - I_min) / 255.0
    
    intensity_values = np.arange(256)
    Mu = np.sum(intensity_values * H) / N

    CP_0 = H[0] / N * 100
    CP_255 = H[255] / N * 100
    CP_total = CP_0 + CP_255
    
    quality_score = ""
    is_flawed = False

    if Mu < 50:
        quality_score += "WADLIWE: Niedostatecznie naswietlone (za ciemne). "
        is_flawed = True
    elif Mu > 200:
        quality_score += "WADLIWE: Przeswietlone (za jasne). "
        is_flawed = True

    if RR < 0.6:
        quality_score += "WADLIWE: Niski kontrast (splaszczony histogram). "
        is_flawed = True
    
    if CP_total > 1.0:
        quality_score += f"WADLIWE: Duze przyciecie (clipping) ({CP_total:.2f}% pikseli na krancach - utrata szczegolow). "
        is_flawed = True
    
    if not is_flawed:
        quality_score = "DOBRA Jakosc (Zrownowazona ekspozycja i kontrast)."

    print("\n--- WERDYKT JAKOSCI NA PODSTAWIE ANALIZY ---")
    print(f"1. Srednia Intensywnosc (Mu): {Mu:.2f} (Ekspozycja)")
    print(f"2. Zakres Tonalny (RR): {RR:.2f} (Kontrast, 1.0 to idealny zakres)")
    print(f"3. Przyciecie (Clipping CP): {CP_total:.2f}% (Utrata detali w cieniach/swiatlach)")
    print(f"WYNIK: {quality_score}")
    
    return is_flawed, quality_score

def improve_quality_with_eq(img_pil):
    print("\n--- Zastosowanie Wyrownania Histogramu (Bonus) ---")
    
    img_gray = img_pil.convert("L")
    img_gray_np = np.array(img_gray)
    
    hist_eq = np.histogram(img_gray_np.flatten(), 256, [0, 256])[0]
    cdf = hist_eq.cumsum()
    
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    
    img_eq_np = cdf[img_gray_np]
    
    img_eq_pil = Image.fromarray(img_eq_np, 'L')
    
    hist_improved, bins_improved = np.histogram(img_eq_np.ravel(), bins=256, range=[0, 256])
    print("Dane Histogramu Po Korekcji:")
    print("Wartosc | Liczba Pikseli")
    print("-------------------------")
    intensities = [0, 1, 50, 127, 200, 254, 255]
    for i in intensities:
        print(f"{i:7} | {hist_improved[i]:15}")

    return img_eq_pil

if __name__ == "__main__":
    img_pil = load_remote_image(IMAGE_URL)
    
    if img_pil is None:
        pass
    else:
        img_np = np.array(img_pil)
        
        hist_flat_data = display_histogram_data(img_np)
        
        is_flawed, quality_score = assess_quality(hist_flat_data)
        
        if is_flawed:
            img_improved = improve_quality_with_eq(img_pil)
            
            hist_improved_data = np.array(img_improved).ravel()
            print("\n--- Ponowna Analiza Poprawionego Zdjecia ---")
            is_flawed_new, quality_score_new = assess_quality(hist_improved_data)
            print(f"NOWY WYNIK PO KOREKCJI: {quality_score_new}")
        else:
            print("\nZdjecie nie wymaga poprawy jakosci.")
