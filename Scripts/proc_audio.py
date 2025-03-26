from utils_proc import unir_csv, add_noise
import os

path_ruido = r'F:\common_voice\Proyecto\ruido_samples'
audios = r'F:\common_voice\Proyecto\audios\{}'
clips = r'F:\common_voice\Proyecto\audios\{}\clips'
idiomas = ['ar','es','fr']

clips_ruido = [os.path.join(path_ruido,i) for i in os.listdir(path_ruido)]

csv_unidos = unir_csv(idiomas,1500)

add_noise(
    df = csv_unidos, 
    lista_ruido=clips_ruido,
    dir_destino=r'F:\common_voice\Proyecto\Scripts\data_proc\audios_proc'
    )

