import pandas as pd
import librosa as lb
import os
import random
import soundfile as sf
import random
import multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from IPython.display import Audio
import uuid


def unir_csv(idiomas, num_muestras):
    csv_0 = None
    for i in idiomas:
        
        ## Leemos tsv
        csv = pd.read_csv(os.path.join(audios.format(i),
                                  'validated.tsv'),sep='\t')[['path','sentence']]
        
        # Separamos el id del audio del path
        csv['id_audio'] = csv['path'].str.split('_').map(lambda x: x[-1].split('.')[0])
        
        # Extraemos el idioma
        csv['idioma'] = csv['path'].str.split('_').str[2]
        
        # Verificamos los que están dentro del conjunto descargado
        csv['presente'] = csv['path'].isin(os.listdir(clips.format(i)))
        
        # Cambiamos de paths relativos a absolutos
        csv['path'] = csv['path'].map(lambda x : os.path.join(clips.format(i),x))
        
        #Filtramos los que no están descargados
        csv = csv[csv['presente'] == True]
        
        #Eliminamos las columnas que no son de interés
        csv = csv[['idioma','id_audio','path','sentence']]
        
        
        csv_0 = pd.concat([csv_0,csv],ignore_index=True)
    
    #Seleccionamos el número deseado de muestras por idioma.
    csv_0 = csv_0.groupby('idioma').apply(lambda x: x.sample(num_muestras)).reset_index(drop=True)
    return csv_0


def add_noise(df, lista_ruido, dir_destino):
    
    """Añade ruido a archivos en función de una lista de rutas con las rutas de los archivos de audio"""

    #Establecemos un SR determinado para todas las grabaciones
    sr = 22050
    
    #Carpeta destino a la que irán los audios
    carpeta_destino = dir_destino
    for _, row in df.iterrows():
        # Escogemos una proporcion de ruido entre el 15% y el 30%
        prop_ruido_audio = random.randint(5,10)/100

        # Cargamos el audio y un archivo de ruido al azar
        y, _ = lb.load(row.iloc[2], sr=sr)
        z, _ = lb.load(random.choice(lista_ruido))


        # Mezclamos los audios según la prop de ruido escogida
        inicio = random.randint(0, z.shape[0]-y.shape[0])

        z_cortado = z[inicio:inicio+y.shape[0]]  

        audio_con_ruido = (y*(1-prop_ruido_audio)+z_cortado*prop_ruido_audio)/2

        # Configuramos el nuevo nombre de archivo

        name_file = Path(row.iloc[2]).name
        name_file = str(name_file).split('.')[0] + '_' + str(int(prop_ruido_audio*100)) + '.mp3'

        # Guardamos en directorio de destino
        sf.write(file= os.path.join(carpeta_destino,name_file), data = audio_con_ruido, samplerate=sr)


def procesar_audio(row, lista_ruido, dir_destino, sr=22050):
    """Agrega ruido a un solo audio y guarda el archivo procesado."""
    print(1)
    prop_ruido_audio = random.randint(15, 30) / 100

    # Cargar el audio original
    y, _ = lb.load(row.iloc[2], sr=sr)
    
    # Cargar un ruido aleatorio
    z, _ = lb.load(random.choice(lista_ruido), sr=sr)

    # Seleccionar un fragmento aleatorio del ruido
    inicio = random.randint(0, z.shape[0] - y.shape[0])
    z_cortado = z[inicio:inicio + y.shape[0]]

    # Mezclar audio y ruido
    audio_con_ruido = (y * (1 - prop_ruido_audio) + z_cortado * prop_ruido_audio) / 2

    # Definir el nuevo nombre del archivo
    name_file = Path(row.iloc[2]).stem + f'_{int(prop_ruido_audio * 100)}.mp3'
    path_final = str(dir_destino / name_file)

    # Guardar el audio procesado
    print(2)
    sf.write(file=path_final, data=audio_con_ruido, samplerate=sr)

    return (path_final, prop_ruido_audio)

def add_noise_parallel(df, lista_ruido):
    """Aplica ruido a los audios en paralelo y devuelve el DataFrame actualizado."""
    carpeta_destino = r'F:\common_voice\Proyecto\Scripts\audios_proc'
    
    with mp.Pool(processes=mp.cpu_count()) as pool:
        print(3)
        results = pool.starmap(procesar_audio, [(row, lista_ruido, carpeta_destino) for _, row in df.iterrows()])

    # Separar los resultados en dos listas
    paths, props = zip(*results)

    df['new_path'] = paths
    df['noise_prop'] = props

    return df


def procesar_ruido(lista_ruido,dir_destino):
    """ """
    sr = 22050
    for i in lista_ruido:
        indice = 0
        y, _ = lb.load(i)
        while indice < len(y):
            duracion = int(random.randint(400,700)*sr/100)
            if indice + duracion > len(y):
                break
            new_audio = y[indice:indice+duracion]
            sf.write(os.path.join(dir_destino,uuid.uuid4().hex+'_ruido.mp3'),new_audio,sr)
            indice += duracion 