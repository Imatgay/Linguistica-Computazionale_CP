#Script per convertire Agender corpus da .raw in .wav + noisereduction + silence trimming

import os
import wave
import numpy as np
import librosa
import noisereduce as nr
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from scipy.io import wavfile

root_directory = r"your_Agender_directory" #cartella in cui ci sono le directories con le singole sessioni dell'Agender Corpus
nfiles = 0
# Impostare i parametri audio (ottimali per agender)
sample_width = 2  
frame_rate = 8000  
channels = 1  

# Iterare il preprocessing in ogni audio delle subdirectory, nella directory_root directory
for root, dirs, files in os.walk(root_directory):
    for directory in dirs:
        directory_path = os.path.join(root,directory)
        for file in os.listdir(directory_path):
            if file.endswith(".raw"):
                nfiles += 1
                input_file_path = os.path.join(directory_path, file)
                output_file_path = os.path.join(directory_path, os.path.splitext(file)[0] + "_simple_nosilences.wav") #attenzione al nome che si aggiunge prima di estensione. Se lo si acmbia, cambiarlo anche in script dei classificatori.

                # Open the .raw file in binary mode
                raw_file = open(input_file_path, 'rb')

                # Creare nuovo file .wav da raw
                with wave.open(output_file_path, 'wb') as wav_file:
                    # Parametri audio prima impostati, qui si settano
                    wav_file.setnchannels(channels)
                    wav_file.setsampwidth(sample_width)
                    wav_file.setframerate(frame_rate)

                    # Leggi i dati del file .wav
                    raw_data = raw_file.read()

                    # Completa conversione. Utile per test su effetti preprocessing
                    wav_file.writeframes(raw_data)


                with wave.open(output_file_path, 'rb') as wav_file:
                    rate, data = wavfile.read(output_file_path)
                    
                    # Noise reduction su .wav
                    reduced_noise = nr.reduce_noise(y=data, sr=rate)

                    #eliminazione silenzi clippando l'audio su base di soglia x (top_db)
                    clips = librosa.effects.split(reduced_noise, top_db=40) #soglia fissata a 40 db
                    wav_data = []
                    for c in clips:
                        data = reduced_noise[c[0]: c[1]]
                        wav_data.extend(data)
                    wav_data = np.array(wav_data)
                  #salva file preprocessato finale
                    wavfile.write(output_file_path, rate, wav_data)
                                 
                    raw_file.close()
