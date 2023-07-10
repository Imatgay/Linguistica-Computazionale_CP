#Modello costruito con ScikitLearn per classificazione binaria gender (Agender Corpus)
## TODO : snellire e semplificare la creazione di sets ed estrazione labs/feats sul modello di TF-model 
#importiamo tutto il necessario

import os
import random
import pandas as pd
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Fissiamo root_directory (dove sono tutte le directories del corpus, una per sessione di registrazion)
# Fissiamo percorso per accedere al file SPEAEXT.tbl (in documentazione del corpus, contiene metadata su sessioni e parlanti), di seguito un esempio.
# Per tutte le informazioni relative al corpus e a metadata, vd, README dell'Agender Corpus nella cartella della Clarin Documentation
root_directory = r"your_corpus_path"
tbl_file = r"your_TBL_path\CLARINDocumentation\TABLES\SPEAEXT.TBL"

# Inizializziamo funzione per estrarre MFCCs da audio pre-enfatizzato
def extract_features(file, preemphasis_coeff=0.97, n_mfcc=80, n_mels=128, fmin=0, fmax=None, window='hann'):
    audio, sr = librosa.load(file)
    preemphasized_audio = np.append(audio[0], audio[1:] - preemphasis_coeff * audio[:-1])  # Apply preemphasis
    mfccs = librosa.feature.mfcc(y=preemphasized_audio, sr=sr, n_mfcc=n_mfcc, n_mels=n_mels, fmin=fmin, fmax=fmax, window=window)
    return np.mean(mfccs.T, axis=0)

def flatten(l): #funzione per sciogliere le liste con sottoliste in liste con semplici elementi
  out = []
  for item in l:
    if isinstance(item, (list, tuple)):
      out.extend(flatten(item))
    else:
      out.append(item)
  return out

#lista per conservare evaluations in caso di più run
accuracy_scores = []
uar_scores = []
all_directories = os.listdir(root_directory)
#sample_size = 400 # alternativa random_directories se si vuole lavorare su sample causale
n_runs = (1)

for exp in range (n_runs):
  ##se random_sample allora: random_directories = random.sample(os.listdir(root_directory), sample_size)
  # Separare le directories (= sessioni di registrazione) per speaker (speaker ID = prime quattro cifre del nome della directory)
  directory_dict = {}
  for directory in all_directories:  # oppure all_directories
      four_digits = directory[:4]
      if four_digits in directory_dict:
          directory_dict[four_digits].append(directory)
      else:
          directory_dict[four_digits] = [directory]

  # Shuffle (non necessario) delle sessioni per ogni speaker
  for key in directory_dict:
      random.shuffle(directory_dict[key])

  # Creare due liste per directories train e test
  train_directories = []
  test_directories = []

  #PUNTO FONDAMENTALE! Questo range determina quali delle sette classe vogliamo considerare. 
    #Ovviamente se non si vuole range, ma alcune specifiche classi, basta sostituire con "for n in (1,4,6)" per esempio  
  for n in range(2,8): #non considerata classe 1
      n_dir_list = []
      for key in directory_dict:
          if key.startswith(str(n)):
              n_dir_list.append(directory_dict[key])
              
      random.shuffle(n_dir_list)
      split_index = int(len(n_dir_list) * 0.85)   #split in train e test set (85% e 15%) all'interno di ogni classe, i cui elementi sono speaker con relative sessioni. Quindi uno stesso speaker non sarà contemporaneame in test e train set.
      train_directories.extend(n_dir_list[:split_index])
      test_directories.extend(n_dir_list[split_index:])              

  #sciogliamo le liste per eliminare i "confini" delle sottoliste, e le rendiamo ciascuna un'unica lista 
  train_directories = flatten(train_directories)
  test_directories = flatten(test_directories)

  #creaiamo liste per percorsi audio del train e del test set
  train_directory_paths = []
  test_directory_paths = []

  for test_directory in test_directories:
      test_directory = str(test_directory).replace('[', '').replace(']', '').replace("'", '')   # Remove square brackets
      test_directory_path = os.path.join(root_directory, test_directory)
      test_directory_paths.append(test_directory_path)
      print (test_directory_path)

  for train_directory in train_directories:
      train_directory = str(train_directory).replace('[', '').replace(']', '').replace("'", '')   # Remove square brackets
      train_directory_path = os.path.join(root_directory, train_directory)
      train_directory_paths.append(train_directory_path)
      print (train_directory_path)

  #  2  -- core script

  train_dataframes = []
  train_features = []
  train_labels = []
  train_file_names  = []

  n_train_data = 0

  # Estrazione features e label del train set. Nella train directory sono storati nomi di directory, adesso andiamo effettivamente a prendere quello che contengono.
  for train_directory in train_directories:
      train_directory = str(train_directory).replace('[', '').replace(']', '').replace("'", '')   # Remove square brackets
      train_directory_path = os.path.join(root_directory, train_directory)
      if os.path.isdir(train_directory_path):  
          speakerID = int(train_directory[:4])#cerca nel TBL file lo speaker
          with open(tbl_file, "r") as f:
              next(f)  # Salta l'header del file
              for line in f:
                  columns = line.strip().split("\t")
                  if int(columns[0]) == speakerID: #la prima colonna conserva lo speakerID
                      current_speaker_sex = columns[3] #fissiamo il sesso del .TBL come label per l'attuale speaker
          subdirectory_dataframes = []
          for file in os.listdir(train_directory_path):
              if file.endswith("nosilences.wav"): #fissiamo una regola per recuperare i file che ci  interessano (qui i preprocessed)
                  file_path = os.path.join(train_directory_path, file)
                  mfcc = extract_features(file_path)
                  # Creare DataFrame per l'attuale file audio
                  audio_dataframe = pd.DataFrame({
                      "MFCC": [mfcc],
                      "Sex": [current_speaker_sex]
                  })
                  #unione il dataframe del file a un dataframe connesso alla sessione
                  subdirectory_dataframes.append(audio_dataframe)
              
          if subdirectory_dataframes:
              subdirectory_dataframe = pd.concat(subdirectory_dataframes)
              train_dataframes.append(subdirectory_dataframe) #creiamo train set con concatenazioni di sub dataframe

  #stessa cosa per il test set
  test_dataframes = []
  test_features = []
  test_labels = []
  test_file_names = []

  n_test_data = 0

  for test_directory in test_directories:
      test_directory = str(test_directory).replace('[', '').replace(']', '').replace("'", '')   
      test_directory_path = os.path.join(root_directory, test_directory)
      if os.path.isdir(test_directory_path):
          speakerID = int(test_directory[:4])
          with open(tbl_file, "r") as f:
              next(f) 
              for line in f:
                  columns = line.strip().split("\t")
                  if int(columns[0]) == speakerID:
                      current_speaker_sex = columns[3]
          subdirectory_dataframes = []
          for file in os.listdir(test_directory_path):
              if file.endswith("nosilences.wav"):
                  n_test_data += 1
                  file_path = os.path.join(test_directory_path, file)
                  mfcc = extract_features(file_path)
                  audio_dataframe = pd.DataFrame({
                      "MFCC": [mfcc],
                      "Sex": [current_speaker_sex]
                  })

                  subdirectory_dataframes.append(audio_dataframe)

              
          if subdirectory_dataframes:
              subdirectory_dataframe = pd.concat(subdirectory_dataframes)
              test_dataframes.append(subdirectory_dataframe)


  train_dataframe = pd.concat(train_dataframes, ignore_index=True)
  train_dataframe.to_pickle('train_dataframe.pkl') #salvataggio di file in locale per accedere in futuro direttamente ai dataframe train e test
  test_dataframe = pd.concat(test_dataframes, ignore_index=True)
  test_dataframe.to_pickle('test_dataframe.pkl')

  #estrazione features e label da train/test dataframes
  train_features = train_dataframe['MFCC'].to_list() 
  test_features = test_dataframe['MFCC'].to_list()
  train_labels = train_dataframe['Sex'].tolist()
  test_labels = test_dataframe['Sex'].tolist()

  #conversione in np.array
  X_train = np.array(train_features)
  X_test = np.array(test_features)
  y_train = np.array(train_labels)
  y_test = np.array(test_labels)

  #setting del MLPClassifier, fitting e predizione
  clf = MLPClassifier(hidden_layer_sizes=(80,40), max_iter=500, activation='relu', solver='adam', random_state=42) #if all_directories, then max_iter = 1000
  clf.fit(X_train, y_train)

  y_pred = clf.predict(X_test)

  #valutazione risultati
  accuracy = accuracy_score(y_test, y_pred)
  precision = precision_score(y_test, y_pred, average='macro')
  recall = recall_score(y_test, y_pred, average='macro')
  uar = np.mean(recall)
  f1 = f1_score(y_test, y_pred, average='macro')
  cm = confusion_matrix(y_test, y_pred)

  accuracy_scores.append(accuracy)
  uar_scores.append(uar)
  print("Train set size: ", n_train_data)
  print("Test set size: ", n_test_data)
  print("Evaluation metrics: ")
  print("Accuracy: ", accuracy)
  print("Precision: ", precision)
  print("Recall: ", recall)
  print("F1 score: ", f1)
  print("Unweighted Average Recall: ", uar)
  print("Confusion matrix: ")
  print(cm)

print("___________________________________________")


#nel caso in cui si facciano più run, qui si stampano delle medie di ACC e UAR
average_accuracy = np.mean(accuracy_scores)
average_uar = np.mean(uar_scores)
print (f"Average metrics - {n_runs} runs ")
print ("Average accuracy score:", average_accuracy)
print ("Average UAR score:", average_uar)
