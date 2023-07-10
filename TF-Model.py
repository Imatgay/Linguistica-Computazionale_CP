#Modello costruito cin Keras-TensorFlow per classificazione multiclass (ottimizzata per 7 classi, quelle presenti in Agender Corpus)

#importiamo tutto il necessario
import os
import random
import numpy as np
import pandas as pd
import librosa
import random
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras import regularizers


# Fissiamo root_directory (dove sono tutte le directories del corpus, una per sessione di registrazion)
# Fissiamo percorso per accedere al file SPEAEXT.tbl (in documentazione del corpus, contiene metadata su sessioni e parlanti), di seguito un esempio.
# Per tutte le informazioni relative al corpus e a metadata, vd, README dell'Agender Corpus nella cartella della Clarin Documentation
root_directory = r"your_path"
tbl_file = r"your_TBL_path\SPEAEXT.TBL"

# Definiamo un dizionario che mappa le classi a nuovi valori indicizzati a zero, per rendere le label processabili dall'algoritmo
label_mapping = {'1':0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6}

# Inizializziamo funzione per estrarre MFCCs da audio pre-enfatizzato
def extract_features(file, preemphasis_coeff=0.97, n_mfcc=80, n_mels=128, fmin=0, fmax=None, window='hann'):
    audio, sr = librosa.load(file)
    #stretched_audio = librosa.effects.time_stretch(audio, rate=0.9)
    preemphasized_audio = np.append(audio[0], audio[1:] - preemphasis_coeff * audio[:-1])   
    mfccs = librosa.feature.mfcc(y=preemphasized_audio, sr=sr, n_mfcc=n_mfcc, n_mels=n_mels, fmin=fmin, fmax=fmax, window=window)
    return np.mean(mfccs.T, axis=0)

##################### RANDOMIZER: fissiamo alcuni parametri per randomizzare l'attuale dataset per costruire train e test set
#all_directories = os.listdir(root_directory)  -- alternativa a random_directories per lavorare sull'intero corpus
sample_size = 800
all_directories = os.listdir(root_directory)

#----------------------

n_runs = (5)   #fissa numero di run, i cui risultati saranno successivamente riuniti in media

for exp in range (n_runs):
# Separare le directories (= sessioni di registrazione) per speaker (speaker ID = prime quattro cifre del nome della directory)
    random_directories = random.sample(os.listdir(root_directory), sample_size)
    directory_dict = {}
    for directory in random_directories: # oppure all_directories
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
    for n in range(1, 8):    
        n_dir_list = []
        for key in directory_dict:
            if key.startswith(str(n)):
                n_dir_list.append(directory_dict[key])

        random.shuffle(n_dir_list)
        split_index = int(len(n_dir_list) * 0.85)  #split in train e test set (85% e 15%) all'interno di ogni classe, i cui elementi sono speaker con relative sessioni. Quindi uno stesso speaker non sarà contemporaneame in test e train set.
        print(int(len(n_dir_list) * 0.85))
        train_directories.extend(n_dir_list[:split_index])
        print(train_directories)
        test_directories.extend(n_dir_list[split_index:])
    #sciogliamo le liste per eliminare i "confini" delle sottoliste, e le rendiamo ciascuna un'unica lista 
    train_directories = [item for sublist in train_directories for item in sublist] 
    test_directories = [item for sublist in test_directories for item in sublist]

    # adesso estraiamo features e labels per ciascuna ciascun set
    train_features = []
    train_labels = []
    test_features = []
    test_labels = []

    n_train_data = 0 #counter per avere conto finale dei audio usati
    n_test_data = 0

    # Estrazione features e label del train set. Nella train directory sono storati nomi di directory, adesso andiamo effettivamente a prendere quello che contengono.
    for train_directory in train_directories:
        train_directory_path = os.path.join(root_directory, train_directory) 
        if os.path.isdir(train_directory_path):
            speakerID = int(train_directory[:4])
            with open(tbl_file, "r") as f: #cerca nel TBL file lo speaker
                next(f)  # Salta l'header del file
                for line in f:
                    columns = line.strip().split("\t")
                    if int(columns[0]) == speakerID: #la prima colonna conserva lo speakerID
                        current_speaker_class = label_mapping[columns[1]] #fissiamo una label per l'attuale speaker sulla base della regola mapping iniziale
            for file in os.listdir(train_directory_path): #entriamo nella directory
                if file.endswith("_simple_nosilences.wav"): #fissiamo una regola per recuperare i file che ci  interessano (qui i preprocessed)
                    n_train_data += 1
                    file_path = os.path.join(train_directory_path, file)
                    mfcc = extract_features(file_path)  # estrai MFCCs
                    train_features.append(mfcc) #popolare features e label
                    train_labels.append(current_speaker_class)

    # La medesima cosa viene fatta per il test set, a partire da test_directories
    for test_directory in test_directories:
        test_directory_path = os.path.join(root_directory, test_directory)
        if os.path.isdir(test_directory_path):
            speakerID = int(test_directory[:4])
            with open(tbl_file, "r") as f:
                next(f)  
                for line in f:
                    columns = line.strip().split("\t")
                    if int(columns[0]) == speakerID:
                        current_speaker_class = label_mapping[columns[1]]
            for file in os.listdir(test_directory_path):
                if file.endswith("_simple_nosilences.wav") and not file.endswith("_simple.wav") or file.endswith("nosilences.wav"):
                    n_test_data += 1
                    file_path = os.path.join(test_directory_path, file)
                    mfcc = extract_features(file_path)
                    test_features.append(mfcc)
                    test_labels.append(current_speaker_class)

    # Convertiamo i dati in np.array per renderli processabili
    train_features = np.array(train_features)
    train_labels = np.array(train_labels)
    test_features = np.array(test_features)
    test_labels = np.array(test_labels)

    # Normalizziamo i dati
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    test_features = scaler.transform(test_features)

    # Encode labels
    label_encoder = LabelEncoder()
    train_labels_encoded = to_categorical(label_encoder.fit_transform(train_labels))
    test_labels_encoded = to_categorical(label_encoder.transform(test_labels))
    #fissiamo parametri, per fare diversi test possiamo aggiungerne nelle rispettive liste
    learning_rates = [0.001]
    dropout_rates = [0.3]
    epochs = [30]
    results = []

    #creiamo modello per ogni combinazione di parametri
    random_seed = 42
    for learning_rate in learning_rates:
        for dropout_rate in dropout_rates:
            for epoch in epochs:
                model = Sequential()
                model.add(Dense(80, activation='relu', input_shape=(train_features.shape[1],)))
                model.add(Dense(40, activation='relu'))
                model.add(Dropout(dropout_rate))
                model.add(Dense(20, activation='relu'))
                model.add(Dropout(dropout_rate))
                model.add(Dense(len(label_mapping), activation='softmax'))
                

                model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

                early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
                combined_data = list(zip(train_features, train_labels_encoded))
                random.Random(random_seed).shuffle(combined_data)
                shuffled_features, shuffled_labels = zip(*combined_data)
                shuffled_labels = np.array(shuffled_labels)

                # Alleniamo il modello
                model.fit(np.array(shuffled_features), shuffled_labels, epochs=epoch, batch_size=32, verbose=1, validation_split=0.2, callbacks=[early_stopping])

                # Valutiamo il modello su predizioni
                predictions = model.predict(test_features)
                predicted_labels = np.argmax(predictions, axis=1)
                accuracy = accuracy_score(test_labels, predicted_labels)
                precision = precision_score(test_labels, predicted_labels, average='macro')
                recall = recall_score(test_labels, predicted_labels, average='macro')
                uar = np.mean(recall)
                f1 = f1_score(test_labels, predicted_labels, average='macro')
                cm = confusion_matrix(test_labels, predicted_labels)


                results.append({
                    'learning_rate': learning_rate,
                    'dropout_rate': dropout_rate,
                    'epoch': epoch,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'uar': uar,
                    'confusion_matrix': cm
                })

    for result in results:
        accuracy_scores.append(result['accuracy'])
        uar_scores.append(result['uar'])
        print("Learning Rate:", result['learning_rate'])
        print("Dropout Rate:", result['dropout_rate'])
        print("Epoch:", result['epoch'])
        print("Accuracy:", result['accuracy'])
        print("Precision:", result['precision'])
        print("Recall:", result['recall'])
        print("F1 score:", result['f1'])
        print("Unweighted Average Recall:", uar)
        print("Confusion matrix:")
        print(result['confusion_matrix'])
        print()


    
print("___________________________________________")


#nel caso in cui si facciano più run, qui si stampano delle medie di ACC e UAR
average_accuracy = np.mean(accuracy_scores)
average_uar = np.mean(uar_scores)
print (f"Average metrics - {n_runs} runs ")
print ("Average accuracy score:", average_accuracy)
print ("Average UAR score:", average_uar)
