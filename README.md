# Linguistica-Computazionale_CP
1- scaricare corpus da https://clarin.phonetik.uni-muenchen.de/BASRepository/ 


2- run di preprocessing.py su cartella che contiene le varie sub-directories (una per sessione di registrazione)


  [ a questo punto in ogni subdirectory avremo tre copie dello stesso file: il .raw originale, il .wav semplice (convertito e non preprocessato, con stesso nome
  del .raw, solo con estensione diversa), il .wav preprocessato (chiamato [nome]_simple_nosilences.wav). Richiamare i nomi corretti di questi file è fondamentale in fase
  di classificazione]

  
3- a seconda di cosa si vuole classificare, run di TF-Model (7 classi di default, facilmente customizzabile modificando linea 71) o SK-Model (di default
    si basa sull'intero corpus esclusa la classe 1 di bambini 7-14 anni, ma anche qui si può modificare l'insieme di classi del corpus da considerare modificando linea 66)
