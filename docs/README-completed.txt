# TITLE OF PROJECT
Speech Emotion Recognition Voice Dataset


## GENERAL INFORMATION


### COLLABORATORS


Name:  Kucev Roman
    Role:  Data creator
    ORCID:  N/A
    Institution:  N/A
    Email:  N/A


Name:  Summer Mengarelli
    Role:  Data curator
    ORCID:  0009-0005-3368-022X
    Institution:  University of Notre Dame
    Email:  smengare@nd.edu


Name: Mikala Narlock
    Role: Data curator
    ORCID:  0000-0002-2730-7542
    Institution:  Indiana University Bloomington
    Email:  mnarlock@iu.edu


### FUNDING


Funder(s):  N/A
Funding period: N/A


## FILE OVERVIEW


* working-data/ (main project directory)
   * README.txt  
      * Description:  This README document.
      * Format:  TXT
      * Creation Date:  2025-04-26 
      * Update Date(s): 
         * 2026-01-03 - filled out template
   * speech_emotions.csv
      * Description:  Tabular data containing information about the participants in the project.
      * Format:  CSV
      * Creation Date:  2025-04-26
   * data-raw/ 
      * 0euphoric.wav
         * Description:  Audio recording of a person’s voice.
         * Format:  WAV
         * Creation Date:  2025-04-26
      * 0joyfully.wav
         * Description:  Audio recording of a person’s voice.
         * Format:  WAV
         * Creation Date:  2025-04-26
      * …
      * 4jsad.mpeg
         * Description:  Audio recording of a person’s voice.
         * Format:  MPEG
         * Creation Date:  2025-04-26
      * …
   * scripts/ 
      * 20240317_DNN_sp4_CS.py
         * Description:  Python script to train a speech-emotion recognition model.
         * Format:  PY
         * Creation Date:  2025-04-28
      * 20241103_LSTM_fa24_NS.py
         * Description: Python script to train a speech-emotion recognition model.
         * Format: PY
         * Creation Date: 2025-08-17


### FILE NAMING CONVENTIONS


#### For Python scripts:
Author name will be a capitalized 2-letter abbreviation of the author's first and last name (Carolyn Stazio = CS). Semester will be made up of a lowercase 2-letter abbreviation of the season and the last 2 numbers of the year (fall 2026 = fa26). Model is a capitalized 3- or 4-letter abbreviation (dense neural network = DNN, long short-term memory = LSTM). Components will be ordered date, model, semester, author with underscores between each: "YYYYMMDD_MOD_se##_AU.py". An example is "20240317_DNN_sp24_CS.py".


#### For audio files:
1-digit participant ID followed by emotion expressed (euphoric, joyfully, sad, surprised): 4euphoric.wav.


## METHODOLOGY & ACCESS INFORMATION


Description of collection methods:
The files were provided by users that passed the test. These data were validated by in-house reviewers.


## CHANGE LOG


Changes:


Scratchpad:
* This data is derived from the Kaggle dataset “Speech Emotion Recognition Voice Dataset,” uploaded by Kucev Roman in 2023. The original dataset and its documentation can be found here: https://www.kaggle.com/datasets/tapakah68/emotions-on-audio-dataset