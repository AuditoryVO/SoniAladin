# SoniAladin
SoniAladin is an application developed in the context of the Open Science initiative of the Spanish Virtual Observatory (SVO). It allows the transformation of Aladin’s virtual sky into audible representations.

It is also available as a web app at: https://auditoryvo.github.io/SoniAladin/

This research has made use of "Aladin sky atlas" developed at CDS, Strasbourg Observatory, France
[[2000A&AS..143...33B](url)](https://ui.adsabs.harvard.edu/abs/2000A%26AS..143...33B/abstract) (Aladin Desktop), [2014ASPC..485..277B](url) (Aladin Lite v2), and [2022ASPC..532....7B](url) (Aladin Lite v3).

This research has made use of the Spanish Virtual Observatory (https://svo.cab.inta-csic.es) project funded by MCIN/AEI/10.13039/501100011033 through grant PID2023-146210NB-I00.

<img width="1361" height="955" alt="SoniAladin" src="https://github.com/user-attachments/assets/b6df18d9-b346-4c38-bd4f-a6525561910b" />


INSTALLATION:

The voice recognition module requires the local model vosk-model-small-en-us-0.15.

	1- Download and unzip it from: https://alphacephei.com/vosk/models
	
	2- Copy the unzipped vosk-model-small-en-us-0.15 folder in the SoniAladin folder.
	
US English model for mobile Vosk applications.
Copyright 2020 Alpha Cephei Inc.
Accuracy: 10.38 (tedlium test) 9.85 (librispeech test-clean).
Speed: 0.11xRT (desktop).
Latency: 0.15s (right context).

	3- On mac, open GarageBand an load a virtual instrument, for instance a 'Classic Electric Piano'.
	   You'll also need to activate the IAC driver in MIDI System preferences.
	   On Windows, SoniAladin uses GM directly so you don't need an additional sound engine.  

	4- Run SoniAladin.py from the Terminal or run the Jupyter notebook SoniAladin.ipynb
	
	5- Say 'music' or press 'right arrow' to start the sonification. 
	   Say 'stop' or press 'left arrow' to stop the sonification. 
	   Say 'exit' or press'Esc' to finish.
