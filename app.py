from flask import Flask, render_template, request, flash, redirect, url_for, send_file
#import requests
import pandas as pd
import numpy as np
from musicalNeuralNet.generate import *
from pianoComposer.predict import *
import os
from werkzeug.utils import secure_filename
import os
import glob

app = Flask(__name__, static_url_path='/static')


song_list = ['joplin_sunflowerDrag', 'joplin_entertainer', 'memphisSlim_cowCowBlues']
advanced_song_list = ['oscar_peterson', 'bach_goldbergVariations', 'theolonious_monk']
training_level_list = ['light', 'light', 'extra']


@app.route('/', methods=['GET', 'POST'])
def index():
    print("In index")
    global SONG_CHOICE_INDEX
    global ADVANCED_SONG_CHOICE_INDEX
    SONG_CHOICE_INDEX = None
    ADVANCED_SONG_CHOICE_INDEX = None
    return render_template('index.html')

@app.route('/downloads', methods=['GET', 'POST'])
def downloads():
    #clear existing contents of midi and pdf output folders      
    midi_files = glob.glob('beginner/_midi_output/*')
    for f in midi_files:
        os.remove(f)
        
    pdf_files = glob.glob('beginner/_pdf_output/*')
    for f in pdf_files:
        os.remove(f)

    oscar_peterson_files = glob.glob('advanced/data/output/oscar_peterson/*')
    for f in oscar_peterson_files: 
        os.remove(f)

    theolonious_monk_files = glob.glob('advanced/data/output/theolonious_monk/*')
    for f in theolonious_monk_files: 
        os.remove(f)

    bach_goldbergVariations_files = glob.glob('advanced/data/output/bach_goldbergVariations/*')
    for f in bach_goldbergVariations_files: 
        os.remove(f)       

    #Generate midi and pdf output of music in the style of the chosen song
    for i in range(len(song_list)):
        if request.form['song_name'] == song_list[i]:
            global SONG_CHOICE_INDEX
            SONG_CHOICE_INDEX = int(i)
            
            song_name = song_list[i]
            notes_filepath = '/home/ubuntu/application/beginner/_notes/' + song_name + '_notes'
            weights_filepath = '/home/ubuntu/application/beginner/_weights/' + song_name + '_weights.hdf5'
            midi_filepath = '/home/ubuntu/application/beginner/_midi_output/' + song_name + '_output'
            pdf_filepath = '/home/ubuntu/application/beginner/_pdf_output/' + song_name + '_output'

            generate(notes_filepath, weights_filepath, midi_filepath, pdf_filepath)
            return render_template('downloads.html')
    for i in range(len(advanced_song_list)):
        if request.form['song_name'] == advanced_song_list[i]:
            global ADVANCED_SONG_CHOICE_INDEX
            ADVANCED_SONG_CHOICE_INDEX = int(i)

            #default variables for inputting into compose() below
            test = 'test'    
            train = 'train'
            gen_size = 2000
            sample_freq = 12
            chordwise = False
            note_offset = 45
            use_test_prompt = False
            generator_bs = 1
            trunc = 5
            random_freq = .5
            prompt_size = None
            
            model_to_load = advanced_song_list[ADVANCED_SONG_CHOICE_INDEX]
            training = training_level_list[ADVANCED_SONG_CHOICE_INDEX]
            output_folder = advanced_song_list[ADVANCED_SONG_CHOICE_INDEX]
                
            compose(model_to_load, training, test, train, gen_size, sample_freq, chordwise, note_offset, use_test_prompt, output_folder, generator_bs, trunc, random_freq, prompt_size)
            
            return render_template('downloads.html', message='')
    
    if SONG_CHOICE_INDEX==None and ADVANCED_SONG_CHOICE_INDEX==None:
        return render_template('index.html')
 

@app.route('/return-file/', methods=['GET', 'POST'])
def return_file():
    global SONG_CHOICE_INDEX
    global ADVANCED_SONG_CHOICE_INDEX
    
    if SONG_CHOICE_INDEX != None:
        try:
            return send_file('beginner/_midi_output/' + song_list[SONG_CHOICE_INDEX] + '_output.mid')
        except:
            midi_error_message = 'Sorry, DeepRiff had trouble generating the midi file for this composition. Please try again.'
            return render_template('downloads.html', message=midi_error_message)
        
    if ADVANCED_SONG_CHOICE_INDEX != None:
        try:
            return send_file('advanced/data/output/' + advanced_song_list[ADVANCED_SONG_CHOICE_INDEX] + '/00.mid')
        except:
            midi_error_message = 'Sorry, DeepRiff had trouble generating the midi file for this composition. Please try again.'
            return render_template('downloads.html', message=midi_error_message)

        
@app.route('/return-pdf/', methods=['GET', 'POST'])
def return_pdf():
    global SONG_CHOICE_INDEX
    global ADVANCED_CHOICE_INDEX
    if SONG_CHOICE_INDEX != None:
        try:
            return send_file('beginner/_pdf_output/' + song_list[SONG_CHOICE_INDEX] + '_output.pdf')
        except:
            pdf_error_message = "Sorry, DeepRiff had trouble generating the pdf for this composition. Please try again."
            render_template('downloads.html', message=pdf_error_message)           
    
    if ADVANCED_SONG_CHOICE_INDEX != None:
        try:
            return send_file('advanced/data/output/' + advanced_song_list[ADVANCED_SONG_CHOICE_INDEX] + '/00.pdf')
        except:
            pdf_error_message = "Sorry, DeepRiff had trouble generating the pdf for this composition. Please try again."
            render_template('downloads.html', message=pdf_error_message)
            

if __name__ == '__main__':
    #app.run(host='0.0.0.0', port=5000, debug=True, threaded=False)
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=False)
    
