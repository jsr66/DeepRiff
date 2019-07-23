""" This module generates notes for a midi file using the
    trained neural network """
import pickle
import numpy
from music21 import instrument, note, stream, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import Lambda
from keras.optimizers import Adam
import os, shutil
from midi2audio import FluidSynth
import time

def generate(notes_filepath, weights_filepath, midi_filepath, pdf_filepath):
    #print("IN GENERATE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    """ Generate a piano midi file """
    #load the notes used to train the model
    with open(notes_filepath, 'rb') as filepath:
        notes = pickle.load(filepath)

    # Get all pitch names
    pitchnames = sorted(set(item for item in notes))
    # Get all pitch names
    n_vocab = len(set(notes))

    print("")
    print("n_vocab: " + str(n_vocab))
    print("")

    network_input, normalized_input = prepare_sequences(notes, pitchnames, n_vocab)
    model = create_network(normalized_input, n_vocab, weights_filepath)
    print("model created. model: " + str(model.summary()))  
    prediction_output = generate_notes(model, network_input, pitchnames, n_vocab)
    create_midi(prediction_output, midi_filepath)
    create_pdf(prediction_output, pdf_filepath)
    del model, network_input, normalized_input, prediction_output

def prepare_sequences(notes, pitchnames, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    # map between notes and integers and back
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    sequence_length = 150
    network_input = []
    output = []
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    normalized_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
    normalized_input = normalized_input / float(n_vocab)

    return (network_input, normalized_input)

def create_network(network_input, n_vocab, weights_filename):
    """ create the structure of the neural network """
    try:
        if weights_filename=='/home/ubuntu/application/pianoComposer/_weights/joplin_sunflowerDrag_weights.hdf5':
            T=1
            model = Sequential()
            model.add(LSTM(
                512,
                input_shape=(network_input.shape[1], network_input.shape[2]),
                return_sequences=True
            ))
            model.add(Dropout(0.3))
            model.add(LSTM(512, return_sequences=True))
            model.add(Dropout(0.3))
            #model.add(LSTM(512, return_sequences=True))
            #model.add(Dropout(0.3))
            model.add(LSTM(512))
            model.add(Dense(256))
            model.add(Dropout(0.3))
            model.add(Dense(n_vocab))
            model.add(Lambda(lambda x: x / T))
            model.add(Activation('softmax'))
            model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

            # Load the weights to each node
            model.load_weights(weights_filename)
            print("Joplin Sunflower Slow Drag model compiled")
            return model
            
        if weights_filename=='/home/ubuntu/application/pianoComposer/_weights/joplin_entertainer_weights.hdf5':
            T=1
            model = Sequential()
            model.add(LSTM(
                512,
                input_shape=(network_input.shape[1], network_input.shape[2]),
                return_sequences=True
            ))
            model.add(Dropout(0.3))
            model.add(LSTM(512, return_sequences=True))
            model.add(Dropout(0.3))
            #model.add(LSTM(512, return_sequences=True))
            #model.add(Dropout(0.3))
            model.add(LSTM(512))
            model.add(Dense(256))
            model.add(Dropout(0.3))
            model.add(Dense(n_vocab))
            model.add(Lambda(lambda x: x / T))
            model.add(Activation('softmax'))
            model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

            # Load the weights to each node
            model.load_weights(weights_filename)

            return model
        
        elif weights_filename=='/home/ubuntu/application/pianoComposer/_weights/memphisSlim_cowCowBlues_weights.hdf5':
            T = 1.1 #temperature for softmax, higher temperature leads to more randomness in sampling. T=1 corresponds to usual softmax
            model = Sequential()
            model.add(LSTM(1024,input_shape=(network_input.shape[1], network_input.shape[2]),return_sequences=True))
            #model.add(Dropout(0.3))
            model.add(LSTM(1024, return_sequences=True))
            #model.add(Dropout(0.3))
            model.add(LSTM(1024, return_sequences=True))
            model.add(LSTM(512))
            model.add(Dense(256))
            #model.add(Dropout(0.3))
            model.add(Dense(n_vocab))
            model.add(Lambda(lambda x: x / T))
            model.add(Activation('softmax'))
            adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
            model.compile(loss='categorical_crossentropy', optimizer=adam)
        
            # Load the weights to each node
            model.load_weights(weights_filename)

            return model

    except FileNotFoundError:
        print("Model weight file not found.")
        return None

def generate_notes(model, network_input, pitchnames, n_vocab):
    """ Generate notes from the neural network based on a sequence of notes """
    # pick a random sequence from the input as a starting point for the prediction
    start = numpy.random.randint(0, len(network_input)-1)

    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    pattern = network_input[start]
    prediction_output = []

    # generate 200 notes
    len_seq = 300
    print("length of generated sequence: " + str(len_seq))
    for note_index in range(len_seq):
        prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)
        prediction = numpy.squeeze(prediction)
        
        #####JR
        if note_index%20 == 0:
            print("prediction.shape:" + str(prediction.shape))
            print("prediction:")
            print(str(prediction))
        #####
        '''
        index = numpy.argmax(prediction)
        print("index: " + str(index))
        '''
        #####JR
        index = numpy.random.choice(numpy.arange(len(prediction)), p=prediction)
        print("index: " + str(index))
        #####
        
        result = int_to_note[index]
        prediction_output.append(result)

        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    return prediction_output

def create_midi(prediction_output, midi_filepath):
    """ convert the output from the prediction to notes and create a midi file
        from the notes """
    offset = 0
    output_notes = []

    # create note and chord objects based on the values generated by the model
    for pattern in prediction_output:
        # pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # increase offset each iteration so that notes do not stack
        offset += 0.5

    midi_stream = stream.Stream(output_notes)

    midi_stream.write('midi', fp= midi_filepath + '.mid')
    #convert to audio
    #fs = FluidSynth()
    #fs.midi_to_audio(midi_filepath + '.mid', midi_filepath + '.mp3')
    #time.sleep(5)

def create_pdf(prediction_output, pdf_filepath):
    """ convert the output from the prediction to notes and create a midi file
        from the notes """
    print("IN CREATE_PDF")
    offset = 0
    output_notes = []

    # create note and chord objects based on the values generated by the model
    for pattern in prediction_output:
        # pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # increase offset each iteration so that notes do not stack
        offset += 0.5

    midi_stream = stream.Stream(output_notes)

    try:
        midi_stream.write('lily.pdf', fp= pdf_filepath)
    except:
        print("Lilypond failed to print pdf.")
    

'''
if __name__ == '__main__':
    notes_filepath = '_notes/joplin_sunflowerDrag_notes'
    weights_filepath = '_weights/joplin_sunflowerDrag_weights.hdf5'
    midi_filepath = '_midi_output/joplin_sunflowerDrag_output'
    #generate(notes_filepath, weights_filepath, midi_filepath)
'''
