from os import path
import pickle
import glob
import numpy as np
from keras.utils import np_utils

class DataProcessor():

    @staticmethod
    def prepare_sequences(self, notes, n_vocab, sequence_length = 50):
        
        pitchnames = sorted(set(item for item in notes))
        note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
        network_input = []
        network_output = []

        for i in range(len(notes) - sequence_length):

            sequence_in = notes[i:i + sequence_length]
            sequence_out = notes[i + sequence_length]
            network_input.append([note_to_int[char] for char in sequence_in])
            network_output.append(note_to_int[sequence_out])

        n_patterns = len(network_input)
        network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
        network_input = network_input / float(n_vocab)
        network_output = np_utils.to_categorical(network_output)
        return (network_input, network_output)

    @staticmethod
    def get_parsed_data(self, files_dir, output_dir = config.ARRAY_DIR, write_flag = True):

        songs = self.__load_songs(output_dir)
        notes = self.__load_notes(output_dir)

        if songs is None or notes is None:

            songs, notes = self.__parse_files(files_dir, output_dir, write_flag)

        return songs, notes

    def __parse_files(self, songs_dir, output_dir, write_flag):

        notes = []
        files = {}

        for file in glob.glob(songs_dir + "*.mid"):

            midi = converter.parse(file)
            files[file] = []
            notes_to_parse = None

            try:

                s2 = instrument.partitionByInstrument(midi)
                notes_to_parse = s2.parts[0].recurse() 

            except:

                notes_to_parse = midi.flat.notes

            for element in notes_to_parse:

                thing_to_add = None

                if isinstance(element, note.Note):

                    thing_to_add = str(element.pitch)

                elif isinstance(element, chord.Chord):

                    thing_to_add = '.'.join(str(n) for n in element.normalOrder)

                if thing_to_add is not None:
                    
                    files[file].append(thing_to_add)
                    notes.append(thing_to_add)

        if write_flag:

            with open(output_dir + 'notes', 'wb') as filepath:

                pickle.dump(notes, filepath)

            with open(output_dir + 'songs', 'wb') as filepath:

                pickle.dump(files, filepath)

        return notes, files

    def __load_songs(self, directory):

        return self.__load_files(directory, 'songs')

    def __load_notes(self, directory):

        return self.__load_files(directory, 'notes')

    def __load_files(self, directory, file):

        files = None

        if path.exists(directory + file):

            with open(directory + file, 'rb') as filepath:

                files = pickle.load(filepath)

        return files