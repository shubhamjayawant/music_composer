import numpy as np
import random
from music21 import instrument, note, stream, chord
from data_processor import DataProcessor

class SongWriter():

    def __init__(self, model, notes, songs):

        self.model = model
        self.notes = notes
        self.songs = songs

    def write_song(self, output_file_name):

        pitchnames = sorted(set(item for item in self.notes))
        n_vocab = len(set(self.notes))
        network_input, normalized_input = DataProcessor.prepare_sequences(pitchnames, n_vocab)
        prediction_output = self.__generate_notes(network_input, pitchnames, n_vocab)
        self.create_midi(prediction_output, output_file_name)

    def __pick_a_song(self):

        return random.choice(list(self.songs.items()))[0]

    def __get_part_of_song(self, song, part):

        index = int(part * len(self.songs[song]))

        if index == len(self.songs[song]):

            index -= 1

        return self.songs[song][index]

    def __get_start(self, song):

        return self.__get_part_of_song(song, 0)

    def __get_current_part(self, note_index, total_notes):

        return note_index / total_notes

    def __generate_notes(self, network_input, pitchnames, n_vocab):

        int_to_note = list(pitchnames)
        song = self.__pick_a_song()
        print('Picked ', song)
        start_part = self.__get_start(song)
        start = int_to_note.index(start_part)
        pattern = network_input[start]
        prediction_output = []
        total_notes = 500

        for note_index in range(1,total_notes + 1):

            if note_index % 40 == 0:

                pattern = pattern[-70:] + network_input[int_to_note.index(self.__get_part_of_song(song, self.__get_current_part(note_index, total_notes)))][:30]

            prediction_input = np.reshape(pattern, (1, len(pattern), 1))
            prediction_input = prediction_input / float(n_vocab)
            prediction = self.model.predict(prediction_input, verbose=0)
            index = np.argmax(prediction)
            result = int_to_note[index]
            prediction_output.append(result)
            pattern.append(index)
            pattern = pattern[1:len(pattern)]

        return prediction_output

    def __create_midi(self, prediction_output, output_file_name):

        offset = 0
        output_notes = []

        for pattern in prediction_output:

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

            else:

                new_note = note.Note(pattern)
                new_note.offset = offset
                new_note.storedInstrument = instrument.Piano()
                output_notes.append(new_note)

            offset += 0.5

        midi_stream = stream.Stream(output_notes)
        midi_stream.write('midi', fp= output_file_name)
