import config
from data_processor import DataProcessor
from lstm import LSTM
from song_writer import SongWriter

def main():

    songs, notes = DataProcessor.get_parsed_data(config.MIDI_FILES_DIR)
    vocab = len(set(notes))

    network_input, network_output = DataProcessor.prepare_sequences(notes, vocab)
    lstm = LSTM(network_input, vocab, config.WEIGHTS_DUMP, config.PRETRAINED_MODEL)
    # lstm = LSTM(network_input, vocab, config.WEIGHTS_DUMP)
    # lstm.train(network_input, network_output)

    song_writer = SongWriter(lstm, notes, songs)
    song_writer.write_song(config.OUTPUT_DIR + config.SONG_NAME + config.MIDI_EXTENSION)

if __name__ == '__main__':
    
    main()