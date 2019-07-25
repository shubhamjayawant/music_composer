# Music Composer

This project makes use of LSTM to compose music. The network is trained over midi files which consist of sequences of notes and chords. Upon minimizing the training error we observe that the network starts producing melodious sounds. Currently the only supported audio format is midi but, with the help of music transcription the support can be extended to variety of other formats.

[Click here](/outputs/great_song.mid) to listen to a sample.

## How does it work:
1. Midi files (ie., training data) are parsed to create sequences and metadata.
2. A bi-axial LSTM is designed by making use of the metadata. The design of the network is as follows,
	1. Input layer, modelled using the information about notes in training data
	2. LSTM layer with 512 nodes
	3. Dropout layer having 0.3 rate
	4. LSTM layer with 512 nodes
	5. Dropout layer having 0.3 rate
	6. LSTM layer with 512 nodes
	7. Fully connected layer with 256 nodes
	8. Dropout layer having 0.3 rate
	7. Fully connected softmax layer which produces the final output
3. This model is then trained over the parsed sequences for nearly 30000 iterations while checking the categorical cross-entropy loss.
4. Upon completion of training, the model is fed with a random seed note and the length of sequence to be produced in order to get the final output.

The weights of the model which was trained over [Coldplay](https://coldplay.com/) songs can be found over [here](/weights/coldplay_weights.hdf5).

## References:
1. LSTM [[paper]](https://www.bioinf.jku.at/publications/older/2604.pdf)
2. [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
3. Generating Music using an LSTM Network [[paper]](https://arxiv.org/pdf/1804.07300.pdf)
