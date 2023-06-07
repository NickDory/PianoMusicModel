import numpy as np
import tensorflow as tf
from music21 import converter, instrument, note, chord

def main():

    

    # Step 1: Data Collection
    # Load the MIDI files
    midi_files = glob.glob("Users/nickdory/Documents/Pianomusicmodel/BachInventions")

    # Step 2: Data Preprocessing
    notes = []

    # Extract notes and chords from MIDI files
    for file in midi_files:
        midi = converter.parse(file)
        notes_to_parse = None

        try:
            # Piano instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse()
        except:
            # If no piano parts found, take all notes
            notes_to_parse = midi.flat.notes

        # Store pitch and duration of each note/chord
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    # Step 3: Feature Extraction
    # Define vocabulary of unique notes and chords
    pitch_names = sorted(set(notes))
    n_vocab = len(pitch_names)

    # Map notes/chords to numerical representation
    note_to_int = dict((note, number) for number, note in enumerate(pitch_names))

    # Create input sequences and corresponding output labels
    sequence_length = 100  # Number of previous notes to consider
    network_input = []
    network_output = []

    for i in range(len(notes) - sequence_length):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    # Reshape and normalize input data
    n_patterns = len(network_input)
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    network_input = network_input / float(n_vocab)

    # Convert output labels to categorical format
    network_output = tf.keras.utils.to_categorical(network_output)

    # Step 4: Model Architecture
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(256, input_shape=(network_input.shape[1], network_input.shape[2])))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(n_vocab, activation='softmax'))

    # Step 5: Model Training
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(network_input, network_output, epochs=50, batch_size=64)

    # Step 6: Model Evaluation

    # Step 7: Music Generation
    # Generate new music
    start_sequence = network_input[0]  # Initial input sequence
    generated_notes = []

    for _ in range(500):
        input_sequence = np.reshape(start_sequence, (1, len(start_sequence), 1))
        input_sequence = input_sequence / float(n_vocab)

        # Predict the next note
        predicted_probs = model.predict(input_sequence)[0]
        predicted_index = np.argmax(predicted_probs)
        predicted_note = pitch_names[predicted_index]
        generated_notes.append(predicted_note)

    # Slide the input sequence window by one note
    start_sequence.append(predicted_index)
    start_sequence = start_sequence[1:]

if __name__ == "__main__":
    main()


# Step 8: Postprocessing
# Convert numerical notes to MIDI format
# You can use the `music21` library here to create MIDI files from the generated notes