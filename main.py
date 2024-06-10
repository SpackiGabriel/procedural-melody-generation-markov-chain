import random

from music21.note import Note
from music21 import stream, metadata
# from music21 import configure
from MarkovChainMelodyGenerator import MarkovChainMelodyGenerator


def create_training_data_little_star():
    '''
    Creates a list of sample training notes for the melody of "Twinkle, Twinkle, Little Star".

    Returns:
        list: List of music21.note.Note objects.
    
    '''
    return [
        Note("C5", quarterLength=1),
        Note("C5", quarterLength=1),
        Note("G5", quarterLength=1),
        Note("G5", quarterLength=1),
        Note("A5", quarterLength=1),
        Note("A5", quarterLength=1),
        Note("G5", quarterLength=2),
        Note("F5", quarterLength=1),
        Note("F5", quarterLength=1),
        Note("E5", quarterLength=1),
        Note("E5", quarterLength=1),
        Note("D5", quarterLength=1),
        Note("D5", quarterLength=1),
        Note("C5", quarterLength=2),
    ]

def create_training_data():
    '''
    Creates a list of sample training notes based on a melody composed by me.

    Returns:
        list: List of music21.note.Note objects.
    
    ''' 
    return [
        Note("C5", quarterLength=1),
        Note("E5", quarterLength=1),
        Note("G5", quarterLength=1),
        Note("F5", quarterLength=1),
        Note("A5", quarterLength=2),
        Note("G5", quarterLength=0.5),
        Note("F5", quarterLength=0.5),
        Note("E5", quarterLength=1),
        Note("D5", quarterLength=1),
        Note("E5", quarterLength=1),
        Note("D5", quarterLength=2),
        Note("C5", quarterLength=2),
        Note("E5", quarterLength=0.5),
        Note("F5", quarterLength=0.5),
        Note("D5", quarterLength=0.5),
        Note("E5", quarterLength=0.5),
        Note("G5", quarterLength=0.5),
        Note("B5", quarterLength=0.5),
        Note("A5", quarterLength=1),
        Note("G5", quarterLength=1),
        Note("F5", quarterLength=0.5),
        Note("A5", quarterLength=0.5),
        Note("G5", quarterLength=2),
        Note("E5", quarterLength=2),
        Note("C5", quarterLength=2),
    ]
    
def visualize_melody(melody):
    '''
    Visualize a sequence of (pitch, duration) pairs using music21.

    Parameters:
        melody (list): List of (pitch, duration) pairs.
    '''

    #configure.run()

    score = stream.Score()
    score.metadata = metadata.Metadata(title="Generated Melody")
    part = stream.Part()

    for n, d in melody:
        part.append(Note(n, quarterLength=d))
    score.append(part)
    score.show()

def main():
    '''
    Main function for training the chain, generating a melody and visualizing the result
    '''

    states = [
        ('C5', 0.5), ('D5', 0.5), ('E5', 0.5), ('F5', 0.5), ('G5', 0.5), ('A5', 0.5), ('B5', 0.5),
        ('C5', 1), ('D5', 1), ('E5', 1), ('F5', 1), ('G5', 1), ('A5', 1), ('B5', 1),
        ('C5', 2), ('D5', 2), ('E5', 2), ('F5', 2), ('G5', 2), ('A5', 2), ('B5', 2),
    ]

    training_data = create_training_data2()

    model = MarkovChainMelodyGenerator(states)
    model.train(training_data)

    generated_modely = model.generate(128)
    visualize_melody(generated_modely)


if __name__ == '__main__':
    main()