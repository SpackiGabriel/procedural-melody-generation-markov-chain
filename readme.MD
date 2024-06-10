# Markov Chain Melody Generator

This Python repository provides a simple implementation of a Markov Chain model for melody generation. The Markov Chain is trained on a given set of musical notes and can generate new melodies based on the learned patterns.

## Usage

To use this package, follow these steps:

1. **Installation**: Install the required dependencies using pip:

    ```bash
    pip install numpy music21
    ```

2. **Training**: Create training data by defining a list of music21.note.Note objects. You can use the provided `create_training_data()` or `create_training_data2()` functions as examples.

3. **Instantiate Model**: Create an instance of the `MarkovChainMelodyGenerator` class by passing a list of states (pitch-duration pairs) to the constructor.

4. **Training the Model**: Train the model using the `train()` method and passing the training data.

5. **Generating Melodies**: Generate melodies using the `generate()` method, specifying the length of the melody.

6. **Visualization**: Visualize the generated melody using the `visualize_melody()` function.

## Example

Here's a simple example demonstrating how to use the Markov Chain Melody Generator:

```python
from MarkovChainMelodyGenerator import MarkovChainMelodyGenerator, visualize_melody, create_training_data2

def main():
    # Define states (pitch-duration pairs)
    states = [
        ('C5', 0.5), ('D5', 0.5), ('E5', 0.5), ('F5', 0.5), ('G5', 0.5), ('A5', 0.5), ('B5', 0.5),
        ('C5', 1), ('D5', 1), ('E5', 1), ('F5', 1), ('G5', 1), ('A5', 1), ('B5', 1),
        ('C5', 2), ('D5', 2), ('E5', 2), ('F5', 2), ('G5', 2), ('A5', 2), ('B5', 2),
    ]

    # Create training data
    training_data = create_training_data2()

    # Instantiate and train the model
    model = MarkovChainMelodyGenerator(states)
    model.train(training_data)

    # Generate a melody
    generated_melody = model.generate(128)

    # Visualize the generated melody
    visualize_melody(generated_melody)

if __name__ == '__main__':
    main()
```

## Note

This repository assumes familiarity with the music21 library for music notation handling. Ensure you have the library installed and configured appropriately for accurate results.