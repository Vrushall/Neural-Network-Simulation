# Neural-Network-Simulation

This simulation shows how a feedforward neural network learns to classify 2D data points (Class 0 and Class 1) through training.

Each neuron and connection is animated to reflect:
- Weight magnitude (line thickness)
- Weight polarity (black for positive, red for negative)
- Neuron activation (bluer = stronger activation)

# Features
- Custom Neural Network built using Numpy
- Real-time visualisation of network structure, weights, and activations
- Interactive controls: sliders to set iterations and alpha | buttons to start and reset the model
- Legend to help understand the simulation
- Final accuracy displayed after training

# Preview



https://github.com/user-attachments/assets/b62e07ef-28aa-497b-b3dd-1675a0ca38f8



# Workflow

## Libraries
- Numpy
- Pygame
- Pygame_gui

## Data Generation 

- Generated a 2-D normally distributed dataset
- Contains two classes with labels 0 and 1

## Model

- A basic neural network with one hidden layer
- 10 embeddings in the hidden layer and 2 neurons for the input and output layer
- Defined functions init_params, softmax, deriv_relu, dropout, forward_prop, one_hot, back_prop, update_param, get_prediction, get_accuracy and gradient_descent

## Training and Visualisation

- The screen starts with a visualisation of the neural network and sliders 
- The neural network starts training after the user clicks start on the screen
- Each layer is rendered on the screen
- Connections update thickness and colour based on weight
- Neurons glow depending on activation
- Accuracy is printed and displayed after training










This project is open-source and available for everyone to use.

