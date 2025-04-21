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
- Final accuracy is displayed after training

# Preview



https://github.com/user-attachments/assets/b62e07ef-28aa-497b-b3dd-1675a0ca38f8



# Workflow

## Libraries
- Numpy
- Pygame
- Pygame_gui

## Data Generation 

- Generated a 2-D dataset using normally distributed points
- Contains two classes with labels 0 and 1

## Model

- A basic feedforward neural network with two input neurons, one hidden layer with 10 neurons and two output neurons (binary classification)
- Defined core functions from scratch using Nunpy: init_params, softmax, deriv_relu, dropout, forward_prop, one_hot, back_prop, update_param, get_prediction, get_accuracy and gradient_descent

## Training and Visualisation
 
- A static network is displayed on launch with interactive sliders for iterations (training steps) and alpha (learning rate), along with buttons for start and reset
- The neural network starts training after the user clicks Start on the screen
- During training, the connections (weights) are visualised as lines whose thickness and colour change based on magnitude and sign and Neurons glow with blue intensity to reflect their activation values
- After training, the final accuracy is calculated and displayed on-screen and users can reset the model and start over with new settings.











Please let me know if you find a bug!
This project is open-source and available for everyone to use.

