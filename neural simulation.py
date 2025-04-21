import numpy as np
import matplotlib.pyplot as plt
import pygame
import pygame_gui


# # Generate class 0: points centered around (1, 1)
# class_0 = np.random.randn(100, 2) * 0.5 + np.array([1, 1])
# labels_0 = np.zeros((100, 1))

# # Generate class 1: points centered around (3, 3)
# class_1 = np.random.randn(100, 2) * 0.5 + np.array([3, 3])
# labels_1 = np.ones((100, 1))  dataset too small

def generate_class(center, n_points, std = 0.5):       #2d normally dist data
    return np.random.randn(n_points, 2) * std + np.array(center)   #returns an array of shape (points, 2)

class_0 = generate_class([1, 1], 1000) #class centered around (1, 1)
labels_0 = np.zeros((1000, 1))

class_1 = generate_class([3, 3], 1000) #class centered around (3, 3)
labels_1 = np.ones((1000, 1))

# Combine the data, stacking vertically 
X = np.vstack((class_0, class_1))
y = np.vstack((labels_0, labels_1))

# Shuffle the dataset
shuffle = np.random.permutation(len(X))
X, y = X[shuffle], y[shuffle]


# # Visualize
# plt.scatter(X[:, 0], X[:, 1], c = y.squeeze(), cmap = "bwr")
# plt.title("2D Classification Dataset")
# plt.show()


#Split the data
split_idx = int(0.8 * len(X))
#80% for training 
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

#djusting the dataset for the model

# NOISE_STD = 0.2  #adding noise to avoid overfitting

# X_train_noisy = X_train + np.random.normal(0, NOISE_STD, X_train.shape)
# X_val_noisy = X_val + np.random.normal(0, NOISE_STD, X_val.shape)

#model expects input to be (features, samples)
V_train = X_train.T #becomes (2, N)
VV_train = y_train.astype(int).flatten() #becomes (N,), 1D array ints

V_val = X_val.T
VV_val = y_val.astype(int).flatten()

#making the model

def init_params():
    W1 = np.random.rand(10, 2) - 0.5  #inputs to hidden (10x2)
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(2, 10) - 0.5  #hidden to outputs (2x10)
    b2 = np.random.rand(2, 1) - 0.5
    return W1, b1, W2, b2
# -0.5 to centre weights around zero, so that there are negative values for the relu to not be biased/gradients flow in the same direction


def ReLu(Z):
    return np.maximum(0, Z)

# def Softmax(Z):
#     return np.exp(Z) / np.sum(np.exp(Z))

def Softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis = 0, keepdims = True))  #so it works column-wise
    return exp_Z / np.sum(exp_Z, axis = 0, keepdims = True)

def dropout(A, drop_prob):
    mask = (np.random.rand(*A.shape) > drop_prob).astype(float)
    return A * mask / (1.0 - drop_prob), mask

def forward_prop(W1, b1, W2, b2, V, Training =  True):
    Z1 = W1.dot(V) + b1
    A1 = ReLu(Z1)
    if Training:
        A1, dropout_mask = dropout(A1, 0.2)  # 20% dropout
    Z2 = W2.dot(A1) + b2
    A2 = Softmax(Z2)
    return Z1, A1, Z2, A2

def one_hot(VV):
    one_hot_VV = np.zeros((VV.size, VV.max() + 1))
    one_hot_VV[np.arange(VV.size), VV] = 1
    one_hot_VV = one_hot_VV.T
    return one_hot_VV

def deriv_relu(Z):
    return Z > 0 #simple mask so negative values go zero

def back_prop(Z1, A1, Z2, A2, W2, V, VV):
    m = VV.size
    one_hot_VV = one_hot(VV)
    dZ2 = A2 - one_hot_VV    #cross-entropy loss for softmax outputs
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis = 1, keepdims = True)
    dZ1 = W2.T.dot(dZ2) * deriv_relu(Z1)
    dW1 = 1 / m * dZ1.dot(V.T)
    db1 = 1 / m * np.sum(dZ1, axis = 1, keepdims = True)
    return dW1, db1, dW2, db2

def update_param(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha): #gradient descent 
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)  #predict class with highest softmax output

def get_accuracy(predictions, VV):
    preds = predictions.tolist()
    print("\nPredictions:", preds[:10])
    acts = VV.tolist()
    print("True Labels:", acts[:10])
    return np.sum(predictions == VV) / VV.size

def gradient_descent(V, VV, U, UU, iterations, alpha):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, V, Training = True)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, V, VV)
        W1, b1, W2, b2 = update_param(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 50 == 0:
            print("Iteration", i)
            train_acc = get_accuracy(get_predictions(A2), VV)
            print("Accuracy", train_acc)
            _, _, _, A2_val = forward_prop(W1, b1, W2, b2, U, Training = False)
            val_acc = get_accuracy(get_predictions(A2_val), UU)
            print("Validation Accuracy:", val_acc)
    return W1, b1, W2, b2

def draw_weighted_line(surface, start_pos, end_pos, weight, max_thickness = 5):
    thickness = int(min(max_thickness, abs(weight) * max_thickness))
    color = (0, 0, 0) if weight >= 0 else (150, 0, 0)  # darker red for negative weights and black for positive 
    if thickness > 0: #skips line if weight zero
        pygame.draw.line(surface, color, start_pos, end_pos, thickness)

def draw_activated_neuron(surface, pos, activation, base_color = (0, 0, 0)):
    # activation in [0,1] scaled to 255
    intensity = int(min(255, activation * 255)) #higher activation better glow
    color = (intensity, intensity, 255)  # bluish glow
    pygame.draw.circle(surface, color, pos, 10)
    pygame.draw.circle(surface, base_color, pos, 10, 2)


# def draw_network(screen, A1, A2, W1, W2):
#     input_layer_pos = [(100, 200), (100, 300)]
#     hidden_layer_pos = [(300, 100 + i * 40) for i in range(10)]
#     output_layer_pos = [(500, 250), (500, 350)]

#     # Draw neurons
#     for pos in input_layer_pos + hidden_layer_pos + output_layer_pos:
#         pygame.draw.circle(screen, (0, 0, 0), pos, 10, 2)

#     # Draw lines from input to hidden
#     for inp in input_layer_pos:
#         for hid in hidden_layer_pos:
#             pygame.draw.line(screen, (150, 150, 150), inp, hid, 1)

#     # Draw lines from hidden to output
#     for hid in hidden_layer_pos:
#         for out in output_layer_pos:
#             pygame.draw.line(screen, (150, 150, 150), hid, out, 1)

#     return input_layer_pos, hidden_layer_pos, output_layer_pos

def draw_network(screen, A1, A2, W1, W2):
    input_layer_pos = [(100, 200), (100, 300)]
    hidden_layer_pos = [(300, 100 + i * 40) for i in range(10)]
    output_layer_pos = [(500, 250), (500, 350)]
    # Draw weighted lines from input → hidden
    for i, inp in enumerate(input_layer_pos):
        for j, hid in enumerate(hidden_layer_pos):
            weight = W1[j, i]                       #weight from input i to hidden j 
            draw_weighted_line(screen, inp, hid, weight)

    # Draw weighted lines from hidden → output
    for i, hid in enumerate(hidden_layer_pos):
        for j, out in enumerate(output_layer_pos):
            weight = W2[j, i]                      #weight from hidden j to output i 
            draw_weighted_line(screen, hid, out, weight)

    # Draw neurons with activation
    for i, pos in enumerate(input_layer_pos):
        draw_activated_neuron(screen, pos, 1.0)  # input always "active"

    for i, pos in enumerate(hidden_layer_pos):
        activation = A1[i, 0] if A1.shape[1] > 0 else 0  # first sample
        draw_activated_neuron(screen, pos, activation)

    for i, pos in enumerate(output_layer_pos):
        activation = A2[i, 0] if A2.shape[1] > 0 else 0
        draw_activated_neuron(screen, pos, activation)


pygame.init()

WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Neural Network Visualization") 

manager = pygame_gui.UIManager((WIDTH, HEIGHT))

sliders = {
    "Iterations": pygame_gui.elements.UIHorizontalSlider(
        relative_rect = pygame.Rect((620, 50), (150, 30)),
        start_value = 100,
        value_range = (10, 1000),
        manager = manager
    ),
    "Alpha": pygame_gui.elements.UIHorizontalSlider(
        relative_rect = pygame.Rect((620, 100), (150, 30)),
        start_value = 0.01,
        value_range = (0.001, 1.0),
        manager = manager
    ),
    "Start": pygame_gui.elements.UIButton(
        relative_rect = pygame.Rect((620, 160), (150, 40)),
        text = "Start",
        manager = manager
    ),
    "Reset": pygame_gui.elements.UIButton(
        relative_rect = pygame.Rect((620, 250), (150, 40)),
        text = "Reset",
        manager = manager
    )
}

font = pygame.font.SysFont(None, 24)
labels = {
    "Iterations": font.render("Iterations", True, (0, 0, 0)),
    "Alpha": font.render("Alpha", True, (0, 0, 0))
}

value_labels = {
    "Iterations": pygame_gui.elements.UILabel(
        relative_rect = pygame.Rect((780, 50), (60, 30)),
        text = "100",
        manager = manager
    ),
    "Alpha": pygame_gui.elements.UILabel(
        relative_rect = pygame.Rect((780, 100), (60, 30)),
        text = "0.010",
        manager = manager
    )
}

def draw_legend(surface):
    font = pygame.font.SysFont(None, 20)
    x, y = 560, 400
    spacing = 25

    pygame.draw.rect(surface, (230, 230, 230), (x - 10, y - 10, 220, 170), border_radius=8)

    legend_items = [
        ("Neuron (circle)", (0, 0, 0)),
        ("Positive weight", (0, 0, 0)),
        ("Negative weight", (150, 0, 0)),
        ("Activation glow", (100, 100, 255)),
        ("Class 0 Label", (200, 0, 0)),
        ("Class 1 Label", (0, 200, 0)),
    ]

    for i, (label, color) in enumerate(legend_items):
        item_y = y + i * spacing
        if "Neuron" in label:
            pygame.draw.circle(surface, color, (x + 10, item_y + 5), 6, 2)
        elif "weight" in label:
            pygame.draw.line(surface, color, (x, item_y + 5), (x + 20, item_y + 5), 3)
        elif "Activation" in label:
            pygame.draw.circle(surface, color, (x + 10, item_y + 5), 6)
        elif "Class 0" in label:
            pygame.draw.circle(surface, color, (x + 10, item_y + 5), 6)
        elif "Class 1" in label:
            pygame.draw.circle(surface, color, (x + 10, item_y + 5), 6)

        label_surface = font.render(label, True, (0, 0, 0))
        surface.blit(label_surface, (x + 30, item_y))

clock = pygame.time.Clock()   #default 60fps
running = True
iteration = 0
state = "idle"
test_input = None
px, py = 0, 0
pred_class = 0

W1, b1, W2, b2 = init_params()
while running:
    time_delta = clock.tick(60) / 1000.0 #time between each frame 
    screen.fill((255, 255, 255))  #screen goes white between the frames 

    for event in pygame.event.get():
        if event.type == pygame.QUIT:   #x to quit
            running = False
        manager.process_events(event)

        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element == sliders["Start"]:   #press start and sees where the sliders are gets their values and stores them to be used for training
                state = "training"
                iteration = 0
                ITER = int(sliders["Iterations"].get_current_value())
                ALPHA = sliders["Alpha"].get_current_value()

            elif event.ui_element == sliders["Reset"]: #reset everything sliders, parameters and all so that a new model can run
                state = "idle"
                sliders["Iterations"].set_current_value(100)
                sliders["Alpha"].set_current_value(0.01)
                value_labels["Iterations"].set_text("100")
                value_labels["Alpha"].set_text("0.010")
                W1, b1, W2, b2 = init_params()
                iteration = 0
                screen.fill((255, 255, 255))
        if event.type == pygame.MOUSEBUTTONDOWN and state == "done":  #after training when you click at a point
            mx, my = pygame.mouse.get_pos()                         #get the coords of where clicked 
            x = mx / WIDTH * 4
            y = my / HEIGHT * 4
            test_input = np.array([[x], [y]])
            _, A1_tmp, _, A2_tmp = forward_prop(W1, b1, W2, b2, test_input, Training = False)  #pass that through the loop to see what class it predicts
            pred_class = np.argmax(A2_tmp)
            px, py = mx, my

    value_labels["Iterations"].set_text(str(int(sliders["Iterations"].get_current_value())))
    value_labels["Alpha"].set_text(f"{sliders['Alpha'].get_current_value():.3f}")  #keeps labels in sync as we drag the slider

    # manager.update(time_delta)

    # manager.draw_ui(screen)
    # pygame.display.update()

    if state == "idle":
        draw_network(screen, np.ones((10, 1)), np.ones((2, 1)), W1, W2)  # dummy activations for initial display
        # screen.blit(labels["Iterations"], (620, 30))
        # screen.blit(labels["Alpha"], (620, 80))
    
    elif state == "training":
        #one training step per frame
        test_input = None
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, V_train, Training = True)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, V_train, VV_train)
        W1, b1, W2, b2 = update_param(W1, b1, W2, b2, dW1, db1, dW2, db2, ALPHA)
        draw_network(screen, A1, A2, W1, W2)

        # manager.update(time_delta)

        # screen.blit(labels["Iterations"], (620, 30))
        # screen.blit(labels["Alpha"], (620, 80))

        # manager.draw_ui(screen)
        # pygame.display.update()

        iteration += 1
        if iteration >= ITER:
            state = "done"
            final_preds = get_predictions(A2)
            acc = get_accuracy(final_preds, VV_train)
            print(f"\nFinal Training Accuracy: {acc:.2f}")
            #stores the final accuracy and state switches to done

    elif state == "done":
        # time_delta = clock.tick(60) / 1000.0
        # screen.fill((255, 255, 255))

        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT:
        #         running = False

        # Final forward pass without dropout
        _, A1_final, _, A2_final = forward_prop(W1, b1, W2, b2, V_train, Training = False)

        draw_network(screen, A1_final, A2_final, W1, W2)

        # px = int(test_input[0, 0] / 4 * WIDTH)
        # py = int(test_input[1, 0] / 4 * HEIGHT)
        if test_input is not None:
            color = [(200, 0, 0), (0, 200, 0)][pred_class]
            pygame.draw.circle(screen, color, (px, py), 6)
            class_text = font.render(f"Prediction: {pred_class}", True, (0, 0, 0))
            class_label = font.render(f"Class {pred_class}", True, (0, 0, 0))
            screen.blit(class_label, (px + 10, py - 10))
            screen.blit(class_text, (620, 180))

        accuracy_text = font.render(f"Final Accuracy: {acc:.2f}", True, (0, 0, 0))  
        screen.blit(accuracy_text, (620, 220))

    manager.update(time_delta)

    screen.blit(labels["Iterations"], (620, 30))
    screen.blit(labels["Alpha"], (620, 80))
    manager.draw_ui(screen)
    draw_legend(screen)
    pygame.display.update()
pygame.quit()


