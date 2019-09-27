import numpy as np
from operator import add
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import random

class DQL(object):
    def __init__(self):
        self.reward = 0;
        self.gamma = 0.9
        self.short_memory = np.array([])
        self.agent_target = 1
        self.agent_predict = 0
        self.learning_rate = 0.0005
        self.model = self.network()
        self.epsilon = 0
        self.actual = []
        self.memory = []

    def get_state(self, snake, food):

            snake_x = snake.body[0][0]
            snake_y = snake.body[0][1]

            food_x = food.x
            food_y = food.y

            state = [

                # danger top:
                ((snake.x_change == 0 and snake.y_change == -1 and (snake_y - 1 < 0 or list(map(add, snake.body[0], [0,-1])) in snake.body)) or # heading top and saw your tail or wall
                 (snake.x_change == 1 and snake.y_change == 0 and ((snake.body[0][0] == 0) or list(map( add, snake.body[0], [0,-1])) in snake.body)) or # heading right and on top there is ur tail and wall,
                 (snake.x_change == -1 and snake.y_change == 0 and ((snake.body[0][0] == 0) or list(map(add, snake.body[0], [0,-1])) in snake.body))),
                # danger bottom
                ((snake.x_change == 0 and snake.y_change == 1  and (snake_y + 1 == 25 or list(map(add, snake.body[0], [0,1])) in snake.body)) or
                 (snake.x_change == 1 and snake.y_change == 0 and ((snake_y + 1 == 25) or list(map(add, snake.body[0], [0,1])) in snake.body)) or
                  (snake.x_change == -1 and snake.y_change == 0 and ( (snake_y + 1 == 25) or list(map(add, snake.body[0], [0,1])) in snake.body ) )),
                # danger right
                ((snake.x_change == 1 and snake.y_change == 0 and (snake_x + 1 == 25 or list(map(add, snake.body[0],[1,0])) in snake.body)) or
                 (snake.x_change == 0 and snake.y_change == -1 and (snake_x + 1 == 25 or list(map(add, snake.body[0],[1,0])) in snake.body)) or
                 (snake.x_change == 0 and snake.y_change == 1 and (snake_x + 1 == 25 or list(map(add, snake.body[0],[1,0])) in snake.body))),
                # danger left
                ((snake.x_change == -1 and snake.y_change == 0 and (snake_x - 1 < 0 or list(map(add, snake.body[0], [-1,0])) in snake.body)) or
                 (snake.x_change == 0 and snake.y_change == -1 and (snake_x - 1 < 0 or list(map(add, snake.body[0], [-1,0])) in snake.body)) or
                 (snake.x_change == 0 and snake.y_change == 1 and (snake_x - 1 < 0 or list(map(add, snake.body[0], [-1,0])) in snake.body))),

                snake.x_change == 1, # moving right
                snake.x_change == -1, # moving left
                snake.y_change == 1, # moving down
                snake.y_change == -1, # moving up

                food_x > snake_x, # food at the right
                food_x < snake_x, # food at the left
                food_y > snake_y, # food at the bottom
                food_y < snake_y  # food at the top
            ]

            for i in range(len(state)):
                if(state[i]): state[i] = 1
                else : state[i] = 0
            return state

    def set_reward(self,snake ,food, crash, old, new):
        self.reward = 0
        if crash :
            self.reward = -10
            print("Crashed !!")
            return self.reward
        if(food):
            self.reward = 5
        if(not food):
            if(new > old): # if the snake is moving far away from the food
                self.reward = -3
            else:
                self.reward = 3
        return self.reward

    def network(self, weights = None):
        model = Sequential([
            Dense(12, activation="relu", input_shape=(12,)),
            Dense(120, activation="relu"), Dropout(0.15),
            Dense(120, activation="relu"), Dropout(0.15),
            Dense(120, activation="relu"), Dropout(0.15),
            Dense(3, activation="softmax")
        ])

        model.compile(loss = 'mse', optimizer=Adam(self.learning_rate))

        if(weights):
            model.load_weights(weights)

        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action,reward, next_state, done)) # (s,a,R,s')

    def replay_new(self, memory):
        if(len(memory) > 1000):
            minibatch = random.sample(memory, 1000)
        else:
            minibatch = memory
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if(not done):
                target = reward + self.gamma * np.amax(self.model.predict(np.array([next_state]))[0])
            target_f = self.model.predict(np.array([state]))
            target_f[0][np.argmax(action)] = target
            self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)

    def train_short_memory(self, state, action, reward, next_state, done):
        target = reward
        if(not done):
            target = reward + self.gamma * np.amax(self.model.predict(np.array([next_state]).reshape((1,12)))[0])
        target_f = self.model.predict(np.array([state]).reshape((1,12)))
        target_f[0][np.argmax(action)] = target
        self.model.fit(np.array([state]).reshape((1,12)), target_f, epochs=1, verbose=0)

        # state will include danger zones, snake's directions
        # and relative body to the food
