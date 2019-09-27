import pygame
import numpy as np
import pandas as pd
import DQL
from keras.utils import to_categorical

pygame.init()
win = pygame.display.set_mode((500,500))
pygame.display.set_caption("Snake ML")
pygame.display.update()

def init_screen(win):
    line_interval = 20
    line_color = (255,255,255)

    win.fill((0,0,0))
    for i in range(25):
        pygame.draw.line(win,line_color,(0,i*line_interval),(500,i*line_interval))
        pygame.draw.line(win,line_color,(i*line_interval,0),(i*line_interval,500))
# init_screen(win)

class food:
    def __init__(self, default_x = None, default_y = None):
        self.color = (0,255,0)
        if(default_x != None):
            self.x = default_x
        else:
            self.x = np.random.randint(0,25)
        if(default_y != None):
            self.y = default_y
        else:
            self.y = np.random.randint(0,25)
        self.width = 20
        self.height = 20
        self.eaten = False

    def spawn(self, win):
        pygame.draw.rect(win, self.color, (self.width * self.x, self.height*self.y,self.width,self.height))

class snake:
    def __init__(self):
        self.bite = False
        self.length = 1;
        self.body = [[np.random.randint(0,25),np.random.randint(0,25)]]
        self.width = 20
        self.height = 20
        self.velocity = 20
        self.color = (255,0,0)
        self.x_change = 1 # Heading right
        self.y_change = 0 # Not changing

    def check_collide(self):
        return (self.body[0] in self.body[1::]) or self.body[0][0] < 0 or\
            self.body[0][1] < 0 or self.body[0][0] == 25 or self.body[0][1] == 25
    def move_by_keys(self, win):
        keys = pygame.key.get_pressed()
        if(keys[pygame.K_DOWN]):
            self.y_change = 1
            self.x_change = 0
        if(keys[pygame.K_UP]):
            self.y_change = -1
            self.x_change = 0
        if(keys[pygame.K_LEFT]):
            self.x_change = -1
            self.y_change = 0
        if(keys[pygame.K_RIGHT]):
            self.x_change = 1
            self.y_change = 0
        self.body.insert(0, [(self.body[0][0] + self.x_change), (self.body[0][1] + self.y_change)])
        self.body.pop(len(self.body) - 1)

        for node in self.body:
            pygame.draw.rect(win, self.color, (node[0]*20, node[1]*20, self.width, self.height))

    def move(self, win, key):
        move_array = [self.x_change, self.y_change]

        if np.array_equal(key ,[1, 0, 0]):
            move_array = self.x_change, self.y_change
        elif np.array_equal(key,[0, 1, 0]) and self.y_change == 0:  # right - going horizontal
            move_array = [0, self.x_change]
        elif np.array_equal(key,[0, 1, 0]) and self.x_change == 0:  # right - going vertical
            move_array = [-self.y_change, 0]
        elif np.array_equal(key, [0, 0, 1]) and self.y_change == 0:  # left - going horizontal
            move_array = [0, -self.x_change]
        elif np.array_equal(key,[0, 0, 1]) and self.x_change == 0:  # left - going vertical
            move_array = [self.y_change, 0]
        self.x_change, self.y_change = move_array

        self.body.insert(0, [(self.body[0][0] + self.x_change), (self.body[0][1] + self.y_change)])
        self.body.pop(len(self.body) - 1)

        for node in self.body:
            pygame.draw.rect(win, self.color, (node[0] * 20, node[1] * 20, self.width, self.height))

    def spawn(self, win):
        pygame.draw.rect(win, self.color, (self.body[0][0]*20, self.body[0][1]*20, self.width, self.height))

def run_manually():
    food_ = food()
    snake_ = snake()
    snake_.spawn(win)
    run = True

    while run:
        pygame.time.delay(100)
        init_screen(win) # draw black screen
        food_.spawn(win) # then draw food on top
        snake_.move_by_keys(win) # then draw snake
        pygame.display.update()

        if(food_.x == snake_.body[0][0] and food_.y == snake_.body[0][1]):
            food_ = food() # if food is eaten, get new food
            snake_.body.append(last_tail)

        if(snake_.check_collide()):
            run = False

        last_tail = [snake_.body[0][0], snake_.body[0][1]]

        for event in pygame.event.get():
            if(event.type == pygame.QUIT):
                pygame.display.quit()
                run = False
    pygame.quit()

def run_auto():
    counter_game = 0;
    agent = DQL.DQL()

    while(counter_game < 150):
        food_ = food()
        snake_ = snake()
        snake_.spawn(win)
        crash = False
        print("Epsilon = " + str(agent.epsilon))
        while not crash:
            food_eaten = False
            pygame.time.delay(0)
            init_screen(win)
            food_.spawn(win)
            agent.epsilon = 80 - counter_game

            state_old = agent.get_state( snake = snake_, food= food_)
            prediction = None
            old_distance = np.sqrt(np.square(snake_.body[0][0] - food_.x) + np.square(snake_.body[0][1] - food_.y))

            if(np.random.randint(0,1000) < agent.epsilon):
                final_move = to_categorical(np.random.randint(0,2), num_classes=3)
            else:
                prediction = agent.model.predict(np.array([state_old]).reshape((1,12)))
                final_move = to_categorical(np.argmax(prediction[0]), num_classes=3)
            snake_.move(win, final_move)
            new_distance = np.sqrt(np.square(snake_.body[0][0] - food_.x) + np.square(snake_.body[0][1] - food_.y))
            if (food_.x == snake_.body[0][0] and food_.y == snake_.body[0][1]):
                food_ = food()  # if food is eaten, get new food
                snake_.body.append(last_tail)
                food_eaten = True

            if(snake_.check_collide()):
                crash = True

            reward = agent.set_reward(snake_, food_eaten, crash, old_distance, new_distance)
            last_tail = [snake_.body[0][0], snake_.body[0][1]]

            pygame.display.update()
            state_new = agent.get_state(snake_, food_)

            agent.train_short_memory(state_old, final_move, reward, state_new, crash)

            agent.remember(state_old, final_move, reward, state_new, crash)

            for event in pygame.event.get():
                if (event.type == pygame.QUIT):
                    pygame.display.quit()
                    crash = True

        agent.replay_new(agent.memory)
        counter_game += 1
        crash = False
        print('Game', counter_game, '     Score : ', len(snake_.body))

run_auto()
