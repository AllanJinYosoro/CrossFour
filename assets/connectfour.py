import gymnasium as gym
import numpy as np
import pygame
import time

class ConnectFourEnv(gym.Env):
    def __init__(self, rows=6, cols=7, render_mode=None, win_condition="four_in_a_row"):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self._board = np.zeros((self.rows, self.cols), dtype=int)  # Private board
        self.current_player = 1
        self.render_mode = render_mode
        self.win_condition = win_condition  # "four_in_a_row" 

        # Define action and observation space
        self.action_space = gym.spaces.Discrete(self.cols)
        self.observation_space = gym.spaces.Box(low=0, high=2, shape=(self.rows, self.cols), dtype=int)

        if self.render_mode == "human":
            pygame.init()
            screen_info = pygame.display.Info()
            
            # Determine max cell size based on screen width and height
            max_width = screen_info.current_w // self.cols
            max_height = (screen_info.current_h - 100) // (self.rows + 2)  # Leave extra space for both images
            self.cell_size = min(max_width, max_height, 100)  # Cap cell size at 100
            
            # Set up the screen with calculated dimensions
            self.screen_width = self.cols * self.cell_size
            self.screen_height = (self.rows + 2) * self.cell_size
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Connect Four")

            # Load and scale the background and logo images
            self.load_images()
    
    @property
    def board(self):
        return self._board.copy()  # Always return a copy of the board
    
    @board.setter
    def board(self, value):
        raise AttributeError("Direct modification of the board is not allowed.")

    def load_images(self):
        # Load and scale ACEMbuilding.png as the main background image
        building_path = "assets/ACEMbuilding.png"  # Path to ACEMbuilding.png
        self.building_image = pygame.image.load(building_path)
        self.building_image = pygame.transform.scale(self.building_image, (self.screen_width, self.screen_height - self.cell_size))

        # Load ACEMlogo.png and preserve its aspect ratio
        logo_path = "assets/ACEMlogo.png"  # Path to ACEMlogo.png
        self.logo_image = pygame.image.load(logo_path)
        logo_original_width, logo_original_height = self.logo_image.get_size()
        
        # Scale the logo to a maximum width (e.g., half the board width) while maintaining aspect ratio
        max_logo_width = self.screen_width // 2
        scale_factor = max_logo_width / logo_original_width
        logo_width = int(logo_original_width * scale_factor)
        logo_height = int(logo_original_height * scale_factor)
        self.logo_image = pygame.transform.scale(self.logo_image, (logo_width, logo_height))
        
        # Calculate position to center the logo horizontally at the top
        self.logo_x = (self.screen_width - logo_width) // 2
        self.logo_y = 0  # Position at the top

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._board = np.zeros((self.rows, self.cols), dtype=int)
        self.current_player = 1
        if self.render_mode == "human":
            self.render()  # Render the initial empty board
        return self.board.copy()

    def step(self, action):
        # Check if the chosen column is full
        if self.board[0, action] != 0:
            # Column is full; current player loses the game
            done = True
            reward = -1  # Assign a negative reward to indicate a loss for the current player
            return self.board.copy(), reward, done, False

        # Drop the piece in the chosen column
        for row in range(self.rows - 1, -1, -1):
            if self.board[row, action] == 0:
                self._board[row, action] = self.current_player
                break

        # Check for win based on selected condition
        if self.win_condition == "four_in_a_row":
            won = self.check_winner(self.current_player)
        else:
            print("Invalid win condition")

        full_board = np.all(self.board[0, :] != 0)  # Check if the top row is full (indicating a full board)
        done = won or full_board
        reward = 1 if won else 0  # Reward is 1 if the player meets the win condition, 0 for a draw

        # Toggle player if the game is not over
        if not done:
            self.current_player = 3 - self.current_player  # Toggle player

        if self.render_mode == "human":
            self.render()  # Render after each move

        return self.board.copy(), reward, done, False

    def check_winner(self, player):
        # Check all win conditions (horizontal, vertical, diagonal)
        for row in range(self.rows):
            for col in range(self.cols - 3):
                if np.all(self.board[row, col:col + 4] == player):
                    return True
        for row in range(self.rows - 3):
            for col in range(self.cols):
                if np.all(self.board[row:row + 4, col] == player):
                    return True
        for row in range(self.rows - 3):
            for col in range(self.cols - 3):
                if np.all([self.board[row + i, col + i] == player for i in range(4)]):
                    return True
                if np.all([self.board[row + 3 - i, col + i] == player for i in range(4)]):
                    return True
        return False


    def render(self):
        if self.render_mode == "human":
            # Draw the logo image at the top center
            self.screen.blit(self.logo_image, (self.logo_x, self.logo_y))

            # Draw the background image below the logo
            self.screen.blit(self.building_image, (0, self.cell_size))

            # Draw the Connect Four grid on top
            for row in range(self.rows):
                for col in range(self.cols):
                    color = (255, 255, 255)  # Empty cell
                    if self.board[row, col] == 1:
                        color = (255, 0, 0)  # Player 1
                    elif self.board[row, col] == 2:
                        color = (255, 255, 0)  # Player 2
                    pygame.draw.circle(self.screen, color,
                                       (col * self.cell_size + self.cell_size // 2,
                                        (row + 2) * self.cell_size + self.cell_size // 2),
                                       self.cell_size // 2 - 5)
            pygame.display.flip()

    def get_human_action(self):
        """ Waits for a mouse click and returns the corresponding column if valid. """
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    x, _ = pygame.mouse.get_pos()
                    col = x // self.cell_size  # Determine the clicked column
                    if 0 <= col < self.cols and self.board[0, col] == 0:
                        return col  # Return the selected column if valid

    def close(self):
        if self.render_mode == "human":
            pygame.quit()


# Play Match Function with Flexible Player Types
def play_match(env, player1, player2):
    env.reset()
    done = False
    winner = None
    
    # Map player functions for each turn
    players = {1: player1, 2: player2}

    while not done:
        current_player = env.current_player
        player = players[current_player]

        # Determine action based on player type
        if player == "human":
            action = env.get_human_action()
        else:
            action = player(env.board.copy())  # Call the policy function
            
        # print(f"Player {current_player} selects column {action}")

        # Take a step
        _, reward, done, _ = env.step(action)
        
        # Render if applicable
        if env.render_mode == "human":
            env.render()  # Render each move to show the board update immediately
            time.sleep(0.3)

        # Determine winner at the end of the game
        if done:
            if reward == 1:
                winner = current_player  # Current player is the winner
            elif reward == -1:
                winner = 3 - current_player   # Other player wins if reward is -1
            else:
                winner = 0  # Indicate a draw if reward is 0

    env.close()
    return winner