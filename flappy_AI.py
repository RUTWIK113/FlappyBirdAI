import random, pygame, neat

pygame.init()

CLOCK = pygame.time.Clock()
RED = (255, 0, 0)
BLACK = (0, 0, 0)
FPS = 60

WN_WIDTH = 400
WN_HEIGHT = 500
WN = pygame.display.set_mode((WN_WIDTH, WN_HEIGHT))
pygame.display.set_caption("AI Plays Flappy Bird")

BG = pygame.image.load("assets/bird_bg.png")
BIRD_IMG = pygame.image.load("assets/bird.png")
BIRD_SIZE = (40, 26)
BIRD_IMG = pygame.transform.scale(BIRD_IMG, BIRD_SIZE)

GRAVITY = 1
JUMP = 8

PIPE_X0 = 400
PIPE_BOTTOM_IMG = pygame.image.load("assets/pipe.png")
PIPE_TOP_IMG = pygame.transform.flip(PIPE_BOTTOM_IMG, False, True)
PIPE_BOTTOM_HEIGHTS = [90, 122, 154, 186, 218, 250]
GAP_PIPE = 150
PIPE_EVENT = pygame.USEREVENT
pygame.time.set_timer(PIPE_EVENT, 1000)

FONT = pygame.font.SysFont("comicsans", 30)
SCORE_INCREASE = 0.01
GEN = 0


class Pipe:
    def __init__(self, height):
        bottom_midtop = (PIPE_X0, WN_HEIGHT - height)
        top_midbottom = (PIPE_X0, WN_HEIGHT - height - GAP_PIPE)
        self.bottom_pipe_rect = PIPE_BOTTOM_IMG.get_rect(midtop=bottom_midtop)
        self.top_pipe_rect = PIPE_TOP_IMG.get_rect(midbottom=top_midbottom)

    def display_pipe(self):
        WN.blit(PIPE_BOTTOM_IMG, self.bottom_pipe_rect)
        WN.blit(PIPE_TOP_IMG, self.top_pipe_rect)


class Bird:
    def __init__(self):
        self.bird_rect = BIRD_IMG.get_rect(center=(WN_WIDTH // 2, WN_HEIGHT // 2))
        self.dead = False
        self.score = 0
        self.velocity = 0

    def collision(self, pipes):
        for pipe in pipes:
            if self.bird_rect.colliderect(pipe.bottom_pipe_rect) or self.bird_rect.colliderect(pipe.top_pipe_rect):
                return True
        if self.bird_rect.bottom >= WN_HEIGHT or self.bird_rect.top <= 0:
            return True
        return False

    def find_nearest_pipes(self, pipes):
        min_distance = float('inf')
        nearest = None
        for pipe in pipes:
            dist = pipe.bottom_pipe_rect.x - self.bird_rect.x
            if dist > 0 and dist < min_distance:
                min_distance = dist
                nearest = pipe
        return nearest

    def get_normalized_distances(self, pipe):
        dx = (pipe.bottom_pipe_rect.centerx - self.bird_rect.centerx) / WN_WIDTH
        dy_top = (self.bird_rect.top - pipe.top_pipe_rect.bottom) / WN_HEIGHT
        dy_bottom = (pipe.bottom_pipe_rect.top - self.bird_rect.bottom) / WN_HEIGHT
        return [dx, dy_top, dy_bottom]

    def update(self):
        self.velocity += GRAVITY
        self.bird_rect.centery += self.velocity


def game_loop(genomes, config):
    global GEN
    GEN += 1

    birds = []
    nets = []
    ge = []

    pipe_list = []

    for _, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = 0
        birds.append(Bird())
        nets.append(net)
        ge.append(genome)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == PIPE_EVENT:
                pipe_list.append(Pipe(random.choice(PIPE_BOTTOM_HEIGHTS)))

        WN.blit(BG, (0, 0))

        remove_pipes = []
        for pipe in pipe_list:
            pipe.top_pipe_rect.x -= 3
            pipe.bottom_pipe_rect.x -= 3
            pipe.display_pipe()
            if pipe.top_pipe_rect.right < 0:
                remove_pipes.append(pipe)
        for pipe in remove_pipes:
            pipe_list.remove(pipe)

        alive_birds = 0
        max_score = 0

        for i, bird in enumerate(birds):
            if not bird.dead:
                bird.update()
                WN.blit(BIRD_IMG, bird.bird_rect)

                ge[i].fitness += bird.score
                bird.score += SCORE_INCREASE
                alive_birds += 1
                max_score = max(max_score, bird.score)

                if bird.collision(pipe_list):
                    bird.dead = True
                    continue

                nearest_pipe = bird.find_nearest_pipes(pipe_list)
                if nearest_pipe:
                    inputs = bird.get_normalized_distances(nearest_pipe)
                    output = nets[i].activate(inputs)
                    if output[0] > 0.5:
                        bird.velocity = -JUMP

        if alive_birds == 0:
            return

        msg = f"Gen: {GEN} Birds Alive: {alive_birds} Score: {int(max_score)}"
        text = FONT.render(msg, True, BLACK)
        WN.blit(text, (40, 20))

        pygame.display.update()
        CLOCK.tick(FPS)


# NEAT setup
neat_config = neat.config.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    "config.txt"
)

population = neat.Population(neat_config)
population.add_reporter(neat.StatisticsReporter())
population.run(game_loop, 50)
