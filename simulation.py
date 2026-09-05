import pygame
import random
import math

# ============================================================
# 2D AUTONOMOUS VEHICLE SIMULATION
# Indian Traffic Scenario
# ============================================================

pygame.init()

# -------------------- WINDOW --------------------
WIDTH = 900
HEIGHT = 700

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Indian Traffic - Autonomous Vehicle Simulation")

clock = pygame.time.Clock()

# -------------------- COLORS --------------------
ROAD_COLOR = (55, 55, 55)
ROAD_EDGE = (220, 220, 220)
LANE_COLOR = (180, 180, 180)

CAR_COLOR = (40, 120, 255)
BIKE_COLOR = (255, 180, 30)
WRONG_SIDE_COLOR = (220, 60, 60)
POTHOLE_COLOR = (30, 30, 30)

COST_LOW = (40, 180, 80)
COST_MEDIUM = (255, 200, 40)
COST_HIGH = (220, 50, 50)

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# -------------------- ROAD --------------------
ROAD_LEFT = 180
ROAD_RIGHT = 720
ROAD_WIDTH = ROAD_RIGHT - ROAD_LEFT

LANE_WIDTH = ROAD_WIDTH // 3

# Ego vehicle starts at bottom center
EGO_WIDTH = 42
EGO_HEIGHT = 70

ego_x = WIDTH // 2
ego_y = HEIGHT - 110

ego_speed = 3.0

# Goal at top
goal_x = WIDTH // 2
goal_y = 50

# -------------------- PLANNER PARAMETERS --------------------
SAFETY_DISTANCE = 130
MAX_LATERAL_SHIFT = 3.5

# -------------------- OBSTACLES --------------------
obstacles = []


class Obstacle:
    def __init__(self, x, y, obstacle_type, speed=0):
        self.x = x
        self.y = y
        self.type = obstacle_type
        self.speed = speed

        if obstacle_type == "bike":
            self.width = 28
            self.height = 45

        elif obstacle_type == "wrong_side":
            self.width = 35
            self.height = 55

        elif obstacle_type == "pothole":
            self.width = 45
            self.height = 25

    def update(self):
        if self.type == "bike":
            self.y += self.speed

        elif self.type == "wrong_side":
            self.y += self.speed

        # potholes remain static

    def draw(self):
        if self.type == "bike":
            color = BIKE_COLOR

        elif self.type == "wrong_side":
            color = WRONG_SIDE_COLOR

        else:
            color = POTHOLE_COLOR

        pygame.draw.rect(
            screen,
            color,
            (
                int(self.x - self.width / 2),
                int(self.y - self.height / 2),
                self.width,
                self.height
            )
        )


# -------------------- SPAWN OBSTACLES --------------------
def spawn_bike_cutting_in():
    """
    Two-wheeler enters from the left or right side.
    """

    side = random.choice(["left", "right"])

    if side == "left":
        x = ROAD_LEFT - 20
        target_x = random.randint(ROAD_LEFT + 80, ROAD_RIGHT - 80)

    else:
        x = ROAD_RIGHT + 20
        target_x = random.randint(ROAD_LEFT + 80, ROAD_RIGHT - 80)

    bike = Obstacle(
        x,
        random.randint(100, 450),
        "bike",
        random.uniform(1.5, 3.0)
    )

    bike.target_x = target_x
    bike.cut_speed = random.uniform(1.0, 2.0)

    obstacles.append(bike)


def spawn_wrong_side_vehicle():
    """
    Vehicle travelling against the normal traffic direction.
    """

    x = random.randint(ROAD_LEFT + 40, ROAD_RIGHT - 40)

    vehicle = Obstacle(
        x,
        random.randint(50, 250),
        "wrong_side",
        random.uniform(1.0, 2.0)
    )

    obstacles.append(vehicle)


def spawn_pothole():
    """
    Static pothole placed on the road.
    """

    x = random.randint(
        ROAD_LEFT + 50,
        ROAD_RIGHT - 50
    )

    y = random.randint(150, HEIGHT - 150)

    pothole = Obstacle(
        x,
        y,
        "pothole"
    )

    obstacles.append(pothole)


# -------------------- COSTMAP --------------------
def calculate_costmap():
    """
    Simple dynamic costmap.

    Each obstacle increases the cost around itself.
    Higher cost = more dangerous area.
    """

    costmap = []

    grid_size = 30

    for y in range(0, HEIGHT, grid_size):

        row = []

        for x in range(ROAD_LEFT, ROAD_RIGHT, grid_size):

            cost = 0

            for obstacle in obstacles:

                distance = math.sqrt(
                    (x - obstacle.x) ** 2 +
                    (y - obstacle.y) ** 2
                )

                if distance < 120:
                    cost += max(0, 120 - distance)

            row.append(cost)

        costmap.append(row)

    return costmap


# -------------------- COLLISION CHECK --------------------
def check_collision(x1, y1, obstacle):

    ego_rect = pygame.Rect(
        x1 - EGO_WIDTH // 2,
        y1 - EGO_HEIGHT // 2,
        EGO_WIDTH,
        EGO_HEIGHT
    )

    obstacle_rect = pygame.Rect(
        obstacle.x - obstacle.width // 2,
        obstacle.y - obstacle.height // 2,
        obstacle.width,
        obstacle.height
    )

    return ego_rect.colliderect(obstacle_rect)


# -------------------- LOCAL PLANNER --------------------
def local_obstacle_avoidance():

    global ego_x

    desired_shift = 0

    closest_obstacle = None
    closest_distance = float("inf")

    # Find obstacle in front of ego vehicle
    for obstacle in obstacles:

        dx = obstacle.x - ego_x
        dy = obstacle.y - ego_y

        # Only consider obstacles ahead
        if dy < 0:

            distance = math.sqrt(dx * dx + dy * dy)

            if distance < closest_distance:
                closest_distance = distance
                closest_obstacle = obstacle

    # No obstacle nearby
    if closest_obstacle is None:
        return

    # Safety threshold
    if closest_distance < SAFETY_DISTANCE:

        obstacle_x = closest_obstacle.x

        # Obstacle is on the left
        if obstacle_x < ego_x:

            desired_shift = MAX_LATERAL_SHIFT

        # Obstacle is on the right
        else:

            desired_shift = -MAX_LATERAL_SHIFT

        # If obstacle is almost directly ahead,
        # select safer side using road boundaries
        if abs(obstacle_x - ego_x) < 50:

            distance_left = ego_x - ROAD_LEFT
            distance_right = ROAD_RIGHT - ego_x

            if distance_left > distance_right:
                desired_shift = -MAX_LATERAL_SHIFT
            else:
                desired_shift = MAX_LATERAL_SHIFT

    # Apply lateral movement
    ego_x += desired_shift

    # Keep ego vehicle inside road
    ego_x = max(
        ROAD_LEFT + EGO_WIDTH // 2,
        min(ROAD_RIGHT - EGO_WIDTH // 2, ego_x)
    )


# -------------------- DRAW ROAD --------------------
def draw_road():

    pygame.draw.rect(
        screen,
        ROAD_COLOR,
        (
            ROAD_LEFT,
            0,
            ROAD_WIDTH,
            HEIGHT
        )
    )

    # Road edges
    pygame.draw.line(
        screen,
        ROAD_EDGE,
        (ROAD_LEFT, 0),
        (ROAD_LEFT, HEIGHT),
        5
    )

    pygame.draw.line(
        screen,
        ROAD_EDGE,
        (ROAD_RIGHT, 0),
        (ROAD_RIGHT, HEIGHT),
        5
    )

    # Lane markings
    for lane in range(1, 3):

        x = ROAD_LEFT + lane * LANE_WIDTH

        for y in range(0, HEIGHT, 50):

            pygame.draw.rect(
                screen,
                LANE_COLOR,
                (x - 2, y, 4, 25)
            )


# -------------------- DRAW EGO CAR --------------------
def draw_ego_car():

    pygame.draw.rect(
        screen,
        CAR_COLOR,
        (
            int(ego_x - EGO_WIDTH / 2),
            int(ego_y - EGO_HEIGHT / 2),
            EGO_WIDTH,
            EGO_HEIGHT
        )
    )

    # Front windshield
    pygame.draw.rect(
        screen,
        (150, 220, 255),
        (
            int(ego_x - 14),
            int(ego_y - 20),
            28,
            15
        )
    )


# -------------------- DRAW GOAL --------------------
def draw_goal():

    pygame.draw.circle(
        screen,
        (50, 220, 100),
        (goal_x, goal_y),
        18
    )

    font = pygame.font.SysFont(None, 25)

    text = font.render(
        "GOAL",
        True,
        WHITE
    )

    screen.blit(
        text,
        (goal_x - 25, goal_y - 40)
    )


# -------------------- DRAW COSTMAP --------------------
def draw_costmap():

    grid_size = 30

    for y in range(0, HEIGHT, grid_size):

        for x in range(ROAD_LEFT, ROAD_RIGHT, grid_size):

            cost = 0

            for obstacle in obstacles:

                distance = math.sqrt(
                    (x - obstacle.x) ** 2 +
                    (y - obstacle.y) ** 2
                )

                if distance < 100:

                    cost = max(
                        cost,
                        100 - distance
                    )

            if cost > 0:

                if cost > 70:
                    color = COST_HIGH

                elif cost > 35:
                    color = COST_MEDIUM

                else:
                    color = COST_LOW

                surface = pygame.Surface(
                    (grid_size, grid_size),
                    pygame.SRCALPHA
                )

                surface.fill(
                    (*color, 50)
                )

                screen.blit(
                    surface,
                    (x, y)
                )


# -------------------- MAIN LOOP --------------------
running = True

spawn_timer = 0
pothole_timer = 0
wrong_side_timer = 0

while running:

    clock.tick(60)

    # ---------------- EVENTS ----------------
    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

        # Press R to restart
        if event.type == pygame.KEYDOWN:

            if event.key == pygame.K_r:

                ego_x = WIDTH // 2
                obstacles.clear()

    # ---------------- SPAWN ----------------
    spawn_timer += 1
    pothole_timer += 1
    wrong_side_timer += 1

    # Two-wheeler cutting in
    if spawn_timer > 100:

        if random.random() < 0.65:
            spawn_bike_cutting_in()

        spawn_timer = 0

    # Wrong-side vehicle
    if wrong_side_timer > 180:

        if random.random() < 0.45:
            spawn_wrong_side_vehicle()

        wrong_side_timer = 0

    # Potholes
    if pothole_timer > 240:

        if random.random() < 0.5:
            spawn_pothole()

        pothole_timer = 0

    # ---------------- UPDATE OBSTACLES ----------------
    for obstacle in obstacles:

        # Cutting-in bikes move toward road
        if obstacle.type == "bike":

            if abs(obstacle.x - obstacle.target_x) > 5:

                if obstacle.x < obstacle.target_x:
                    obstacle.x += obstacle.cut_speed
                else:
                    obstacle.x -= obstacle.cut_speed

            obstacle.y += obstacle.speed

        else:

            obstacle.update()

    # Remove objects that leave screen
    obstacles = [
        obstacle
        for obstacle in obstacles
        if obstacle.y < HEIGHT + 100
    ]

    # ---------------- PLANNING ----------------
    calculate_costmap()

    local_obstacle_avoidance()

    # ---------------- EGO MOVEMENT ----------------
    ego_y -= ego_speed

    # Goal reached
    if ego_y < goal_y + 20:

        ego_y = HEIGHT - 110
        ego_x = WIDTH // 2
        obstacles.clear()

    # ---------------- COLLISION ----------------
    collision = False

    for obstacle in obstacles:

        if check_collision(
            ego_x,
            ego_y,
            obstacle
        ):

            collision = True

    # ---------------- DRAW ----------------
    screen.fill((30, 150, 30))

    draw_road()

    # Costmap visualization
    draw_costmap()

    draw_goal()

    # Obstacles
    for obstacle in obstacles:
        obstacle.draw()

    # Ego vehicle
    draw_ego_car()

    # ---------------- UI ----------------
    font = pygame.font.SysFont(None, 26)

    title = font.render(
        "Indian Traffic Autonomous Vehicle",
        True,
        WHITE
    )

    screen.blit(
        title,
        (20, 20)
    )

    status = "AVOIDING OBSTACLE" if not collision else "COLLISION!"

    status_text = font.render(
        status,
        True,
        WHITE
    )

    screen.blit(
        status_text,
        (20, 50)
    )

    controls = font.render(
        "R = Restart",
        True,
        WHITE
    )

    screen.blit(
        controls,
        (20, HEIGHT - 35)
    )

    pygame.display.flip()


pygame.quit()
