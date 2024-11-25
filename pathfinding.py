# Import the libraries
import pygame
from queue import PriorityQueue, deque, LifoQueue
import random

# Define constants
WIDTH = 800
WIN = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption("Pathfinding Visualizer")

# Define colors
RED = (255, 0, 0)
ORANGE = (255, 165, 0)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
PURPLE = (128, 0, 128)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (128, 128, 128)

class Node:
  def __init__(self, row, col, width, total_rows):
    self.row = row
    self.col = col
    self.x = row * width
    self.y = col * width
    self.width = width
    self.total_rows = total_rows
    self.color = WHITE
    self.neighbors = []

  def get_pos(self):
    return self.row, self.col

  def is_start(self):
    return self.color == RED

  def is_target(self):
    return self.color == ORANGE

  def is_barrier(self):
    return self.color == BLACK

  def is_open(self):
    return self.color == YELLOW

  def is_closed(self):
    return self.color == GREEN
  
  def reset(self):
    self.color = WHITE

  def make_start(self):
    self.color = RED

  def make_target(self):
    self.color = ORANGE

  def make_barrier(self):
    self.color = BLACK

  def make_open(self):
    self.color = YELLOW

  def make_closed(self):
    self.color = GREEN

  def make_path(self):
    self.color = BLUE

  def draw(self, win):
    pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))

  def update_neighbors(self, grid):
    self.neighbors = []

    if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrier(): # DOWN
      self.neighbors.append(grid[self.row + 1][self.col])

    if self.row > 0 and not grid[self.row - 1][self.col].is_barrier(): # UP
      self.neighbors.append(grid[self.row - 1][self.col])

    if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_barrier(): # RIGHT
      self.neighbors.append(grid[self.row][self.col + 1])

    if self.col > 0 and not grid[self.row][self.col - 1].is_barrier(): # LEFT
      self.neighbors.append(grid[self.row][self.col - 1])

  def __lt__(self, other):
    return False

# h is the heuristic function. h(p1, p2) estimates the cost to reach p2 from p1.
def h(p1, p2):
  x1, y1 = p1
  x2, y2 = p2
  return abs(x1 - x2) + abs(y1 - y2)

def reconstruct_path(came_from, current, draw):
  while current in came_from:
    current = came_from[current]
    current.make_path()
    draw()

# A* finds a path from start to target
def a_star(draw, grid, start, target):
  # Break ties with nodes with the same f_score value in the priority queue
  count = 0

  # The set of discovered nodes that may need to be (re-)expanded.
  # Initially, only the start node is known.
  open_set = PriorityQueue()
  open_set.put((0, count, start))

  # For node n, came_from[n] is the node immediately preceding it on the cheapest path from the start.
  came_from = {}

  # For node n, g_score[n] is the cost of the cheapest path from start to n currently known.
  g_score = {node: float("inf") for row in grid for node in row}
  g_score[start] = 0

  # For node n, f_score[n] = g_score[n] + h(p1, p2). f_score[n] represents our current best guess as to how cheap a path could be from start to finish if it goes through n.
  f_score = {node: float("inf") for row in grid for node in row}
  f_score[start] = 0

  # open_set_hash synchronizes with open_set.
  open_set_hash = {start}

  while not open_set.empty():
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        pygame.quit()

    # The node in open_set having the lowest f_score value
    current = open_set.get()[2]
    open_set_hash.remove(current)

    if current == target:
      reconstruct_path(came_from, target, draw)
      start.make_start()
      target.make_target()
      return True


    for neighbor in current.neighbors:
      # The weight of the edge from current to neighbor is 1.
      # tentative_g_score is the distance from start to the neighbor.
      tentative_g_score = g_score[current] + 1 
      if tentative_g_score < g_score[neighbor]:
        # This path to neighbor is better than any previous one.
        came_from[neighbor] = current
        g_score[neighbor] = tentative_g_score
        f_score[neighbor] = tentative_g_score + h(neighbor.get_pos(), target.get_pos()) 
        if neighbor not in open_set_hash:
          count += 1
          open_set.put((f_score[neighbor], count, neighbor))
          open_set_hash.add(neighbor)
          neighbor.make_open()

    draw()

    if current != start:
      current.make_closed()

  # open_set is empty but goal was never reached.
  return False

def bfs(draw, grid, start, target):
  # Contains the frontier along with the algorithm is currently searching
  queue = deque()
  queue.append(start)

  # Trace the shortest path back to start
  parent = {}

  # Label start as explored
  explored = {node: False for row in grid for node in row}
  explored[start] = True

  while len(queue) > 0:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        pygame.quit()

    current = queue.popleft()

    if current == target:
      reconstruct_path(parent, current, draw)
      start.make_start()
      target.make_target()
      return True 
    
    for neighbor in current.neighbors:
      if not explored[neighbor]:
        # Label neighbor as explored
        explored[neighbor] = True
        parent[neighbor] = current
        queue.append(neighbor)
        neighbor.make_open()

    draw()

    if current != start:
      current.make_closed()
  
  return False

def dijkstra(draw, grid, start, target):
  # Break ties with nodes with the same distance in the priority queue
  count = 0

  # Set of all the unvisited nodes
  unvisited_set = PriorityQueue()
  unvisited_set.put((0, count, start))

  # Contains pointers to previous-hop nodes on the shortest path from the start to the given node
  prev = {}

  # Contains the current distances from the start to other nodes
  dist = {node: float('inf') for row in grid for node in row}
  dist[start] = 0

  # unvisited_set_hash synchronizes with unvisited_set
  unvisited_set_hash = {start}

  while not unvisited_set.empty():
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        pygame.quit()

    # Searches for the node in the unvisited set that has the least distance
    current = unvisited_set.get()[2]
    unvisited_set_hash.remove(current)

    if current == target:
      reconstruct_path(prev, current, draw)
      start.make_start()
      target.make_target()
      return True

    for neighbor in current.neighbors:
      # The distance between neighbor and current is 1
      # alt is the length of the path from the start node to the neighbor node if
      alt = dist[current] + 1
      # If the path is shorter than the current shortest path, then the distance of neighbor is updated to alt.
      if alt < dist[neighbor]:
        count += 1
        prev[neighbor] = current
        dist[neighbor] = alt
        unvisited_set.put((alt, count, neighbor))
        unvisited_set_hash.add(neighbor)
        neighbor.make_open()

    draw () 

    if current != start:
      current.make_closed()

  # unvisited_set is empty or contains only nodes with infinite distance (which are unreachable)
  return False

# Randomized depth-first search

# Recursive division method

def make_grid(rows, width):
  grid = []
  gap = width // rows

  for i in range(rows):
    grid.append([])
    for j in range(rows):
      node = Node(i, j, gap, rows)
      grid[i].append(node)

  return grid

def draw_grid(win, rows, width):
  gap = width // rows

  for i in range(rows):
    pygame.draw.line(win, GREY, (0, i * gap), (width, i * gap))
    for j in range(rows):
      pygame.draw.line(win, GREY, (j * gap, 0), (j * gap, width))

def draw(win, grid, rows, width):
  win.fill(WHITE)

  for row in grid:
    for node in row:
      node.draw(win)

  draw_grid(win, rows, width)
  pygame.display.update()

def get_clicked_pos(pos, rows, width):
  gap = width // rows
  y, x = pos

  row = y // gap
  col = x // gap

  return row, col

def main(win, width):
  ROWS = 50
  grid = make_grid(ROWS, width)

  start = None
  target = None

  row = random.randint(0, 49)
  col = random.randint(0, 49)
  
  run = True
  while run:
    draw(win, grid, ROWS, width)
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        run = False

      if pygame.mouse.get_pressed()[0]: # LEFT
        pos = pygame.mouse.get_pos()
        row, col = get_clicked_pos(pos, ROWS, width)
        node = grid[row][col]
        if not start and node != target:
          start = node
          start.make_start()

        elif not target and node != start:
          target = node
          target.make_target()

        elif node != target and node != start:
          node.make_barrier()

      elif pygame.mouse.get_pressed()[2]: # RIGHT
        pos = pygame.mouse.get_pos()
        row, col = get_clicked_pos(pos, ROWS, width)
        node = grid[row][col]
        node.reset()
        if node == start:
          start = None
        elif node == target:
          target = None

      if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_c: # CLEAR
          start = None
          target = None
          grid = make_grid(ROWS, width)

        if event.key == pygame.K_a and start and target: # A* SEARCH ALGORITHM
          for row in grid:
            for node in row:
              node.update_neighbors(grid)

          a_star(lambda: draw(win, grid, ROWS, width), grid, start, target)

        if event.key == pygame.K_b and start and target: # BREADTH-FIRST SEARCH
          for row in grid:
            for node in row:
              node.update_neighbors(grid)

          bfs(lambda: draw(win, grid, ROWS, width), grid, start, target)

        if event.key == pygame.K_d and start and target: # DIJKSTRA'S ALGORITHM
          for row in grid:
            for node in row:
              node.update_neighbors(grid)

          dijkstra(lambda: draw(win, grid, ROWS, width), grid, start, target)
  pygame.quit()

main(WIN, WIDTH)
