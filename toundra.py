import pygame_gui
import pygame
import cv2
import pandas as pd
import yaml
import numpy as np
from scipy.linalg import block_diag
from scipy.integrate import odeint
import os
import random
import math
from tqdm import tqdm
import sys


########################################################################################
# ----------------------- Définition des constantes / semi-variables
# NOTA : NE PAS MODIFIER ICI. CHANGER LE FICHIER CONFIG.YAML
########################################################################################


SIM_NAME = sys.argv[2] # nom de la simulation
RUN_SIM = bool(sys.argv[1])   # faire tourner ou réutiliser ?

os.makedirs("Simulations", exist_ok=True)
SIMULATION_SAVE_FOLDER = os.path.join("Simulations", SIM_NAME)

ISO_SCALE = 18 # échelle du rendu
MAX_ELMTS_PER_TILE = 12 # ratio par rapport à iso_scale
ELMT_SIZE = .6 * ISO_SCALE  # taille des arbres en ratio (peut être > 1) par rapport à iso_scale
TPS = 16 # ticks par seconde (=vitesse de lecture)
FIRE_DURATION = 5 # durée en pas de temps (integer)
DEAD_TREE_DURATION = 15 # idem

# Généralités pygame
SCREEN_WIDTH = 1680
SCREEN_HEIGHT = 937

SHOW_GUI = True

# Paramètres rendu isométrique
ANGLE = 23
COS = math.cos(math.radians(ANGLE))
SIN = math.sin(math.radians(ANGLE))

rot = -110
cos_rot = math.cos(math.radians(rot))
sin_rot = math.sin(math.radians(rot))

ORIGIN_X = SCREEN_WIDTH // 2.7
ORIGIN_Y = SCREEN_HEIGHT * (1 - 0.1)

AREA_SCALE = 1  # hectares par pixel

# Max d'arbres (réel)
TREES_T_MIN = 0
TREES_T_MAX = 1.3
# MAX T est valué par data_util lors du chargement des données (voir fonction "build_from_solution")
MAX_T = -1
TILE_SIZE = 1

# constantes couleurs
BROWN = (143, 50, 43)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
DARK_GREEN = (35, 94, 35)
LIGHT_GREEN = (86, 179, 86)
RED = (255, 0, 0)
SNOW = (197, 205, 217)
ICE = (160, 190, 235)

########################################################################################
# ----------------------- Initialisation de Pygame et du videoWriter
########################################################################################

print("Program started.")
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption('IsoForest3')

video_path = os.path.join(SIMULATION_SAVE_FOLDER, "video.avi")
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
video_writer = cv2.VideoWriter(video_path, fourcc, 45, (SCREEN_WIDTH, SCREEN_HEIGHT))


########################################################################################
# ----------------------- Chargement des assets et sources externes
########################################################################################

def read_params_from(path):
    formatted_lines = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                # c'est un param
                param, val = parts
                formatted_lines.append((param + ': ' + str(val)))
            elif len(parts) == 1:
                # c'est un break
                formatted_lines.append("----------")
    
    return formatted_lines


def load_and_scale_image(image_path, size):
    image = pygame.image.load(image_path).convert_alpha()
    return pygame.transform.scale(image, (size, size))


images = {
    "boreal_tree": load_and_scale_image("assets/boreal.png", ELMT_SIZE),
    "dead_tree": load_and_scale_image("assets/deadtree.png", ELMT_SIZE),
    "burning_tree": load_and_scale_image("assets/burning_tree.png", ELMT_SIZE),
}

########################################################################################
# ----------------------- Partie simulation - Différences finies
########################################################################################

# Chargement des paramètres depuis le fichier YAML (config)
alpha = ["alpha"]
beta = ["beta"]
a = ["a"]
b = ["b"]
c = ["c"]
delta = ["delta"]
L = ["L"]
J = ["J"]
T = ["T"]
N = ["N"]
p = ["p"]
freq = ["freq"]
intensity = ["intensity"]
minree = ["minree"]
maxree = ["maxree"]



params = {}
with open("params.txt", 'r') as f:
    for ligne in f:
        # Enlever les espaces et les retours à la ligne
        ligne = ligne.strip()
        
        # Ignore les lignes vides ou les commentaires
        if not ligne or ligne.startswith("#"):
            continue
        
        # Séparer la ligne en clé et valeur
        cle, valeur = ligne.split(":", 1)
        params[cle.strip()] = valeur.strip()
    

# Calcul des paramètres dérivés
dx = L / J
dy = L / J
dt = T / N
dtau = dt / 2
kx = delta * dtau / dx ** 2
ky = delta * dtau / dy ** 2

init_method = "naive"


def randomize_starter(uplus, wplus, umoins, wmoins):
    
    if init_method == "naive":
        # méthode naive : état instable +- 0.5 partout, sur chaque point
        u_0 = np.zeros(J * J)
        w_0 = np.zeros(J * J)
        for i in range(J):
            for j in range(J):
                u_0[i * J + j] = umoins + (-1 + 2 * random.random()) * 0.5
                w_0[i * J + j] = wmoins + (-1 + 2 * random.random()) * 0.5
    
        return u_0, w_0
    
    elif init_method == "bumps":
        # méthode "à bosses"
        max_bosses = 50
        max_height = uplus
        max_d = 20
        
        nb_bosses = random.randint(1, max_bosses)
        u_0 = np.zeros(J * J)
        w_0 = np.zeros(J * J)
        
        for _ in range(nb_bosses):
            
            h = random.random() * max_height
            d = random.randint(1, max_d)
            cx = random.randint(0, J)
            cy = random.randint(0, J)
            
            print(f"h = {h}, d = {d}, cx = {cx}, cy = {cy}")
            
            bosse_func = lambda i, j: h * d / ((cx - i) ** 2 + (cy - j) ** 2 + d)
            
            for i in range(J):
                for j in range(J):
                    u_0[i * J + j] += bosse_func(i, j)
                    w_0[i * J + j] += bosse_func(i, j)
        
        return u_0, w_0
    

# Termes de réaction du modèle
def reaction(X, t):
    u, w = X
    du = alpha * w - q(u)
    dw = -beta * w + alpha * u
    return [du, dw]


def q(u):
    return u * (a * (u - b) ** 2 + c)


def run_simulation(save_folder="Simulations/default"):
    
    os.makedirs(save_folder, exist_ok=True)
    
    dat_file = os.path.join(save_folder, "data.csv")
    param_file = os.path.join(save_folder, "params.txt")
    fires_file = os.path.join(save_folder, "fires.dat")
    
    ### Écriture des paramètres dans un fichier texte
    with open(param_file, "w") as f_:
        f_.write("Simulation " + SIMULATION_SAVE_FOLDER + "\n")
        f_.write("----------\n")
        for key, value in config["params_simulation"].items():
            f_.write(f"{key} {value}\n")
        # Saut de ligne avant les paramètres dérivés
        f_.write("----------\n")
        # Écriture des paramètres dérivés
        derived_params = {
            "dx": dx,
            "dy": dy,
            "dt": dt,
            "dtau": dtau,
            "kx": kx,
            "ky": ky
        }
        for key, value in derived_params.items():
            f_.write(f"{key} {value}\n")
    
    print(f"Paramètres enregistrés dans {param_file}")
    
    # Équilibres du modèle
    uplus = b + np.sqrt((alpha ** 2 - beta * c) / (a * beta))
    wplus = alpha * uplus / beta
    
    umoins = b - np.sqrt((alpha ** 2 - beta * c) / (a * beta))
    wmoins = alpha * umoins / beta
    
    print("Uplus = (", uplus, ", ", wplus, ")")
    print("Umoins = (", umoins, ", ", wmoins, ")")
    
    # Condition initiale
    x = np.linspace(0.0, L, J)
    y = np.linspace(0.0, L, J)
    X = np.linspace(0.0, L * L, J * J)
    
    u0, w0 = randomize_starter(uplus, wplus, umoins, wmoins)
    
    # Matrice du schéma
    diag = np.ones(J)
    diagsup = np.ones(J - 1)
    D = np.diag(diag * (1 + 2 * (kx + ky)), 0) + np.diag(diagsup * (-ky), 1) + np.diag(diagsup * (-ky), -1)
    A = block_diag(D)
    for i in range(J - 1):
        A = block_diag(A, D)
    
    # Conditions au bord de Neumann
    for k in [0, J - 1, J * J - J, J * J - 1]:
        A[k][k] = 1 + kx + ky
    for n in range(1, J - 1):
        n1 = J * n
        n2 = J * n + J - 1
        A[n1][n1] = 1 + 2 * kx + ky
        A[n2][n2] = 1 + 2 * kx + ky
    for k in range(1, J - 1):
        A[k][k] = 1 + kx + 2 * ky
    for k in range(J * J - J + 1, J * J - 1):
        A[k][k] = 1 + kx + 2 * ky
    
    grandediag = np.ones(J * (J - 1))
    A = A + np.diag(grandediag * (-ky), J) + np.diag(grandediag * (-ky), -J)
    
    # Calculs
    print("Inversion de matrices...")
    invA = np.linalg.inv(A)
    
    u = u0
    w = w0
    
    print('Calculs en cours...')
    file = open(dat_file, 'w')
    file_fire = open(fires_file, 'w')
    file.write('t i j u w \n')
    
    for t in tqdm(range(T)):
        for i in range(J):
            for j in range(J):
                file.write(
                    str(t) + ' ' + str(i) + ' ' + str(j) + ' ' + str(u[i * J + j]) + ' ' + str(w[i * J + j]) + '\n')
        
        """Méthode de Strang"""
        # diffusion 1/2 pas
        u = np.dot(invA, u)
        w = np.dot(invA, w)
        
        # réaction 1 pas
        for i in range(J):
            for j in range(J):
                X0 = [u[i * J + j], w[i * J + j]]
                orbit = odeint(reaction, X0, [0, dt])
                newpoint = orbit[-1]
                u[i * J + j], w[i * J + j] = newpoint.T
        
        # diffusion 1/2 pas
        u = np.dot(invA, u)
        w = np.dot(invA, w)
        
        if t % freq == 0:
            if np.random.binomial(1, p) == 1:
                
                xee = random.random() * J
                yee = random.random() * J
                
                ree = minree + random.random() * (maxree - minree)
                
                file_fire.write(str(t) + ' ' + str(xee) + ' ' + str(yee) + ' '
                                + str(ree) + ' ' + str(intensity) + '\n')
                
                var_ee = lambda i, j, var: var[i * J + j] * (1 - intensity) if ((i - xee) ** 2 + (
                        j - yee) ** 2) < ree ** 2 else var[i * J + j]
                
                for i in range(J):
                    for j in range(J):
                        u[i * J + j] = var_ee(i, j, u)
                        w[i * J + j] = var_ee(i, j, w)
    
    print('Fin du programme de simulation.')


def get_dfs(from_folder="Simulations/default"):
    df = pd.read_csv(os.path.join(from_folder, "data.csv"),
                     names=["t", "x", "y", "u", "w"], header=0, sep=" ", index_col=False)
    df = df.drop_duplicates()
    
    df_feux = pd.read_csv(os.path.join(from_folder, "fires.dat"),
                          names=['t', 'x', 'y', 'r', 'I'], header=None, sep=" ", index_col=False)
    
    return df, df_feux


########################################################################################
# ----------------------- Gestion rendu isométrique
########################################################################################

# utilisé dans Tile.draw_tile. Les coordonnées de Nodes et Fires sont cartésiennes.
def projeter_en_isometrique(x, y):
    cx = x
    cy = y
    
    xr = (cx * cos_rot) - (cy * sin_rot)
    yr = (cx * sin_rot) + (cy * cos_rot)
    
    x_proj = (-xr - yr) * COS * ISO_SCALE + ORIGIN_X
    y_proj = (-xr + yr) * SIN * ISO_SCALE + ORIGIN_Y
    
    return x_proj, y_proj


def grid_to_iso(i, j):
    """ Convertit les coordonnées de la grille (i, j) en coordonnées isométriques. """
    
    # Coordonnées centrées sur l'origine
    cx = i * TILE_SIZE / 2
    cy = j * TILE_SIZE / 2
    
    # Appliquer la rotation et la mise à l'échelle isométrique
    xr = (cx * cos_rot) - (cy * sin_rot)
    yr = (cx * sin_rot) + (cy * cos_rot)
    
    x_iso = (-xr - yr) * COS * ISO_SCALE + ORIGIN_X
    y_iso = (-xr + yr) * SIN * ISO_SCALE + ORIGIN_Y
    
    return x_iso, y_iso


########################################################################################
# ----------------------- Construction de l'UI
########################################################################################

manager = pygame_gui.UIManager((SCREEN_WIDTH, SCREEN_HEIGHT))
ui_height = 25

slider_rect = pygame.Rect((10, SCREEN_HEIGHT - ui_height),
                          (SCREEN_WIDTH - 20, ui_height))
play_button_rect = pygame.Rect((10, SCREEN_HEIGHT - ui_height - 60 - 10), (60, 60))

slider = pygame_gui.elements.UIHorizontalSlider(
    relative_rect=slider_rect,
    start_value=0,
    value_range=(0, 0),  # value range adaptée à MAX_T dans main seulement (MAX_T pas encore valué)
    manager=manager,
)

play_button = pygame_gui.elements.UIButton(
    relative_rect=play_button_rect,
    text="autoplay",
    manager=manager,
)


########################################################################################
# ----------- Importation des résultats FreeFem++ et construction des nodes
########################################################################################


# Node : sommet de maille, avec vecteurs des valeurs en fonction du temps
class Node:
    """ Une node est d'abord un point réel / non-interpolé, donné par la sortie freefem"""
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.u = None
        
        self.fires = None


# build_from_solution : construction des nodes à partir du dossier de sortie FreeFem++
def build_from_solution(saved_simulation_folder="Simulations/default"):
    global MAX_T
    
    print(f"Building from solution folder \"{saved_simulation_folder}\"...")
    nodes__ = []
    
    df, df_feux = get_dfs(saved_simulation_folder)
    MAX_T = df["t"].max()
    # nota : le df_feux est passé en retour, il n'est pas utilisé avant que les mailles soient définies
    
    print(f"\tMax time is {MAX_T}")
    
    u_min, u_max = df["u"].min(), df["u"].max()
    
    unique_xy_list = list(df[['x', 'y']].drop_duplicates().itertuples(index=False, name=None))
    
    print("TILE_SIZE = ", TILE_SIZE)
    
    for xy in unique_xy_list:
        x = xy[0]
        y = xy[1]
        node = Node(x, y)
        
        filtered_df = df[(df['x'] == x) & (df['y'] == y)]
        
        node.u = filtered_df['u'].to_numpy()
        nodes__.append(node)
    
    print(f"\tBuilt {len(nodes__)} nodes.")
    print(f"\tu_min = {u_min}")
    print(f"\tu_max = {u_max}")
    print("\n")
    
    return nodes__, df_feux


########################################################################################
# ----------- Gestion des mailles à partir des nodes récupérées ci-dessus
########################################################################################

# classe simple pour stocker les feux proprement
class Fire:
    
    def __init__(self, t_start, x, y, r, I):
        self.t_start = t_start
        # coordonnées cartésiennes
        self.x = x
        self.y = y
        self.r = r
        self.I = I


# class::Tree : arbre avec sa position (isométrique) son type et affiché ou non. Méthode d'affichage incluse.
class Tree:
    
    def __init__(self, state, *pos):
        
        self.pos_x = pos[0]
        self.pos_y = pos[1]
        
        self.iso_pos = (pos[0] - ELMT_SIZE // 2, pos[1] - ELMT_SIZE)
        # états : 'hide', 'boreal', 'burning', 'dead'
        self.states = ['hide'] * MAX_T
    
    # affichage de l'arbre en fonction du state et du temps
    def blit(self, screen, t):
        if self.states[t] == 'boreal':
            screen.blit(images["boreal_tree"], self.iso_pos)
        elif self.states[t] == 'burning':
            screen.blit(images["burning_tree"], self.iso_pos)
        elif self.states[t] == 'dead':
            screen.blit(images["dead_tree"], self.iso_pos)
        elif self.states[t] == 'hide':
            pass


# class::Tile : Une maille. Vecteurs moyens des nodes qui la compose. Contient la liste de ses instances d'arbres,
# gère ses feux, l'affichage de ses arbres et son affichage (fond, bordure).
class Tile:
    """ Représente une zone polygonale (triangle) dans l'écran avec ses nœuds et ses propriétés."""
    
    def __init__(self, topleft_node):
        
        self.topleft_node = topleft_node
        self.iso_x, self.iso_y = projeter_en_isometrique(self.topleft_node.x, self.topleft_node.y)
        
        self.corners = []
        self.corners.append((self.topleft_node.x, self.topleft_node.y))
        self.corners.append((self.topleft_node.x + TILE_SIZE, self.topleft_node.y))
        self.corners.append((self.topleft_node.x + TILE_SIZE, self.topleft_node.y + TILE_SIZE))
        self.corners.append((self.topleft_node.x, self.topleft_node.y + TILE_SIZE))
        
        self.center = (self.topleft_node.x + TILE_SIZE // 2, self.topleft_node.y + TILE_SIZE // 2)
        
        self.area = TILE_SIZE * TILE_SIZE
        self.u = topleft_node.u
        
        # nombre véritable d'arbres
        self.true_nb_trees = [u_t * self.area * AREA_SCALE for u_t in self.u]
        
        # nombre d'arbres à blitter (fonction du nombre max d'arbres à montrer)
        self.screen_nb_trees = [int((true_nb_trees_t - TREES_T_MIN) * MAX_ELMTS_PER_TILE
                                    / (TREES_T_MAX - TREES_T_MIN))
                                for true_nb_trees_t in self.true_nb_trees]
        
        self.trees = []  # liste d'instance de trees
        self.generate_trees()
    
    # generate_trees : construction des positions d'abres (avec random_point_in_triangle) et affectation des positions
    # à des instances d'arbres. Par défaut, tous les arbres sont .show=False
    def generate_trees(self):
        
        ## 1 - génération des positions
        generated_points = []
        
        while len(generated_points) < MAX_ELMTS_PER_TILE:
            x, y = (random.uniform(self.topleft_node.x, self.topleft_node.x + TILE_SIZE),
                    random.uniform(self.topleft_node.y, self.topleft_node.y + TILE_SIZE))
            
            iso_x, iso_y = projeter_en_isometrique(x, y)
            iso_point = (int(iso_x), int(iso_y))
            
            if iso_point not in generated_points:
                generated_points.append(iso_point)
        
        ## 2- construction des arbres
        for point in generated_points:
            tree = Tree('hide', *point)
            self.trees.append(tree)
        
        # construction des showtimes
        for t in range(MAX_T):
            nb_t = self.screen_nb_trees[t]
            for tree in self.trees[:nb_t]:
                tree.states[t] = 'boreal'
    
    # dessin de la maille, et de son contour si décommenté ci-dessous
    def draw_tile(self, screen):
        
        points_iso = [projeter_en_isometrique(corner[0], corner[1]) for corner in self.corners]
        pygame.draw.polygon(screen, SNOW, points_iso)
    
    def draw_tile_borders(self, screen):
        points_iso = [projeter_en_isometrique(corner[0], corner[1]) for corner in self.corners]
        pygame.draw.polygon(screen, BLACK, points_iso, 1)
    
    # affichage des arbres ayant .show=True (appelé après draw_tiles pour bliter par dessus la maille)
    def draw_trees(self, screen, t):
        for tree in self.trees:
            tree.blit(screen, t)


# create_tiles : Création des mailles à partir des nodes (issues de build_from_solution a priori)
def create_tiles(nodes):
    print("Generating tiles...")
    
    tiles_ = []
    test = True
    for node in nodes:
        tile = Tile(node)
        if test:
            print(len(tile.trees[0].states))
            test = False
        tiles_.append(tile)
    
    print(f"\tGenerated {len(tiles_)} tiles.")
    
    mean_area = np.mean([tile.area for tile in tiles_])
    max_true_nb = np.max([tile.true_nb_trees for tile in tiles_])
    max_screen_nb = np.max([tile.screen_nb_trees for tile in tiles_])
    
    print("\nTILES PROPERTIES")
    print(f"\tmean_area: {mean_area}")
    print(f"\tmax_true_nb_trees: {max_true_nb}")
    print(f"\tmax_screen_nb_trees: {max_screen_nb}")
    
    return tiles_


# resolve_burning_trees : fonction qui relie les feux et les tiles. Détermine les arbres qui brûlent.
def resolve_burning_trees(tiles, df_feux):
    ## 1 - construction des feux
    fires = []
    for _, fire in df_feux.iterrows():
        fire = Fire(fire["t"], fire["x"], fire["y"], fire["r"], fire["I"])
        fires.append(fire)
    
    ## 2 - affectation des feux aux arbres
    for fire in fires:
        
        # on pose les temps de transition pour aérer le code plus bas
        # pourquoi le +1 ? parce que lorsque t_start est à x, les arbres disparaissent à x+1 dans la simu.
        # on utilise toujours t_start pour regarder "eligible_trees" (cf plus bas) ; mais les images de feu doivent
        # commencer lorsque les arbres ont disparus, pour les remplacer, càd à x+1.
        
        fire_start = int(fire.t_start) + 1
        fire_end = min(fire_start + FIRE_DURATION, MAX_T)
        dead_start = fire_end
        dead_end = min(dead_start + DEAD_TREE_DURATION, MAX_T)
        
        for tile in tiles:
            
            dif_x = abs(tile.center[0] - fire.x)
            dif_y = abs(tile.center[1] - fire.y)
            
            # vérifier que le centre de la tuile est dans le feu = tuile en feu
            if dif_x ** 2 + dif_y ** 2 < fire.r ** 2:
                
                # arbres eligibles à brûler si pas 'hide' ou 'dead' au temps du feu
                eligible_trees = [tree for tree in tile.trees
                                  if tree.states[int(fire.t_start) - 1] not in ['hide', 'dead']]
                
                nb_to_burn = int(len(eligible_trees) * fire.I)  # nb d'arbres qui brûlent parmi les eligibles
                selected_trees = random.sample(eligible_trees, nb_to_burn)
                
                for tree in selected_trees:
                    for fire_time in range(fire_start, fire_end):
                        tree.states[fire_time] = 'burning'
                    for dead_time in range(dead_start, dead_end):
                        tree.states[dead_time] = 'dead'


########################################################################################
# ------------------------ Main
########################################################################################


if __name__ == "__main__":
    
    # C'est parti
    print(" ###### STARTED.")
    print("Running main.\n")
    
    if not os.path.exists(SIMULATION_SAVE_FOLDER) and not MAKE_NEW_SIMULATION:
        print(f"ERROR : Simulation {SIMULATION_SAVE_FOLDER} not found.\n"
              f" Try setting \"MAKE_NEW_SIMULATION\" to \"True\", or select another folder.")
        quit(1)
    
    if MAKE_NEW_SIMULATION:
        # lance la simulation et récupère les paramètres
        run_simulation(SIMULATION_SAVE_FOLDER)

    
    # Gestion de l'affichage des paramètres
    lines = read_params_from(os.path.join(SIMULATION_SAVE_FOLDER, "params.txt"))  # à adapter
    font = pygame.font.Font(None, 26)
    text_params = [font.render(line, True, BLACK) for line in lines]
    
    # Construction des instances du rendu
    nodes, df_feux = build_from_solution(SIMULATION_SAVE_FOLDER)
    tiles = create_tiles(nodes)
    resolve_burning_trees(tiles, df_feux)
    
    # finaliser la construction du slider (MAX_T dépend de la simulation)
    slider.value_range = (0, MAX_T - 1)
    print(f"MAX_ELMTS_PER_TILE is set to {MAX_ELMTS_PER_TILE}.")
    
    ########################################################################################
    # -------------------- Boucle principale de pygame
    ########################################################################################
    
    clock = pygame.time.Clock()
    running = True
    play_mode = True
    said_last_words = False
    
    frames_folder = os.path.join(SIMULATION_SAVE_FOLDER, "Frames")
    os.makedirs(frames_folder, exist_ok=True)
    
    t = 0
    FPS = 60
    
    # tick_it permet de régler la vitesse de simulation sans affecter le vrai fps de pygame (= GUI fluide)
    tick_it = 1
    
    print("Running pygame loop...")
    while running:
        
        screen.fill(ICE)
        
        if play_mode:
            # en play_mode (default) le slider suit la simulation
            slider.set_current_value(t)
            # tick_it augmente. Si tick_it est 0 modulo truc, avancer d'un pas de temps, et remettre tick à 0
            tick_it += 1
            if not tick_it % (FPS // TPS):
                t = min(t + 1, MAX_T - 1)
                tick_it = 1
        else:
            
            # hors play_mode, c'est le slider qui contrôle le temps
            t = int(slider.get_current_value())
        
        ##########################################
        ## Partie affichage
        ##########################################
        
        for tile in tiles:
            tile.draw_tile(screen)
        """
        for tile in tiles:
            tile.draw_tile_borders(screen)
        """
        for tile in tiles:
            tile.draw_trees(screen, t)
        
        # affichage des paramètres
        # screen.blit(text_surface, (20, 50))
        
        # affichage de t
        t_surface = font.render(f"T = {t / 10:.1f}", True, BLACK)
        screen.blit(t_surface, (SCREEN_WIDTH // 2 - 30, 50))
        y_offset = 10
        for surface in text_params:
            screen.blit(surface, (10, y_offset))
            y_offset += surface.get_height() + 5
        
        ####################################################
        ## Gestion fin de simulation / enregistrement video
        ####################################################
        
        # si t est maximum, on affiche un cercle rouge, on arrête l'enregistrement vidéo
        if t == MAX_T - 1:
            pygame.draw.circle(screen, (255, 0, 0), (SCREEN_WIDTH - 40, 40), 30)
            
            if not said_last_words:
                print(f"Max time ({t + 1}) reached.")
                print("Writing video...")
                video_writer.release()
                print(f"\t Wrote video as \"{video_path}\".\n")
                said_last_words = True
        
        # sinon, on continue d'enregistrer la video
        else:
            frame = pygame.surfarray.array3d(screen)
            # transformations pour OpenCV
            frame = np.rot90(frame)
            frame = np.flipud(frame)
            # conversion OpenCV
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # écriture de la frame
            video_writer.write(frame)
            
            # Sauvegarde en JPEG (plan B)
            frame_filename = os.path.join(frames_folder, f"frame_{t:02d}.jpg")
            cv2.imwrite(frame_filename, frame)
        ##########################################
        ## Gestion des évenements
        ##########################################
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # si le slider bouge, on arrête le play mode
            elif event.type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
                if event.ui_element == slider:
                    if said_last_words:
                        play_mode = False
            # on alterne play_mode lorsqu'on clique le bouton associé
            elif event.type == pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == play_button:
                    if said_last_words:
                        play_mode = not play_mode
            
            manager.process_events(event)
        
        ###########################################
        
        time_delta = clock.tick(FPS) / 1000.0
        
        manager.update(time_delta)
        manager.draw_ui(screen)
        
        pygame.display.flip()
    
    pygame.quit()
    print("###### END OF PROGRAM.")
