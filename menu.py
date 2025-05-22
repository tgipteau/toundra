import os
import re
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk

DOSSIER_SIMULATIONS = "Simulations"
PARAMETRES_DEFAUT = {
    "sim_name": "default",
    "alpha": 1.0,
    "beta": 1.0,
    "delta": 0.5,
    "a": 3.0,
    "b": 1.0,
    "c": 0.5,
    "T": 100,
    "N": 1000,
    "L": 50.0,
    "J": 40.0,
    "p": 0.1,
    "freq": 1,
    "minree": 3.0,
    "maxree": 8.0,
    "minI": 0.3,
    "maxI": 1.0,
    "intensity": 0.8,
}

progress_history = []


def get_progress(sim_name, max_iter):
    global progress_history

    pattern = re.compile(r"u-(\d+)\.dat$")
    max_n = -1
    output_folder = DOSSIER_SIMULATIONS + "/" + sim_name + "/output"

    if not os.path.exists(output_folder):
        # si pas encore de dossier ouput, attendre...
        return 0, "inf"

    for filename in os.listdir(output_folder):
        match = pattern.match(filename)
        if match:
            n = int(match.group(1))
            max_n = max(max_n, n)

    percentage_complete = (max_n / max_iter) * 100

    # Ajout à l'historique
    now = time.time()
    progress_history.append((now, max_n))

    # Garder un historique sur les 10 dernières secondes
    progress_history = [(t, v) for t, v in progress_history if now - t <= 40]

    # Calcul de la vitesse
    if len(progress_history) >= 2:
        t0, v0 = progress_history[0]
        t1, v1 = progress_history[-1]
        speed = (v1 - v0) / (t1 - t0) if t1 > t0 else 0
    else:
        speed = 0

    # ETA en secondes
    remaining = max_iter - max_n
    eta = remaining / speed if speed > 0 else float(1000)
    eta_minutes = int(eta / 60.0)
    eta_seconds = int(eta % 60.0)
    eta = f"{eta_minutes:02d}:{eta_seconds:02d}"

    return percentage_complete, eta


def geometry_centered(root, window_width, window_height):
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2

    root.geometry("%dx%d+%d+%d" % (window_width, window_height, x, y))


def lister_dossiers_simulations():
    if not os.path.exists(DOSSIER_SIMULATIONS):
        os.makedirs(DOSSIER_SIMULATIONS)
    dossiers = [
        d
        for d in os.listdir(DOSSIER_SIMULATIONS)
        if os.path.isdir(os.path.join(DOSSIER_SIMULATIONS, d))
    ]
    return sorted(dossiers)


def creer_nouvelle_simulation(parametres):
    # Nettoyage
    for widget in root.winfo_children():
        widget.destroy()
    root.update_idletasks()

    geometry_centered(root, 400, 200)
    print("Création d'une nouvelle simulation avec paramètres :")
    for k, v in parametres.items():
        print(f" - {k}: {v}")

    sim_name = parametres["sim_name"]
    sim_folder = os.path.join(DOSSIER_SIMULATIONS, sim_name)

    if os.path.exists(sim_folder):
        # si existe déjà, le vider
        shutil.rmtree(sim_folder)
    os.makedirs(sim_folder)

    label_loading_sim = tk.Label(
        root, text=f'Construction de la simulation "{sim_name}"...', font=("Arial", 14)
    )
    label_loading_sim.pack(pady=10)

    progress = ttk.Progressbar(
        root, orient="horizontal", length=300, mode="determinate"
    )
    progress.pack(pady=10)
    progress["maximum"] = 100
    label_progress = tk.Label(root, text="")
    label_progress.pack(pady=10)
    label_eta = tk.Label(root, text="")
    label_eta.pack(pady=10)

    params_path = os.path.join(sim_folder, "params.txt")
    with open(params_path, "w") as f:
        for cle, valeur in parametres.items():
            if not cle == "sim_name":
                f.write(f"{cle}: {valeur}\n")

    simulation_thread = threading.Thread(
        target=lambda: subprocess.run([sys.executable, "toundra.py", "1", sim_folder])
    )
    simulation_thread.start()

    max_iter = int(parametres["N"])
    print(max_iter)

    def update_progress():
        if not os.path.exists(sim_folder + "/output/" + "EOP"):
            percentage_complete, eta = get_progress(sim_name, max_iter)
            print(percentage_complete)

            progress["value"] = percentage_complete
            label_progress.config(text=f"{percentage_complete:.1f} %")
            label_eta.config(text=f"ETA : {eta}")

            root.after(500, update_progress)

        else:
            progress["value"] = 100
            label_loading_sim.config(text="Simulation terminée.")
            print("[TK] simulation terminée")
            pygame_time(sim_name)

    time.sleep(1)
    update_progress()


def charger_simulation_existante(sim_name):
    # Nettoyage
    for widget in root.winfo_children():
        widget.destroy()
    root.update_idletasks()

    print(f"Chargement de la simulation : {sim_name}")
    label_loading_sim = tk.Label(
        root, text=f'Chargement de la simulation "{sim_name}"...', font=("Arial", 14)
    )
    label_loading_sim.pack(pady=10)
    root.update_idletasks()

    pygame_time(sim_name)


def valider_formulaire(champs):
    parametres = {k: champs[k].get() for k in champs}
    creer_nouvelle_simulation(parametres)


def afficher_formulaire_creation():
    # Nettoyage
    for widget in root.winfo_children():
        widget.destroy()
    root.update_idletasks()

    # Agrandissement de la fenêtre
    geometry_centered(root, 900, 700)

    titre = tk.Label(root, text="Paramètres de simulation", font=("Arial", 14))
    titre.pack(pady=10)

    champs = {}

    # Conteneur principal (pas de scroll ici, car 2 colonnes évitent la saturation verticale)
    content_frame = tk.Frame(root)
    content_frame.pack(pady=10)

    # Répartition sur 2 colonnes
    keys = list(PARAMETRES_DEFAUT.items())
    demi = (len(keys) + 1) // 2  # coupe en 2 colonnes
    colonne_gauche = keys[:demi]
    colonne_droite = keys[demi:]

    for col, bloc in enumerate([colonne_gauche, colonne_droite]):
        for row, (cle, valeur) in enumerate(bloc):
            label = tk.Label(content_frame, text=f"{cle} :", anchor="w", width=20)
            label.grid(row=row, column=col * 2, sticky="w", padx=5, pady=2)

            entry = tk.Entry(content_frame, width=20)
            entry.insert(0, str(valeur))
            entry.grid(row=row, column=col * 2 + 1, sticky="w", padx=5, pady=2)

            champs[cle] = entry

    # Bouton Valider
    btn_valider = tk.Button(
        root, text="Valider", command=lambda: valider_formulaire(champs)
    )
    btn_valider.pack(pady=20)

    root.update_idletasks()


def valider_selection():
    selection = combo.get()
    if selection == "Nouvelle simulation":
        afficher_formulaire_creation()
    elif selection != "":
        charger_simulation_existante(selection)
    else:
        print("Aucune sélection effectuée.")


def pygame_time(sim_name):

    root.destroy()
    root.update_idletasks()

    pygame_thread = threading.Thread(
        target=lambda: subprocess.run([sys.executable, "toundra.py", sim_name])
    )
    pygame_thread.start()


if __name__ == "__main__":
    # --- Interface Tkinter ---
    root = tk.Tk()
    root.title("Sélecteur de simulation")
    geometry_centered(root, 400, 300)
    root.focus_force()

    # Interface de départ
    label = tk.Label(root, text="Choisir une simulation :", font=("Arial", 14))
    label.pack(pady=10)

    options = ["Nouvelle simulation"] + lister_dossiers_simulations()
    combo = ttk.Combobox(root, values=options, state="readonly", width=30)
    combo.pack()
    combo.set("Nouvelle simulation")

    btn_valider = tk.Button(root, text="Valider", command=valider_selection)
    btn_valider.pack(pady=15)

    # Boucle principale
    root.mainloop()
