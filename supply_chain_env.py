import gymnasium as gym
from gymnasium import spaces
import numpy as np
import itertools

class SupplyChainEnv(gym.Env):
    def __init__(self):
        super(SupplyChainEnv, self).__init__()
        
        # --- PARAMÈTRES DU PROBLÈME ---
        self.n_stores = 3
        self.max_age = 5       # Durée de vie (L)
        self.lead_time = 2     # Délai de livraison
        self.max_stock_cap = 50 
        
        # Coûts
        self.h = 0.1   # Stockage
        self.p = 1.0  # Rupture (Penality)
        self.w = 0.5   # Déchets (Waste)
        self.c_fix = 0.2 # Coût fixe de commande (incite à commander par gros blocs)

        # --- DISCRÉTISATION DE L'ACTION POUR DQN ---
        # On autorise seulement 3 niveaux de commande pour simplifier : 0, 20, 40 unités
        self.possible_orders = [0, 20, 40]
        # On génère toutes les combinaisons possibles pour les 3 magasins (produit cartésien)
        # Ex: (0,0,0), (0,0,20), (0,20,40)... Ça fait 3^3 = 27 actions possibles.
        self.action_mapping = list(itertools.product(self.possible_orders, repeat=self.n_stores))
        
        self.action_space = spaces.Discrete(len(self.action_mapping))
        
        # --- OBSERVATION ---
        # Pour chaque magasin : [Stock_age_0, ..., Stock_age_4, Transit_1, Transit_2]
        # Taille = 3 * (5 + 2) = 21 variables
        obs_dim = self.n_stores * (self.max_age + self.lead_time)
        self.observation_space = spaces.Box(low=0, high=200, shape=(obs_dim,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialisation : stocks vides
        self.stocks = np.zeros((self.n_stores, self.max_age)) 
        self.orders_transit = np.zeros((self.n_stores, self.lead_time))
        self.day = 0
        return self._get_obs(), {}

    def step(self, action_idx):
        self.day += 1
        # 1. DÉCODAGE DE L'ACTION
        # L'IA donne un chiffre entre 0 et 26, on récupère le vecteur (q1, q2, q3)
        orders_today = np.array(self.action_mapping[action_idx])
        
        rewards = 0
        info = {"holding": 0, "shortage": 0, "waste": 0, "ordering": 0}
        
        # Génération demande (Moyenne 15, écart-type 5)
        demands = np.maximum(0, np.random.normal(15, 5, self.n_stores)).astype(int)
        
        # BOUCLE SUR CHAQUE MAGASIN
        for i in range(self.n_stores):
            # A. Réception commande (arrivant de la veille, délai fini)
            arriving_qty = self.orders_transit[i, 0]
            
            # Mise à jour file d'attente commandes
            self.orders_transit[i, :-1] = self.orders_transit[i, 1:]
            self.orders_transit[i, -1] = orders_today[i]
            
            # B. Intégration au stock (On suppose réception le matin = Age 0)
            # On crée un buffer temporaire pour gérer le FIFO
            current_stock = self.stocks[i].copy()
            # On décale les âges : ce qui était age 'a' devient 'a+1'
            # Mais d'abord, on satisfait la demande
            
            # C. SATISFACTION DEMANDE (FIFO : On vend le plus vieux d'abord)
            rem_demand = demands[i]
            # On parcourt du plus vieux (fin de liste) au plus jeune
            for age in reversed(range(self.max_age)):
                if rem_demand > 0:
                    available = current_stock[age]
                    taken = min(rem_demand, available)
                    current_stock[age] -= taken
                    rem_demand -= taken
            
            # Si demande > stock total, le reste est pris sur l'arrivage du matin
            if rem_demand > 0:
                taken_fresh = min(rem_demand, arriving_qty)
                arriving_qty -= taken_fresh
                rem_demand -= taken_fresh
            
            shortage = rem_demand # Ce qui reste est une vente perdue
            
            # D. VIEILLISSEMENT & PÉREMPTION
            # Le stock d'âge max (4) qui n'a pas été vendu est jeté
            waste = current_stock[-1]
            
            # Décalage des stocks : I_a(t+1) = I_{a-1}(t)
            new_stock = np.zeros(self.max_age)
            new_stock[1:] = current_stock[:-1] # Shift
            new_stock[0] = arriving_qty        # Le reste de l'arrivage devient Age 0
            
            self.stocks[i] = new_stock
            
            # E. CALCUL COÛTS
            c_hold = self.h * np.sum(new_stock)
            c_short = self.p * shortage
            c_waste = self.w * waste
            c_order = self.c_fix if orders_today[i] > 0 else 0
            
            total_cost = c_hold + c_short + c_waste + c_order
            rewards -= total_cost
            
            # Logs
            info["holding"] += c_hold
            info["shortage"] += c_short
            info["waste"] += c_waste
            info["ordering"] += c_order

        truncated = False
        terminated = False
        if self.day >= 365: # Horizon 1 an
            terminated = True
            
        return self._get_obs(), rewards, terminated, truncated, info

    def _get_obs(self):
        # On aplatit tout en un seul vecteur pour le réseau de neurones
        return np.concatenate([self.stocks.flatten(), self.orders_transit.flatten()]).astype(np.float32)