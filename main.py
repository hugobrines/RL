import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from supply_chain_env import SupplyChainEnv

def run_heuristic(env, s=20, S=50):
    """
    Simule l'environnement avec une politique (s, S).
    Si Stock + Commandes < s, on commande pour atteindre S.
    """
    obs, _ = env.reset()
    done = False
    total_cost = 0
    history = {"cost": [], "shortage": [], "waste": []}
    
    while not done:
        # Reconstruire l'état pour l'heuristique
        # L'obs est plate, on doit retrouver la structure par magasin
        # Structure: [Stock_Mag1, Stock_Mag2, ..., Transit_Mag1...]
        n = env.n_stores
        l_stock = n * env.max_age
        stocks_flat = obs[:l_stock]
        transits_flat = obs[l_stock:]
        
        actions_qty = []
        
        # Logique (s, S) pour chaque magasin
        for i in range(n):
            # Récupérer stock total magasin i
            start_s = i * env.max_age
            stock_i = np.sum(stocks_flat[start_s : start_s + env.max_age])
            
            # Récupérer commandes en transit magasin i
            start_t = i * env.lead_time
            transit_i = np.sum(transits_flat[start_t : start_t + env.lead_time])
            
            inventory_position = stock_i + transit_i
            
            q = 0
            if inventory_position < s:
                needed = S - inventory_position
                # On arrondit à nos actions discrètes (0, 20, 40) pour être juste dans la comparaison
                if needed < 10: q = 0
                elif needed < 30: q = 20
                else: q = 40
            actions_qty.append(q)
            
        # Trouver l'index de cette action dans le mapping de l'environnement
        try:
            action_idx = env.action_mapping.index(tuple(actions_qty))
        except ValueError:
            # Si l'heuristique veut une combinaison non prévue (rare), on ne fait rien
            action_idx = 0 

        obs, reward, done, _, info = env.step(action_idx)
        total_cost += -reward # Reward est négatif
        
        history["cost"].append(-reward)
        history["shortage"].append(info["shortage"] / 10.0) # On divise par le prix pour avoir la quantité approx
        history["waste"].append(info["waste"] / 5.0)

    return total_cost, history

def train_dqn():
    print("--- Démarrage de l'entraînement DQN ---")
    env = SupplyChainEnv()
    
    # Création du modèle DQN
    # policy="MlpPolicy" : Réseau de neurones classique
    # learning_rate : Vitesse d'apprentissage
    # buffer_size : Mémoire des expériences passées
    model = DQN("MlpPolicy", env, verbose=1, learning_rate=0.01, buffer_size=50000, gamma=0.95)
    
    # Entraînement (augmenter total_timesteps pour de meilleurs résultats, ex: 100000)
    model.learn(total_timesteps=100_000) 
    model.save("dqn_supply_chain")
    print("--- Entraînement terminé ---")
    return model

def evaluate(model):
    env = SupplyChainEnv()
    
    # 1. Évaluation DQN
    obs, _ = env.reset()
    done = False
    dqn_history = {"cost": [], "shortage": [], "waste": []}
    total_cost_dqn = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        total_cost_dqn += -reward
        dqn_history["cost"].append(-reward)
        dqn_history["shortage"].append(info["shortage"]/10)
        dqn_history["waste"].append(info["waste"]/5)
        
    # 2. Évaluation Heuristique
    total_cost_heur, heur_history = run_heuristic(env)
    
    print(f"\nRÉSULTATS SUR 1 AN (365 jours):")
    print(f"Coût Total DQN        : {total_cost_dqn:.2f} €")
    print(f"Coût Total Heuristique: {total_cost_heur:.2f} €")
    
    # 3. Graphiques
    plt.figure(figsize=(12, 6))
    
    plt.subplot(3, 1, 1)
    plt.plot(dqn_history["cost"], label="DQN Cost", alpha=0.7)
    plt.plot(heur_history["cost"], label="Heuristic Cost", alpha=0.7, linestyle="--")
    plt.title("Coûts journaliers cumulés")
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(dqn_history["waste"], label="DQN Waste (qty)", color="red")
    plt.plot(heur_history["waste"], label="Heuristic Waste (qty)", color="orange", linestyle="--")
    plt.title("Quantité de déchets (Péremption)")
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(dqn_history["shortage"], label="DQN Shortage (qty)", color="blue")
    plt.plot(heur_history["shortage"], label="Heuristic Shortage (qty)", color="cyan", linestyle="--")
    plt.title("Quantité de ruptures (Demande non servie)")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Si le modèle existe déjà, on peut le charger direct avec DQN.load("dqn_supply_chain")
    # Ici on ré-entraîne à chaque fois pour la démo
    model = train_dqn()
    evaluate(model)