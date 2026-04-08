# antigravity

Implémentation d'un pipeline **Transformer + SAC** et d'un environnement EV/PV/Grid/Batterie.

## Fichiers

- `sac_transformer.py`: modèles PyTorch (`TransformerFeatureExtractor`, `SACActor`, `SACCritic`, `SACModel`)
- `charging_station_env.py`: environnement station de charge (sans dépendance externe)
- `simulate_strategy.py`: simulation heuristique rapide
- `train_transformer_sac.py`: entraînement SAC (séquence de 24 états) sur 10 épisodes


## Notebook Colab (sans GitHub)

- `colab_transformer_sac_ev.ipynb`: notebook autonome à copier/importer directement dans Colab.
- Inclut: environnement, agent Transformer+SAC, entraînement 10 épisodes, et graphiques reward/cost par épisode.

## Exécution

### 1) Démo modèle

```bash
python sac_transformer.py
```

### 2) Test heuristique EV/PV/Grid/Batterie

```bash
python simulate_strategy.py
```

### 3) Entraînement Transformer + SAC (10 épisodes)

```bash
python train_transformer_sac.py
```

> Note: `sac_transformer.py` et `train_transformer_sac.py` nécessitent `torch`.
