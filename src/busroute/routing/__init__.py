"""Routing tools.

``orienteering`` and ``metaheuristics`` operate on the OD matrix. The
``ppo_od`` module trains SBX PPO on that matrix. ``ppo_road`` trains CPU SB3 PPO
over the full road graph; ``road_ppo_env`` defines routing/Gymnasium rules and
``road_ppo_policy`` defines the Torch networks. ``env``, ``gnn`` and ``ppo_train``
are legacy teaching scaffolds, not the active training pipeline. ``baselines``
and ``evaluate`` use fixed-weight NetworkX graphs.
"""
