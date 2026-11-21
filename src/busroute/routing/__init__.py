"""Routing tools.

``orienteering``, ``metaheuristics``, and ``od_evaluation`` operate on the OD
matrix. ``ppo_road`` trains CPU SB3 edge-level PPO over the full road graph
(with ``road_ppo_env`` and ``road_ppo_policy``), and ``ppo_waypoint`` trains
CPU SB3 waypoint-level PPO (with ``waypoint_env``). ``baselines`` and
``evaluate`` use fixed-weight NetworkX graphs.
"""
