# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Tous droits réservés.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script pour entraîner un agent RL avec RSL-RL."""

"""Lancez d'abord le simulateur Isaac Sim."""

# Importation des modules nécessaires
import argparse
import sys

from isaaclab.app import AppLauncher

# Imports locaux
import cli_args  # est exclu de l'analyse d'importation automatique par isort

# Configuration des arguments de la ligne de commande avec argparse
parser = argparse.ArgumentParser(description="Entraîner un agent RL avec RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Enregistrer des vidéos pendant l'entraînement.")
parser.add_argument("--video_length", type=int, default=200, help="Longueur de la vidéo enregistrée (en étapes).")
parser.add_argument("--video_interval", type=int, default=2000, help="Intervalle entre chaque enregistrement vidéo (en étapes).")
parser.add_argument("--num_envs", type=int, default=None, help="Nombre d'environnements à simuler.")
parser.add_argument("--task", type=str, default=None, help="Nom de la tâche.")
parser.add_argument("--seed", type=int, default=None, help="Graine utilisée pour l'environnement.")
parser.add_argument("--max_iterations", type=int, default=None, help="Nombre d'itérations d'entraînement pour la politique RL.")
# Ajouter les arguments de la ligne de commande spécifiques à RSL-RL
cli_args.add_rsl_rl_args(parser)
# Ajouter les arguments de la ligne de commande pour AppLauncher
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Si l'option vidéo est activée, on s'assure que les caméras sont activées pour l'enregistrement vidéo
if args_cli.video:
    args_cli.enable_cameras = True

# Modifier sys.argv pour gérer les arguments non pris en charge par Hydra
sys.argv = [sys.argv[0]] + hydra_args

# Lancer l'application Omniverse
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""La suite du script suit."""

import gymnasium as gym
import os
import torch
from datetime import datetime

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Configuration des paramètres CUDA pour PyTorch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

# Fonction pour ajouter des objets dans la scène
import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

def design_scene():
    """Conçoit la scène en ajoutant des objets comme des cônes, des cuboïdes, etc."""
    # Créer le plancher
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

    # Ajouter une lumière distante
    cfg_light_distant = sim_utils.DistantLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),
    )
    cfg_light_distant.func("/World/lightDistant", cfg_light_distant, translation=(1, 0, 10))

    # Créer un prim Xform pour les objets à ajouter
    prim_utils.create_prim("/World/Objects", "Xform")
    
    # Ajouter un cône rouge
    cfg_cone = sim_utils.ConeCfg(
        radius=0.15,
        height=0.5,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
    )
    cfg_cone.func("/World/Objects/Cone1", cfg_cone, translation=(-1.0, 1.0, 1.0))
    cfg_cone.func("/World/Objects/Cone2", cfg_cone, translation=(-1.0, -1.0, 1.0))

    # Ajouter un cône vert avec un corps rigide et des propriétés de collision
    cfg_cone_rigid = sim_utils.ConeCfg(
        radius=0.15,
        height=0.5,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
    )
    cfg_cone_rigid.func(
        "/World/Objects/ConeRigid", cfg_cone_rigid, translation=(-0.2, 0.0, 2.0), orientation=(0.5, 0.0, 0.5, 0.0)
    )

    # Ajouter un cuboïde bleu avec un corps déformable
    cfg_cuboid_deformable = sim_utils.MeshCuboidCfg(
        size=(0.2, 0.5, 0.2),
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        physics_material=sim_utils.DeformableBodyMaterialCfg(),
    )
    cfg_cuboid_deformable.func("/World/Objects/CuboidDeformable", cfg_cuboid_deformable, translation=(0.15, 0.0, 2.0))

    # Ajouter un fichier USD représentant une table
    cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
    cfg.func("/World/Objects/Table", cfg, translation=(0.0, 0.0, 1.05))

# Fonction principale d'entraînement de l'agent RL avec RSL-RL
@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Entraînement avec l'agent RSL-RL."""
    
    # Mise à jour des configurations avec les arguments passés en ligne de commande
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # Initialisation de la graine pour l'environnement
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Définir le répertoire pour la journalisation des expériences
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Journalisation de l'expérience dans le répertoire : {log_root_path}")
    # Créer un sous-répertoire spécifique pour cette exécution avec la date et l'heure
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Nom exact de l'expérience demandé depuis la ligne de commande : {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # Créer l'environnement Isaac Gym avec les paramètres définis
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # Ajouter des objets à la scène avant de commencer l'entraînement
    design_scene()

    # Si l'environnement est multi-agent, le convertir en un environnement à agent unique
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # Sauvegarder le chemin du point de reprise avant de créer un nouveau répertoire de journalisation
    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # Configurer l'enregistrement vidéo si nécessaire
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Enregistrement vidéo pendant l'entraînement.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Emballer l'environnement pour le rendre compatible avec RSL-RL
    env = RslRlVecEnvWrapper(env)

    # Créer le runner RSL-RL pour entraîner l'agent
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # Ajouter l'état du dépôt git dans les journaux
    runner.add_git_repo_to_log(__file__)
    # Charger le modèle à partir d'un point de reprise si nécessaire
    if agent_cfg.resume:
        print(f"[INFO]: Chargement du point de contrôle du modèle depuis : {resume_path}")
        runner.load(resume_path)

    # Sauvegarder les configurations dans les fichiers YAML et Pickle pour la journalisation
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # Lancer l'entraînement
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # Fermer l'environnement et la simulation à la fin de l'entraînement
    env.close()


if __name__ == "__main__":
    # Lancer la fonction principale
    main()
    # Fermer l'application de simulation
    simulation_app.close()

