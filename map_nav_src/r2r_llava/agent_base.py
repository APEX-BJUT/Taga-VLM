import json
import os
import sys
import warnings
from collections import defaultdict

import h5py
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.distributed import is_default_gpu
from utils.logger import print_progress

from llava.model.builder import load_pretrained_model

warnings.filterwarnings("ignore")
from tqdm import tqdm


class BaseAgent(object):
    """Base class for an REVERIE agent to generate and save trajectories."""

    def __init__(self, env):
        self.env = env
        self.results = {}

    def get_results(self, detailed_output=False):
        output = []
        for k, v in self.results.items():
            output.append({"instr_id": k, "trajectory": v["path"]})
            if detailed_output:
                output[-1]["details"] = v["details"]
        return output

    def rollout(self, **args):
        """Return a list of dicts containing instr_id:'xx', path:[(viewpointId, heading_rad, elevation_rad)]"""
        raise NotImplementedError

    @staticmethod
    def get_agent(name):
        return globals()[name + "Agent"]

    def test(self, iters=None, **kwargs):
        self.env.reset_epoch(shuffle=(iters is not None))  # If iters is not none, shuffle the env batch
        self.losses = []
        self.results = {}
        # We rely on env showing the entire batch before repeating anything
        looped = False
        self.loss = 0
        count = 0
        with tqdm(total=3520, desc="trajs", leave=False) as pbar:
            if iters is not None:
                # For each time, it will run the first 'iters' iterations. (It was shuffled before)
                for i in range(iters):
                    for traj in self.rollout(**kwargs):
                        self.loss = 0
                        self.results[traj["instr_id"]] = traj
            else:  # Do a full round
                while True:
                    for traj in self.rollout(**kwargs):
                        if traj["instr_id"] in self.results:
                            looped = True
                        else:
                            self.loss = 0
                            self.results[traj["instr_id"]] = traj
                    pbar.update(1)
                    # print("count:"+str(count))
                    if looped:
                        pbar.update(1)
                        # print("count:"+str(count))
                        break

    def test_viz(self, iters=None, **kwargs):
        self.env.reset_epoch(shuffle=(iters is not None))  # If iters is not none, shuffle the env batch
        self.losses = []
        self.results = {}
        # We rely on env showing the entire batch before repeating anything
        looped = False
        self.loss = 0
        if iters is not None:
            # For each time, it will run the first 'iters' iterations. (It was shuffled before)
            for i in range(iters):
                for traj in self.rollout(**kwargs):
                    self.loss = 0
                    self.results[traj["instr_id"]] = traj
        else:  # Do a full round
            while True:
                for traj in self.rollout_viz(**kwargs):
                    if traj["instr_id"] in self.results:
                        looped = True
                    else:
                        self.loss = 0
                        self.results[traj["instr_id"]] = traj
                if looped:
                    break


class Seq2SeqAgent(BaseAgent):
    env_actions = {
        "left": (0, -1, 0),  # left
        "right": (0, 1, 0),  # right
        "up": (0, 0, 1),  # up
        "down": (0, 0, -1),  # down
        "forward": (1, 0, 0),  # forward
        "<end>": (0, 0, 0),  # <end>
        "<start>": (0, 0, 0),  # <start>
        "<ignore>": (0, 0, 0),  # <ignore>
    }
    for k, v in env_actions.items():
        env_actions[k] = [[vx] for vx in v]

    def __init__(self, args, env, rank=0):
        super().__init__(env)
        self.args = args

        self.default_gpu = is_default_gpu(self.args)
        self.rank = rank

        # Models
        self._build_model()

        if self.args.world_size > 1:
            self.vln_bert = DDP(self.vln_bert, device_ids=[self.rank], find_unused_parameters=True)
            self.critic = DDP(self.critic, device_ids=[self.rank], find_unused_parameters=True)

        self.models = (self.vln_bert, self.critic)
        self.device = torch.device("cuda:%d" % self.rank)
        data_root = self.args.root_dir
        project_root = os.path.dirname(data_root)
        self.scanvp_cands = json.load(open(os.path.join(data_root, "R2R/annotations/scanvp_candview_relangles.json")))
        self.views_file = h5py.File(os.path.join(data_root, "view_images_bgr_from_mattersim.h5"), "r")
        self.hm3d_views_file = os.path.join(data_root, "view_images_hm3d/")
        self.mp3d_views_file = os.path.join(data_root, "mp3d_data/")
        connectivity_dir = os.path.join(data_root, "R2R/connectivity")
        pretrained = os.path.join(project_root, "model_zoo/TagaVLM-qwen2-0.5b")
        model_name = "llava_qwen"
        device_map = "auto"
        llava_model_args = {
            "multimodal": True,
            "attn_implementation": "sdpa",
        }
        overwrite_config = {}
        overwrite_config["image_aspect_ratio"] = "pad"
        llava_model_args["overwrite_config"] = overwrite_config
        self.llava_tokenizer, self.llava_model, self.llava_image_processor, self.llava_max_length = (
            load_pretrained_model(pretrained, None, model_name, device_map=device_map, **llava_model_args)
        )

        self.graphs, self.shortest_distances, self.shortest_paths = self.load_nav_graphs(connectivity_dir)
        # self.llava_tokenizer.add_tokens([NODE_BEGIIN_TOKEN], special_tokens=True)
        # self.llava_tokenizer.add_tokens([NODE_END_TOKEN], special_tokens=True)
        # self.llava_tokenizer.add_tokens([STOP_TOKEN], special_tokens=True)
        # self.llava_tokenizer.add_tokens([NAV_INSTR_BEGIN_TOKEN], special_tokens=True)
        # self.llava_tokenizer.add_tokens([NAV_INSTR_END_TOKEN], special_tokens=True)
        # self.llava_tokenizer.add_tokens([IDX_BEGIN_TOKEN], special_tokens=True)
        # self.llava_tokenizer.add_tokens([IDX_END_TOKEN], special_tokens=True)

        self.llava_model.eval()
        self.conv_template = "qwen_nav"

        # self.pbar = progressbar.ProgressBar(maxval=len(6000), widgets=[
        #         progressbar.Bar('=', '[', ']'),
        #         ' ', progressbar.Percentage(),
        #         ' ', progressbar.ETA(),
        #         ' ', progressbar.FileTransferSpeed(unit="iter")
        #     ])
        # self.pbar.start()

        # Optimizers
        if self.args.optim == "rms":
            optimizer = torch.optim.RMSprop
        elif self.args.optim == "adam":
            optimizer = torch.optim.Adam
        elif self.args.optim == "adamW":
            optimizer = torch.optim.AdamW
        elif self.args.optim == "sgd":
            optimizer = torch.optim.SGD
        else:
            assert False
        if self.default_gpu:
            print("Optimizer: %s" % self.args.optim)

        # self.vln_bert_optimizer = optimizer(self.vln_bert.parameters(), lr=self.args.lr)
        # self.critic_optimizer = optimizer(self.critic.parameters(), lr=self.args.lr)
        # self.optimizers = (self.vln_bert_optimizer, self.critic_optimizer)

        # Evaluations
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.args.ignoreid, reduction="sum")

        # Logs
        sys.stdout.flush()
        self.logs = defaultdict(list)

    def _build_model(self):
        raise NotImplementedError("child class should implement _build_model: self.vln_bert & self.critic")

    def test(self, use_dropout=False, feedback="argmax", allow_cheat=False, iters=None, viz=False):
        """Evaluate once on each instruction in the current environment"""
        self.feedback = feedback
        if use_dropout:
            self.vln_bert.train()
            self.critic.train()
        else:
            self.vln_bert.eval()
            self.critic.eval()
        if viz:
            super().test_viz(iters=iters)
        else:
            super().test(iters=iters)

    def train(self, n_iters, feedback="teacher", **kwargs):
        """Train for a given number of iterations"""
        self.feedback = feedback

        self.vln_bert.train()
        self.critic.train()

        self.losses = []
        for iter in range(1, n_iters + 1):
            self.vln_bert_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            self.loss = 0

            if self.args.train_alg == "imitation":
                self.feedback = "teacher"
                self.rollout(train_ml=1.0, train_rl=False, **kwargs)
            elif self.args.train_alg == "dagger":
                if self.args.ml_weight != 0:
                    self.feedback = "teacher"
                    self.rollout(train_ml=self.args.ml_weight, train_rl=False, **kwargs)
                self.feedback = "expl_sample" if self.args.expl_sample else "sample"
                self.rollout(train_ml=1, train_rl=False, **kwargs)
            else:
                if self.args.ml_weight != 0:
                    self.feedback = "teacher"
                    self.rollout(train_ml=self.args.ml_weight, train_rl=False, **kwargs)
                self.feedback = "sample"
                self.rollout(train_ml=None, train_rl=True, **kwargs)

            # print(self.rank, iter, self.loss)
            self.loss.backward()

            torch.nn.utils.clip_grad_norm_(self.vln_bert.parameters(), 40.0)

            self.vln_bert_optimizer.step()
            self.critic_optimizer.step()

            if self.args.aug is None:
                print_progress(iter, n_iters + 1, prefix="Progress:", suffix="Complete", bar_length=50)

    def save(self, epoch, path):
        """Snapshot models"""
        the_dir, _ = os.path.split(path)
        os.makedirs(the_dir, exist_ok=True)
        states = {}

        def create_state(name, model, optimizer):
            states[name] = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }

        all_tuple = [
            ("vln_bert", self.vln_bert, self.vln_bert_optimizer),
            ("critic", self.critic, self.critic_optimizer),
        ]
        for param in all_tuple:
            create_state(*param)
        torch.save(states, path)

    def load(self, path):
        """Loads parameters (but not training state)"""
        states = torch.load(path, map_location=lambda storage, loc: storage)

        def recover_state(name, model, optimizer):
            state = model.state_dict()
            model_keys = set(state.keys())
            load_keys = set(states[name]["state_dict"].keys())
            state_dict = states[name]["state_dict"]
            if model_keys != load_keys:
                print("NOTICE: DIFFERENT KEYS IN THE LISTEREN")
                if not list(model_keys)[0].startswith("module.") and list(load_keys)[0].startswith("module."):
                    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                if list(model_keys)[0].startswith("module.") and (not list(load_keys)[0].startswith("module.")):
                    state_dict = {"module." + k: v for k, v in state_dict.items()}
                same_state_dict = {}
                extra_keys = []
                for k, v in state_dict.items():
                    if k in model_keys:
                        same_state_dict[k] = v
                    else:
                        extra_keys.append(k)
                state_dict = same_state_dict
                print("Extra keys in state_dict: %s" % (", ".join(extra_keys)))
            state.update(state_dict)
            model.load_state_dict(state)
            if self.args.resume_optimizer:
                optimizer.load_state_dict(states[name]["optimizer"])

        all_tuple = [
            ("vln_bert", self.vln_bert, self.vln_bert_optimizer),
            ("critic", self.critic, self.critic_optimizer),
        ]
        for param in all_tuple:
            recover_state(*param)
        return states["vln_bert"]["epoch"] - 1

    def load_nav_graphs(self, connectivity_dir):
        """Load connectivity graph for each scan"""

        def distance(pose1, pose2):
            """Euclidean distance between two graph poses"""
            return (
                (pose1["pose"][3] - pose2["pose"][3]) ** 2
                + (pose1["pose"][7] - pose2["pose"][7]) ** 2
                + (pose1["pose"][11] - pose2["pose"][11]) ** 2
            ) ** 0.5

        scans = [x.strip() for x in open(os.path.join(connectivity_dir, "scans.txt")).readlines()]
        graphs = {}
        for scan in scans:
            with open(os.path.join(connectivity_dir, "%s_connectivity.json" % scan)) as f:
                G = nx.Graph()
                positions = {}
                data = json.load(f)
                for i, item in enumerate(data):
                    if item["included"]:
                        for j, conn in enumerate(item["unobstructed"]):
                            if conn and data[j]["included"]:
                                positions[item["image_id"]] = np.array(
                                    [item["pose"][3], item["pose"][7], item["pose"][11]]
                                )
                                assert data[j]["unobstructed"][i], "Graph should be undirected"
                                G.add_edge(item["image_id"], data[j]["image_id"], weight=distance(item, data[j]))
                nx.set_node_attributes(G, values=positions, name="position")
                graphs[scan] = G

        shortest_distances = {}
        shortest_paths = {}
        for scan, G in graphs.items():  # compute all shortest paths
            shortest_distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))
            shortest_paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
        return graphs, shortest_distances, shortest_paths
