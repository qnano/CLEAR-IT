# clearit/trainers/head_trainer.py
import yaml
import torch
import torch.nn as nn
from time import time
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve
import numpy as np

from clearit.models.resnet      import ResNetEncoder
from clearit.models.head        import MLPHead
from clearit.models.composite   import EncoderClassifier
from torch.optim                import Adam

class HeadTrainer:
    def __init__(
        self,
        encoder_dir: str,
        head_dir: str,
        overrides: dict = None,
    ):
        from pathlib import Path
        import yaml
        from time import time

        self.encoder_dir = Path(encoder_dir)
        self.head_dir    = self.encoder_dir / head_dir
        self.head_dir.mkdir(parents=True, exist_ok=True)

        # 1) load encoder config so we know proj_layers, mlp_layers, etc.
        enc_cfg = yaml.safe_load((self.encoder_dir / "conf_enc.yaml").read_text())
        self.encoder_config = enc_cfg

        # 2) load our complete defaults (including id, base_encoder, …)
        defaults_path = Path(__file__).parent.parent / "configs" / "headtrainer_defaults.yaml"
        defaults      = yaml.safe_load(defaults_path.read_text())

        # 3) if there’s already a saved head-config, layer it on top
        user_cfg = {}
        existing = self.head_dir / "conf_head.yaml"
        if existing.exists():
            user_cfg = yaml.safe_load(existing.read_text())

        # 4) merge defaults ← user_cfg ← overrides  
        cfg = { **defaults, **user_cfg }
        if overrides:
            # special-case: if pos_weight explicitly None, make it all-ones
            if overrides.get("pos_weight") is None:
                overrides["pos_weight"] = [1.0] * cfg["num_classes"]
            cfg.update(overrides)

        # 5) fill in any last-moment required defaults
        cfg.setdefault("init_time", int(time()))
        cfg.setdefault("status", 0)

        self.config      = cfg
        self.device      = "cuda" if torch.cuda.is_available() else "cpu"
        self.model       = None
        self.best_thresh = None

    def save_config(self):
        """Write out conf_head.yaml with our updated self.config."""
        path = self.head_dir / "conf_head.yaml"
        with open(path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
        print(f"Saved config to {path}")

    def save_model(self):
        """Save head and, if unfrozen, encoder weights."""
        if not self.config['freeze_encoder']:
            torch.save(self.model.encoder.state_dict(), self.head_dir / "enc.pt")
        torch.save(self.model.classification_head.state_dict(), self.head_dir / "head.pt")

    def initialize_model(self):
        # pick how many SimCLR projections to use
        proj = min(self.config['proj_layers'], len(self.encoder_config.get('mlp_layers', [])))
        # determine its feature dim
        if proj == len(self.encoder_config.get('mlp_layers', [])):
            feat_dim = self.encoder_config['mlp_features']
        else:
            feat_dim = self.encoder_config['encoder_features']

        # build & load encoder
        enc = ResNetEncoder(
            encoder_name     = self.encoder_config['encoder_name'],
            encoder_features = self.encoder_config['encoder_features'],
            mlp_layers       = proj,
            mlp_features     = feat_dim,
            in_channels      = 3
        )
        ckpt = self.encoder_dir / "enc.pt"
        if ckpt.exists():
            state = torch.load(str(ckpt), map_location=self.device)
            enc.load_state_dict(state, strict=False)
            print(f"Loaded encoder from {ckpt}")
        if self.config['freeze_encoder']:
            enc.eval()

        # build head
        feature_size = enc.get_feature_size(proj) * self.config['num_channels']
        head = MLPHead(
            input_size  = feature_size,
            num_classes = self.config['num_classes'],
            dropout     = self.config['dropout'],
            head_layers = self.config['head_layers']
        )

        # composite & to device
        self.model = EncoderClassifier(encoder=enc, classification_head=head)
        self.model = self.model.to(self.device)

    def initialize_optimizer(self):
        if self.config['optimizer'] == 'ADAM':
            self.optimizer = Adam(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )

    def get_criterion(self):
        if self.config['label_mode'] == 'multilabel':
            return nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(self.config['pos_weight'], device=self.device)
            )
        else:
            return nn.CrossEntropyLoss()

    def _compute_thresholds(self, outputs, targets):
        """
        For each class, find the threshold that maximizes F1
        on the validation set.
        """
        thresholds = []
        probs = outputs.sigmoid().cpu().numpy()
        targs = targets.cpu().numpy()
        for c in range(self.config['num_classes']):
            p, r, thr = precision_recall_curve(targs[:,c], probs[:,c])
            f1 = 2 * p * r / (p + r + 1e-8)
            best = thr[np.nanargmax(f1[:-1])]  # ignore last point
            thresholds.append(float(best))
        return thresholds

    def train(self, dataloader_train, dataloader_val):
        criterion = self.get_criterion()
        best_val = float('inf')
        best_epoch = self.config.get('best_epoch', 0)

        # prepare logging
        self.model.train()
        pbar = tqdm(total=self.config['epochs'], desc="Training Heads")
        start = time()

        for epoch in range(self.config['epochs_trained'], self.config['epochs']):
            # --- train ---
            total_loss = 0.0
            for x,y,_,_ in dataloader_train:
                x,y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                out = self.model(x)
                loss = criterion(out, y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * x.size(0)

            # --- validate ---
            self.model.eval()
            val_loss = 0.0
            outs, tars = [], []
            with torch.no_grad():
                for x,y,_,_ in dataloader_val:
                    x,y = x.to(self.device), y.to(self.device)
                    o = self.model(x)
                    val_loss += criterion(o, y).item() * x.size(0)
                    outs.append(o)
                    tars.append(y)
            self.model.train()

            val_loss /= len(dataloader_val.dataset)
            # check for improvement
            if val_loss < best_val:
                best_val    = val_loss
                best_epoch  = epoch + 1
                # compute thresholds on this validation set
                all_out = torch.cat(outs, dim=0)
                all_tar = torch.cat(tars, dim=0)
                self.best_thresh = self._compute_thresholds(all_out, all_tar)

                # save checkpoint
                self.save_model()
                # record in config
                self.config['best_epoch']  = best_epoch
                self.config['thresholds']  = self.best_thresh

            # advance progress bar
            pbar.set_description(
              f"Epoch {epoch+1}/{self.config['epochs']}  "
              f"Val_loss {val_loss:.4f}  Best {best_val:.4f}@{best_epoch}"
            )
            pbar.update(1)
            self.config['epochs_trained'] = epoch+1

        pbar.close()
        # finalize
        self.config['status'] = 1
        self.save_config()
