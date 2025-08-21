# clearit/trainers/encoder_trainer.py
import yaml
from pathlib import Path
from time import time
import os
import torch
from datetime import timedelta
from tqdm import tqdm
import torch.nn.functional as F
from clearit.augmentations.factory import get_augmentations
from clearit.models.resnet import ResNetEncoder
from clearit.objectives.objective import compute_loss
from torchlars import LARS
from torch.optim import SGD


class EncoderTrainer:
    def __init__(self, model_dir: str, **overrides):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # 1) load defaults
        defaults_path = Path(__file__).parent.parent / "configs" / "pretrainer_defaults.yaml"
        defaults = yaml.safe_load(defaults_path.read_text())

        # 2) load existing user config if present
        user_cfg_path = self.model_dir / "conf_enc.yaml"
        user_cfg = yaml.safe_load(user_cfg_path.read_text()) if user_cfg_path.exists() else {}

        # 3) merge: defaults ← user_cfg ← overrides
        cfg = {**defaults, **user_cfg}
        if 'transforms' in overrides:
            cfg['transforms'].update(overrides.pop('transforms'))
        cfg.update(overrides)

        # 4) sanity defaults
        cfg.setdefault('init_time', int(time()))
        cfg.setdefault('status', 0)

        self.config = cfg
        self.model  = None

    def load_config(self):
        """
        Loads configuration from a YAML file within the model directory, merging it with defaults.
        """
        config_path = os.path.join(self.model_dir, 'config.yaml')
        if os.path.exists(config_path):
            with open(config_path, 'r') as file:
                file_config = yaml.safe_load(file)
                self.config.update(file_config)  # Merge file config into default config


    def save_config(self):
        """
        Write out exactly what's in self.config to conf_enc.yaml
        """
        out = self.model_dir / "conf_enc.yaml"
        out.write_text(yaml.dump(self.config, default_flow_style=False, sort_keys=False))
            
    def update_config(self, overrides):
        """
        Updates the configuration with any overrides provided at initialization.
        Applies additional checks and updates specific fields if necessary.
        """
        if 'transforms' in overrides:  # Check if transforms are overridden
            for key, value in overrides['transforms'].items():
                self.config['transforms'][key] = value  # Update individual transform values
            del overrides['transforms']  # Remove transforms from overrides
        self.config.update(overrides)  # Update the rest of the configuration
        self.config.update(overrides)
        if self.config['init_time'] is None: # Set initialization time
            self.config['init_time'] = int(time())
        if self.config['name'] is None: # Set model name to directory name
            self.config['name'] = os.path.dirname(self.model_dir).split("/")[-1]
            
    def initialize_model(self, device='cuda'):
        """
        Initializes the model with an option to specify the computation device.
        """
        cfg = self.config

        # derive how many projector‐layers we actually configured
        mlp_sizes   = cfg.get('mlp_layers', [])
        n_proj_layers = len(mlp_sizes)

        # if mlp_features isn’t explicitly given, fall back to last size
        mlp_feat = cfg.get('mlp_features', mlp_sizes[-1] if mlp_sizes else cfg['encoder_features'])

        # now instantiate with the integer count + feature dim
        self.model = ResNetEncoder(
            encoder_name     = cfg['encoder_name'],
            encoder_features = cfg['encoder_features'],
            mlp_layers       = n_proj_layers,
            mlp_features     = mlp_feat,
            in_channels      = cfg['num_channels']
        ).to(device)

    def initialize_optimizer(self):
        """
        Initializes the optimizer
        """
        if self.config['optimizer'] == "LARS": # If optimizer is LARS,
            if self.config['learning_rate'] is None: # use learning rate of 0.3*B/256
                self.config['learning_rate'] = 0.3*self.config['batch_size']/256
            base_optimizer = SGD(self.model.parameters(), lr=self.config['learning_rate'])
            self.optimizer = LARS(optimizer=base_optimizer, eps=1e-8, trust_coef=0.001)
        # todo: implement other optimizers
        
    def save_model(self,verbose=True):
        """
        Saves the model state to 'enc.pt' in the model directory.
        """
        model_path = os.path.join(self.model_dir, "enc.pt")
        torch.save(self.model.state_dict(), model_path)
        if verbose: print(f"Model saved to {model_path}")        
        
    def load_model(self):
        """
        Loads the model state from a file named 'enc.pt' in the model directory.
        """
        model_path = os.path.join(self.model_dir, 'enc.pt')
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            print(f"Model loaded from {model_path}")
        else:
            print("Model checkpoint does not exist. Ensure that the model has been saved.")
        
    def log_progress(self, logfile, epoch, step, loss, start_time):
        """
        Logs training progress to a specified logfile.
        """
        elapsed_time = time() - start_time
        formatted_time = str(timedelta(seconds=int(elapsed_time)))  # Convert elapsed time to HH:MM:SS format
        with open(logfile, 'a') as log_file:
            log_file.write(f"Time {formatted_time}, Epoch {epoch + 1}, Step {step}, Loss {loss}\n")
        
    def train(self, dataloader):
        """
        Executes the training loop with appropriate logging and state management.
        """
        if self.config['status'] in [0.5, 1]:
            print("Training is either in progress or already completed.")
            return

        # Prepare paths for logging and saving
        log_outname = os.path.join(self.model_dir, "log.txt")

        # Write dataset information into config
        self.config['dataset_size'] = len(dataloader.dataset)
        self.config['dataset_name'] = dataloader.dataset.dataset_name
        # Setting the training in progress
        self.config['status'] = 0.5
        self.save_config()

        epochs = self.config['epochs']
        tau = self.config['tau']
        step = 1
        start = time()
        
        aug_list = get_augmentations(self.config['transforms'],self.config['img_size'])
                    
        pbar = tqdm(total=epochs * len(dataloader), desc="Training Progress")
            
        for epoch in range(epochs):
            for imgs in dataloader:
                self.optimizer.zero_grad()
                batch_size = imgs.shape[0]
                imgs2 = torch.repeat_interleave(imgs,2,dim=0).cuda()
                imgs = aug_list(imgs2)
                outputs = self.model(imgs)
                outputs = F.normalize(outputs, dim=1)

                labels = torch.arange(batch_size).repeat_interleave(2).cuda() # initialize labels vector with size 2*batch_size counting up every 2 elements
                                                                              # i.e. [0 0 1 1 2 2...] such that every image augmentation pair has same label
                labels_matrix = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()  # reshape labels vector into a matrix with 2*2 blocks containing "True" on its diagonal
                
                sim_matrix = torch.matmul(outputs, outputs.T)
                #loss = F.mse_loss(sim_matrix, labels_matrix)  # Example loss computation
                loss = compute_loss("ntxent", sim_matrix, labels_matrix, batch_size, tau)

                loss.backward()
                self.optimizer.step()

                # Update tqdm bar with the latest loss value
                pbar.set_description(f"Epoch {epoch + 1}/{epochs} - Loss: {loss.item():.4f}")
                pbar.update()
                
                # Error handling and logging
                if step % 100 == 0 or step == 1:
                    self.log_progress(log_outname, epoch, step, loss.item(), start)
                    self.save_model(verbose=False)  # Save periodically or on conditions

                step += 1

            # Update config after each epoch
            self.config['epochs_trained'] += 1
            self.save_config()
            
        # Finalize training
        self.config['status'] = 1
        self.save_config() 
        pbar.set_description(f"Epoch {epoch + 1}/{epochs} - Loss: {loss.item():.4f} - Training completed")
        pbar.close()