# tools/print_all_configs.py
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print("==== data ====")
    print(OmegaConf.to_yaml(cfg.data))
    print("==== model ====")
    print(OmegaConf.to_yaml(cfg.model))
    print("==== train ====")
    print(OmegaConf.to_yaml(cfg.train))
    print("==== task ====")
    print(OmegaConf.to_yaml(cfg.task))
    print("==== run ====")
    print(OmegaConf.to_yaml(cfg.run))
    print("==== eval ====")
    print(OmegaConf.to_yaml(cfg.eval))
    print("==== logging ====")
    print(OmegaConf.to_yaml(cfg.logging))
    if "paths" in cfg:
        print("==== paths ====")
        print(OmegaConf.to_yaml(cfg.paths))

if __name__ == "__main__":
    main()
