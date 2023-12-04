
import importlib
import sys
sys.path.append("../")


if __name__ == "__main__":
    if len(sys.argv)==3:
        module_name = sys.argv[2]
    elif len(sys.argv)==2:
        module_name = sys.argv[1]
    module = importlib.import_module(f"configs.{module_name}")
    base_obj = module.Configs()
    from trainer import Trainer
    trainer = Trainer(base_obj)

    trainer.train(base_obj.PRUNING_TOLERANCE)
    print(f"Completed Training, and validation for {module_name}!")
