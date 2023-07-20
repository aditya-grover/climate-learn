import climate_learn as cl
import argparse
from util import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='lowres')
    parser.add_argument('--zero_shot_only', default=False, action='store_true')
    args = parser.parse_args()
    cfg = load_config(f'configs/{args.config}.yaml')

    dm = load_data_numpy(cfg)

    # climatology is the average value over the training period
    climatology = cl.load_forecasting_module(data_module=dm, preset="climatology")
    # persistence returns its input as its prediction
    persistence = cl.load_forecasting_module(data_module=dm, preset="persistence")

    # load module
    module = cl.load_forecasting_module(
        data_module=dm, 
        preset=cfg['model'], 
        cfg=cfg,
    )
    
    # load trainer
    trainer = load_trainer(cfg)    

    trainer.fit(module, dm, ckpt_path=f'lightning_logs/{cfg['ckpt']}/checkpoints/last.ckpt')

    # test climatolgy
    print('Testing Climatology')
    trainer.test(climatology, dm)

    # test persistence
    print('Testing Persistence')
    trainer.test(persistence, dm)

    # test module
    print('Testing Module')
    trainer.test(module, dm)


if __name__ == "__main__":
    main()