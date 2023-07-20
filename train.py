
# In order to run this with conda-installed cuda, you may need to run the following line first
# export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/"

# Test GPU with
# python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"


import deeplabcut 
import os
import argparse

from deeplabcut.utils import edit_config, read_config, read_plainconfig, write_plainconfig

from glob import glob
from pathlib import Path

# Hot fix for truncated images
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


parser = argparse.ArgumentParser(
    prog = 'telfer-dlc-runner',
    description = 'Trains, evaluates, and runs deeplabcut projects',
    epilog = 'contact: andretelfer@cmail.carleton.ca')

parser.add_argument('project', type=Path)
parser.add_argument('--iteration', default=None, type=int)
parser.add_argument('--duration-ratio', default=0.2, type=float)
parser.add_argument('--train-batch-size', default=1, type=int)
parser.add_argument('--test-batch-size', default=16, type=int)
parser.add_argument('--shuffles', default=1, type=int)
parser.add_argument('--displayiters', default=100, type=int)
parser.add_argument('--saveiters', default=50000, type=int)
parser.add_argument('--eval-only', default=False, type=bool)
parser.add_argument('--init-weights', default=None, type=str, help="weights can be changed to restart from a particular snapshot e.g. '«full path>-snapshot-5000'")
args = vars(parser.parse_args())

if __name__ == '__main__':
    project = Path(args.get('project')).absolute()
    shuffles = args.get('shuffles')
    config_path = project / 'config.yaml'

    import tensorflow as tf
    assert tf.test.is_gpu_available()

    # Update the config file
    # edit_config(config_path, {'batch_size': args.get('batch_size')})
    if args.get('iteration'):
        edit_config(config_path, {'iteration': args.get('iteration')})
                
    # Create training dataset
    deeplabcut.create_training_dataset(config_path, num_shuffles=shuffles)

    for shuffle in range(1,shuffles+1):
        if not args.get('eval_only'):
            # Update training times
            train_pose_cfg_path, _, _ = deeplabcut.return_train_network_path(
                config_path, shuffle=shuffle)
            train_pose_cfg = read_plainconfig(train_pose_cfg_path)
            train_pose_cfg['multi_step'] = (
                [[lr, int(epoch*args.get('duration_ratio'))] for lr, epoch in train_pose_cfg['multi_step']]
            )
            train_pose_cfg['batch_size'] = args.get('train_batch_size')
            train_pose_cfg['scale_jitter_up'] = 1.05
            if args.get('init_weights'):
                train_pose_cfg['init_weights'] = args.get('init_weights')
            write_plainconfig(train_pose_cfg_path, train_pose_cfg)
        
            # Train
            deeplabcut.train_network(
                config_path, 
                displayiters=args.get('displayiters'), 
                saveiters=args.get('saveiters'),
                shuffle=shuffle)

        # Analyze
        cfg = read_config(config_path)
        analysis_folder = config_path.parent / 'analysis' / f'iteration-{cfg["iteration"]}' / f'shuffle{shuffle}'
        if not analysis_folder.exists():
            analysis_folder.mkdir(parents=True)

        videos = list(map(str, project.glob('videos/*.mp4')))
        videos += list(map(str, project.glob('videos/*.MP4'))) 
        deeplabcut.analyze_videos(
            config_path, videos, destfolder=analysis_folder, shuffle=shuffle, batchsize=args.get('test_batch_size'))
        deeplabcut.create_labeled_video(
            config_path, videos, destfolder=analysis_folder, shuffle=shuffle)

    # Evaluate
    deeplabcut.evaluate_network(config_path, Shuffles=list(range(1, shuffles+1)))
    
