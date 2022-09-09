import os.path as osp
import torch
import shutil
import numpy as np
import argparse
import mmcv


def parse_args():
    parser = argparse.ArgumentParser(description='Convert HSICityV2 to dicts')
    parser.add_argument('hsicity_path', help='hsicity data path')
    parser.add_argument('-o', '--out-dir', help='output path')
    parser.add_argument(
        '--nproc', default=1, type=int, help='number of process')
    args = parser.parse_args()
    return args


def read_hsd(args):
    save_dir, base_dir, file = args
    abs_file = osp.join(base_dir, file)
    result = dict()

    data = np.fromfile(abs_file, dtype=np.int32)
    height = data[0]
    width = data[1]
    SR = data[2]
    D = data[3]

    result['height'] = height
    result['width'] = width
    result['SR'] = SR
    result['D'] = D

    data = np.fromfile(abs_file, dtype=np.float32)
    a = 7
    average = data[a:a + SR]
    a = a + SR
    coeff = data[a:a + D * SR].reshape((D, SR))
    a = a + D * SR
    scoredata = data[a:a + height * width * D].reshape((height * width, D))

    result['average'] = torch.from_numpy(average)
    result['scoredata'] = torch.from_numpy(scoredata)
    result['coeff'] = torch.from_numpy(coeff)

    save_file = osp.join(save_dir, file.replace('.hsd', '.pt'))
    torch.save(result, save_file)
    # with open(save_file, 'wb') as f:
    #     pickle.dump(result, f)


def main():
    args = parse_args()
    root = args.hsicity_path
    out_dir = args.out_dir if args.out_dir else root

    for split in ['train', 'test']:
        tasks = []
        print('Processing {}'.format(split))
        base_dir = osp.join(root, split)
        save_dir = osp.join(out_dir, split)
        mmcv.mkdir_or_exist(save_dir)

        for file in mmcv.scandir(base_dir, '.hsd'):
            tasks.append((save_dir, base_dir, file))

        if args.nproc > 1:
            mmcv.track_parallel_progress(read_hsd, tasks, args.nproc)
        else:
            mmcv.track_progress(read_hsd, tasks)

        if out_dir != root:
            for file in mmcv.scandir(base_dir, '.png'):
                shutil.copy(osp.join(base_dir, file), osp.join(save_dir, file))
            for file in mmcv.scandir(base_dir, '.jpg'):
                shutil.copy(osp.join(base_dir, file), osp.join(save_dir, file))


if __name__ == '__main__':
    main()
