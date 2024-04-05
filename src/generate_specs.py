import argparse
import numpy as np
from torchvision import transforms, datasets
import os
import onnxruntime as rt
import re
from PIL import Image
import pandas as pd


def get_testing_dataset(img_size):
    ds = pd.read_csv('datasets/GTSRB_dataset/Test.csv')
    y_test = ds["ClassId"].values
    imgs = ds["Path"].values

    data = []
    for img in imgs:
        image = Image.open('datasets/GTSRB_dataset/' + img)
        image = image.resize(img_size)
        data.append(np.array(image))
    X_test = np.array(data).astype(np.float32)

    return X_test, y_test


def write_vnn_spec(x_ds, y_ds, index, eps, dir_path="./", prefix="spec", data_lb=0, data_ub=1, n_class=10, mean=0.0, std=1.0, negate_spec=False):
    x = x_ds[index]
    y = y_ds[index]
    x = np.array(x)
    x_lb = np.clip(x - eps, data_lb, data_ub)
    x_lb = ((x_lb-mean)/std).reshape(-1)
    x_ub = np.clip(x + eps, data_lb, data_ub)
    x_ub = ((x_ub - mean) / std).reshape(-1)

    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    if np.all(mean == 0.) and np.all(std == 1.):
        spec_name = f"{prefix}_idx_{index}_eps_{eps:.5f}.vnnlib"
    else:
        existing_specs = os.listdir(dir_path)
        competing_norm_ids = [int(re.match(f"{prefix}_idx_{index}_eps_{eps:.5f}_n([0-9]+).vnnlib", spec).group(
            1)) for spec in existing_specs if spec.startswith(f"{prefix}_idx_{index}_eps_{eps:.5f}_n")]
        norm_id = 1 if len(competing_norm_ids) == 0 else max(
            competing_norm_ids)+1
        spec_name = f"{prefix}_idx_{index}_eps_{eps:.5f}_n{norm_id}.vnnlib"

    spec_path = os.path.join(dir_path, spec_name)

    with open(spec_path, "w") as f:
        f.write(f"; Spec for sample id {index} and epsilon {eps:.5f}\n")

        f.write(f"\n; Definition of input variables\n")
        for i in range(len(x_ub)):
            f.write(f"(declare-const X_{i} Real)\n")

        f.write(f"\n; Definition of output variables\n")
        for i in range(n_class):
            f.write(f"(declare-const Y_{i} Real)\n")

        f.write(f"\n; Definition of input constraints\n")
        for i in range(len(x_ub)):
            f.write(f"(assert (<= X_{i} {x_ub[i]:.8f}))\n")
            f.write(f"(assert (>= X_{i} {x_lb[i]:.8f}))\n")

        f.write(f"\n; Definition of output constraints\n")
        if negate_spec:
            for i in range(n_class):
                if i == y:
                    continue
                f.write(f"(assert (<= Y_{i} Y_{y}))\n")
        else:
            f.write(f"(assert (or\n")
            for i in range(n_class):
                if i == y:
                    continue
                f.write(f"\t(and (>= Y_{i} Y_{y}))\n")
            f.write(f"))\n")
    return spec_name


def get_sample_idx(n, block=True, seed=42, n_max=10000, start_idx=None):
    np.random.seed(seed)
    assert n <= n_max, f"only {n_max} samples are available"
    if block:
        if start_idx is None:
            start_idx = np.random.choice(n_max, 1, replace=False)
        else:
            start_idx = start_idx % n_max
        idx = list(np.arange(start_idx, min(start_idx+n, n_max)))
        idx += list(np.arange(0, n-len(idx)))
    else:
        idx = list(np.random.choice(n_max, n, replace=False))
    return idx


def main():
    parser = argparse.ArgumentParser(description='VNN spec generator',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--epsilon', type=float, required=True,
                        help='The epsilon for L_infinity perturbation')
    parser.add_argument('--n', type=int, default=25,
                        help='The number of specs to generate')
    parser.add_argument('--block', action="store_true",
                        default=False, help='Generate specs in a block')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for idx generation')
    parser.add_argument('--start_idx', type=int, default=None,
                        help='Enforce block mode and return deterministic indices')
    parser.add_argument("--network", type=str, default=None,
                        help="Network to evaluate as .onnx file.")
    parser.add_argument('--img_size', type=int, default=30,
                        help='Image size used in the model')
    parser.add_argument('--mean', nargs='+', type=float, default=0.0,
                        help='the mean used to normalize the data with')
    parser.add_argument('--std', nargs='+', type=float, default=1.0,
                        help='the standard deviation used to normalize the data with')
    parser.add_argument('--negate_spec', action="store_true", default=False,
                        help='Generate spec that is violated for correct certification')
    parser.add_argument('--dont_extend', action="store_true", default=False,
                        help='Do not filter for naturally correctly classified images')

    args = parser.parse_args()

    if args.start_idx is not None:
        args.block = True
        print(
            f"Generating {args.n} deterministic specs starting from index {args.start_idx}.")
    else:
        print(f"Generating {args.n} random specs using seed {args.seed}.")

    x_test, y_test = get_testing_dataset((args.img_size, args.img_size))

    if args.network is not None:
        sess = rt.InferenceSession(args.network)
        input_name = sess.get_inputs()[0].name

    mean = np.array(args.mean).reshape((1, -1, 1, 1)).astype(np.float32)
    std = np.array(args.std).reshape((1, -1, 1, 1)).astype(np.float32)

    idxs = get_sample_idx(args.n, block=args.block, seed=args.seed, n_max=len(
        x_test), start_idx=args.start_idx)
    spec_path = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "../specs", "traffic")

    i = 0
    ii = 1
    n_ok = 0
    while i < len(idxs):
        idx = idxs[i]
        i += 1
        if args.network is not None:
            x = x_test[idx]
            y = y_test[idx]
            x_new = x[np.newaxis, ...]
            pred_onx = sess.run(None, {input_name: x_new})[0]
            y_pred = np.argmax(pred_onx, axis=-1)
            n_ok += sum(y == y_pred)

        if args.network is None or y == y_pred:
            # spec_i = write_vnn_spec(x_test, y_test, idx, args.epsilon, dir_path=spec_path, prefix="img_size_"+str(args.img_size),
            #                         data_lb=0, data_ub=255, n_class=43, mean=mean, std=std, negate_spec=args.negate_spec)
            spec_i = write_vnn_spec(x_test, y_test, idx, 1, dir_path=spec_path, prefix="img_size_"+str(args.img_size),
                                    data_lb=0, data_ub=255, n_class=43, mean=mean, std=std, negate_spec=args.negate_spec)
            spec_i = write_vnn_spec(x_test, y_test, idx, 3, dir_path=spec_path, prefix="img_size_"+str(args.img_size),
                                    data_lb=0, data_ub=255, n_class=43, mean=mean, std=std, negate_spec=args.negate_spec)
            spec_i = write_vnn_spec(x_test, y_test, idx, 5, dir_path=spec_path, prefix="img_size_"+str(args.img_size),
                                    data_lb=0, data_ub=255, n_class=43, mean=mean, std=std, negate_spec=args.negate_spec)
            spec_i = write_vnn_spec(x_test, y_test, idx, 10, dir_path=spec_path, prefix="img_size_"+str(args.img_size),
                                    data_lb=0, data_ub=255, n_class=43, mean=mean, std=std, negate_spec=args.negate_spec)
            spec_i = write_vnn_spec(x_test, y_test, idx, 15, dir_path=spec_path, prefix="img_size_"+str(args.img_size),
                                    data_lb=0, data_ub=255, n_class=43, mean=mean, std=std, negate_spec=args.negate_spec)
            spec_i = write_vnn_spec(x_test, y_test, idx, 20, dir_path=spec_path, prefix="img_size_"+str(args.img_size),
                                    data_lb=0, data_ub=255, n_class=43, mean=mean, std=std, negate_spec=args.negate_spec)
        elif not args.dont_extend:
            # only sample idxs while there are still new samples to be found
            print(idxs[-1])
            if len(idxs) < len(x_test):
                if args.block:  # if we want samples in a block, just get the next one
                    idxs.append(*get_sample_idx(1, True,
                                                n_max=len(x_test), start_idx=idxs[-1]+1))
                # otherwise sample deterministicly (for given seed) until we find a new sample
                else:
                    tmp_idx = get_sample_idx(
                        1, False, seed=args.seed+ii, n_max=len(x_test))
                    ii += 1
                    while tmp_idx in idxs:
                        tmp_idx = get_sample_idx(
                            1, False, seed=args.seed + ii, n_max=len(x_test))
                        ii += 1
                    idxs.append(*tmp_idx)
    print(f"{len(idxs)-n_ok} samples were misclassified{''if args.dont_extend else ' and replacement samples drawn'}.")


if __name__ == "__main__":
    main()
