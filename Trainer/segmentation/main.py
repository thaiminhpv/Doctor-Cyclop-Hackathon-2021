from dump_mask import dump_mask
from config import resume_fold
from seg_train import run

if __name__ == '__main__':
    for fold in range(resume_fold, 5):
        run(fold)

    dump_mask()
