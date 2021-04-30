from config import resume_fold
from cls_train import run

if __name__ == '__main__':
    for fold in range(resume_fold, 5):
        run(fold)
