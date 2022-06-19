import fastai
print(fastai.__version__)
from fastai.vision.all import *
import gc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fastai.vision import *
from fastai.metrics import accuracy, top_k_accuracy
from PIL import Image

PATH = r'C:\Users\Diana\Desktop\Wardrobe-login\Wardrobe-logn'
PATH1 = r"C:\Users\Diana\Desktop\Wardrobe-login\Wardrobe-logn"
def load_model():
    path = r'C:\Users\Diana\Desktop\Wardrobe-login\Wardrobe-logn\atr-recognition-stage-3-resnet34.pth'
    # assert os.path.isfile(path)
    # map_location = torch.device('cpu')
    learn = load_learner(path, cpu=True)
    print(learn)
    return learn

def get_x(r):
  new_path = r["image_name"].replace('\\', '//')
  one_path = os.path.join(PATH1,new_path)
  filename = Path(one_path)
  # print(filename)
  return filename

def get_y(r): return r['labels'].split(',')
def splitter(df):
    train = df.index[df['is_valid']==0].tolist()
    valid = df.index[df['is_valid']==1].tolist()
    return train,valid


def predict_attribute(model, path, display_img=True):
    predicted = model.predict(path)
    if display_img:
        size = 244,244
        img=Image.open(path)
        # img.thumbnail(size,Image.ANTIALIAS)
        img.show()
    return predicted[0]

def accuracy_multi(inp, targ, thresh=0.5, sigmoid=True):
    "Compute accuracy when `inp` and `targ` are the same size."
    if sigmoid: inp = inp.sigmoid()
    return ((inp>thresh)==targ.bool()).float().mean()


class LabelSmoothingBCEWithLogitsLossFlat(BCEWithLogitsLossFlat):
    def __init__(self, eps: float = 0.1, **kwargs):
        self.eps = eps
        super().__init__(thresh=0.2, **kwargs)

    def __call__(self, inp, targ, **kwargs):
        # https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/166833#929222
        targ_smooth = targ.float() * (1. - self.eps) + 0.5 * self.eps
        return super().__call__(inp, targ_smooth, **kwargs)

    def __repr__(self):
        return "FlattenedLoss of LabelSmoothingBCEWithLogits()"

if __name__ == '__main__':

    TRAIN_PATH = "multilabel-train.csv"
    TEST_PATH = "multilabel-test.csv"
    CLASSES_PATH = "attribute-classes.txt"

    train_df = pd.read_csv(TRAIN_PATH)
    train_df.head()
    wd = 5e-7  # weight decay parameter
    opt_func = partial(ranger, wd=wd)

    dblock = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),
                       splitter=splitter,
                       get_x=get_x,
                       get_y=get_y,
                       item_tfms=RandomResizedCrop(224, min_scale=0.8),
                       batch_tfms=aug_transforms())

    dls = dblock.dataloaders(train_df, num_workers=0)
    dls.show_batch(nrows=1, ncols=6)

    dsets = dblock.datasets(train_df)
    metrics = [FBetaMulti(2.0, 0.2, average='samples'), partial(accuracy_multi, thresh=0.2)]

    test_df = pd.read_csv(TEST_PATH)
    test_df.head()
    dblock = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),
                       get_x=get_x,
                       get_y=get_y,
                       item_tfms=Resize(224))  # Not Sure)

    test_dls = dblock.dataloaders(test_df, num_workers=0)

    print("alo")
    print(dls.vocab)
    learn = vision_learner(dls, resnet34, loss_func=LabelSmoothingBCEWithLogitsLossFlat(),
                          metrics=metrics, opt_func=opt_func).to_fp16()

    path = r'C:\Users\Diana\Desktop\Wardrobe-login\Wardrobe-logn\atr-recognition-stage-3-resnet34.pth'
    # # assert os.path.isfile(path)
    # # map_location = torch.device('cpu')
    # map_location = torch.device('cpu')
    # learn = load_learner(path, cpu=False)
    # learn = torch.load(path, map_location='cpu')
    print(fastai.__version__)
    learn.load_state_dict(torch.load(path,
                                     map_location=torch.device('cpu'))['model'])

    # learn.data = test_dls
    # learn.validate()

    # learn.show_results(figsize=(12, 12))
    image_path = PATH + '/4.jpg'
    print(predict_attribute(learn, image_path))
