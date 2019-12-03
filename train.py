from dataset import DataSplit

images_path = "/home/layely/Myprojects/datasets/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages"
txt_file = "voc_2012.txt"
train = 0.6
val = 0.2
test = 0.2
input_shape = (464,464)

dataloader = DataSplit(images_path, txt_file, train, val, test, input_shape, seed=1)