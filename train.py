import paddlehub as hub
from paddlehub.dataset.base_cv_dataset import BaseCVDataset

class DemoDataset(BaseCVDataset):
    def __init__(self):
        # 数据集存放位置

        self.dataset_dir = ""
        super(DemoDataset, self).__init__(
            base_path=self.dataset_dir,
            train_list_file="dataset/train_list.txt",
            validate_list_file="dataset/validate_list.txt",
            test_list_file="dataset/test_list.txt",
            label_list_file="dataset/label_list.txt",
        )

module = hub.Module(name="resnet_v2_50_imagenet")
dataset = DemoDataset()

data_reader = hub.reader.ImageClassificationReader(
    image_width=module.get_expected_image_width(),
    image_height=module.get_expected_image_height(),
    images_mean=module.get_pretrained_images_mean(),
    images_std=module.get_pretrained_images_std(),
    dataset=dataset)

config = hub.RunConfig(
    use_cuda=False,  # 是否使用GPU训练，默认为False；
    num_epoch=5,  # Fine-tune的轮数；
    checkpoint_dir="cv_finetune_turtorial_demo",  # 模型checkpoint保存路径, 若用户没有指定，程序会自动生成；
    batch_size=10,  # 训练的批大小，如果使用GPU，请根据实际情况调整batch_size；
    eval_interval=10,  # 模型评估的间隔，默认每100个step评估一次验证集；
    strategy=hub.finetune.strategy.DefaultFinetuneStrategy())  #Fine-tune优化策略；
    #strategy=hub.finetune.strategy.AdamWeightDecayStrategy())

input_dict, output_dict, program = module.context(trainable=True)
img = input_dict["image"]
feature_map = output_dict["feature_map"]
feed_list = [img.name]

task = hub.ImageClassifierTask(
    data_reader=data_reader,
    feed_list=feed_list,
    feature=feature_map,
    num_classes=dataset.num_labels,
    config=config)

run_states = task.finetune_and_eval()