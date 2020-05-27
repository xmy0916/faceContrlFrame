import paddlehub as hub
from paddlehub.dataset.base_cv_dataset import BaseCVDataset
import numpy as np
import cv2
import threading
import os

flag = 0

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


def showImg():
    global flag
    cap = cv2.VideoCapture(0)
    while 1:
        ret,frame = cap.read()
        cv2.imshow("cap",frame)

        if flag is 0:
            cv2.imwrite("./temp_out/cap.jpg",frame)
            flag = 1

        if cv2.waitKey(100) & 0xff == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def recognize():
    global flag
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

    label_map = dataset.label_dict()
    #run_states = task.finetune_and_eval()
    while 1:
        if flag is 1:
            data = []
            data.append("/home/xmy/PycharmProjects/test/paddle/proj3_recognizeMyself/temp_out/cap.jpg")
            index = 0
            run_states = task.predict(data=data)
            results = [run_state.run_results for run_state in run_states]

            for batch_result in results:
                batch_result = np.argmax(batch_result, axis=2)[0]
                for result in batch_result:
                    index += 1
                    result = label_map[result]
                    #print("input %i is %s, and the predict result is %s" %
                        #(index, data[index - 1], result))

            if "科比" in result:
                os.system("wmctrl -a \"pycharm\"")
            elif "库里" in result:
                os.system("wmctrl -a \"chrome\"")
            flag = 0



if __name__ == '__main__':
    t1 = threading.Thread(target=showImg)
    t2 = threading.Thread(target=recognize)
    t1.start()
    t2.start()