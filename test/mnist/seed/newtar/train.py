import logging
import sys
#sys.path.append('/media/sheetal/project_space/FL/code/fedbird')
from kerasweights_modeltar.py import KerasWeightsHelper

from client1_new import TrainingProcess, Model, TrainDataReader


if __name__ == '__main__':
    logger = logging.getLogger('__name__')
    logger.info("Calling the train function")
    helper = KerasWeightsHelper()
    model = Model()
    data = TrainDataReader()
    start_process = TrainingProcess(data, model)
                                    #,classes_path ='/app/client/model_data/seabird_classes.txt', anchors_path = '/app/client/model_data/tiny_yolo_anchors.txt', data_path ='/app/data/Annotation/list1.txt')
    #load global model weights here
    #global_model_weights = load_weights()
    #helper.load_model(start_process.local_model)
    outer_model = helper.load_model(sys.argv[1])
    model = outer_model['model']
    #local_model_weights = start_process.train(sys.argv[1])
    local_model = start_process.train()
    helper.save_model(local_model, path='package')


