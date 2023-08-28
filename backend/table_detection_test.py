import os
import warnings
from pathlib import Path

import cv2
import torch
import torchvision


warnings.filterwarnings('ignore')


class tableDetectionTest:
    # The init method or constructor
    def __init__(self, input_file, output_dir, model_path):
        # Instance Variable
        self.input_file = input_file
        self.output_dir = output_dir
        self.model_path = model_path

    """  table detection """

    def table_detection(self, cd_model):
        try:
            detection_path = "table_detection"
            detection_path = os.path.join(self.output_dir, detection_path)
            os.makedirs(detection_path, exist_ok=True)

            # get a file name and file extension
            file_name = Path(self.input_file).stem
            # file_extension = Path(self.input_file).suffix

            stripped = file_name.split('_', 1)[0]

            detection_path = os.path.join(detection_path, stripped)
            os.makedirs(detection_path, exist_ok=True)

            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

            # model = torch.load(self.model_path)
            model = cd_model
            cd_model = torch.load('table_attribution_col_v1.0.pth', map_location=torch.device('cuda:0'))
            #cd_model = torch.load(CONFIG['CD_MODEL_PATH'], map_location=torch.device('cuda:0'))
            # model.load_state_dict(torch.load(self.model_path))
            model.eval()

            input_image = cv2.imread(self.input_file)
            input_img = torchvision.transforms.functional.to_tensor(input_image).to(device)

            table_tensors = model([input_img])[0]["boxes"]
            prediction = table_tensors.data.cpu().numpy()

            file_name_add = []
            count = 0
            detected_file = None
            if prediction is not None and len(prediction) > 0:
                for table_tensor in prediction:
                    table_tensor = [int(i) for i in table_tensor]
                    detected_file = f'{detection_path}/{file_name}_{count}.jpg'
                    file_name_add.append(file_name)
                    count += 1
                if len(file_name_add) == 1:
                    cv2.imwrite(detected_file, input_image[table_tensor[1]:table_tensor[3], table_tensor[0]:table_tensor[2]])
            return detected_file, file_name_add
        except Exception as ex:
            raise ex