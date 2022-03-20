import os
import io
import json
import torch
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from model import resnet_34

app = Flask(__name__)
CORS(app)

weight_path = "./ResNet34-byMyTrainV2.pth"
class_json_path = "./class_indices.json"
assert os.path.exists(weight_path), "weights path does not exist."
assert os.path.exists(class_json_path), "class json path does not exist."


model = resnet_34(num_classes=60)
model.load_state_dict(torch.load(weight_path))
model.eval()

json_file = open(class_json_path, 'r')
class_indict = json.load(json_file)


def transforms_image(image_bytes):
    my_transforms = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != "RGB":
        image = image.convert("RGB")
    img = my_transforms(image)
    return torch.unsqueeze(img, dim=0)


def get_prediction(image_bytes):
    try:
        img = transforms_image(image_bytes)
        with torch.no_grad():
            output = torch.squeeze(model(img))
            predict = torch.softmax(output, dim=0)
            predict_class = torch.argmax(predict).numpy()
            index_pre = [class_indict[str(predict_class)], float(predict[predict_class].numpy())]
            return_info = {"result": index_pre}
    except Exception as e:
        return_info = {"result": [str(e)]}
    return return_info


@app.route("/predict", methods=["POST"])
def predict():
    image = request.files["file"]
    img_bytes = image.read()
    info = get_prediction(image_bytes=img_bytes)
    return jsonify(info)


@app.route("/", methods=["GET", "POST"])
def root():
    return render_template("index.html")


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=9000)


