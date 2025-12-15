## for 128x128

from Imports import *
from model_loading import *

app = Flask(__name__)

device = torch.device('cpu')

model = Handwritten().to(device)
model.load_state_dict(torch.load("best_model2.pth", map_location=device))
model.eval()

def preprocess(img):
    img = img.convert("L")
    img = ImageOps.invert(img)

    # Resize the digit to 128x128
    img = img.resize((128,128))

    arr = np.array(img).astype("float32") / 255.0
    arr = (arr - 0.1307) / 0.3081

    tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0)
    return tensor

@app.route("/")
def index():
    return render_template("UI/index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["image"]
    data = data.split(",")[1]
    img_bytes = base64.b64decode(data)
    img = Image.open(io.BytesIO(img_bytes))

    tensor = preprocess(img)

    with torch.no_grad():
        out = model(tensor)
        pred = out.argmax(1).item()

    return jsonify({"prediction": pred})

if __name__ == "__main__":
    app.run(debug=True)
