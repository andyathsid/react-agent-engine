import torch
from model import LVL
from transformers import RobertaTokenizer
from PIL import Image
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = LVL()
model.load_state_dict(torch.load("scold.pth", map_location=device))
model.to(device)
model.eval()

# Load tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


def predict(image_path, text):
    image = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        img_feat, txt_feat = model(image, tokens["input_ids"], tokens["attention_mask"])
        similarity = torch.matmul(img_feat, txt_feat.T).squeeze()

    return similarity.item()
