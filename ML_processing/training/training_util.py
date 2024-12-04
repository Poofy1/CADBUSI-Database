import random, torch, os
from PIL import Image




    
    
class StaticNoise:
    def __init__(self, intensity_min=0.0, intensity_max=0.2):
        self.intensity_min = intensity_min
        self.intensity_max = intensity_max

    def __call__(self, img):
        intensity = random.uniform(self.intensity_min, self.intensity_max)
        noise = torch.randn(*img.size()) * intensity
        img = torch.clamp(img + noise, 0, 1)

        return img
    
def get_image_size(image_path):
    with Image.open(image_path) as img:
        return img.size

def find_max_dimensions(df):
    max_width = max_height = 0
    for size in df['image_size']:
        width, height = eval(size)  # Convert string tuple to actual tuple
        max_width = max(max_width, width)
        max_height = max(max_height, height)
    return max_width, max_height



def load_model(model, model_path):
    if os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Model loaded from {model_path}")
    else:
        print("No previous model found, starting training from scratch.")
    return model