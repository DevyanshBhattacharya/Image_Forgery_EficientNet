import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from model import SegmentationModel

# Function to load and preprocess the image
def load_image(image_path, transform=None):
    image = Image.open(image_path).convert('RGB')  
    
    if transform:
        image = transform(image) 
    
    return image


def visualize_segmentation(model, input_image, device='cuda'):
    model.eval()  

    with torch.no_grad():
       
        input_image = input_image.to(device)

        # Generate prediction
        prediction = model(input_image.unsqueeze(0))  
        prediction = torch.sigmoid(prediction)  
        predicted_mask = prediction.squeeze(0).squeeze(0).cpu().numpy()  
        
        
        predicted_mask = np.where(predicted_mask > 0.5, 1, 0)

       
        input_image_np = input_image.cpu().numpy().transpose(1, 2, 0) 

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        # Plot input image
        axs[0].imshow(input_image_np)
        axs[0].set_title('Input Image')
        axs[0].axis('off')

        # Plot predicted mask
        axs[1].imshow(predicted_mask, cmap='gray')
        axs[1].set_title('Predicted Mask')
        axs[1].axis('off')

        plt.show()


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Path to the test image
test_image_path = "/Users/hiteshgupta/Documents/ML-CV/Image-Segement-Forgery/Dataset/train/img/img/0_000000064070.tif" 

# Load and preprocess the image
input_image = load_image(test_image_path, transform=transform)
device = 'cuda'

model = SegmentationModel(input_size=(256, 256))  
model.load_state_dict(torch.load('models/segmentation_model.pth', map_location=torch.device('mps')))  
model = model.to(device)

visualize_segmentation(model, input_image, device=device)