import torch
import cv2
import os
import csv
from PIL import Image
from torchvision import models, transforms
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

# Paths and runtime settings.
MODEL_PATH = r"D:\Projects\AI_ML_DL\SmartCropDoc-AI\model\mobilenetv3_phase_2.pth"
TEST_DIR = r"D:\Projects\AI_ML_DL\SmartCropDoc-AI\Dataset\test"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class order must match the training checkpoint exactly.
CLASSES = [
    'Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple_healthy',
    'Banana_Fusarium_wilt', 'Banana_Healthy','Banana_Panama','Banana_Sigatoka',
    'Cauliflower_Bacterial_spot_rot','Cauliflower_Black_Rot','Cauliflower_Downy_Mildew','Cauliflower_Healthy',
    'Corn_(maize)_Cercospora_leaf_spot','Corn_(maize)_Common_rust', 'Corn_(maize)_Northen_Leaf_Blight','Corn_(maize)_healthy',
    'Grape_Black_rot', 'Grape_Esca(Black_Measles)', 'Grape_Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape_healthy',
    'Mango_Anthracnose','Mango_Gall Midge','Mango_Healthy','Mango_Powdery Mildew',
    'Potato_Early_blight', 'Potato_Late_blight', 'Potato_healthy',
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Two_spotted_Spider_mites',
    'Tomato_healthy'
]

# Initialize Real-ESRGAN for optional enhancement before classification.
model_esr = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
upsampler = RealESRGANer(
    scale=4, 
    model_path=r'D:\Projects\AI_ML_DL\SmartCropDoc-AI\RealESRGAN_x4plus.pth', 
    model=model_esr, 
    device=DEVICE,
    tile=400, tile_pad=10, half=True 
)

# Load MobileNetV3 classifier.
model_cls = models.mobilenet_v3_large()
model_cls.classifier[3] = torch.nn.Linear(model_cls.classifier[3].in_features, len(CLASSES))
model_cls.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model_cls.to(DEVICE).eval()

def run_inference():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    valid_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    images = []
    for root, dirs, files in os.walk(TEST_DIR):
        for file in files:
            if file.endswith(valid_extensions):
                images.append(os.path.join(root, file))
    
    if not images:
        print(f"❌ No images found in {TEST_DIR}.")
        return

    print(f"🚀 Found {len(images)} images. Starting inference...\n")

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Write results as we process each image.
    with open("field_test_results.csv", "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Actual_Folder", "File_Name", "Predicted_Class", "Confidence"])

        for img_path in images:
            true_folder = os.path.basename(os.path.dirname(img_path))
            img_name = os.path.basename(img_path)
            
            img = cv2.imread(img_path)
            if img is None: continue
            
            try:
                # Step A: enhance image quality.
                enhanced_img, _ = upsampler.enhance(img, outscale=2)
                
                # Step B: preprocess for classifier input.
                img_rgb = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)
                input_tensor = preprocess(Image.fromarray(img_rgb)).unsqueeze(0).to(DEVICE)
                
                # Step C: run prediction.
                with torch.no_grad():
                    output = model_cls(input_tensor)
                    prob = torch.nn.functional.softmax(output, dim=1)
                    conf, pred = torch.max(prob, 1)
                
                pred_label = CLASSES[pred.item()]
                conf_val = f"{conf.item()*100:.2f}%"
                
                # Log result in terminal.
                print(f"Folder: {true_folder} | File: {img_name}")
                print(f"   >>> Predicted: {pred_label} ({conf_val})")
                print("-" * 50)
                
                # Persist row immediately to avoid data loss on interruption.
                csv_writer.writerow([true_folder, img_name, pred_label, conf_val])
                
            except torch.cuda.OutOfMemoryError:
                print(f"⚠️ Memory Error for {img_name}, skipping.")
                torch.cuda.empty_cache()
                continue

    print("✅ All results saved to field_test_results.csv")

if __name__ == '__main__':
    run_inference()
