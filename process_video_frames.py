import os
import glob
import torch
import gc
import numpy as np
from PIL import Image
import depth_pro
import matplotlib.pyplot as plt

def process_frames():
    # 1. Define Paths 
    input_folder = "./_1t6z04Ir0g_0_100-20260602T072351Z-3-001/" 
    output_folder = "./_1t6z04Ir0g_0_100-20260602T072351Z-3-001_visuals/"
    
    os.makedirs(output_folder, exist_ok=True)

    # 2. Load Model in Half-Precision (Crucial for 6GB VRAM)
    print("Loading model...")
    model, transform = depth_pro.create_model_and_transforms()
    model = model.to(device="cuda", dtype=torch.float16)
    model.eval()

    # Grab all PNG frames
    frame_paths = sorted(glob.glob(os.path.join(input_folder, "*.png")))
    total_frames = len(frame_paths)
    print(f"Found {total_frames} frames to process.")

    # 3. Processing Loop
    with torch.no_grad(): 
        for count, frame_path in enumerate(frame_paths):
            
            # Load and preprocess image
            image, _, f_px = depth_pro.load_rgb(frame_path)
            image = transform(image)
            
            # Move input to GPU and cast to float16
            image = image.to(device="cuda", dtype=torch.float16)

            # Run Inference
            prediction = model.infer(image, f_px=f_px)
            
            # Extract depth map
            depth_map = prediction["depth"].detach().cpu().numpy().squeeze()
            
            # Normalize the depth map to a 0.0 to 1.0 range
            depth_min = depth_map.min()
            depth_max = depth_map.max()
            depth_normalized = (depth_map - depth_min) / (depth_max - depth_min + 1e-8)
            
            # Invert depth: closer objects = warm/red, far objects = cool/blue
            depth_inverted = 1.0 - depth_normalized
            
            # Apply 'turbo' colormap and convert to 8-bit unsigned integers
            depth_colored = (plt.cm.turbo(depth_inverted)[:, :, :3] * 255).astype(np.uint8)
            
            # Save visual image
            base_filename = os.path.splitext(os.path.basename(frame_path))[0]
            output_image = Image.fromarray(depth_colored)
            output_image.save(os.path.join(output_folder, f"vis_{base_filename}.png"))

            # 4. Memory Management
            if count % 20 == 0:
                gc.collect()
                torch.cuda.empty_cache()
                print(f"Processed {count}/{total_frames} frames...")

    print(f"All frames processed successfully!\nVisuals saved to: {output_folder}")

if __name__ == "__main__":
    process_frames()