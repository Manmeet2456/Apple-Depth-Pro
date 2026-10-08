import os
import glob
import torch
import gc
import numpy as np
import depth_pro

def process_frames():
    # 1. Define Paths 
    input_folder = "./_1t6z04Ir0g_0_100-20260602T072351Z-3-001/"  # Using the folder from your screenshot
    output_folder = "./_1t6z04Ir0g_0_100-20260602T072351Z-3-001_output/"
    os.makedirs(output_folder, exist_ok=True)

    # 2. Load Model in Half-Precision
    print("Loading model...")
    model, transform = depth_pro.create_model_and_transforms()
    model = model.to(device="cuda", dtype=torch.float16)
    model.eval()

    # Grab all png frames (matching your folder structure)
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
            
            # Extract RAW depth map (absolute distance in meters)
            # We cast it back to float32 before saving so your downstream smoke simulator reads standard floats
            depth_map = prediction["depth"].detach().cpu().to(torch.float32).numpy().squeeze()
            
            # Extract original filename without the extension
            filename, _ = os.path.splitext(os.path.basename(frame_path))
            
            # Save the result as a .npy file
            output_path = os.path.join(output_folder, f"{filename}_depth.npy")
            np.save(output_path, depth_map)

            # 4. Memory Management
            if count % 20 == 0:
                gc.collect()
                torch.cuda.empty_cache()
                print(f"Processed {count}/{total_frames} frames...")

    print("All frames processed and saved as .npy successfully!")

if __name__ == "__main__":
    process_frames()