import yaml
import os
from ultralytics import YOLO

def main():
    dataset_dir = os.path.abspath(os.path.join("Data", "dataset"))
    yaml_path = os.path.join(dataset_dir, "data.yaml")

    # Ensure the yaml exists
    if not os.path.exists(yaml_path):
        print(f"Error: {yaml_path} does not exist.")
        return

    # Fix the yaml
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    # Update the paths to be absolute based on current directory
    data['train'] = os.path.join(dataset_dir, "images", "train")
    data['val'] = os.path.join(dataset_dir, "images", "train") # Validation set is missing, using train

    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

    print("Updated data.yaml successfully.")

    print("Starting YOLO training...")
    # Load a model
    model = YOLO('models/yolov8n.pt')

    # Train the model
    # Note: 10 epochs used for demonstration/quick training.
    # We will let YOLO run in the console.
    results = model.train(
        data=yaml_path,
        epochs=10,
        imgsz=640,
        device='0',
        project='models',
        name='drishti_custom',
        exist_ok=True,
        workers=0,
        batch=-1
    )
    
    print("Training complete! Model saved in models/drishti_custom")

if __name__ == "__main__":
    main()
