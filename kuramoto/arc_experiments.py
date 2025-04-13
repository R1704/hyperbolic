"""
Example demonstrating how to use the ARC framework from an external project.

This script shows how to:
1. Import the ARC module
2. Load datasets
3. Get and visualize tasks
4. Evaluate models on tasks
"""


# Import from the ARC module
from arc import (
    load_dataset, 
    get_task,
    visualize,
    evaluate_model,
    download_arc_dataset,
    ARCTask,
    BaseModel
)

# Example implementation of a simple model
class ExampleModel(BaseModel):
    def predict(self, input_data):
        # This is just a dummy implementation
        # A real model would do actual predictions
        return input_data  # Just return the input as a placeholder

def main():
    print("ARC Framework External Usage Example")
    print("-" * 40)
    
    # 1. Download dataset (if not already present)
    print("Ensuring dataset is available...")
    download_arc_dataset("arc-agi")
    
    # 2. Load a dataset
    print("\nLoading dataset...")
    dataset = load_dataset("arc-agi", split="training")
    print(f"Loaded dataset with {len(dataset)} tasks")
    
    # 3. Get a specific task
    print("\nLoading a specific task...")
    task_id = dataset.task_ids[0]  # Get the first task ID
    task = get_task("arc-agi", task_id)
    print(f"Loaded task: {task}")
    
    # 4. Visualize the task
    print("\nVisualizing task...")
    visualize(task, save=True)
    print(f"Task visualization saved")
    
    # 5. Create a model and evaluate it
    print("\nEvaluating model on task...")
    model = ExampleModel()
    metrics = evaluate_model(model, task)
    print("Evaluation results:")
    metrics.report()

if __name__ == "__main__":
    main()