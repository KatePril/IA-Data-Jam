import colorsys
import os

from PIL import Image, ImageDraw, ImageFont
import torch
from torch.utils.data import Dataset
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor, RTDetrV2Config, Trainer, TrainingArguments


MODEL_START_PATH= "./rtdetr_finetuned" # "checkpoint-7000"


class YOLOtoRTDETRDataset(Dataset):
    def __init__(self, images_dir, labels_dir, image_processor, class_names, split='train'):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_processor = image_processor
        self.class_names = class_names
        self.id2label = {i: name for i, name in enumerate(class_names)}
        self.label2id = {name: i for i, name in enumerate(class_names)}
        
        self.image_files = [f for f in os.listdir(images_dir) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        self.valid_images = []
        for img_file in self.image_files:
            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(labels_dir, label_file)
            if os.path.exists(label_path):
                self.valid_images.append(img_file)
        
        print(f"Found {len(self.valid_images)} valid image-label pairs")
    
    def __len__(self):
        return len(self.valid_images)
    
    def yolo_to_coco_bbox(self, yolo_bbox, img_width, img_height):
        """Convert YOLO format (x_center, y_center, width, height) to COCO format (x_min, y_min, width, height)"""
        x_center, y_center, width, height = yolo_bbox
        
        x_center *= img_width
        y_center *= img_height
        width *= img_width
        height *= img_height
        
        # Convert to COCO format
        x_min = x_center - width / 2
        y_min = y_center - height / 2
        
        return [x_min, y_min, width, height]
    
    def __getitem__(self, idx):
        img_file = self.valid_images[idx]
        img_path = os.path.join(self.images_dir, img_file)
        
        image = Image.open(img_path).convert('RGB')
        img_width, img_height = image.size
        
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(self.labels_dir, label_file)
        
        boxes = []
        labels = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    if line:
                        parts = line.split()
                        class_id = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:5])
                        
                        # Convert YOLO bbox to COCO format
                        coco_bbox = self.yolo_to_coco_bbox([x_center, y_center, width, height], 
                                                         img_width, img_height)
                        boxes.append(coco_bbox)
                        labels.append(class_id)
        
        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.tensor([], dtype=torch.int64)
        
        annotations = []
        if len(boxes) > 0:
            for i in range(len(boxes)):
                annotation = {
                    'image_id': idx,
                    'bbox': boxes[i].tolist(),
                    'category_id': labels[i].item(),
                    'area': float(boxes[i][2] * boxes[i][3]),
                    'iscrowd': 0,
                    'id': i
                }
                annotations.append(annotation)
        
        target = {
            'image_id': idx,
            'annotations': annotations
        }
        
        encoding = self.image_processor(images=image, annotations=target, return_tensors="pt")
        
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0] if "labels" in encoding else target
        
        return {
            "pixel_values": pixel_values,
            "labels": target
        }


def collate_fn(batch):
    """Custom collate function for batching"""
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = [item["labels"] for item in batch]
    
    return {
        "pixel_values": pixel_values,
        "labels": labels
    }


def setup_fine_tuning(dataset_path, class_names, model_name="PekingU/rtdetr_v2_r18vd"):
    """
    Set up RT-DETR fine-tuning
    
    Args:
        dataset_path: Path to your out_archive folder
        class_names: List of class names in order (e.g., ['person', 'car', 'bicycle'])
        model_name: Pre-trained model to use
    """
    
    images_dir = os.path.join(dataset_path, 'images')
    labels_dir = os.path.join(dataset_path, 'labels')
    
    image_processor = RTDetrImageProcessor.from_pretrained(MODEL_START_PATH)
    config = RTDetrV2Config.from_pretrained(MODEL_START_PATH)
    
    config.num_labels = len(class_names)
    config.id2label = {i: name for i, name in enumerate(class_names)}
    config.label2id = {name: i for i, name in enumerate(class_names)}
    
    model = RTDetrV2ForObjectDetection.from_pretrained(
        MODEL_START_PATH, 
        config=config,
        ignore_mismatched_sizes=True
    )

    #dataset
    dataset = YOLOtoRTDETRDataset(images_dir, labels_dir, image_processor, class_names)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Classes: {class_names}")
    
    return model, image_processor, train_dataset, val_dataset


def train_model(model, train_dataset, val_dataset, output_dir="./rtdetr_finetuned"):
    """Train the model"""
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=20,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=500,
        save_total_limit=3,
        remove_unused_columns=False,
        push_to_hub=False,
        dataloader_num_workers=16,
        learning_rate=1e-6,
        lr_scheduler_type="cosine",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        tokenizer=image_processor,
    )
    
    print("Starting training...")
    trainer.train()

    trainer.save_model()
    print(f"Model saved to {output_dir}")
    
    return trainer


def load_finetuned_model(model_path):
    """Load fine-tuned model for inference"""
    
    # Load the fine-tuned model and processor
    model = RTDetrV2ForObjectDetection.from_pretrained(model_path)
    image_processor = RTDetrImageProcessor.from_pretrained(model_path)
    
    model.eval()  # Set to evaluation mode
    
    return model, image_processor


def inference_with_finetuned_model(model_path, image_path, threshold=0.3):
    """Run inference with your fine-tuned model"""
    
    # Load model and processor
    model, image_processor = load_finetuned_model(model_path)
    
    # Load and process image
    image = Image.open(image_path).convert('RGB')
    inputs = image_processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Post-process results
    results = image_processor.post_process_object_detection(
        outputs, 
        target_sizes=torch.tensor([(image.height, image.width)]), 
        threshold=threshold
    )
    
    # Print results
    print(f"Results for {image_path}:")
    for result in results:
        for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
            score, label = score.item(), label_id.item()
            box = [round(i, 2) for i in box.tolist()]
            class_name = model.config.id2label[label]
            print(f"{class_name}: {score:.3f} {box}")
    
    return results


# Batch inference for multiple images
def batch_inference(model_path, images_folder, threshold=0.3):
    """Run inference on multiple images"""
    
    model, image_processor = load_finetuned_model(model_path)
    
    # Get all image files
    image_files = [f for f in os.listdir(images_folder) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    results_dict = {}
    
    for image_file in image_files:
        image_path = os.path.join(images_folder, image_file)
        
        try:
            image = Image.open(image_path).convert('RGB')
            inputs = image_processor(images=image, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            results = image_processor.post_process_object_detection(
                outputs, 
                target_sizes=torch.tensor([(image.height, image.width)]), 
                threshold=threshold
            )
            
            results_dict[image_file] = results[0]  # Store results for this image
            
            print(f"\n{image_file}:")
            for score, label_id, box in zip(results[0]["scores"], results[0]["labels"], results[0]["boxes"]):
                score, label = score.item(), label_id.item()
                box = [round(i, 2) for i in box.tolist()]
                class_name = model.config.id2label[label]
                print(f"  {class_name}: {score:.3f} {box}")
                
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
    
    return results_dict


def debug_inference(model_path, image_path, threshold=0.1):
    """Debug inference to see what's happening"""
    
    print(f"Loading model from: {model_path}")
    model, image_processor = load_finetuned_model(model_path)
    
    print(f"Model config:")
    print(f"  Number of labels: {model.config.num_labels}")
    print(f"  Classes: {model.config.id2label}")
    
    # Load and process image
    image = Image.open(image_path).convert('RGB')
    print(f"Image size: {image.size}")
    
    inputs = image_processor(images=image, return_tensors="pt")
    print(f"Input tensor shape: {inputs['pixel_values'].shape}")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    print(f"Raw output keys: {outputs.keys()}")
    if hasattr(outputs, 'logits'):
        print(f"Logits shape: {outputs.logits.shape}")
    if hasattr(outputs, 'pred_boxes'):
        print(f"Pred boxes shape: {outputs.pred_boxes.shape}")
    
    # Check raw predictions before post-processing
    if hasattr(outputs, 'logits'):
        # Get class probabilities
        class_probs = torch.softmax(outputs.logits, dim=-1)
        max_probs, predicted_classes = torch.max(class_probs, dim=-1)
        
        print(f"Max class probabilities (before threshold): {max_probs.max().item():.4f}")
        print(f"Min class probabilities: {max_probs.min().item():.4f}")
        print(f"Average class probabilities: {max_probs.mean().item():.4f}")
    
    # Try with very low threshold first
    results = image_processor.post_process_object_detection(
        outputs, 
        target_sizes=torch.tensor([(image.height, image.width)]), 
        threshold=threshold
    )
    
    print(f"\nResults with threshold {threshold}:")
    if results and len(results[0]["scores"]) > 0:
        for score, label_id, box in zip(results[0]["scores"], results[0]["labels"], results[0]["boxes"]):
            score, label = score.item(), label_id.item()
            box = [round(i, 2) for i in box.tolist()]
            class_name = model.config.id2label.get(label, f"class_{label}")
            print(f"  {class_name}: {score:.4f} {box}")
    else:
        print("  No detections found!")
        
        # Try even lower thresholds
        for test_threshold in [0.05, 0.01, 0.001]:
            test_results = image_processor.post_process_object_detection(
                outputs, 
                target_sizes=torch.tensor([(image.height, image.width)]), 
                threshold=test_threshold
            )
            if test_results and len(test_results[0]["scores"]) > 0:
                print(f"  Found {len(test_results[0]['scores'])} detections with threshold {test_threshold}")
                print(f"  Highest score: {test_results[0]['scores'][0].item():.4f}")
                break
        else:
            print("  No detections even with very low thresholds!")
    
    return results


def visualize_detections(model_path, image_path, threshold=0.3, save_path=None):
    """
    Clean function to visualize object detections
    
    Args:
        model_path: Path to fine-tuned model directory
        image_path: Path to input image
        threshold: Confidence threshold for detections
        save_path: Where to save annotated image (optional, auto-generates if None)
    
    Returns:
        PIL Image with drawn bounding boxes
    """
    
    # Load model and processor
    model = RTDetrV2ForObjectDetection.from_pretrained(model_path)
    processor = RTDetrImageProcessor.from_pretrained(model_path)
    model.eval()
    
    # Load and process image
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt")
    
    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Post-process results
    results = processor.post_process_object_detection(
        outputs, 
        target_sizes=torch.tensor([(image.height, image.width)]), 
        threshold=threshold
    )
    
    # Draw on image
    draw = ImageDraw.Draw(image)
    
    # Try to load font
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    # Generate colors for classes
    def get_color(class_id):
        hue = (class_id * 0.618033988749895) % 1  # Golden ratio for good color distribution
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 1.0)
        return tuple(int(c * 255) for c in rgb)
    
    # Draw detections
    if results and len(results[0]["scores"]) > 0:
        for score, label_id, box in zip(results[0]["scores"], results[0]["labels"], results[0]["boxes"]):
            score_val = score.item()
            label = label_id.item()
            x1, y1, x2, y2 = [int(coord) for coord in box.tolist()]
            
            # Get class name and color
            class_name = model.config.id2label.get(label, f"class_{label}")
            color = get_color(label)
            
            # Draw box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # Draw label
            text = f"{class_name}: {score_val:.2f}"
            bbox = draw.textbbox((x1, y1), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            draw.rectangle([x1, y1 - text_height - 4, x1 + text_width + 4, y1], 
                         fill=color, outline=color)
            draw.text((x1 + 2, y1 - text_height - 2), text, fill='white', font=font)
        
        print(f"Found {len(results[0]['scores'])} detections")
    else:
        print("No detections found")
    
    # Save image
    if save_path is None:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        save_path = f"{base_name}_detections.jpg"
    
    image.save(save_path)
    print(f"Saved to: {save_path}")
    
    return image

if __name__ == "__main__":
    DATASET_PATH = "kaggle/working/out_archive"
    CLASS_NAMES = ["Explosives", "Anti-personnel mine", "Anti-vehicle mine"]
      
    model, image_processor, train_dataset, val_dataset = setup_fine_tuning(
        DATASET_PATH, 
        CLASS_NAMES
    )
    
    trainer = train_model(model, train_dataset, val_dataset)


    #check_model_training_state("rtdetr_finetuned/checkpoint-7000")
    #image_path=r"uadamage-demining-competition\train\images\0a1fc6b8346e0a357b79f960809b74b518f1ae601d813e8cb2492182aa3278bf.jpg"
    image_path = r"kaggle\working\preprocessed_object_present\val\images\0d664fed2765b49c93d9d5e9c46124342d552401d7e14dd6a3038b3314e55a7b_2_6.jpg"


#     visualize_detections(
#     model_path="./rtdetr_finetuned",
#     image_path=image_path, 
#     threshold=0.1,
#     save_path="my_results.jpg"
# )