import gradio as gr
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import timm


class MultiPlaneMRNet(nn.Module):
    """
    Processes 3 MRI planes (axial, coronal, sagittal) separately then fuses
    Architecture: EfficientNet-B0 backbone per plane + fusion layer
    """
    def __init__(self, backbone='tf_efficientnet_b0_ns', num_classes=3, 
                 dropout=0.5, fusion='concat'):
        super(MultiPlaneMRNet, self).__init__()
        
        self.fusion = fusion
        
        self.axial_backbone = timm.create_model(backbone, pretrained=True, num_classes=0)
        self.coronal_backbone = timm.create_model(backbone, pretrained=True, num_classes=0)
        self.sagittal_backbone = timm.create_model(backbone, pretrained=True, num_classes=0)
        
        self.feature_dim = self.axial_backbone.num_features
        
        if fusion == 'concat':
            fusion_dim = self.feature_dim * 3
        elif fusion == 'average':
            fusion_dim = self.feature_dim
        elif fusion == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(self.feature_dim * 3, 3),
                nn.Softmax(dim=1)
            )
            fusion_dim = self.feature_dim
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, axial, coronal, sagittal):
        feat_axial = self.axial_backbone(axial)
        feat_coronal = self.coronal_backbone(coronal)
        feat_sagittal = self.sagittal_backbone(sagittal)
        
        if self.fusion == 'concat':
            features = torch.cat([feat_axial, feat_coronal, feat_sagittal], dim=1)
        elif self.fusion == 'average':
            features = (feat_axial + feat_coronal + feat_sagittal) / 3.0
        elif self.fusion == 'attention':
            all_features = torch.cat([feat_axial, feat_coronal, feat_sagittal], dim=1)
            attention_weights = self.attention(all_features)
            stacked_features = torch.stack([feat_axial, feat_coronal, feat_sagittal], dim=1)
            features = torch.sum(stacked_features * attention_weights.unsqueeze(2), dim=1)
        
        output = self.classifier(features)
        return output

def get_transforms(input_size=(256, 320)):
    return transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def load_model(model_path, backbone='tf_efficientnet_b0_ns', num_classes=3, fusion='concat', device='cuda'):
    model = MultiPlaneMRNet(backbone=backbone, num_classes=num_classes, fusion=fusion)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def predict_mri(axial_image, coronal_image, sagittal_image, model_path='/kaggle/input/mriscan/pytorch/default/1/MRI_model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = get_transforms()
    
    try:
        axial = Image.open(axial_image).convert('RGB')
        coronal = Image.open(coronal_image).convert('RGB')
        sagittal = Image.open(sagittal_image).convert('RGB')
        
        axial = transform(axial).unsqueeze(0).to(device)
        coronal = transform(coronal).unsqueeze(0).to(device)
        sagittal = transform(sagittal).unsqueeze(0).to(device)
    except Exception as e:
        return f"❌ Error processing images: {str(e)}"
    
    try:
        model = load_model(model_path, device=device)
    except Exception as e:
        return f"❌ Error loading model: {str(e)}"
    
    optimal_thresholds = {
        'abnormal': 0.600000000000002,
        'acl': 0.20000000000000004,
        'meniscus': 0.25000000000000006
    }
    
    with torch.no_grad():
        outputs = model(axial, coronal, sagittal)
        probs = torch.sigmoid(outputs).cpu().numpy()[0]
    
    detected_conditions = []
    label_names = ['abnormal', 'acl', 'meniscus']
    
    for i, name in enumerate(label_names):
        prob = probs[i]
        threshold = optimal_thresholds[name]
        if prob >= threshold:
            detected_conditions.append(name)
    
    if not detected_conditions:
        output = "✅ NO ABNORMALITIES DETECTED\n\n"
        output += "The MRI scan analysis shows no signs of:\n"
        output += "  • General abnormalities\n"
        output += "  • ACL (Anterior Cruciate Ligament) injury\n"
        output += "  • Meniscus tear\n"
    else:
        output = "⚠️ ABNORMALITIES DETECTED\n\n"
        output += "The following conditions were identified:\n\n"
        
        for condition in detected_conditions:
            if condition == 'abnormal':
                output += "🔴 ABNORMAL\n"
            elif condition == 'acl':
                output += "🔴 ACL INJURY - Anterior Cruciate Ligament tear detected\n"
            elif condition == 'meniscus':
                output += "🔴 MENISCUS TEAR - Meniscus damage detected\n"
        
    
    return output

interface = gr.Interface(
    fn=predict_mri,
    inputs=[
        gr.Image(type="filepath", label="Axial MRI Image"),
        gr.Image(type="filepath", label="Coronal MRI Image"),
        gr.Image(type="filepath", label="Sagittal MRI Image"),
        gr.Textbox(label="Model Path", value="/kaggle/input/mriscan/pytorch/default/1/MRI_model.pth")
    ],
    outputs=gr.Textbox(label="Diagnosis Results", lines=10),
    title="🏥 Multi-Plane MRI Diagnosis Tool",
    description="Upload Axial, Coronal, and Sagittal MRI images to detect knee abnormalities, ACL injuries, and meniscus tears.",
    theme="default"
)

if __name__ == "__main__":
    interface.launch()




