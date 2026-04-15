import json
import uuid

def gen_id():
    return str(uuid.uuid4())

elements = []

# ================================
# Layout & Theme Settings
# ================================
W = 260
H = 80
X_GAP = 120
Y_GAP = 80

C1 = 100
C2 = C1 + W + X_GAP
C3 = C2 + W + X_GAP
C4 = C3 + W + X_GAP
C5 = C4 + W + X_GAP
C6 = C5 + W + X_GAP

R2 = 250
R3 = R2 + H + Y_GAP
R4 = R3 + H + Y_GAP
R5 = R4 + H + Y_GAP
R6 = R5 + H + Y_GAP

C_INPUT = "#f8f9fa"
C_BACKBONE = "#d0ebff" 
C_NECK = "#d8f5a2" 
C_HEAD = "#ffd8a8" 
C_OUTPUT = "#ffc9c9" 
C_SPPF = "#eebefa" 

# ================================
# Helper Functions
# ================================
def add_label(x, y, text, font_size=28, color="#000000"):
    elements.append({
        "id": gen_id(),
        "type": "text",
        "x": x,
        "y": y,
        "width": len(text) * font_size * 0.6,
        "height": font_size * 1.2,
        "angle": 0,
        "strokeColor": color,
        "backgroundColor": "transparent",
        "fillStyle": "hachure",
        "strokeWidth": 1,
        "strokeStyle": "solid",
        "roughness": 0,
        "opacity": 100,
        "groupIds": [],
        "roundness": None,
        "boundElements": [],
        "updated": 1,
        "link": None,
        "locked": False,
        "text": text,
        "fontSize": font_size,
        "fontFamily": 5,
        "textAlign": "left",
        "verticalAlign": "top",
        "baseline": font_size
    })

def add_region(x, y, w, h, text, color):
    r_id = gen_id()
    elements.append({
        "id": r_id,
        "type": "rectangle",
        "x": x,
        "y": y,
        "width": w,
        "height": h,
        "angle": 0,
        "strokeColor": "transparent",
        "backgroundColor": color,
        "fillStyle": "solid",
        "strokeWidth": 1,
        "strokeStyle": "solid",
        "roughness": 0,
        "opacity": 40,
        "groupIds": [],
        "roundness": {"type": 3},
        "boundElements": [],
        "updated": 1,
        "link": None,
        "locked": True
    })
    add_label(x + 20, y + 20, text, 24, "#495057")

def add_rect(x, y, w, h, text, bg_color="#ffffff", text_color="#000000", font_size=18, border_color="#1e1e1e"):
    rect_id = gen_id()
    text_id = gen_id()
    
    lines = text.split('\n')
    text_h = len(lines) * font_size * 1.2
    
    # Text element
    elements.append({
        "id": text_id,
        "type": "text",
        "x": x + 10, 
        "y": y + (h - text_h) / 2 + 5,
        "width": w - 20,
        "height": text_h,
        "angle": 0,
        "strokeColor": text_color,
        "backgroundColor": "transparent",
        "fillStyle": "hachure",
        "strokeWidth": 1,
        "strokeStyle": "solid",
        "roughness": 0,
        "opacity": 100,
        "groupIds": [],
        "roundness": None,
        "boundElements": [{"id": rect_id, "type": "rectangle"}],
        "updated": 1,
        "link": None,
        "locked": False,
        "text": text,
        "fontSize": font_size,
        "fontFamily": 5, 
        "textAlign": "center",
        "verticalAlign": "middle",
        "baseline": font_size
    })
    
    # Rectangle element
    elements.append({
        "id": rect_id,
        "type": "rectangle",
        "x": x,
        "y": y,
        "width": w,
        "height": h,
        "angle": 0,
        "strokeColor": border_color,
        "backgroundColor": bg_color,
        "fillStyle": "solid",
        "strokeWidth": 2,
        "strokeStyle": "solid",
        "roughness": 0,
        "opacity": 100,
        "groupIds": [],
        "roundness": {"type": 3},
        "boundElements": [{"id": text_id, "type": "text"}],
        "updated": 1,
        "link": None,
        "locked": False
    })
    
    return (rect_id, x, y, w, h)

def add_arrow_between(r1, r2, text=None, dash=False):
    id1, x1, y1, w1, h1 = r1
    id2, x2, y2, w2, h2 = r2
    
    cx1 = x1 + w1 / 2
    cy1 = y1 + h1 / 2
    cx2 = x2 + w2 / 2
    cy2 = y2 + h2 / 2
    
    arrow_id = gen_id()
    arrow = {
        "id": arrow_id,
        "type": "arrow",
        "x": cx1,
        "y": cy1,
        "width": abs(cx2 - cx1),
        "height": abs(cy2 - cy1),
        "angle": 0,
        "strokeColor": "#1e1e1e",
        "backgroundColor": "transparent",
        "fillStyle": "hachure",
        "strokeWidth": 2,
        "strokeStyle": "dashed" if dash else "solid",
        "roughness": 0,
        "opacity": 100,
        "groupIds": [],
        "roundness": {"type": 2},
        "boundElements": [],
        "updated": 1,
        "link": None,
        "locked": False,
        "points": [
            [0, 0],
            [cx2 - cx1, cy2 - cy1]
        ],
        "startBinding": {"elementId": id1, "focus": 0, "gap": 1},
        "endBinding": {"elementId": id2, "focus": 0, "gap": 1},
        "lastCommittedPoint": None,
        "startArrowhead": None,
        "endArrowhead": "arrow"
    }
    
    if text:
        text_id = gen_id()
        elements.append({
            "id": text_id,
            "type": "text",
            "x": (cx1 + cx2) / 2 + 5,
            "y": (cy1 + cy2) / 2 - 20,
            "width": len(text) * 12,
            "height": 20,
            "angle": 0,
            "strokeColor": "#e03131",
            "backgroundColor": "#ffffff",
            "fillStyle": "solid",
            "strokeWidth": 1,
            "strokeStyle": "solid",
            "roughness": 0,
            "opacity": 100,
            "groupIds": [],
            "roundness": None,
            "boundElements": [],
            "updated": 1,
            "link": None,
            "locked": False,
            "text": text,
            "fontSize": 14,
            "fontFamily": 5,
            "textAlign": "center",
            "verticalAlign": "middle",
            "baseline": 14
        })
        arrow["boundElements"].append({"id": text_id, "type": "text"})
        
    elements.append(arrow)

# ================================
# Build the Diagram
# ================================

# Title
add_label(C2, 50, "YOLO26 Architecture Diagram (Enhanced)", 36, "#1864ab")

# Regions (Added first so they render in background)
PADDING = 40
add_region(C1 - PADDING, R2 - 80, W + PADDING*2, (R6 - R2) + H + 120, "1. Input", "#f1f3f5")
add_region(C2 - PADDING, R2 - 80, W + PADDING*2, (R6 - R2) + H + 120, "2. Backbone", "#a5d8ff")
add_region(C3 - PADDING, R2 - 80, W * 2 + X_GAP + PADDING*2, (R6 - R2) + H + 120, "3. Neck (FPN + PAN)", "#b2f2bb")
add_region(C5 - PADDING, R2 - 80, W + PADDING*2, (R6 - R2) + H + 120, "4. Head", "#ffec99")
add_region(C6 - PADDING, R2 - 80, W + PADDING*2, (R6 - R2) + H + 120, "5. Output", "#ffc9c9")

# 1. Input Nodes
node_in = add_rect(C1, R3, W, H, "Input Image\n(640 x 640 x 3)", C_INPUT)
node_aug = add_rect(C1, R4, W, H, "Data Augmentation\n(Mosaic, MixUp)", C_INPUT)

# 2. Backbone Nodes
node_stem = add_rect(C2, R2, W, H, "Stem\n(Conv Layers)", C_BACKBONE)
node_c2f_1 = add_rect(C2, R3, W, H, "C2f Block (P3)\n(Stride 8, Small)", C_BACKBONE)
node_c2f_2 = add_rect(C2, R4, W, H, "C2f Block (P4)\n(Stride 16, Medium)", C_BACKBONE)
node_c2f_3 = add_rect(C2, R5, W, H, "C2f Block (P5)\n(Stride 32, Large)", C_BACKBONE)
node_sppf = add_rect(C2, R6, W, H, "SPPF Block\n(Spatial Pyramid Pooling)", C_SPPF)

# 3. Neck Nodes
# FPN (Top-Down)
node_up1 = add_rect(C3, R4, W, H, "FPN (P4)\nUpsample + Concat + C2f", C_NECK)
node_up2 = add_rect(C3, R3, W, H, "FPN (P3)\nUpsample + Concat + C2f", C_NECK)

# PAN (Bottom-Up)
node_down1 = add_rect(C4, R4, W, H, "PAN (P4)\nConv + Concat + C2f", C_NECK)
node_down2 = add_rect(C4, R6, W, H, "PAN (P5)\nConv + Concat + C2f", C_NECK)

# 4. Head Nodes
node_head_s = add_rect(C5, R3, W, H, "Detect Head (Small)\nGrid: 80x80", C_HEAD)
node_head_m = add_rect(C5, R4, W, H, "Detect Head (Medium)\nGrid: 40x40", C_HEAD)
node_head_l = add_rect(C5, R6, W, H, "Detect Head (Large)\nGrid: 20x20", C_HEAD)

# 5. Output Nodes
node_loss = add_rect(C6, R4, W, 140, "Output Processing\n\n- Decoupled Classification\n- BBox Regression\n- DFL + CIoU Loss\n- NMS (Non-Max Suppress)", C_OUTPUT)


# ================================
# Connect the Nodes (Arrows)
# ================================

# Input -> Backbone
add_arrow_between(node_in, node_aug)
add_arrow_between(node_aug, node_stem)

# Backbone Path
add_arrow_between(node_stem, node_c2f_1)
add_arrow_between(node_c2f_1, node_c2f_2)
add_arrow_between(node_c2f_2, node_c2f_3)
add_arrow_between(node_c2f_3, node_sppf)

# Neck Path: Top-Down (FPN)
add_arrow_between(node_sppf, node_up1, "Up")
add_arrow_between(node_c2f_2, node_up1, "Lateral", dash=True)

add_arrow_between(node_up1, node_up2, "Up")
add_arrow_between(node_c2f_1, node_up2, "Lateral", dash=True)

# Neck Path: Bottom-Up (PAN)
add_arrow_between(node_up2, node_down1, "Down")
add_arrow_between(node_up1, node_down1, "Lateral", dash=True)

add_arrow_between(node_down1, node_down2, "Down")
add_arrow_between(node_sppf, node_down2, "Lateral", dash=True)

# Head Connections
add_arrow_between(node_up2, node_head_s)
add_arrow_between(node_down1, node_head_m)
add_arrow_between(node_down2, node_head_l)

# Output Connections
add_arrow_between(node_head_s, node_loss)
add_arrow_between(node_head_m, node_loss)
add_arrow_between(node_head_l, node_loss)

# ================================
# Save Excalidraw JSON
# ================================
excalidraw_data = {
    "type": "excalidraw",
    "version": 2,
    "source": "https://excalidraw.com",
    "elements": elements,
    "appState": {
        "viewBackgroundColor": "#ffffff",
        "gridSize": 20
    },
    "files": {}
}

with open("yolo26-architecture.excalidraw", "w", encoding="utf-8") as f:
    json.dump(excalidraw_data, f, indent=2)

print("Diagram generated successfully with dynamic binding: yolo26-architecture.excalidraw")
