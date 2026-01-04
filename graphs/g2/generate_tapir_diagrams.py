import os
from graphviz import Digraph

# Add Graphviz to PATH
graphviz_path = r'C:\Users\PC\Downloads\tools\Graphviz-12.2.1-win64\bin'
if graphviz_path not in os.environ['PATH']:
    os.environ['PATH'] += os.pathsep + graphviz_path

def create_tapir_initialization_diagram():
    """Generates the TAPIR Initialization (Cost Volume) diagram."""
    
    # --- CONFIGURATION AREA ---
    # Edit these labels to change the text (supports Vietnamese if font is compatible)
    LABELS = {
        'Video': 'Khung hình Video',
        'QueryPoints': 'Điểm truy vấn\n(t, y, x)',
        'ResNet': 'ResNet Backbone',
        'FeatureGrid': 'Lưới đặc trưng\n(Nhiều mức phân giải)',
        'QueryFeat': 'Trích xuất\nĐặc trưng truy vấn',
        'CostVol': 'Tính khối chi phí\n',
        'ConvProc': 'Lớp tích chập\n',
        'Heatmap': 'Bản đồ nhiệt / Softmax',
        'SoftArgmax': 'Soft Argmax\n',
        'InitTraj': 'Quỹ đạo ban đầu',
        'InitOcc': 'Điểm bị che ban đầu',
        'InitUnc': 'Độ không chắc chắn ban đầu',
        
        # Cluster Labels
        'c_inputs': 'Đầu vào',
        'c_backbone': 'Trích xuất đặc trưng',
        'c_costvol': 'Khởi tạo dày đặc',
        'c_outputs': 'Đầu ra',
        
        # Edges
        'e_extract': 'Trích xuất đặc trưng',
        'e_sample': 'Lấy mẫu tại vị trí truy vấn',
        'e_dot': 'Độ tương đồng',
        'e_process': 'Xử lý',
        'e_decode': 'Giải mã'
    }

    # Consistent Color Palette
    COLORS = {
        'input': '#E0E0E0',
        'encoder': '#ADD8E6',
        'feat': '#B0E0E6',
        'emb': '#FFFACD',
        'corr': '#87CEFA',
        'trans': '#FFD700',
        'trans_block': '#FFA500',
        'proxy': '#D8BFD8',
        'head': '#FFA500',
        'update': '#90EE90',
        'output': '#32CD32',
        'conv_proc': '#00BFFF',
        'heatmap': '#1E90FF',
        'softmax': '#4169E1'
    }

    dot = Digraph(name='TAPIR_Initialization', comment='TAPIR Initialization Process', format='png')
    # Updated for A4 fit and legibility
    dot.attr(rankdir='TB', splines='ortho', dpi='300', size='6.5,9.0')
    dot.attr('node', shape='rect', style='filled', fontname='Arial', fontsize='24', height='0.8')
    dot.attr('edge', fontname='Arial', fontsize='20')

    # Inputs
    with dot.subgraph(name='cluster_inputs') as c:
        c.attr(label=LABELS['c_inputs'], color='white', rank='same', fontname='Arial')
        c.node('Video', LABELS['Video'], fillcolor=COLORS['input'])
        c.node('QueryPoints', LABELS['QueryPoints'], fillcolor=COLORS['input'])

    # Feature Extraction
    with dot.subgraph(name='cluster_backbone') as c:
        c.attr(label=LABELS['c_backbone'], color='lightgrey', style='rounded', fontname='Arial')
        c.node('ResNet', LABELS['ResNet'], fillcolor=COLORS['encoder'])
        c.node('FeatureGrid', LABELS['FeatureGrid'], fillcolor='#E6E6FA')

    # Query Feature Extraction
    dot.node('QueryFeat', LABELS['QueryFeat'], fillcolor=COLORS['emb'])

    # Cost Volume Block
    with dot.subgraph(name='cluster_costvol') as c:
        c.attr(label=LABELS['c_costvol'], color='lightblue', style='rounded', fontname='Arial')
        c.node('CostVol', LABELS['CostVol'], fillcolor=COLORS['corr'])
        c.node('ConvProc', LABELS['ConvProc'], fillcolor=COLORS['conv_proc'])
        c.node('Heatmap', LABELS['Heatmap'], fillcolor=COLORS['heatmap'], fontcolor='white')
        c.node('SoftArgmax', LABELS['SoftArgmax'], fillcolor=COLORS['softmax'], fontcolor='white')

    # Outputs
    with dot.subgraph(name='cluster_outputs') as c:
        c.attr(label=LABELS['c_outputs'], color='white', rank='same', fontname='Arial')
        c.node('InitTraj', LABELS['InitTraj'], fillcolor=COLORS['output'])
        c.node('InitOcc', LABELS['InitOcc'], fillcolor=COLORS['output'])
        c.node('InitUnc', LABELS['InitUnc'], fillcolor=COLORS['output'])

    # Edges
    dot.edge('Video', 'ResNet', label=LABELS['e_extract'])
    dot.edge('ResNet', 'FeatureGrid')
    
    # Query Feat needs grid and points
    dot.edge('FeatureGrid', 'QueryFeat')
    dot.edge('QueryPoints', 'QueryFeat', label=LABELS['e_sample'])

    # Cost Volume needs grid and query feats
    dot.edge('FeatureGrid', 'CostVol')
    dot.edge('QueryFeat', 'CostVol', label=LABELS['e_dot'])

    dot.edge('CostVol', 'ConvProc')
    dot.edge('ConvProc', 'Heatmap')
    dot.edge('Heatmap', 'SoftArgmax')

    dot.edge('SoftArgmax', 'InitTraj', label=LABELS['e_decode'])
    
    # Occlusion/Uncertainty branches from ConvProc/Heatmap area usually, 
    # but strictly in code it comes from the same Conv block heads
    dot.edge('ConvProc', 'InitOcc')
    dot.edge('ConvProc', 'InitUnc')

    # Save
    dot.save('tapir_initialization.dot')
    try:
        output_path = dot.render('tapir_initialization', view=False)
        print(f"Initialization graph generated at: {output_path}")
    except Exception as e:
        print(f"Error rendering initialization graph: {e}")

def create_tapir_refinement_diagram():
    """Generates the TAPIR Iterative Refinement (PIPs) diagram."""

    # --- CONFIGURATION AREA ---
    LABELS = {
        'PrevState': 'Trạng thái trước\n(Vị trí, Che, KCC)',
        'VideoPyr': 'Đặc trưng Video\n(Pyramid)',
        'QueryCtx': 'Đặc trưng truy vấn\n',
        'Sampling': 'Bilinear Sampling\n',
        'SampledFeat': 'Đặc trưng được lấy mẫu',
        'Concat': 'Nối chập\n(Vị trí, Che, Đặc trưng)',
        'Mixer': 'Bộ trộn PIPS\n(Các khối MLP-Mixer)',
        'Residuals': 'Độ lệch dự đoán\n(dPos, dOcc, dUnc)',
        'Update': 'Cập nhật trạng thái\n',
        'NewState': 'Trạng thái mới\n(Vị trí, Che, KCC)',
        'NextIter': 'Lần lặp kế tiếp',
        
        # Cluster Labels
        'c_state': 'Trạng thái hiện tại',
        'c_context': 'Dữ liệu ngữ cảnh',
        'c_sampling': 'Trích xuất đặc trưng',
        'c_mixer': 'Nhân tinh chỉnh',
        
        # Edges
        'e_guide': 'Hướng dẫn lấy mẫu',
        'e_extract': 'Trích xuất',
        'e_input': 'Đầu vào',
        'e_refine': 'Tinh chỉnh',
        'e_loop': 'Lặp'
    }
    
    # Consistent Color Palette
    COLORS = {
        'input': '#E0E0E0',
        'encoder': '#ADD8E6',
        'feat': '#B0E0E6',
        'emb': '#FFFACD',
        'corr': '#87CEFA',
        'trans': '#FFD700',
        'trans_block': '#FFA500',
        'mixer': '#FF8C00',
        'head': '#FFA500',
        'update': '#90EE90',
        'output': '#32CD32'
    }

    dot = Digraph(name='TAPIR_Refinement', comment='TAPIR Refinement Process', format='png')
    # Updated for A4 fit and legibility
    dot.attr(rankdir='TB', splines='ortho', dpi='300', size='6.5,9.0')
    dot.attr('node', shape='rect', style='filled', fontname='Arial', fontsize='24', height='0.8')
    dot.attr('edge', fontname='Arial', fontsize='20')

    # Inputs / State
    with dot.subgraph(name='cluster_state') as c:
        c.attr(label=LABELS['c_state'], color='gold', style='dashed', fontname='Arial')
        c.node('PrevState', LABELS['PrevState'], fillcolor=COLORS['emb'])

    # Context Data
    with dot.subgraph(name='cluster_context') as c:
        c.attr(label=LABELS['c_context'], color='lightgrey', style='rounded', fontname='Arial')
        c.node('VideoPyr', LABELS['VideoPyr'], fillcolor='#E6E6FA')
        c.node('QueryCtx', LABELS['QueryCtx'], fillcolor='#E6E6FA')

    # Sampling Mechanism
    with dot.subgraph(name='cluster_sampling') as c:
        c.attr(label=LABELS['c_sampling'], color='lightblue', style='rounded', fontname='Arial')
        c.node('Sampling', LABELS['Sampling'], fillcolor=COLORS['corr'], shape='trapezium')
        c.node('SampledFeat', LABELS['SampledFeat'], fillcolor=COLORS['encoder'])

    # Mixing Block
    with dot.subgraph(name='cluster_mixer') as c:
        c.attr(label=LABELS['c_mixer'], color='orange', style='rounded', fontname='Arial')
        c.node('Concat', LABELS['Concat'], shape='circle', width='1.0', fillcolor=COLORS['trans_block'])
        c.node('Mixer', LABELS['Mixer'], fillcolor=COLORS['mixer'], shape='box3d')
        c.node('Residuals', LABELS['Residuals'], fillcolor='#FF7F50')

    # Update
    dot.node('Update', LABELS['Update'], shape='oval', fillcolor=COLORS['update'])
    dot.node('NewState', LABELS['NewState'], fillcolor=COLORS['output'])

    # Edges
    # Sampling needs Video Pyramid and Current Position (from PrevState)
    dot.edge('VideoPyr', 'Sampling')
    dot.edge('PrevState', 'Sampling', label=LABELS['e_guide'])
    dot.edge('Sampling', 'SampledFeat', label=LABELS['e_extract'])

    # Mixer Input Construction
    dot.edge('PrevState', 'Concat', label=LABELS['e_input'])
    dot.edge('QueryCtx', 'Concat')
    dot.edge('SampledFeat', 'Concat')

    # Mixer Processing
    dot.edge('Concat', 'Mixer')
    dot.edge('Mixer', 'Residuals', label=LABELS['e_refine'])

    # Update State
    dot.edge('PrevState', 'Update') # Base for addition
    dot.edge('Residuals', 'Update') # Residual
    dot.edge('Update', 'NewState')

    # Loop back (Conceptual)
    dot.edge('NewState', 'PrevState', label=LABELS['e_loop'], style='dashed', constraint='false', color='grey')

    # Save
    dot.save('tapir_refinement.dot')
    try:
        output_path = dot.render('tapir_refinement', view=False)
        print(f"Refinement graph generated at: {output_path}")
    except Exception as e:
        print(f"Error rendering refinement graph: {e}")

if __name__ == '__main__':
    create_tapir_initialization_diagram()
    create_tapir_refinement_diagram()
