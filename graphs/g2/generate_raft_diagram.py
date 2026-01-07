import os
from graphviz import Digraph

# Add Graphviz to PATH
graphviz_path = r'C:\Users\PC\Downloads\tools\Graphviz-12.2.1-win64\bin'
if graphviz_path not in os.environ['PATH']:
    os.environ['PATH'] += os.pathsep + graphviz_path

def create_raft_diagram():
    # --- CONFIGURATION AREA ---
    # Edit these labels to change the text (supports Vietnamese if font is compatible)
    LABELS = {
        'I1': 'Ảnh 1\n(t)',
        'I2': 'Ảnh 2\n(t+1)',
        'FNet': 'Mạng trích xuất đặc trưng',
        'CNet': 'Mạng ngữ cảnh',
        'FMap1': 'Bản đồ đặc trưng 1',
        'FMap2': 'Bản đồ đặc trưng 2',
        'Net_Init': 'Trạng thái ẩn (h0)',
        'Ctx_Feat': 'Đặc trưng ngữ cảnh',
        'CorrVol': 'Khối tương quan 4 chiều',
        'Pooling': 'Gộp (Pooling)\n(Pyramid)',
        'Lookup': 'Tra cứu tương quan\n(Bilinear)',
        'Flow_Prev': 'Ước tính độ di dời\n(k)',
        'MotionEnc': 'Mã hoá chuyển động\n(Các lớp Conv)',
        'ConvGRU': 'Khối ConvGRU\nCập nhật trạng thái ẩn',
        'FlowHead': 'Đầu dự đoán độ di dời\n(Các lớp Conv)',
        'DeltaFlow': 'Độ lệch di dời\n(d_flow)',
        'Flow_Next': 'Ước tính độ di dời\n(k+1)',
        'Sum': 'Cộng',
        'Upsample': 'Upsample\n',
        'HighResFlow': '(Độ phân giải gốc)\n',
        
        # Cluster Labels
        'c_inputs': 'Đầu vào',
        'c_encoders': 'Bộ mã hoá',
        'c_features': 'Đặc trưng trung gian',
        'c_corr': 'Trích xuất tương quan',
        'c_update': 'Khối cập nhật lặp lại (GRU)',
        'c_output': 'Đầu ra',
        
        # Edge Labels
        'e_split': 'Tách',
        'e_pyramid': 'Truy cập tháp',
        'e_index': 'Chỉ số (Pyramid)',
        'e_corr_feat': 'Đặc trưng tương quan',
        'e_flow_feat': 'Đặc trưng di dời',
        'e_motion_feat': 'Đặc trưng chuyển động',
        'e_context': 'Ngữ cảnh',
        'e_hidden_init': '(Khởi tạo)',
        'e_hidden_recur': '(t-1) -> (t)',
        'e_hidden_curr': '(t)',
        'e_next_iter': 'Lần lặp kế tiếp',
        'e_mask': 'Trọng số'
    }

    # Consistent Color Palette
    COLORS = {
        'input': '#E0E0E0',
        'encoder': '#ADD8E6',
        'feat': '#B0E0E6',
        'emb': '#FFFACD',
        'corr': '#87CEFA',
        'trans': '#FFD700', # Used for GRU block/Update block
        'trans_block': '#FFA500',
        'proxy': '#D8BFD8',
        'head': '#FFA500',
        'update': '#90EE90',
        'output': '#32CD32'
    }

    # Create Digraph - Vertical Layout (TB)
    dot = Digraph(name='RAFT_Architecture', comment='RAFT Optical Flow Model', format='png')
    # Updated for A4 fit and legibility
    dot.attr(rankdir='TB', splines='ortho', dpi='300', size='6.5,9.0')
    dot.attr('node', shape='rect', style='filled', fontname='Arial', fontsize='24', height='0.8')
    dot.attr('edge', fontname='Arial', fontsize='20')

    # 1. Inputs
    with dot.subgraph(name='cluster_inputs') as c:
        c.attr(label=LABELS['c_inputs'], color='white', rank='same', fontname='Arial')
        c.node('I1', LABELS['I1'], fillcolor=COLORS['input'])
        c.node('I2', LABELS['I2'], fillcolor=COLORS['input'])

    # 2. Encoders
    with dot.subgraph(name='cluster_encoders') as c:
        c.attr(label=LABELS['c_encoders'], color='lightgrey', style='rounded', fontname='Arial')
        
        # Feature Encoder (Shared)
        c.node('FNet', LABELS['FNet'], fillcolor=COLORS['encoder'])
        
        # Context Encoder
        c.node('CNet', LABELS['CNet'], fillcolor=COLORS['encoder'])

    # 3. Intermediate Features
    with dot.subgraph(name='cluster_features') as c:
        c.attr(rank='same')
        c.node('FMap1', LABELS['FMap1'], shape='parallelogram', fillcolor='#E6E6FA')
        c.node('FMap2', LABELS['FMap2'], shape='parallelogram', fillcolor='#E6E6FA')
        c.node('Net_Init', LABELS['Net_Init'], fillcolor=COLORS['emb'])
        c.node('Ctx_Feat', LABELS['Ctx_Feat'], fillcolor=COLORS['emb'])

    # 4. Correlation Block
    with dot.subgraph(name='cluster_corr') as c:
        c.attr(label=LABELS['c_corr'], color='lightblue', style='rounded', fontname='Arial')
        c.node('CorrVol', LABELS['CorrVol'], fillcolor=COLORS['corr'])
        c.node('Pooling', LABELS['Pooling'], fillcolor=COLORS['corr'])
        c.node('Lookup', LABELS['Lookup'], fillcolor='#00BFFF')

    # 5. Iterative Update Block
    with dot.subgraph(name='cluster_update') as c:
        c.attr(label=LABELS['c_update'], color='gold', style='dashed', fontname='Arial')
        
        # Inputs to update block
        c.node('Flow_Prev', LABELS['Flow_Prev'], fillcolor='#98FB98')
        
        # Motion Encoder
        c.node('MotionEnc', LABELS['MotionEnc'], fillcolor='#FFA500')
        
        # ConvGRU
        c.node('ConvGRU', LABELS['ConvGRU'], fillcolor='#FF8C00')
        
        # Flow Head
        c.node('FlowHead', LABELS['FlowHead'], fillcolor='#FF4500')
        
        c.node('DeltaFlow', LABELS['DeltaFlow'], fillcolor='#FF6347')

    # 6. Flow Update & Output
    with dot.subgraph(name='cluster_output') as c:
        c.attr(label=LABELS['c_output'], color='white', fontname='Arial')
        c.node('Sum', LABELS['Sum'], shape='circle', width='0.5', fixedsize='true')
        c.node('Flow_Next', LABELS['Flow_Next'], fillcolor='#90EE90')
        c.node('Upsample', LABELS['Upsample'], fillcolor='#32CD32')
        c.node('HighResFlow', LABELS['HighResFlow'], fillcolor='#32CD32', shape='folder')

    # --- Edges ---

    # Encoder paths
    dot.edge('I1', 'FNet')
    dot.edge('I2', 'FNet')
    dot.edge('FNet', 'FMap1')
    dot.edge('FNet', 'FMap2')

    dot.edge('I1', 'CNet')
    dot.edge('CNet', 'Net_Init', label=LABELS['e_split'])
    dot.edge('CNet', 'Ctx_Feat', label=LABELS['e_split'])

    # Correlation construction
    dot.edge('FMap1', 'CorrVol')
    dot.edge('FMap2', 'CorrVol')
    dot.edge('CorrVol', 'Pooling')

    # Iteration Loop Logic
    # Lookup needs Coordinates derived from Flow + Grid
    dot.edge('Pooling', 'Lookup', label=LABELS['e_pyramid'])
    dot.edge('Flow_Prev', 'Lookup', label=LABELS['e_index'])

    # Motion Encoder
    dot.edge('Lookup', 'MotionEnc', label=LABELS['e_corr_feat'])
    dot.edge('Flow_Prev', 'MotionEnc', label=LABELS['e_flow_feat'])

    # ConvGRU inputs
    dot.edge('MotionEnc', 'ConvGRU', label=LABELS['e_motion_feat'])
    dot.edge('Ctx_Feat', 'ConvGRU', label=LABELS['e_context'])
    dot.edge('Net_Init', 'ConvGRU', label=LABELS['e_hidden_init'])

    # Recursive Hidden State connection (conceptual)
    dot.edge('ConvGRU', 'ConvGRU', label=LABELS['e_hidden_recur'], style='dotted', constraint='false')

    # Output head
    dot.edge('ConvGRU', 'FlowHead', label=LABELS['e_hidden_curr'])
    dot.edge('FlowHead', 'DeltaFlow')

    # Flow Update
    dot.edge('Flow_Prev', 'Sum')
    dot.edge('DeltaFlow', 'Sum')
    dot.edge('Sum', 'Flow_Next')
    
    # Loop back (make it go up)
    dot.edge('Flow_Next', 'Flow_Prev', label=LABELS['e_next_iter'], style='dashed', constraint='false')

    # Upsampling
    dot.edge('FlowHead', 'Upsample', label=LABELS['e_mask'])
    dot.edge('Flow_Next', 'Upsample')
    dot.edge('Upsample', 'HighResFlow')

    # Save and Render
    dot.save('raft_flow_diagram.dot')
    try:
        output_path = dot.render('raft_flow_diagram', view=False)
        print(f"Graph generated at: {output_path}")
    except Exception as e:
        print(f"Error rendering graph: {e}")
        print("DOT source saved to raft_flow_diagram.dot")

if __name__ == '__main__':
    create_raft_diagram()
