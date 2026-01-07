import os
from graphviz import Digraph

# Add Graphviz to PATH (adjust as needed for the user's environment)
graphviz_path = r'C:\Users\PC\Downloads\tools\Graphviz-12.2.1-win64\bin'
if graphviz_path not in os.environ['PATH']:
    os.environ['PATH'] += os.pathsep + graphviz_path

def create_cotracker3_diagram():
    # --- CONFIGURATION AREA ---
    LABELS = {
        # Inputs
        'Video': 'Video đầu vào\n(B x T x 3 x H x W)',
        'Queries': 'Điểm truy vấn\n(B x N x 3)',
        
        # Encoder & Pyramid
        'CNN': 'Mạng CNN\n(Dùng chung)',
        'Pyramid': 'Pyramid đặc trưng\nđa tầng',
        
        # Track Features
        'Sampler': 'Lấy mẫu bilinear\n(Đặc trưng điểm)',
        
        # 4D Correlation
        'CorrVol': 'Khối tương quan 4 chiều',
        'CorrMLP': 'MLP Tương quan',
        'CorrEmb': 'Mã hoá\nTương quan',
        
        # Transformer Input
        'PosEnc': 'Mã hoá\nVị trí',
        'Concat': 'Nối chập\nToken',
        'InputTrans': 'Biến đổi đầu vào',
        
        # UpdateFormer
        'UpdateFormer': 'EfficientUpdateFormer',
        'TimeAttn': 'Attention thời gian',
        
        # Proxy / Spatial Attention
        'ProxyAttn': 'Attention ảo',
        'P2V': 'Điểm -> Token Ảo',
        'V2V': 'Self-Attn Token Ảo',
        'V2P': 'Token Ảo -> Điểm',
        
        # Heads
        'FlowHead': 'Đầu dự đoán độ di dời',
        'VisHead': 'Đầu dự đoán Hiển thị/Tin cậy',
        
        # Updates
        'UpdateState': 'Cập nhật trạng thái\n(Toạ độ, Hiển thị, Tin cậy)',
        
        # Cluster Labels
        'c_inputs': 'Đầu vào',
        'c_encoder': 'Trích xuất đặc trưng',
        'c_corr': 'Lấy mẫu & Tương quan',
        'c_prep': 'Chuẩn bị Token',
        'c_transformer': 'UpdateFormer',
        'c_proxy': 'Attention ảo',
        'c_heads': 'Các đầu dự đoán',
        
        # Edges
        'e_feat': 'Đặc trưng',
        'e_sample': 'Lấy mẫu',
        'e_corr': 'Tính\nTương quan',
        'e_tokens': 'Token',
        'e_delta': 'Độ lệch',
        'e_iter': 'Cập nhật\nLặp lại',
        'e_proxy_flow': 'N Điểm <-> M Token ảo',
        'e_init_coords': '(Toạ độ khởi tạo)',
        'e_track_feats': 'Đặc trưng điểm',
        'e_local_grid': 'Lưới cục bộ',
        'e_point_tokens': 'Token điểm',
        'e_recur_state': 'Cập nhật'
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
        'proxy_sub': '#DDA0DD',
        'head': '#FFA500',
        'update': '#90EE90',
        'output': '#32CD32'
    }

    # Create Digraph - Vertical Layout (TB)
    dot = Digraph(name='CoTracker3_Architecture', comment='CoTracker3 Architecture', format='png')
    # Updated for A4 fit and legibility
    dot.attr(rankdir='TB', splines='ortho', nodesep='0.6', ranksep='0.6', dpi='300', size='6.5,9.0')
    dot.attr('node', shape='rect', style='filled', fontname='Arial', fontsize='24', height='0.8')
    dot.attr('edge', fontname='Arial', fontsize='20')

    # 1. Inputs
    with dot.subgraph(name='cluster_0_inputs') as c:
        c.attr(label=LABELS['c_inputs'], penwidth='0', fontname='Arial')
        c.node('Video', LABELS['Video'], fillcolor=COLORS['input'], shape='box3d')
        c.node('Queries', LABELS['Queries'], fillcolor=COLORS['input'], shape='component')

    # 2. Encoder Pyramid (Simplified)
    with dot.subgraph(name='cluster_1_encoder') as c:
        c.attr(label=LABELS['c_encoder'], color='lightgrey', style='rounded', fontname='Arial')
        c.node('CNN', LABELS['CNN'], fillcolor=COLORS['encoder'])
        c.node('Pyramid', LABELS['Pyramid'], fillcolor=COLORS['feat'], shape='trapezium')

    # 3. Track Features & Correlation
    with dot.subgraph(name='cluster_2_corr') as c:
        c.attr(label=LABELS['c_corr'], color='#E6E6FA', style='dashed', fontname='Arial')
        
        c.node('Sampler', LABELS['Sampler'], fillcolor='#98FB98')
        
        # 4D Correlation Visualization
        c.node('CorrVol', LABELS['CorrVol'], fillcolor=COLORS['corr'], shape='box3d')
        c.node('CorrMLP', LABELS['CorrMLP'], fillcolor=COLORS['trans_block'])
        c.node('CorrEmb', LABELS['CorrEmb'], fillcolor=COLORS['trans_block'])

    # 4. Transformer Input
    with dot.subgraph(name='cluster_3_prep') as c:
        c.attr(label=LABELS['c_prep'], color='white', penwidth='0', fontname='Arial')
        c.node('PosEnc', LABELS['PosEnc'], fillcolor=COLORS['emb'])
        c.node('Concat', LABELS['Concat'], fillcolor='#F0E68C', shape='trapezium')

    # 5. EfficientUpdateFormer
    with dot.subgraph(name='cluster_4_transformer') as c:
        c.attr(label=LABELS['UpdateFormer'], color=COLORS['trans'], style='rounded', penwidth='2', fontname='Arial')
        
        # 5a. Temporal Attention (Separate)
        c.node('TimeAttn', LABELS['TimeAttn'], fillcolor=COLORS['trans_block'])
        
        # 5b. Proxy Attention (Leaner)
        with c.subgraph(name='cluster_proxy') as p:
            p.attr(label=LABELS['ProxyAttn'], color='#9370DB', style='solid', bgcolor='#F5F5F5', fontname='Arial')
            
            # Simplified Flow: Point -> Proxy -> Point
            # Using rank=same to make it look compact if desired, or just vertical flow
            p.node('P2V', LABELS['P2V'], fillcolor=COLORS['proxy_sub'])
            p.node('V2V', LABELS['V2V'], fillcolor='#BA55D3')
            p.node('V2P', LABELS['V2P'], fillcolor=COLORS['proxy_sub'])
            
            p.edge('P2V', 'V2V')
            p.edge('V2V', 'V2P')

    # 6. Heads
    with dot.subgraph(name='cluster_5_heads') as c:
        c.attr(label=LABELS['c_heads'], color='lightgrey', fontname='Arial')
        c.node('FlowHead', LABELS['FlowHead'], fillcolor=COLORS['head'])
        c.node('VisHead', LABELS['VisHead'], fillcolor=COLORS['head'])
        c.node('UpdateState', LABELS['UpdateState'], fillcolor=COLORS['output'], shape='folder')

    # --- EDGES ---

    # Input -> CNN
    dot.edge('Video', 'CNN')
    dot.edge('CNN', 'Pyramid')
    
    # Pyramid -> Sampler & Corr
    dot.edge('Pyramid:s', 'Sampler')
    dot.edge('Pyramid', 'CorrVol', style='dashed')
    
    # Queries -> Initial State
    dot.edge('Queries', 'Sampler', label=LABELS['e_init_coords'])
    dot.edge('Queries', 'CorrVol')
    
    # Sampler -> CorrVol (Support Features used in Correlation)
    dot.edge('Sampler', 'CorrVol', label=LABELS['e_track_feats'])
    
    # Correlation Flow
    dot.edge('CorrVol', 'CorrMLP', label=LABELS['e_local_grid'])
    dot.edge('CorrMLP', 'CorrEmb')
    dot.edge('CorrEmb', 'Concat')
    
    # PosEnc
    dot.edge('Queries', 'PosEnc', style='dotted')
    dot.edge('PosEnc', 'Concat')
    
    # Transformer Flow
    dot.edge('Concat', 'TimeAttn')
    dot.edge('TimeAttn', 'P2V', label=LABELS['e_point_tokens'])
    
    # Heads Flow
    dot.edge('V2P', 'FlowHead')
    dot.edge('V2P', 'VisHead')
    
    # Update State
    dot.edge('FlowHead', 'UpdateState', label=LABELS['e_delta'])
    dot.edge('VisHead', 'UpdateState')
    
    # Iterative Recurrence (The Loop)
    dot.edge('UpdateState', 'CorrVol', color='blue', style='dashed', constraint='false')
    dot.edge('UpdateState', 'PosEnc', color='blue', style='dashed', constraint='false')
    dot.edge('UpdateState', 'Concat', label=LABELS['e_recur_state'], color='blue', style='dashed', constraint='false')

    # Save and Render
    output_filename = 'cotracker3_architecture'
    dot.save(f'{output_filename}.dot')
    try:
        output_path = dot.render(output_filename, view=False)
        print(f"Graph generated at: {output_path}")
    except Exception as e:
        print(f"Error rendering graph: {e}")
        print(f"DOT source saved to {output_filename}.dot")

if __name__ == '__main__':
    create_cotracker3_diagram()
