import os
from graphviz import Digraph

# Add Graphviz to PATH (adjust as needed for the user's environment)
graphviz_path = r'C:\Users\PC\Downloads\tools\Graphviz-12.2.1-win64\bin'
if graphviz_path not in os.environ['PATH']:
    os.environ['PATH'] += os.pathsep + graphviz_path

def create_cotracker_diagram():
    # --- CONFIGURATION AREA ---
    # Simplified labels for clarity and brevity
    LABELS = {
        # Inputs
        'Video': 'Video đầu vào',
        'Queries': 'Điểm truy vấn',
        
        # Encoders & Embeddings
        'CNN': 'Mạng CNN',
        'FMap': 'Bản đồ đặc trưng',
        'TrackFeat': 'Đặc trưng điểm',
        'PosEmb': 'Mã hoá vị trí',
        'TimeEmb': 'Mã hoá thời gian',
        'FlowEmb': 'Mã hoá độ di dời',
        
        # Correlation
        'CorrBlock': 'Khối tương quan',
        'CorrSamp': 'Lấy mẫu lưới 2D',
        
        # Transformer Input
        'TokenConcat': 'Nối chập',
        
        # Transformer Layers
        'TimeAttn': 'Attention thời gian',
        
        # Proxy/Space Attention
        'ProxyInit': 'Token ảo',
        'P2V': 'Điểm -> Token Ảo',
        'V2V': 'Self-Attn Token Ảo',
        'V2P': 'Token Ảo -> Điểm',
        
        # Heads & Update
        'FlowHead': 'Đầu dự đoán độ di dời',
        'VisHead': 'Đầu dự đoán hiển thị',
        'FeatUpdate': 'Cập nhật đặc trưng',
        'CoordUpdate': 'Cập nhật toạ độ',
        
        # Outputs
        'OutTraj': 'Quỹ đạo',
        'OutVis': 'Sự hiển thị',
        
        # Cluster Labels
        'c_inputs': 'Đầu vào',
        'c_encoder': 'Đặc trưng',
        'c_embeddings': 'Mã hoá',
        'c_prep': 'Xử lý',
        'c_transformer': 'UpdateFormer',
        'c_proxy': 'Attention ảo',
        'c_heads': 'Các đầu dự đoán',
        'c_outputs': 'Đầu ra',
        
        # Edges
        'e_vid': 'Video',
        'e_sample': 'Lấy mẫu ban đầu',
        'e_init_coords': '(Toạ độ khởi tạo)',
        'e_curr_coords': '(Toạ độ hiện tại)',
        'e_corr': 'Tương quan',
        'e_delta': 'Độ lệch',
        'e_tokens': 'Token',
        'e_proxy': 'Ảo',
        'e_feat_flow': 'Đặc trưng',
        'e_iter': 'Lặp lại',
        'e_recur': 'Đệ quy'
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
        'head': '#FFA500', # Unified head color
        'update': '#90EE90',
        'output': '#32CD32'
    }

    # Create Digraph - Vertical Layout (TB)
    dot = Digraph(name='CoTracker_Architecture', comment='CoTracker Architecture', format='png')
    # Updated for A4 fit and legibility
    dot.attr(rankdir='TB', splines='ortho', nodesep='0.5', ranksep='0.5', dpi='300', size='6.5,9.0')
    dot.attr('node', shape='rect', style='filled', fontname='Arial', fontsize='24', height='0.8')
    dot.attr('edge', fontname='Arial', fontsize='20')

    # 1. Inputs
    with dot.subgraph(name='cluster_inputs') as c:
        c.attr(label=LABELS['c_inputs'], color='white', rank='same', margin='10', fontname='Arial')
        c.node('Video', LABELS['Video'], fillcolor=COLORS['input'], shape='box3d')
        c.node('Queries', LABELS['Queries'], fillcolor=COLORS['input'], shape='component')

    # 2. Encoders & Feature Extraction
    with dot.subgraph(name='cluster_encoder') as c:
        c.attr(label=LABELS['c_encoder'], color='lightgrey', style='rounded', margin='10', fontname='Arial')
        c.node('CNN', LABELS['CNN'], fillcolor=COLORS['encoder'])
        c.node('FMap', LABELS['FMap'], fillcolor=COLORS['feat'], shape='parallelogram')
        c.node('TrackFeat', LABELS['TrackFeat'], fillcolor=COLORS['feat'])
        
    # 3. Embeddings
    with dot.subgraph(name='cluster_embeddings') as c:
        c.attr(label=LABELS['c_embeddings'], color='lightyellow', style='dashed', margin='10', fontname='Arial')
        c.node('PosEmb', LABELS['PosEmb'], fillcolor=COLORS['emb'])
        c.node('TimeEmb', LABELS['TimeEmb'], fillcolor=COLORS['emb'])
        c.node('FlowEmb', LABELS['FlowEmb'], fillcolor=COLORS['emb'])

    # 4. Correlation & Token Construction
    with dot.subgraph(name='cluster_prep') as c:
        c.attr(label=LABELS['c_prep'], color='white', margin='10', fontname='Arial')
        c.node('CorrBlock', LABELS['CorrBlock'], fillcolor=COLORS['corr'])
        c.node('CorrSamp', LABELS['CorrSamp'], fillcolor=COLORS['corr'])
        c.node('TokenConcat', LABELS['TokenConcat'], fillcolor='#F0E68C', shape='rect')

    # 5. The EfficientUpdateFormer (Core)
    with dot.subgraph(name='cluster_transformer') as c:
        c.attr(label=LABELS['c_transformer'], color=COLORS['trans'], style='rounded', penwidth='2', margin='12', fontname='Arial')
        
        # Time Block
        c.node('TimeAttn', LABELS['TimeAttn'], fillcolor=COLORS['trans_block'])
        
        # Proxy / Space Attention Block
        with dot.subgraph(name='cluster_proxy') as p:
            p.attr(label=LABELS['c_proxy'], color='#9370DB', style='solid', bgcolor='#F5F5F5', margin='10', fontsize='12', fontname='Arial')
            p.node('ProxyInit', LABELS['ProxyInit'], fillcolor=COLORS['proxy'], shape='octagon', width='1.0')
            
            # The Sandwich Pattern
            p.node('P2V', LABELS['P2V'], fillcolor=COLORS['proxy_sub'], width='1.5')
            p.node('V2V', LABELS['V2V'], fillcolor='#BA55D3', width='1.5')
            p.node('V2P', LABELS['V2P'], fillcolor=COLORS['proxy_sub'], width='1.5')
            
            # Edges inside Proxy
            p.edge('ProxyInit', 'V2V', style='dotted')
            p.edge('P2V', 'V2V')
            p.edge('V2V', 'V2P')
            
    # 6. Heads & Updates
    with dot.subgraph(name='cluster_heads') as c:
        c.attr(label=LABELS['c_heads'], color='lightgrey', fontname='Arial')
        c.node('FlowHead', LABELS['FlowHead'], fillcolor=COLORS['head'])
        c.node('VisHead', LABELS['VisHead'], fillcolor=COLORS['head'])
        c.node('CoordUpdate', LABELS['CoordUpdate'], fillcolor=COLORS['update'])
        c.node('FeatUpdate', LABELS['FeatUpdate'], fillcolor=COLORS['update'])

    # 7. Outputs
    with dot.subgraph(name='cluster_outputs') as c:
        c.attr(label=LABELS['c_outputs'], color='white', rank='same', fontname='Arial')
        c.node('OutTraj', LABELS['OutTraj'], fillcolor=COLORS['output'], shape='folder')
        c.node('OutVis', LABELS['OutVis'], fillcolor=COLORS['output'], shape='folder')

    # --- Connections ---

    # Input to Encoder
    dot.edge('Video', 'CNN', label=LABELS['e_vid'])
    dot.edge('CNN', 'FMap')
    dot.edge('Queries', 'TrackFeat', label=LABELS['e_sample'])
    dot.edge('FMap', 'TrackFeat')
    
    # Correlation Path
    dot.edge('FMap', 'CorrBlock')
    dot.edge('TrackFeat', 'CorrBlock')
    dot.edge('CorrBlock', 'CorrSamp', label=LABELS['e_corr'])
    
    # Embedding Path
    dot.edge('Queries', 'FlowEmb', label=LABELS['e_init_coords'])
    
    # Token Construction
    dot.edge('FlowEmb', 'TokenConcat')
    dot.edge('CorrSamp', 'TokenConcat')
    dot.edge('TrackFeat', 'TokenConcat')
    dot.edge('PosEmb', 'TokenConcat', style='dashed')
    dot.edge('TimeEmb', 'TokenConcat', style='dashed')
    
    # Transformer Flow
    dot.edge('TokenConcat', 'TimeAttn')
    dot.edge('TimeAttn', 'P2V', label=LABELS['e_feat_flow'])
    
    # Proxy Interaction
    dot.edge('P2V', 'V2P', style='invis') # For layout alignment
    
    # Heads
    dot.edge('V2P', 'FlowHead')
    dot.edge('V2P', 'VisHead')
    
    # Updates
    dot.edge('FlowHead', 'CoordUpdate', label=LABELS['e_delta'])
    dot.edge('V2P', 'FeatUpdate')
    
    # Iterative Loop Back
    dot.edge('CoordUpdate', 'FlowEmb', label=LABELS['e_curr_coords'], style='dashed', constraint='false', color='blue')
    dot.edge('FeatUpdate', 'TrackFeat', label=LABELS['e_recur'], style='dashed', constraint='false', color='blue')
    dot.edge('CoordUpdate', 'CorrSamp', style='dashed', constraint='false', color='blue')

    # Output
    dot.edge('CoordUpdate', 'OutTraj')
    dot.edge('VisHead', 'OutVis')

    # Save and Render
    dot.save('cotracker_diagram.dot')
    try:
        output_path = dot.render('cotracker_diagram', view=False)
        print(f"Graph generated at: {output_path}")
    except Exception as e:
        print(f"Error rendering graph: {e}")
        print("DOT source saved to cotracker_diagram.dot")

if __name__ == '__main__':
    create_cotracker_diagram()
