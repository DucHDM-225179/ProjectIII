import os
from graphviz import Digraph

# Add Graphviz to PATH
graphviz_path = r'C:\Users\PC\Downloads\tools\Graphviz-12.2.1-win64\bin'
if graphviz_path not in os.environ['PATH']:
    os.environ['PATH'] += os.pathsep + graphviz_path

def create_tapnext_diagram():
    # --- BALANCED LABELS ---
    LABELS = {
        'Video': 'Video',
        'Queries': 'Điểm truy vấn',
        'Grid': 'Token lưới video\n(+ Mã hoá vị trí)',
        'Point': 'Token điểm tryt vấn\n(+ Mã hoá vị trí)',
        'Concat': 'Nối chập',
        
        'TRecViT': 'Khối TRecViT (x12)',
        'ToTime': 'Sắp xếp theo thời gian\n(B*N, T, C)',
        'SSM': 'SSM thời gian\n(RG-LRU)',
        'Hidden': 'Trạng thái ẩn (h_t)',
        'ToSpace': 'Sắp xếp theo không gian\n(B*T, N, C)',
        'Attn': 'Attention không gian\n(Self-Attention)',
        
        'Coord': 'Đầu dự đoán toạ độ',
        'Vis': 'Đầu dự đoán hiển thị',
        'Tracks': 'Quỹ đạo',
        'Visibility': 'Sự hiển thị',
        
        # Cluster Labels
        'c_tokens': 'Token hoá',
        'c_trecvit': 'Khối TRecViT (x12)',
        'c_outputs': 'Các đầu dự đoán & Đầu ra'
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
        'ssm': '#87CEFA',
        'attn': '#32CD32',
        'head': '#FFA500',
        'update': '#90EE90',
        'output': '#FF4500' # Matches original tapnext output color or use standard green? Original was orange/red. I'll stick to palette: Output green usually. But let's respect this file's flow. Wait, standard is #32CD32 (Green). I will use Green for Tracks/Vis.
    }
    # Update output colors to match standard
    COLORS['output'] = '#32CD32'

    dot = Digraph(name='TAPNext_Architecture', comment='TAPNext Balanced', format='png')
    # Updated for A4 fit and legibility
    dot.attr(rankdir='TB', splines='ortho', dpi='300', size='6.5,9.0') 
    dot.attr('node', shape='rect', style='filled', fontname='Arial', fontsize='24', height='0.8')
    dot.attr('edge', fontname='Arial', fontsize='20')

    # 1. Inputs
    dot.node('Video', LABELS['Video'], fillcolor=COLORS['input'])
    dot.node('Queries', LABELS['Queries'], fillcolor=COLORS['input'])

    # 2. Token Group
    with dot.subgraph(name='cluster_tokens') as c:
        c.attr(label=LABELS['c_tokens'], color='lightgrey', style='rounded', fontname='Arial')
        c.node('Grid', LABELS['Grid'], fillcolor=COLORS['encoder'])
        c.node('Point', LABELS['Point'], fillcolor=COLORS['encoder'])
        c.node('Concat', LABELS['Concat'], shape='diamond', fillcolor='#F08080')

    # 3. TRecViT Block
    with dot.subgraph(name='cluster_trecvit') as c:
        c.attr(label=LABELS['c_trecvit'], color='gold', style='bold', penwidth='2', fontname='Arial')
        c.node('ToTime', LABELS['ToTime'], fillcolor=COLORS['feat'])
        c.node('SSM', LABELS['SSM'], fillcolor=COLORS['ssm'])
        c.node('Hidden', LABELS['Hidden'], fillcolor=COLORS['emb'], shape='ellipse')
        c.node('ToSpace', LABELS['ToSpace'], fillcolor='#98FB98')
        c.node('Attn', LABELS['Attn'], fillcolor=COLORS['attn'])
        
        c.edge('ToTime', 'SSM')
        c.edge('SSM', 'ToSpace')
        c.edge('ToSpace', 'Attn')

    # 4. Output Group
    with dot.subgraph(name='cluster_outputs') as c:
        c.attr(label=LABELS['c_outputs'], color='orange', style='rounded', fontname='Arial')
        c.node('Coord', LABELS['Coord'], fillcolor=COLORS['head'])
        c.node('Vis', LABELS['Vis'], fillcolor=COLORS['head'])
        c.node('Tracks', LABELS['Tracks'], fillcolor=COLORS['output'])
        c.node('Visibility', LABELS['Visibility'], fillcolor=COLORS['output'])

    # --- Edges ---
    dot.edge('Video', 'Grid')
    dot.edge('Queries', 'Point')
    dot.edge('Grid', 'Concat')
    dot.edge('Point', 'Concat')
    
    dot.edge('Concat', 'ToTime')
    dot.edge('SSM', 'Hidden', dir='both', style='dashed')
    
    dot.edge('Attn', 'Coord')
    dot.edge('Attn', 'Vis')
    dot.edge('Coord', 'Tracks')
    dot.edge('Vis', 'Visibility')

    dot_file = 'tapnext_architecture.dot'
    dot.save(dot_file)
    try:
        dot.render('tapnext_architecture', view=False)
        print(f"Graph generated: {dot_file}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    create_tapnext_diagram()
