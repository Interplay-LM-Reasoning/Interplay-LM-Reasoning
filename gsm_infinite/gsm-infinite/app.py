import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import uuid
from collections import defaultdict
from scipy import integrate

# -----------------------------
# 1. App Configuration
# -----------------------------

# Set the page configuration
st.set_page_config(
    page_title="GSM-Infinite Data Viewer",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title of the app
st.title("GSM-Infinite Results Viewer")

# -----------------------------
# 2. Data Loading and Processing
# -----------------------------

@st.cache_data
def load_data():
    """Load and process the benchmark results from CSV."""
    try:
        df = pd.read_csv('results/processed_results.csv')
        return df
    except FileNotFoundError:
        st.error("The file 'processed_results.csv' was not found. Please run preprocess.py first.")
        st.stop()

df = load_data()

# Initialize session state for storing selected series
if 'selected_series' not in st.session_state:
    st.session_state.selected_series = []

# -----------------------------
# 3. Series Selection Interface
# -----------------------------

st.sidebar.header("Add New Series")

# a. Dataset Selection
datasets = sorted(df['dataset'].unique())
selected_dataset = st.sidebar.selectbox("Select Dataset", options=datasets, key="dataset_selector")

# b. Model Selection based on Selected Dataset
models = sorted(df[df['dataset'] == selected_dataset]['model'].unique())
selected_model = st.sidebar.selectbox("Select Model", options=models, key="model_selector")

# c. Length Selection
lengths = sorted(df['length'].unique())
selected_length = st.sidebar.selectbox("Select Length", options=lengths, key="length_selector", index=lengths.index(0) if 0 in lengths else 0)

# d. Subset Filters (if available)
has_subset_info = 'template' in df.columns and 'mode' in df.columns

# Function to calculate area under curve
def calculate_auc(x, y):
    """Calculate area under curve using trapezoidal rule and multiply by 100"""
    if len(x) < 2:
        return 0
    # Sort by x values to ensure proper integration
    sorted_pairs = sorted(zip(x, y), key=lambda pair: pair[0])
    sorted_x, sorted_y = zip(*sorted_pairs)
    auc = np.trapz(sorted_y, sorted_x) * 100
    return auc

if has_subset_info:
    # Get available templates and modes for the selected dataset/model/length
    filtered_base = df[
        (df['dataset'] == selected_dataset) & 
        (df['model'] == selected_model) &
        (df['length'] == selected_length)
    ]
    
    # Get all available templates and modes
    available_templates = [t for t in sorted(filtered_base['template'].unique()) if t != 'default']
    available_modes = [m for m in sorted(filtered_base['mode'].unique()) if m != 'default']

    # Replace 'default' with 'all' in the options
    template_options = ['all'] + available_templates
    mode_options = ['all'] + available_modes
    
    # Multi-select for templates
    selected_templates = st.sidebar.multiselect(
        "Select Templates",
        options=template_options,
        default=["all"],
        key="template_selector"
    )
    
    # Multi-select for modes
    selected_modes = st.sidebar.multiselect(
        "Select Modes",
        options=mode_options,
        default=["all"],
        key="mode_selector"
    )
    # Process 'all' selection - convert to actual template/mode values
    if 'all' in selected_templates:
        selected_templates = available_templates
    if 'all' in selected_modes:
        selected_modes = available_modes
# Import a color library if you want more sophisticated palettes
import plotly.express as px
# e. Series Color
# Define color palettes
color_palettes = {
    "Default": ['blue', 'red', 'green', 'purple', 'orange', 'teal', 'pink', 'brown', 'gray', 'black'],
    "Plotly": px.colors.qualitative.Plotly,
    "Pastel": px.colors.qualitative.Pastel,
    "Dark": px.colors.qualitative.Dark24,
    "Light": px.colors.qualitative.Light24
}

# Select palette first
selected_palette = st.sidebar.selectbox(
    "Color Palette",
    options=list(color_palettes.keys()),
    key="palette_selector"
)

# Then select color from that palette
palette_colors = color_palettes[selected_palette]
default_color_index = len(st.session_state.selected_series) % len(palette_colors)
selected_color = st.sidebar.selectbox(
    "Series Color", 
    options=palette_colors, 
    index=default_color_index,
    key="color_selector"
)
# f. Add Series Button
if st.sidebar.button("Add Series to Plot"):
    # Create a filter for the selected series
    series_filter = {
        'dataset': selected_dataset,
        'model': selected_model,
        'length': selected_length
    }
    
    # Filter the data based on dataset, model, and length
    filtered_data = df[
        (df['dataset'] == selected_dataset) & 
        (df['model'] == selected_model) &
        (df['length'] == selected_length)
    ]
    
    if has_subset_info and selected_templates and selected_modes:
        # Further filter by selected templates and modes
        filtered_data = filtered_data[
            (filtered_data['template'].isin(selected_templates)) & 
            (filtered_data['mode'].isin(selected_modes))
        ]
        
        # Group by N and calculate weighted average based on num_examples
        grouped_data = filtered_data.groupby('N').apply(
            lambda x: np.average(x['accuracy'], weights=x['num_examples']),
            include_groups=False  # Add this parameter to fix the deprecation warning
        ).reset_index()
        grouped_data.columns = ['N', 'accuracy']
        
        # Sort by N
        grouped_data = grouped_data.sort_values('N')
        
        # Create a label for the series
        template_str = ", ".join(selected_templates) if len(selected_templates) < len(available_templates) else f"all templates"
        mode_str = ", ".join(selected_modes) if len(selected_modes) < len(available_modes) else f"all modes"
        label = f"{selected_dataset}: {selected_model} (len={selected_length}, {template_str}, {mode_str})"
        
        # Calculate AUC
        auc = calculate_auc(grouped_data['N'].tolist(), grouped_data['accuracy'].tolist())
        
        # Add to session state
        st.session_state.selected_series.append({
            'id': str(uuid.uuid4()),
            'label': label,
            'filter': {
                'dataset': selected_dataset,
                'model': selected_model,
                'length': selected_length,
                'templates': selected_templates,
                'modes': selected_modes
            },
            'x': grouped_data['N'].tolist(),
            'y': grouped_data['accuracy'].tolist(),
            'color': selected_color,
            'auc': auc
        })
    else:
        # If no subset info or no templates/modes selected, use all data
        filtered_data = filtered_data.sort_values('N')
        
        # Create a label for the series
        label = f"{selected_dataset}: {selected_model} (len={selected_length})"
        
        # Calculate AUC
        auc = calculate_auc(filtered_data['N'].tolist(), filtered_data['accuracy'].tolist())
        
        # Add to session state
        st.session_state.selected_series.append({
            'id': str(uuid.uuid4()),
            'label': label,
            'filter': series_filter,
            'x': filtered_data['N'].tolist(),
            'y': filtered_data['accuracy'].tolist(),
            'color': selected_color,
            'auc': auc
        })
    
    st.sidebar.success(f"Added series: {label}")

# -----------------------------
# 4. Series Management
# -----------------------------

st.header("Selected Series")

# Display the selected series in a table
if st.session_state.selected_series:
    # Create a DataFrame for the selected series
    series_df = pd.DataFrame([
        {
            'Series ID': s['id'][:6],  # Truncate UUID for display
            'Label': s['label'],
            'Points': len(s['x']),
            'Min Op': min(s['x']) if s['x'] else 'N/A',
            'Max Op': max(s['x']) if s['x'] else 'N/A',
            'Min Accuracy': min(s['y']) if s['y'] else 'N/A',
            'Max Accuracy': max(s['y']) if s['y'] else 'N/A',
            'Avg Accuracy': np.mean(s['y']) if s['y'] else 'N/A',
            'AUC (×100)': round(s['auc'], 2),
            'Color': s['color']
        }
        for s in st.session_state.selected_series
    ])
    
    # Display the table
    st.dataframe(series_df, use_container_width=True)
    
    # Add a button to clear all series
    if st.button("Clear All Series"):
        st.session_state.selected_series = []
        st.rerun()
    
    # Add a button to remove selected series
    series_to_remove = st.selectbox(
        "Select Series to Remove",
        options=[s['label'] for s in st.session_state.selected_series],
        key="series_to_remove"
    )
    
    if st.button("Remove Selected Series"):
        st.session_state.selected_series = [
            s for s in st.session_state.selected_series if s['label'] != series_to_remove
        ]
        st.rerun()
else:
    st.info("No series selected. Use the sidebar to add series to the plot.")

# -----------------------------
# 5. Main Plot
# -----------------------------

if st.session_state.selected_series:
    st.header("Accuracy vs. Op")
    
    # Display Options
    col1, col2, col3 = st.columns(3)
    with col1:
        show_average = st.checkbox("Show Average Line", value=True)
    with col2:
        line_width = st.slider("Line Width", min_value=1, max_value=5, value=2)
    with col3:
        marker_size = st.slider("Marker Size", min_value=4, max_value=12, value=8)
    
    # Create the main plot
    fig = go.Figure()
    
    # Add each series to the plot
    for series in st.session_state.selected_series:
        fig.add_trace(go.Scatter(
            x=series['x'],
            y=series['y'],
            mode='lines+markers',
            name=f"{series['label']} (AUC={round(series['auc'], 2)})",
            line=dict(width=line_width, color=series['color']),
            marker=dict(size=marker_size, color=series['color'])
        ))
    
    # Add average line if requested
    if show_average and len(st.session_state.selected_series) > 1:
        # Collect all N values
        all_n_values = sorted(set(n for series in st.session_state.selected_series for n in series['x']))
        
        # Calculate average accuracy for each N
        avg_accuracies = []
        for n in all_n_values:
            accuracies = []
            for series in st.session_state.selected_series:
                if n in series['x']:
                    idx = series['x'].index(n)
                    accuracies.append(series['y'][idx])
            
            if accuracies:
                avg_accuracies.append(np.mean(accuracies))
            else:
                avg_accuracies.append(None)
        
        # Calculate average of individual AUCs instead of AUC of the average line
        avg_auc = np.mean([series['auc'] for series in st.session_state.selected_series])
      
        
        # Add average line to plot
        fig.add_trace(go.Scatter(
            x=all_n_values,
            y=avg_accuracies,
            mode='lines+markers',
            name=f'Average (AUC={round(avg_auc, 2)})',
            line=dict(width=line_width+1, dash='dash', color='black'),
            marker=dict(size=marker_size+2, color='black')
        ))
    
    # Update layout
    fig.update_layout(
        title=f"Accuracy vs. Op",
        xaxis_title="Op",
        yaxis_title="Accuracy",
        legend_title="Series",
        hovermode='closest',
        height=600
    )
    
    # Display the plot
    st.plotly_chart(fig, use_container_width=True)
    
    # -----------------------------
    # 6. Series Details
    # -----------------------------
    
    st.header("Series Details")
    
    # Let user select a series to view details
    selected_detail_series = st.selectbox(
        "Select Series to View Details",
        options=[s['label'] for s in st.session_state.selected_series],
        key="detail_series_selector"
    )
    
    # Find the selected series
    detail_series = next((s for s in st.session_state.selected_series if s['label'] == selected_detail_series), None)
    
    if detail_series:
        # Create a DataFrame with the detailed data
        detail_df = pd.DataFrame({
            'Op': detail_series['x'],
            'Accuracy': detail_series['y']
        })
        
        # Display the data
        st.dataframe(detail_df, use_container_width=True)
        
        # Add download button for this series
        csv = detail_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"Download {selected_detail_series} Data",
            data=csv,
            file_name=f"gsm_infinite_{selected_detail_series.replace(' ', '_')}.csv",
            mime='text/csv',
        )

# -----------------------------
# 7. Download All Data
# -----------------------------

if st.session_state.selected_series:
    st.header("Download All Data")
    
    # Prepare data for all series
    all_series_data = []
    for series in st.session_state.selected_series:
        for i, (n, acc) in enumerate(zip(series['x'], series['y'])):
            all_series_data.append({
                'Series': series['label'],
                'Op': n,
                'Accuracy': acc,
                'AUC': series['auc']
            })
    
    all_data_df = pd.DataFrame(all_series_data)
    
    # Download button for all data
    all_csv = all_data_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download All Series Data",
        data=all_csv,
        file_name="gsm_infinite_all_series_data.csv",
        mime='text/csv',
    )