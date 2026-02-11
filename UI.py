import streamlit as st
import requests
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import time
from typing import Optional

st.set_page_config(
    page_title="Advanced eDNA Analyzer", 
    page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
API_URL = "http://127.0.0.1:8000/analyze/"
PREDICT_URL = "http://127.0.0.1:8000/predict_unknowns/"
HEALTH_URL = "http://127.0.0.1:8000/health"

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .prediction-card {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .confidence-high { border-left-color: #28a745; }
    .confidence-medium { border-left-color: #ffc107; }
    .confidence-low { border-left-color: #fd7e14; }
    .confidence-very-low { border-left-color: #dc3545; }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

def check_backend_status() -> bool:
    """Check if the FastAPI backend is running."""
    try:
        response = requests.get(HEALTH_URL, timeout=5)
        return response.status_code == 200
    except:
        return False

def make_api_request(endpoint: str, files: dict, data: dict, timeout: int = 600) -> Optional[dict]:
    """Make API request with proper error handling."""
    try:
        response = requests.post(endpoint, files=files, data=data, timeout=timeout)
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 400:
            error_data = response.json()
            st.error(f"❌ Invalid input: {error_data.get('detail', 'Unknown error')}")
            return None
        elif response.status_code == 422:
            st.error("❌ Request format error. Please check your file and parameters.")
            return None
        elif response.status_code == 500:
            error_data = response.json()
            st.error(f"❌ Server error: {error_data.get('detail', 'Internal server error')}")
            return None
        else:
            st.error(f"❌ API Error (Status {response.status_code})")
            try:
                error_data = response.json()
                st.json(error_data)
            except:
                st.text(response.text[:500])
            return None
            
    except requests.exceptions.Timeout:
        st.error("⏰ Request timed out. BLAST searches can take a long time. Try with a smaller file or use 'Quick Predictions Only'.")
        return None
    except requests.exceptions.ConnectionError:
        st.error("🚨 Connection Error: Could not connect to the backend API. Please ensure the server is running on http://127.0.0.1:8000")
        return None
    except Exception as e:
        st.error(f"💥 Unexpected error: {str(e)}")
        return None

def display_summary_metrics(results: dict):
    """Display summary metrics in columns."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "📄 File", 
            results.get('filename', 'N/A')[:15] + "..." if len(results.get('filename', '')) > 15 else results.get('filename', 'N/A')
        )
    
    with col2:
        st.metric(
            "🧬 Total Sequences", 
            results.get('total_sequences_processed', results.get('total_sequences', '0'))
        )
    
    with col3:
        st.metric(
            "✅ Known Sequences",
            results.get('known_sequences_count', results.get('known_sequences', 'N/A'))
        )
    
    with col4:
        st.metric(
            "❓ Unknown Sequences", 
            results.get('unknown_sequences_count', results.get('unknown_sequences', 'N/A'))
        )

def create_pie_chart(pie_data: dict, title: str):
    """Create a pie chart from the data."""
    if not pie_data:
        st.warning("No data available for pie chart")
        return None
        
    fig = px.pie(
        values=list(pie_data.values()),
        names=list(pie_data.keys()),
        title=title,
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig.update_traces(
        textposition='inside', 
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>Percentage: %{percent}<br>Count: %{value}<extra></extra>'
    )
    return fig

def display_confidence_breakdown(cluster_summary: dict):
    """Display confidence breakdown chart."""
    confidence_data = {}
    
    for cluster_info in cluster_summary.values():
        conf = cluster_info.get('confidence', 0)
        if conf > 0.5:
            level = "High Confidence (>50%)"
        elif conf > 0.3:
            level = "Medium Confidence (30-50%)"
        elif conf > 0.1:
            level = "Low Confidence (10-30%)"
        else:
            level = "Very Low Confidence (<10%)"
        
        confidence_data[level] = confidence_data.get(level, 0) + cluster_info.get('num_sequences', 0)
    
    if confidence_data:
        fig = px.bar(
            x=list(confidence_data.keys()),
            y=list(confidence_data.values()),
            title="Prediction Confidence Levels",
            color=list(confidence_data.values()),
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(showlegend=False, xaxis_tickangle=-45)
        return fig
    return None

def display_cluster_predictions(cluster_summary: dict):
    """Display detailed cluster predictions."""
    if not cluster_summary:
        st.warning("No cluster predictions available")
        return
    
    for cluster_name, cluster_info in cluster_summary.items():
        prediction = cluster_info.get('predicted_kingdom', 'Unknown')
        confidence = cluster_info.get('confidence', 0)
        num_seqs = cluster_info.get('num_sequences', 0)
        
        # Determine confidence class for styling
        if confidence > 0.5:
            conf_class = "confidence-high"
            conf_emoji = "🎯"
        elif confidence > 0.3:
            conf_class = "confidence-medium" 
            conf_emoji = "🎲"
        elif confidence > 0.1:
            conf_class = "confidence-low"
            conf_emoji = "❓"
        else:
            conf_class = "confidence-very-low"
            conf_emoji = "🔍"
        
        st.markdown(f"""
        <div class="prediction-card {conf_class}">
            <h4>{conf_emoji} {cluster_name}</h4>
            <p><strong>AI Prediction:</strong> {prediction}</p>
            <p><strong>Confidence Score:</strong> {confidence:.2%}</p>
            <p><strong>Sequences:</strong> {num_seqs}</p>
        </div>
        """, unsafe_allow_html=True)

def display_traditional_clustering_results(cluster_data: dict):
    """Display traditional clustering results."""
    if not cluster_data:
        st.warning("No traditional clustering data available")
        return
    
    for cluster_name, cluster_info in sorted(cluster_data.items()):
        annotation = cluster_info.get('annotation', 'N/A')
        confidence = cluster_info.get('confidence', 'N/A')
        sequences = cluster_info.get('sequences', [])
        
        # Determine if this is a known or unknown cluster