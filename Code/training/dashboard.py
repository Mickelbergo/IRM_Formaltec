"""
Real-time Training Visualization Dashboard for Wound Segmentation
Launch with: streamlit run Code/training/dashboard.py
"""

import streamlit as st
import json
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import time
from PIL import Image
import pandas as pd

# Page config
st.set_page_config(
    page_title="Wound Segmentation Training Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stMetric {
        background-color: white;
        padding: 1rem;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)


class TrainingDashboard:
    """Real-time training visualization dashboard"""

    def __init__(self, log_dir="training_logs"):
        self.log_dir = Path(log_dir)

    def get_available_runs(self):
        """Get list of available training runs"""
        if not self.log_dir.exists():
            return []
        runs = [d.name for d in self.log_dir.iterdir() if d.is_dir()]
        return sorted(runs, reverse=True)  # Most recent first

    def load_run_data(self, run_name):
        """Load all data for a specific run"""
        run_dir = self.log_dir / run_name

        data = {}

        # Load metrics
        metrics_file = run_dir / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                data['metrics'] = json.load(f)

        # Load hyperparameters
        hp_file = run_dir / "hyperparameters.json"
        if hp_file.exists():
            with open(hp_file, 'r') as f:
                data['hyperparameters'] = json.load(f)

        # Load model info
        model_file = run_dir / "model_info.json"
        if model_file.exists():
            with open(model_file, 'r') as f:
                data['model_info'] = json.load(f)

        # Load stage changes
        stage_file = run_dir / "stage_changes.json"
        if stage_file.exists():
            with open(stage_file, 'r') as f:
                data['stage_changes'] = json.load(f)

        # Load summary
        summary_file = run_dir / "summary.json"
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                data['summary'] = json.load(f)

        data['run_dir'] = run_dir

        return data

    def plot_training_curves(self, metrics):
        """Create training/validation curves plot"""
        epochs = list(range(1, len(metrics['valid']['iou']) + 1))

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Loss', 'IoU Score', 'F1 Score', 'Accuracy'),
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )

        # Loss
        fig.add_trace(
            go.Scatter(x=epochs, y=metrics['train']['loss'], name='Train Loss',
                      line=dict(color='#FF6B6B', width=2)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=metrics['valid']['loss'], name='Valid Loss',
                      line=dict(color='#4ECDC4', width=2)),
            row=1, col=1
        )

        # IoU
        fig.add_trace(
            go.Scatter(x=epochs, y=metrics['train']['iou'], name='Train IoU',
                      line=dict(color='#FF6B6B', width=2)),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=metrics['valid']['iou'], name='Valid IoU',
                      line=dict(color='#4ECDC4', width=2)),
            row=1, col=2
        )

        # F1
        fig.add_trace(
            go.Scatter(x=epochs, y=metrics['train']['f1'], name='Train F1',
                      line=dict(color='#FF6B6B', width=2)),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=metrics['valid']['f1'], name='Valid F1',
                      line=dict(color='#4ECDC4', width=2)),
            row=2, col=1
        )

        # Accuracy
        fig.add_trace(
            go.Scatter(x=epochs, y=metrics['train']['accuracy'], name='Train Acc',
                      line=dict(color='#FF6B6B', width=2)),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=metrics['valid']['accuracy'], name='Valid Acc',
                      line=dict(color='#4ECDC4', width=2)),
            row=2, col=2
        )

        # Update layout
        fig.update_xaxes(title_text="Epoch", row=2, col=1)
        fig.update_xaxes(title_text="Epoch", row=2, col=2)
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="IoU Score", row=1, col=2)
        fig.update_yaxes(title_text="F1 Score", row=2, col=1)
        fig.update_yaxes(title_text="Accuracy", row=2, col=2)

        fig.update_layout(
            height=600,
            showlegend=True,
            legend=dict(x=1.05, y=1),
            hovermode='x unified'
        )

        return fig

    def plot_learning_rate(self, metrics):
        """Plot learning rate schedule"""
        if not metrics['train']['lr']:
            return None

        epochs = list(range(1, len(metrics['train']['lr']) + 1))

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=epochs, y=metrics['train']['lr'],
                      name='Learning Rate',
                      line=dict(color='#95E1D3', width=3),
                      fill='tozeroy')
        )

        fig.update_layout(
            title='Learning Rate Schedule',
            xaxis_title='Epoch',
            yaxis_title='Learning Rate',
            height=300,
            yaxis_type='log',
            hovermode='x'
        )

        return fig

    def plot_per_class_iou(self, metrics, class_names=None):
        """Plot per-class IoU heatmap"""
        if not metrics['valid']['per_class_iou']:
            return None

        per_class_iou = np.array(metrics['valid']['per_class_iou'])
        epochs = list(range(1, len(per_class_iou) + 1))

        if class_names is None:
            class_names = [
                'Background', 'Dermatorrhagia', 'Hematoma', 'Stab', 'Cut',
                'Thermal', 'Skin Abrasion', 'Puncture', 'Contused-Lacerated',
                'Semisharp Force', 'Lacerations'
            ]

        # Ensure we have the right number of class names
        num_classes = per_class_iou.shape[1]
        if len(class_names) > num_classes:
            class_names = class_names[:num_classes]

        fig = go.Figure(data=go.Heatmap(
            z=per_class_iou.T,
            x=epochs,
            y=class_names,
            colorscale='Viridis',
            colorbar=dict(title="IoU Score"),
            hoverongaps=False
        ))

        fig.update_layout(
            title='Per-Class IoU Over Training',
            xaxis_title='Epoch',
            yaxis_title='Class',
            height=400
        )

        return fig

    def plot_latest_per_class_iou_bar(self, metrics, class_names=None):
        """Bar chart of latest per-class IoU"""
        if not metrics['valid']['per_class_iou']:
            return None

        latest_iou = np.array(metrics['valid']['per_class_iou'][-1])

        if class_names is None:
            class_names = [
                'Background', 'Dermatorrhagia', 'Hematoma', 'Stab', 'Cut',
                'Thermal', 'Skin Abrasion', 'Puncture', 'Contused-Lacerated',
                'Semisharp Force', 'Lacerations'
            ]

        # Ensure we have the right number of class names
        if len(class_names) > len(latest_iou):
            class_names = class_names[:len(latest_iou)]

        fig = go.Figure(data=[
            go.Bar(
                x=class_names,
                y=latest_iou,
                marker=dict(
                    color=latest_iou,
                    colorscale='RdYlGn',
                    showscale=True,
                    cmin=0,
                    cmax=1
                ),
                text=[f'{x:.3f}' for x in latest_iou],
                textposition='auto'
            )
        ])

        fig.update_layout(
            title='Latest Per-Class IoU (Validation)',
            xaxis_title='Class',
            yaxis_title='IoU Score',
            height=400,
            xaxis_tickangle=-45
        )

        return fig

    def display_prediction_samples(self, run_dir, epoch=None):
        """Display prediction samples"""
        pred_dir = run_dir / "predictions"

        if not pred_dir.exists():
            st.info("No prediction samples saved yet")
            return

        # Get available epochs
        epoch_dirs = sorted([d for d in pred_dir.iterdir() if d.is_dir()], reverse=True)

        if not epoch_dirs:
            st.info("No prediction samples saved yet")
            return

        # Select epoch
        if epoch is None:
            selected_epoch_dir = epoch_dirs[0]
        else:
            epoch_name = f"epoch_{epoch:03d}"
            selected_epoch_dir = pred_dir / epoch_name
            if not selected_epoch_dir.exists():
                selected_epoch_dir = epoch_dirs[0]

        st.subheader(f"Predictions: {selected_epoch_dir.name}")

        # Load samples
        sample_files = sorted(selected_epoch_dir.glob("*.npz"))

        if not sample_files:
            st.info("No samples in this epoch")
            return

        # Display samples in columns
        cols = st.columns(min(len(sample_files), 3))

        for idx, sample_file in enumerate(sample_files[:6]):  # Show max 6 samples
            with cols[idx % 3]:
                data = np.load(sample_file)

                # Process image
                image = data['image']
                if image.shape[0] == 3:  # CHW -> HWC
                    image = np.transpose(image, (1, 2, 0))
                image = np.clip(image, 0, 1)

                ground_truth = data['ground_truth']
                prediction = data['prediction']

                # Create visualization
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))

                axes[0].imshow(image)
                axes[0].set_title('Input')
                axes[0].axis('off')

                axes[1].imshow(ground_truth, cmap='tab10', vmin=0, vmax=10)
                axes[1].set_title('Ground Truth')
                axes[1].axis('off')

                axes[2].imshow(prediction, cmap='tab10', vmin=0, vmax=10)
                axes[2].set_title('Prediction')
                axes[2].axis('off')

                plt.tight_layout()
                st.pyplot(fig)
                plt.close()


# Main Dashboard
def main():
    st.markdown('<h1 class="main-header">🏥 Wound Segmentation Training Dashboard</h1>', unsafe_allow_html=True)

    dashboard = TrainingDashboard()
    available_runs = dashboard.get_available_runs()

    if not available_runs:
        st.error("❌ No training runs found in 'training_logs/' directory")
        st.info("Start a training run first to see visualizations here!")
        return

    # Sidebar
    with st.sidebar:
        st.title("⚙️ Controls")

        # Run selection
        selected_run = st.selectbox(
            "Select Training Run",
            available_runs,
            help="Choose a training run to visualize"
        )

        # Auto-refresh
        auto_refresh = st.checkbox("🔄 Auto-refresh (every 10s)", value=False)

        if auto_refresh:
            st.info("Dashboard will refresh every 10 seconds")
            time.sleep(10)
            st.rerun()

        # Refresh button
        if st.button("🔄 Refresh Now"):
            st.rerun()

        st.markdown("---")
        st.markdown("### 📊 Dashboard Info")
        st.markdown("""
        - **Real-time metrics** tracking
        - **Per-class IoU** visualization
        - **Learning rate** monitoring
        - **Prediction samples** display
        """)

    # Load run data
    data = dashboard.load_run_data(selected_run)

    if 'metrics' not in data:
        st.error("❌ No metrics found for this run")
        return

    metrics = data['metrics']

    # Top metrics cards
    st.markdown("## 📈 Current Performance")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        latest_train_iou = metrics['train']['iou'][-1] if metrics['train']['iou'] else 0
        best_train_iou = max(metrics['train']['iou']) if metrics['train']['iou'] else 0
        st.metric(
            "Train IoU",
            f"{latest_train_iou:.4f}",
            f"Best: {best_train_iou:.4f}"
        )

    with col2:
        latest_valid_iou = metrics['valid']['iou'][-1] if metrics['valid']['iou'] else 0
        best_valid_iou = max(metrics['valid']['iou']) if metrics['valid']['iou'] else 0
        st.metric(
            "Valid IoU",
            f"{latest_valid_iou:.4f}",
            f"Best: {best_valid_iou:.4f}"
        )

    with col3:
        latest_valid_f1 = metrics['valid']['f1'][-1] if metrics['valid']['f1'] else 0
        best_valid_f1 = max(metrics['valid']['f1']) if metrics['valid']['f1'] else 0
        st.metric(
            "Valid F1",
            f"{latest_valid_f1:.4f}",
            f"Best: {best_valid_f1:.4f}"
        )

    with col4:
        current_epoch = len(metrics['valid']['iou'])
        st.metric(
            "Epochs",
            f"{current_epoch}",
            "Completed"
        )

    st.markdown("---")

    # Training curves
    st.markdown("## 📊 Training Curves")
    fig_curves = dashboard.plot_training_curves(metrics)
    st.plotly_chart(fig_curves, use_container_width=True)

    # Learning rate
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("## 📉 Learning Rate Schedule")
        fig_lr = dashboard.plot_learning_rate(metrics)
        if fig_lr:
            st.plotly_chart(fig_lr, use_container_width=True)
        else:
            st.info("Learning rate data not available")

    with col2:
        st.markdown("## 📊 Latest Per-Class IoU")
        fig_bar = dashboard.plot_latest_per_class_iou_bar(metrics)
        if fig_bar:
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Per-class IoU data not available")

    # Per-class IoU heatmap
    st.markdown("## 🔥 Per-Class IoU Heatmap")
    fig_heatmap = dashboard.plot_per_class_iou(metrics)
    if fig_heatmap:
        st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        st.info("Per-class IoU data not available")

    # Hyperparameters
    with st.expander("⚙️ Hyperparameters", expanded=False):
        if 'hyperparameters' in data:
            st.json(data['hyperparameters'])
        else:
            st.info("No hyperparameters logged")

    # Model info
    with st.expander("🧠 Model Information", expanded=False):
        if 'model_info' in data:
            st.json(data['model_info'])
        else:
            st.info("No model information logged")

    # Stage changes
    if 'stage_changes' in data:
        with st.expander("🎯 Training Stages", expanded=False):
            for stage in data['stage_changes']:
                st.markdown(f"**Epoch {stage['epoch']}**: {stage['stage']}")
                if stage.get('unfrozen_layers'):
                    st.markdown(f"  - Unfrozen layers: {stage['unfrozen_layers']}")

    # Prediction samples
    st.markdown("## 🖼️ Prediction Samples")
    import matplotlib.pyplot as plt
    dashboard.display_prediction_samples(data['run_dir'])


if __name__ == "__main__":
    main()
