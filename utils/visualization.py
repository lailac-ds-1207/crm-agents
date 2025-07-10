"""
Visualization utility for the CRM-Agent system.
Provides functions to create and save various types of visualizations.
"""
import os
from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from pathlib import Path
from datetime import datetime

import config

# Configure logging
logging.basicConfig(
    level=logging.INFO if not config.DEBUG_MODE else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set default styles
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('Set2')

# Default colors for consistent branding
COLOR_PALETTE = {
    'primary': '#1f77b4',  # Blue
    'secondary': '#ff7f0e',  # Orange
    'tertiary': '#2ca02c',  # Green
    'quaternary': '#d62728',  # Red
    'quinary': '#9467bd',  # Purple
    'senary': '#8c564b',  # Brown
    'septenary': '#e377c2',  # Pink
    'octonary': '#7f7f7f',  # Gray
    'nonary': '#bcbd22',  # Yellow-green
    'denary': '#17becf',  # Cyan
}

# Default figure size
DEFAULT_FIG_SIZE = (10, 6)

def setup_visualization_env():
    """Set up the visualization environment and create necessary directories."""
    os.makedirs(config.VISUALIZATIONS_DIR, exist_ok=True)
    logger.info(f"Visualization directory setup at: {config.VISUALIZATIONS_DIR}")

def generate_filename(prefix: str = 'viz', extension: str = 'png') -> str:
    """
    Generate a unique filename for a visualization.
    
    Args:
        prefix: Prefix for the filename
        extension: File extension (without dot)
        
    Returns:
        A unique filename string
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{prefix}_{timestamp}.{extension}"

def save_matplotlib_fig(fig, filename: str = None, dpi: int = 300) -> str:
    """
    Save a matplotlib figure to file.
    
    Args:
        fig: Matplotlib figure object
        filename: Optional filename (if None, generates one)
        dpi: Resolution for the saved image
        
    Returns:
        Path to the saved file
    """
    if filename is None:
        filename = generate_filename('matplotlib')
    
    filepath = os.path.join(config.VISUALIZATIONS_DIR, filename)
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    logger.debug(f"Saved matplotlib figure to {filepath}")
    return filepath

# def save_plotly_fig(fig, filename: str = None, format: str = 'png') -> str:
#     """
#     Save a plotly figure to file.
    
#     Args:
#         fig: Plotly figure object
#         filename: Optional filename (if None, generates one)
#         format: Output format ('png', 'jpg', 'svg', 'pdf', 'html')
        
#     Returns:
#         Path to the saved file
#     """
#     if filename is None:
#         filename = generate_filename('plotly', format)
    
#     filepath = os.path.join(config.VISUALIZATIONS_DIR, filename)
    
#     if format == 'html':
#         fig.write_html(filepath)
#     else:
#         fig.write_image(filepath)
    
#     logger.debug(f"Saved plotly figure to {filepath}")
#     return filepath

def save_plotly_fig(fig, filename: str = None, format: str = 'png', scale: float = 1.0, 
                    width: int = 800, height: int = 500) -> str:
    """
    Save a plotly figure to file with optimized performance.
    
    Args:
        fig: Plotly figure object
        filename: Optional filename (if None, generates one)
        format: Output format ('png', 'jpg', 'svg', 'pdf', 'html')
        scale: Image scale/resolution (lower = faster)
        width: Image width in pixels
        height: Image height in pixels
        
    Returns:
        Path to the saved file
    """
    
    from datetime import datetime
    print(0, datetime.now())
    logger.info(f"Starting To Save plotly figure")
    print(1, datetime.now())
    
    if filename is None:
        print(2, datetime.now())
        filename = generate_filename('plotly', format)
    
    print(3, datetime.now())
    filepath = os.path.join(config.VISUALIZATIONS_DIR, filename)
    print(4, datetime.now())
    
    # 이미 파일이 존재하면 다시 생성하지 않음 (선택적)
    if os.path.exists(filepath):
        print(5, datetime.now())
        logger.debug(f"Using cached image: {filepath}")
        return filepath
    
    print(6, datetime.now())
    # 메모리 최적화: 기존 레이아웃 설정 보존
    original_width = fig.layout.width
    original_height = fig.layout.height
    print(7, datetime.now())
    # 이미지 크기 설정 (렌더링 속도 향상)
    fig.update_layout(width=width, height=height)
    print(8, datetime.now())
    
    if format == 'html':
        print(9, datetime.now())
        fig.write_html(filepath, include_plotlyjs='cdn')  # CDN 사용으로 파일 크기 감소
    else:
        print(10, datetime.now())
        # kaleido 렌더러 최적화 옵션 적용
        fig.write_image(
            filepath,
            scale=scale,  # 해상도 스케일 (1.0 = 100%, 0.5 = 50% 해상도로 더 빠름)
            engine="kaleido",  # 명시적으로 kaleido 엔진 사용
            format=format
        )
    
    print(11, datetime.now())
    # 원래 레이아웃 복원 (다른 곳에서 같은 fig 객체를 사용할 경우)
    if original_width is not None or original_height is not None:
        print(12, datetime.now())
        update_dict = {}
        if original_width is not None:
            update_dict['width'] = original_width
        if original_height is not None:
            update_dict['height'] = original_height
        fig.update_layout(**update_dict)
    
    print(13, datetime.now())
    logger.debug(f"Saved plotly figure to {filepath}")
    
    print(14,datetime.now())
    
    return filepath


def create_time_series_plot(
    df: pd.DataFrame,
    x_column: str,
    y_columns: Union[str, List[str]],
    title: str = 'Time Series Plot',
    xlabel: str = None,
    ylabel: str = None,
    figsize: Tuple[int, int] = DEFAULT_FIG_SIZE,
    use_plotly: bool = True
) -> Union[plt.Figure, go.Figure]:
    """
    Create a time series plot.
    
    Args:
        df: DataFrame containing the data
        x_column: Column name for x-axis (typically date/time)
        y_columns: Column name(s) for y-axis values
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size as (width, height)
        use_plotly: Whether to use plotly (True) or matplotlib (False)
        
    Returns:
        Figure object (either matplotlib or plotly)
    """
    if use_plotly:
        if isinstance(y_columns, list):
            fig = go.Figure()
            for i, col in enumerate(y_columns):
                color = list(COLOR_PALETTE.values())[i % len(COLOR_PALETTE)]
                fig.add_trace(go.Scatter(
                    x=df[x_column],
                    y=df[col],
                    mode='lines',
                    name=col,
                    line=dict(color=color)
                ))
        else:
            fig = px.line(df, x=x_column, y=y_columns, title=title)
        
        fig.update_layout(
            title=title,
            xaxis_title=xlabel or x_column,
            yaxis_title=ylabel or (y_columns if isinstance(y_columns, str) else "Value"),
            template="plotly_white"
        )
        return fig
    else:
        fig, ax = plt.subplots(figsize=figsize)
        
        if isinstance(y_columns, list):
            for i, col in enumerate(y_columns):
                color = list(COLOR_PALETTE.values())[i % len(COLOR_PALETTE)]
                ax.plot(df[x_column], df[col], label=col, color=color)
            ax.legend()
        else:
            ax.plot(df[x_column], df[y_columns], color=COLOR_PALETTE['primary'])
        
        ax.set_title(title)
        ax.set_xlabel(xlabel or x_column)
        ax.set_ylabel(ylabel or (y_columns if isinstance(y_columns, str) else "Value"))
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

def create_bar_chart(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str = 'Bar Chart',
    xlabel: str = None,
    ylabel: str = None,
    color_column: str = None,
    figsize: Tuple[int, int] = DEFAULT_FIG_SIZE,
    horizontal: bool = False,
    use_plotly: bool = True
) -> Union[plt.Figure, go.Figure]:
    """
    Create a bar chart.
    
    Args:
        df: DataFrame containing the data
        x_column: Column name for x-axis
        y_column: Column name for y-axis
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        color_column: Column name for color grouping
        figsize: Figure size as (width, height)
        horizontal: Whether to create a horizontal bar chart
        use_plotly: Whether to use plotly (True) or matplotlib (False)
        
    Returns:
        Figure object (either matplotlib or plotly)
    """
    if use_plotly:
        if horizontal:
            fig = px.bar(
                df, 
                y=x_column, 
                x=y_column, 
                title=title,
                color=color_column,
                orientation='h'
            )
        else:
            fig = px.bar(
                df, 
                x=x_column, 
                y=y_column, 
                title=title,
                color=color_column
            )
        
        fig.update_layout(
            title=title,
            xaxis_title=xlabel or (y_column if horizontal else x_column),
            yaxis_title=ylabel or (x_column if horizontal else y_column),
            template="plotly_white"
        )
        return fig
    else:
        fig, ax = plt.subplots(figsize=figsize)
        
        if color_column:
            grouped = df.groupby(color_column)
            bar_width = 0.8 / len(grouped)
            
            for i, (name, group) in enumerate(grouped):
                offset = (i - len(grouped)/2 + 0.5) * bar_width
                if horizontal:
                    ax.barh(
                        np.arange(len(group)) + offset, 
                        group[y_column], 
                        height=bar_width, 
                        label=name
                    )
                else:
                    ax.bar(
                        np.arange(len(group)) + offset, 
                        group[y_column], 
                        width=bar_width, 
                        label=name
                    )
            ax.set_xticks(np.arange(len(df[x_column].unique())))
            ax.set_xticklabels(df[x_column].unique())
            ax.legend()
        else:
            if horizontal:
                ax.barh(df[x_column], df[y_column], color=COLOR_PALETTE['primary'])
            else:
                ax.bar(df[x_column], df[y_column], color=COLOR_PALETTE['primary'])
        ax.set_title(title)
        ax.set_xlabel(xlabel or (y_column if horizontal else x_column))
        ax.set_ylabel(ylabel or (x_column if horizontal else y_column))
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

def create_pie_chart(
    df: pd.DataFrame,
    values_column: str,
    names_column: str,
    title: str = 'Pie Chart',
    figsize: Tuple[int, int] = (8, 8),
    use_plotly: bool = True
) -> Union[plt.Figure, go.Figure]:
    """
    Create a pie chart.
    
    Args:
        df: DataFrame containing the data
        values_column: Column name for values (sizes)
        names_column: Column name for slice names
        title: Plot title
        figsize: Figure size as (width, height)
        use_plotly: Whether to use plotly (True) or matplotlib (False)
        
    Returns:
        Figure object (either matplotlib or plotly)
    """
    if use_plotly:
        fig = px.pie(
            df, 
            values=values_column, 
            names=names_column, 
            title=title
        )
        fig.update_layout(title=title)
        return fig
    else:
        fig, ax = plt.subplots(figsize=figsize)
        ax.pie(
            df[values_column], 
            labels=df[names_column],
            autopct='%1.1f%%',
            startangle=90,
            colors=list(COLOR_PALETTE.values())[:len(df)]
        )
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        ax.set_title(title)
        fig.tight_layout()
        return fig

def create_heatmap(
    df: pd.DataFrame,
    x_column: str = None,
    y_column: str = None,
    value_column: str = None,
    title: str = 'Heatmap',
    figsize: Tuple[int, int] = (10, 8),
    use_plotly: bool = True
) -> Union[plt.Figure, go.Figure]:
    """
    Create a heatmap.
    
    Args:
        df: DataFrame containing the data
        x_column: Column name for x-axis
        y_column: Column name for y-axis
        value_column: Column name for values (if None, assumes df is already a matrix)
        title: Plot title
        figsize: Figure size as (width, height)
        use_plotly: Whether to use plotly (True) or matplotlib (False)
        
    Returns:
        Figure object (either matplotlib or plotly)
    """
    # If columns are provided, pivot the data
    if x_column and y_column and value_column:
        pivot_df = df.pivot(index=y_column, columns=x_column, values=value_column)
    else:
        pivot_df = df
    
    if use_plotly:
        fig = px.imshow(
            pivot_df,
            title=title,
            color_continuous_scale='Viridis'
        )
        fig.update_layout(title=title)
        return fig
    else:
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(pivot_df, annot=True, cmap="viridis", ax=ax)
        ax.set_title(title)
        fig.tight_layout()
        return fig

def create_scatter_plot(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str = 'Scatter Plot',
    xlabel: str = None,
    ylabel: str = None,
    color_column: str = None,
    size_column: str = None,
    figsize: Tuple[int, int] = DEFAULT_FIG_SIZE,
    use_plotly: bool = True
) -> Union[plt.Figure, go.Figure]:
    """
    Create a scatter plot.
    
    Args:
        df: DataFrame containing the data
        x_column: Column name for x-axis
        y_column: Column name for y-axis
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        color_column: Column name for point colors
        size_column: Column name for point sizes
        figsize: Figure size as (width, height)
        use_plotly: Whether to use plotly (True) or matplotlib (False)
        
    Returns:
        Figure object (either matplotlib or plotly)
    """
    if use_plotly:
        fig = px.scatter(
            df, 
            x=x_column, 
            y=y_column, 
            color=color_column,
            size=size_column,
            title=title
        )
        fig.update_layout(
            title=title,
            xaxis_title=xlabel or x_column,
            yaxis_title=ylabel or y_column,
            template="plotly_white"
        )
        return fig
    else:
        fig, ax = plt.subplots(figsize=figsize)
        
        if color_column:
            scatter = ax.scatter(
                df[x_column], 
                df[y_column],
                c=df[color_column] if color_column in df.columns else None,
                s=df[size_column] * 20 if size_column in df.columns else 50,
                alpha=0.7,
                cmap='viridis'
            )
            
            if color_column in df.columns:
                plt.colorbar(scatter, ax=ax, label=color_column)
        else:
            ax.scatter(
                df[x_column], 
                df[y_column],
                s=df[size_column] * 20 if size_column in df.columns else 50,
                color=COLOR_PALETTE['primary'],
                alpha=0.7
            )
        
        ax.set_title(title)
        ax.set_xlabel(xlabel or x_column)
        ax.set_ylabel(ylabel or y_column)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

def create_histogram(
    df: pd.DataFrame,
    column: str,
    bins: int = 20,
    title: str = 'Histogram',
    xlabel: str = None,
    ylabel: str = 'Count',
    figsize: Tuple[int, int] = DEFAULT_FIG_SIZE,
    use_plotly: bool = True
) -> Union[plt.Figure, go.Figure]:
    """
    Create a histogram.
    
    Args:
        df: DataFrame containing the data
        column: Column name for the histogram
        bins: Number of bins
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size as (width, height)
        use_plotly: Whether to use plotly (True) or matplotlib (False)
        
    Returns:
        Figure object (either matplotlib or plotly)
    """
    if use_plotly:
        fig = px.histogram(
            df, 
            x=column, 
            nbins=bins,
            title=title
        )
        fig.update_layout(
            title=title,
            xaxis_title=xlabel or column,
            yaxis_title=ylabel,
            template="plotly_white"
        )
        return fig
    else:
        fig, ax = plt.subplots(figsize=figsize)
        ax.hist(df[column], bins=bins, color=COLOR_PALETTE['primary'], alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel(xlabel or column)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

def create_box_plot(
    df: pd.DataFrame,
    x_column: str = None,
    y_column: str = None,
    title: str = 'Box Plot',
    xlabel: str = None,
    ylabel: str = None,
    figsize: Tuple[int, int] = DEFAULT_FIG_SIZE,
    use_plotly: bool = True
) -> Union[plt.Figure, go.Figure]:
    """
    Create a box plot.
    
    Args:
        df: DataFrame containing the data
        x_column: Column name for grouping (can be None)
        y_column: Column name for values
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size as (width, height)
        use_plotly: Whether to use plotly (True) or matplotlib (False)
        
    Returns:
        Figure object (either matplotlib or plotly)
    """
    if use_plotly:
        if x_column:
            fig = px.box(
                df, 
                x=x_column, 
                y=y_column, 
                title=title
            )
        else:
            fig = px.box(
                df, 
                y=y_column, 
                title=title
            )
        
        fig.update_layout(
            title=title,
            xaxis_title=xlabel or x_column,
            yaxis_title=ylabel or y_column,
            template="plotly_white"
        )
        return fig
    else:
        fig, ax = plt.subplots(figsize=figsize)
        
        if x_column:
            sns.boxplot(x=x_column, y=y_column, data=df, ax=ax)
        else:
            sns.boxplot(y=y_column, data=df, ax=ax)
        
        ax.set_title(title)
        if x_column:
            ax.set_xlabel(xlabel or x_column)
        ax.set_ylabel(ylabel or y_column)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

def create_correlation_matrix(
    df: pd.DataFrame,
    columns: List[str] = None,
    title: str = 'Correlation Matrix',
    figsize: Tuple[int, int] = (10, 8),
    use_plotly: bool = True
) -> Union[plt.Figure, go.Figure]:
    """
    Create a correlation matrix visualization.
    
    Args:
        df: DataFrame containing the data
        columns: List of columns to include (if None, uses all numeric columns)
        title: Plot title
        figsize: Figure size as (width, height)
        use_plotly: Whether to use plotly (True) or matplotlib (False)
        
    Returns:
        Figure object (either matplotlib or plotly)
    """
    # Select only numeric columns if not specified
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Calculate correlation matrix
    corr_matrix = df[columns].corr()
    
    if use_plotly:
        fig = px.imshow(
            corr_matrix,
            title=title,
            color_continuous_scale='RdBu_r',
            zmin=-1,
            zmax=1
        )
        fig.update_layout(title=title)
        return fig
    else:
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            corr_matrix, 
            annot=True, 
            cmap='RdBu_r', 
            vmin=-1, 
            vmax=1, 
            ax=ax
        )
        ax.set_title(title)
        fig.tight_layout()
        return fig

def create_dashboard(
    plots: List[Dict[str, Any]],
    title: str = 'Dashboard',
    layout: List[List[int]] = None,
    figsize: Tuple[int, int] = (12, 10),
    use_plotly: bool = True
) -> Union[plt.Figure, go.Figure]:
    """
    Create a dashboard with multiple plots.
    
    Args:
        plots: List of dictionaries with plot information
            Each dict should have: 'fig' (figure object), 'title' (optional)
        title: Dashboard title
        layout: Grid layout for subplots, e.g., [[0, 1], [2, 3]] for 2x2 grid
            If None, creates a single column
        figsize: Figure size as (width, height)
        use_plotly: Whether to use plotly (True) or matplotlib (False)
        
    Returns:
        Figure object (either matplotlib or plotly)
    """
    if use_plotly:
        # Determine layout if not provided
        if layout is None:
            rows = len(plots)
            cols = 1
            layout = [[i] for i in range(rows)]
        else:
            rows = len(layout)
            cols = max(len(row) for row in layout)
        
        # Create subplot figure
        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=[plot.get('title', '') for plot in plots],
            vertical_spacing=0.1
        )
        
        # Add each plot to the appropriate subplot
        for i, plot_info in enumerate(plots):
            plot_fig = plot_info['fig']
            
            # Find position in layout
            position = None
            for r, row in enumerate(layout):
                if i in row:
                    position = (r+1, row.index(i)+1)
                    break
            
            if position is None:
                continue
                
            # Add traces from the plot to the dashboard
            for trace in plot_fig.data:
                fig.add_trace(trace, row=position[0], col=position[1])
            
            # Copy layout properties
            for axis in ['xaxis', 'yaxis']:
                axis_props = {}
                for prop in ['title', 'type', 'range']:
                    if prop in plot_fig.layout[axis]:
                        axis_props[prop] = plot_fig.layout[axis][prop]
                
                fig.update_xaxes(**axis_props, row=position[0], col=position[1])
                fig.update_yaxes(**axis_props, row=position[0], col=position[1])
        
        # Update overall layout
        fig.update_layout(
            title=title,
            height=figsize[1] * 100,
            width=figsize[0] * 100,
            template="plotly_white"
        )
        return fig
    else:
        # Determine layout if not provided
        if layout is None:
            rows = len(plots)
            cols = 1
            layout = [[i] for i in range(rows)]
        else:
            rows = len(layout)
            cols = max(len(row) for row in layout)
        
        # Create figure with subplots
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1 or cols == 1:
            axes = axes.reshape(-1, 1) if cols == 1 else axes.reshape(1, -1)
        
        # Add each plot to the appropriate subplot
        for i, plot_info in enumerate(plots):
            plot_fig = plot_info['fig']
            
            # Find position in layout
            position = None
            for r, row in enumerate(layout):
                if i in row:
                    position = (r, row.index(i))
                    break
            
            if position is None:
                continue
            
            # Copy the plot to the dashboard
            ax = axes[position[0], position[1]]
            
            # For matplotlib figures, we need to redraw the plot on the new axis
            # This is a simplified approach and might not work for all plot types
            if hasattr(plot_fig, 'axes') and len(plot_fig.axes) > 0:
                src_ax = plot_fig.axes[0]
                
                # Copy lines
                for line in src_ax.lines:
                    ax.plot(line.get_xdata(), line.get_ydata(), 
                           color=line.get_color(), linestyle=line.get_linestyle(),
                           marker=line.get_marker(), label=line.get_label())
                
                # Copy bars (simplified)
                for patch in src_ax.patches:
                    # This is a very simplified approach and won't work for all cases
                    ax.add_patch(patch)
                
                # Copy other elements as needed
                
                # Copy labels and title
                ax.set_xlabel(src_ax.get_xlabel())
                ax.set_ylabel(src_ax.get_ylabel())
                ax.set_title(plot_info.get('title', src_ax.get_title()))
                
                # Copy legend if it exists
                if src_ax.get_legend() is not None:
                    ax.legend()
            
            # Set title if not already set
            if not ax.get_title() and 'title' in plot_info:
                ax.set_title(plot_info['title'])
        
        # Set overall title
        fig.suptitle(title, fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
        return fig

def create_trend_analysis_dashboard(
    sales_data: pd.DataFrame,
    category_data: pd.DataFrame = None,
    customer_data: pd.DataFrame = None,
    title: str = 'Sales Trend Analysis Dashboard',
    use_plotly: bool = True
) -> Union[plt.Figure, go.Figure]:
    """
    Create a comprehensive trend analysis dashboard.
    
    Args:
        sales_data: DataFrame with sales time series data
        category_data: Optional DataFrame with category-specific data
        customer_data: Optional DataFrame with customer-specific data
        title: Dashboard title
        use_plotly: Whether to use plotly (True) or matplotlib (False)
        
    Returns:
        Dashboard figure object
    """
    plots = []
    
    # Total sales trend
    if 'date' in sales_data.columns and 'sales_amount' in sales_data.columns:
        sales_trend_fig = create_time_series_plot(
            sales_data,
            'date',
            'sales_amount',
            title='Total Sales Trend',
            xlabel='Date',
            ylabel='Sales Amount',
            use_plotly=use_plotly
        )
        plots.append({'fig': sales_trend_fig, 'title': 'Total Sales Trend'})
    
    # Category comparison
    if category_data is not None and 'category' in category_data.columns and 'sales_amount' in category_data.columns:
        category_fig = create_bar_chart(
            category_data,
            'category',
            'sales_amount',
            title='Sales by Category',
            xlabel='Category',
            ylabel='Sales Amount',
            use_plotly=use_plotly
        )
        plots.append({'fig': category_fig, 'title': 'Sales by Category'})
    
    # Customer segment analysis
    if customer_data is not None and 'segment' in customer_data.columns and 'customer_count' in customer_data.columns:
        segment_fig = create_pie_chart(
            customer_data,
            'customer_count',
            'segment',
            title='Customer Segments',
            use_plotly=use_plotly
        )
        plots.append({'fig': segment_fig, 'title': 'Customer Segments'})
    
    # Create a 2x2 dashboard layout
    layout = [[0, 1], [2, 3]] if len(plots) >= 4 else [[0, 1], [2, 2]] if len(plots) == 3 else [[0], [1]] if len(plots) == 2 else [[0]]
    
    # Add a placeholder plot if needed
    while len(plots) < len([item for sublist in layout for item in sublist]):
        if use_plotly:
            placeholder_fig = go.Figure()
            placeholder_fig.update_layout(
                title="No Data Available",
                annotations=[dict(
                    text="No data available for this panel",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False
                )]
            )
        else:
            placeholder_fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No data available for this panel", 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes)
            ax.set_title("No Data Available")
            ax.axis('off')
        
        plots.append({'fig': placeholder_fig, 'title': 'No Data Available'})
    
    # Create the dashboard
    dashboard = create_dashboard(
        plots,
        title=title,
        layout=layout,
        figsize=(14, 10),
        use_plotly=use_plotly
    )
    
    return dashboard

# Initialize visualization environment when module is imported
setup_visualization_env()
