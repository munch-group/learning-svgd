# file: vscode_theme_checker.py
import os
from IPython.core.magic import register_line_magic
from IPython import get_ipython
from IPython.display import display, HTML
import re
import platform
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams

import matplotlib._pylab_helpers as pylab_helpers
import gc

html_style = '''
<style>
/*  make widget background transparent to match vscode theme: */
.cell-output-ipywidget-background {
   background-color: transparent !important;
}
.jp-OutputArea-output {
   background-color: transparent;
}  

/* make tqdm progress bar less intrusive: */
.widget-label,
.widget-html,
.widget-button,
.widget-dropdown,
.widget-text,
.widget-textarea {
    color: #CCCCCC !important;  /* Set to desired color */
    font-size: 10px !important;      /* Font size */

}
div.widget-html-content > progress { /* Outer container */
    height: 10px !important;

}
div.jp-OutputArea-output td.output_html { /* tqdm HTML output area */
    height: 5px !important;
}
div.progress { /* Jupyter Notebook (classic) */
    height: 5px !important;
    min-height: 5px !important;
    margin-top: 12px !important;    
}
div.progress-bar { /* Inner bar */
    height: 5px !important;
    min-height: 5px !important;
    line-height: 5px !important;
    color: 'white' !important;
    padding-bottom: 0px !important;    
}
</style>
'''
display(HTML(html_style))

def get_vscode_user_settings_dir():
    system = platform.system()
    home = Path.home()

    if system == "Darwin":  # macOS
        return home / "Library" / "Application Support" / "Code" / "User"
    elif system == "Windows":
        return Path(os.getenv("APPDATA")) / "Code" / "User"
    elif system == "Linux":
        return home / ".config" / "Code" / "User"
    else:
        raise NotImplementedError(f"Unsupported OS: {system}")

_last_theme = None

def set_plot_style(theme: str):
    """Update rcParams and existing figures based on the theme name."""

    if theme is None: # or "dark" in theme.lower():
        # vscode starts in default theme (dark)
        plt.style.use('dark_background')        
        rcParams.update({
            'figure.facecolor': '#1F1F1F',
            'axes.facecolor': '#1F1F1F',
            'text.color': '#CCCCCC',
        })
    else:
        plt.style.use('default')        
        rcParams.update({
            'figure.facecolor': '#FFFFFF',
            'axes.facecolor': '#FFFFFF',
            'text.color': '#3B3B3B',
        })

    # Update existing open figures
    for manager in matplotlib._pylab_helpers.Gcf.get_all_fig_managers():
        print(manager)
        fig = manager.canvas.figure
        fig.set_facecolor(rcParams["figure.facecolor"])
        for ax in fig.axes:
            ax.set_facecolor(rcParams["axes.facecolor"])
            ax.title.set_color(rcParams["text.color"])
            ax.xaxis.label.set_color(rcParams["axes.labelcolor"])
            ax.yaxis.label.set_color(rcParams["axes.labelcolor"])
            ax.tick_params(colors=rcParams["xtick.color"])

        fig.canvas.draw_idle()

_last_theme = 'default'

def check_and_apply_theme():
    global _last_theme
    # theme = os.environ.get("VSCODE_THEME", "Default Light Modern")

    if not any(x in os.environ for x in ["VSCODE_PID", "VSCODE_CWD", "JPY_PARENT_PID"]):
        return

    settings_path = get_vscode_user_settings_dir() / "settings.json"
    with open(settings_path, 'r') as f:
        theme = None
        for line in f:
            if 'workbench.colorTheme' in line:
                match = re.search(r'"workbench\.colorTheme"\s*:\s*"([^"]+)"', line)
                theme = match.group(1)
                break

    if theme != _last_theme:
        _last_theme = theme
        set_plot_style(theme)
        # force_figure_refresh()
        
def load_ipython_extension(ipython):
    ipython.events.register("pre_run_cell", lambda _: check_and_apply_theme())

def unload_ipython_extension(ipython):
    ipython.events.unregister("pre_run_cell", lambda _: check_and_apply_theme())

# def clear_stale_figures():
#     """Clear all stale matplotlib figures to force updates"""
#     # Get all active figure managers
#     fig_managers = pylab_helpers.Gcf.get_all_fig_managers()
    
#     if fig_managers:
#         print(f"Clearing {len(fig_managers)} active figures...")
        
#         # Close all figures
#         for manager in fig_managers:
#             plt.close(manager.canvas.figure)
        
#         # Force garbage collection
#         gc.collect()
        
#         print("All figures cleared and memory freed")
#     else:
#         print("No active figures to clear")

# def get_figure_info():
#     """Get information about currently active figures"""
#     fig_managers = pylab_helpers.Gcf.get_all_fig_managers()
    
#     print(f"Active figures: {len(fig_managers)}")
#     for i, manager in enumerate(fig_managers):
#         fig = manager.canvas.figure
#         print(f"  Figure {i+1}: {fig.number} - {fig.get_size_inches()}")
    
#     return fig_managers

# def force_figure_refresh():
#     """Force all figures to refresh their display"""
#     fig_managers = pylab_helpers.Gcf.get_all_fig_managers()
    
#     for manager in fig_managers:
#         try:
#             manager.canvas.draw_idle()
#             manager.canvas.flush_events()
#         except Exception as e:
#             print(f"Warning: Could not refresh figure {manager.canvas.figure.number}: {e}")
