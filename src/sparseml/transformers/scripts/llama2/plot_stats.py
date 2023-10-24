import matplotlib.pyplot as plt
import matplotlib.patches as patches

width = 0.1
linewidth = 1

def plot_statistics(stats_arrays, name):
    fig, ax = plt.subplots()
    for x, stats in enumerate(stats_arrays):
        # Plot a rectangle using a Patch object
        rectangle = patches.Rectangle(
            (x-0.5*width, stats["p25"]), 
            width=width, 
            height=stats["p75"] - stats["p25"], 
            linewidth=linewidth, 
            edgecolor='r', 
            facecolor='none',
        )
    
        # Add the rectangle to the plot
        ax.add_patch(rectangle)
    
        # Add a horizontal line at y_median with the same width as the rectangle
        ax.hlines(stats["median"], x-0.5*width, x+0.5*width, color='b', linewidth=linewidth)

        # Add a vertical line through the center of the rectangle
        ax.vlines(
            x, 
            ymin=stats["min_val"], 
            ymax=stats["max_val"], 
            color='g', 
            linewidth=linewidth,
        )

        # Add a horizontal line at y_median with the same width as the rectangle
        ax.hlines(
            stats["min_val"], 
            x-0.5*width, 
            x+0.5*width, 
            color='b', 
            linewidth=linewidth,
        )
    
        # Add a horizontal line at y_median with the same width as the rectangle
        ax.hlines(
            stats["max_val"], 
            x-0.5*width, 
            x+0.5*width, 
            color='b', 
            linewidth=linewidth,
        )

    ax.set_xlim(-1, len(stats_arrays))
    plt.savefig(f"{name}.png")
    plt.show()
    plt.close("all")
