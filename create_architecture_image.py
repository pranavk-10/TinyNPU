import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# TinyNPU Neural Network Visualization
# ============================================================
#
# Architecture:
#
#       21 → 14 → 7 → 1
#
# The image will be saved as:
#
#       docs/tinynpu_architecture.png
#
# ============================================================


# ============================================================
# Create docs directory
# ============================================================

docs_dir = Path("docs")

docs_dir.mkdir(
    exist_ok=True
)


# ============================================================
# Network Configuration
# ============================================================

layers = [
    21,
    14,
    7,
    1
]


layer_names = [
    "Input Layer\n21 Features",
    "Hidden Layer 1\n14 Neurons\nReLU",
    "Hidden Layer 2\n7 Neurons\nReLU",
    "Output Layer\n1 Neuron\nSigmoid"
]


# ============================================================
# Create Figure
# ============================================================

fig, ax = plt.subplots(
    figsize=(16, 10)
)


# ============================================================
# Positions
# ============================================================

x_positions = [
    0,
    1,
    2,
    3
]


# ============================================================
# Draw Neurons
# ============================================================

neuron_positions = []


for layer_index, neuron_count in enumerate(layers):

    positions = []


    # Don't draw all 21 neurons individually.
    # Show representative neurons for large layers.

    if neuron_count == 21:

        visible_neurons = 7

    elif neuron_count == 14:

        visible_neurons = 7

    elif neuron_count == 7:

        visible_neurons = 7

    else:

        visible_neurons = 1


    # Vertical positions

    y_positions = []

    for i in range(
        visible_neurons
    ):

        y = (
            visible_neurons - 1
            - i
        ) / (
            visible_neurons - 1
        ) if visible_neurons > 1 else 0.5

        y_positions.append(y)


    positions = [
        (
            x_positions[layer_index],
            y
        )
        for y in y_positions
    ]


    neuron_positions.append(
        positions
    )


# ============================================================
# Draw Connections
# ============================================================

for layer_index in range(
    len(layers) - 1
):

    current_layer = neuron_positions[
        layer_index
    ]

    next_layer = neuron_positions[
        layer_index + 1
    ]


    for x1, y1 in current_layer:

        for x2, y2 in next_layer:

            ax.plot(
                [x1, x2],
                [y1, y2],
                linewidth=0.7,
                alpha=0.35
            )


# ============================================================
# Draw Neurons
# ============================================================

for layer_index, positions in enumerate(
    neuron_positions
):

    for x, y in positions:

        ax.scatter(
            x,
            y,
            s=1000,
            edgecolors="black",
            linewidths=1.5,
            zorder=5
        )


# ============================================================
# Add Layer Titles
# ============================================================

for i, name in enumerate(
    layer_names
):

    ax.text(
        x_positions[i],
        1.12,
        name,
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold"
    )


# ============================================================
# Add Feature Labels
# ============================================================

ax.text(
    0,
    -0.08,
    "X₁",
    ha="center",
    fontsize=11
)

ax.text(
    0,
    0.92,
    "X₃",
    ha="center",
    fontsize=11
)

ax.text(
    0,
    0.48,
    "⋮",
    ha="center",
    fontsize=20
)

ax.text(
    0,
    0.00,
    "X₂₁",
    ha="center",
    fontsize=11
)


# ============================================================
# Add Hidden Layer Labels
# ============================================================

ax.text(
    1,
    0.45,
    "⋮",
    ha="center",
    fontsize=20
)

ax.text(
    2,
    0.45,
    "⋮",
    ha="center",
    fontsize=20
)


# ============================================================
# Add Weight Information
# ============================================================

ax.text(
    0.5,
    -0.15,
    "294 weights\n21 × 14",
    ha="center",
    fontsize=11
)

ax.text(
    1.5,
    -0.15,
    "98 weights\n14 × 7",
    ha="center",
    fontsize=11
)

ax.text(
    2.5,
    -0.15,
    "7 weights\n7 × 1",
    ha="center",
    fontsize=11
)


# ============================================================
# Add Activation Labels
# ============================================================

ax.text(
    1,
    -0.30,
    "ReLU",
    ha="center",
    fontsize=11
)

ax.text(
    2,
    -0.30,
    "ReLU",
    ha="center",
    fontsize=11
)

ax.text(
    3,
    -0.30,
    "Probability 0–1",
    ha="center",
    fontsize=11
)


# ============================================================
# Title
# ============================================================

ax.set_title(
    "TinyNPU — Neural Network Architecture",
    fontsize=22,
    fontweight="bold",
    pad=30
)


# ============================================================
# Subtitle
# ============================================================

ax.text(
    1.5,
    1.30,
    "21 → 14 → 7 → 1 | Quantized Neural Network Inference Accelerator",
    ha="center",
    fontsize=12
)


# ============================================================
# Remove Axes
# ============================================================

ax.set_xlim(
    -0.5,
    3.5
)

ax.set_ylim(
    -0.45,
    1.45
)

ax.axis(
    "off"
)


# ============================================================
# Save Image
# ============================================================

output_path = (
    docs_dir /
    "tinynpu_architecture.png"
)


plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# Print Result
# ============================================================

print()
print("============================================")
print("TinyNPU Architecture Image Created")
print("============================================")

print()
print("Saved to:")

print(
    output_path
)

print()
print("Your project now contains:")
print()
print("docs/")
print("└── tinynpu_architecture.png")

print()
print("============================================")
print("Done!")
print("============================================")