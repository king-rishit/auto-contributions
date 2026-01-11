# --- Learning Objective: Generate a unique, procedurally generated 2D world map using Perlin noise ---
# This tutorial will guide you through creating a basic 2D world map by leveraging the power of Perlin noise.
# Perlin noise is a gradient noise function developed by Ken Perlin, widely used in computer graphics
# for generating natural-looking textures and patterns. We'll use it to simulate terrain variations,
# creating hills, valleys, and flat areas.

# --- Libraries Needed ---
# We'll need 'numpy' for efficient array manipulation and 'noise' for the Perlin noise generation.
# If you don't have them installed, run:
# pip install numpy noise

import numpy as np  # For numerical operations, especially array handling.
import noise  # The Perlin noise library.
import matplotlib.pyplot as plt  # For visualizing the generated map.

# --- Configuration Parameters ---
# These variables control the characteristics of our generated map.
MAP_WIDTH = 256  # The width of our map in pixels.
MAP_HEIGHT = 256  # The height of our map in pixels.
SCALE = 100.0  # Controls the 'zoom' level of the noise. Higher values mean more zoomed in, more detail.
OCTAVES = 6  # The number of Perlin noise layers. More octaves add more detail and complexity.
PERSISTENCE = 0.5  # Controls how much each octave contributes to the overall noise. Lower values make features smoother.
LACUNARITY = 2.0  # Controls how much the frequency increases for each octave. Higher values add finer details.
SEED = None  # Set to an integer for reproducible maps, or None for a new random map each time.
              # Using a SEED allows us to generate the exact same map multiple times.

def generate_world_map(width, height, scale, octaves, persistence, lacunarity, seed=None):
    """
    Generates a 2D world map using Perlin noise.

    Args:
        width (int): The desired width of the map.
        height (int): The desired height of the map.
        scale (float): The scaling factor for the Perlin noise. Affects the level of detail.
        octaves (int): The number of noise layers (octaves) to combine. More octaves = more detail.
        persistence (float): How much each octave contributes to the overall noise.
        lacunarity (float): How much the frequency increases for each octave.
        seed (int, optional): A seed for the random number generator. If None, a new map is generated each time.

    Returns:
        numpy.ndarray: A 2D array representing the generated world map, with values
                       ranging from approximately -1.0 to 1.0.
    """
    # Create an empty numpy array to store our map data.
    # We'll fill this with noise values.
    world_map = np.zeros((height, width))

    # Iterate over each pixel (x, y) in our map.
    for y in range(height):
        for x in range(width):
            # Calculate the noise value for the current (x, y) coordinates.
            # We divide by 'scale' to control the 'zoom' of the noise.
            # 'octaves', 'persistence', and 'lacunarity' fine-tune the noise's appearance.
            # 'seed' ensures reproducibility if provided.
            noise_value = noise.pnoise2(x / scale,
                                        y / scale,
                                        octaves=octaves,
                                        persistence=persistence,
                                        lacunarity=lacunarity,
                                        base=seed) # 'base' is the parameter for seed in 'noise' library.

            # Assign the calculated noise value to the corresponding pixel in our map.
            # The values from pnoise2 typically range from -1.0 to 1.0.
            world_map[y][x] = noise_value

    return world_map

def map_noise_to_terrain(noise_map):
    """
    Maps raw noise values to terrain types (e.g., water, land, mountains).

    This is a simplified mapping. In a real game, you'd have more nuanced terrain types.
    We're using simple thresholds to divide the noise range.

    Args:
        noise_map (numpy.ndarray): The 2D array of noise values.

    Returns:
        numpy.ndarray: A 2D array representing terrain types, where:
                       0: Water
                       1: Land
                       2: Mountains
    """
    # Normalize the noise map to a range of 0.0 to 1.0 for easier thresholding.
    # The original noise is roughly -1 to 1. Adding 1 shifts it to 0 to 2. Dividing by 2 makes it 0 to 1.
    normalized_map = (noise_map + 1) / 2

    # Define thresholds for different terrain types.
    # These values are empirical and can be adjusted.
    WATER_THRESHOLD = 0.3  # Anything below this is considered water.
    LAND_THRESHOLD = 0.7   # Anything between WATER_THRESHOLD and LAND_THRESHOLD is land.
                           # Anything above LAND_THRESHOLD is mountains.

    # Create a new array to store the terrain types.
    terrain_map = np.zeros_like(normalized_map, dtype=int)

    # Apply the thresholds.
    terrain_map[normalized_map < WATER_THRESHOLD] = 0  # Water
    terrain_map[(normalized_map >= WATER_THRESHOLD) & (normalized_map < LAND_THRESHOLD)] = 1  # Land
    terrain_map[normalized_map >= LAND_THRESHOLD] = 2  # Mountains

    return terrain_map

# --- Example Usage ---
if __name__ == "__main__":
    print("Generating world map...")

    # Generate the raw Perlin noise map.
    # You can change the SEED to generate a different map.
    # For example, try SEED = 42 for a specific reproducible map.
    raw_map = generate_world_map(MAP_WIDTH, MAP_HEIGHT,
                                 SCALE, OCTAVES, PERSISTENCE, LACUNARITY,
                                 seed=SEED)

    # Convert the raw noise into distinct terrain types.
    terrain_data = map_noise_to_terrain(raw_map)

    print("Map generation complete. Displaying map...")

    # --- Visualization ---
    # We'll use matplotlib to display the generated map.
    # We'll create a colormap to represent our terrain types.

    # Define a colormap: Blue for water, Green for land, Brown for mountains.
    cmap = plt.cm.colors.ListedColormap(['#0000FF', '#00FF00', '#A0522D']) # Blue, Green, Brown

    # Display the terrain map.
    plt.figure(figsize=(8, 8)) # Set the figure size for better viewing.
    plt.imshow(terrain_data, cmap=cmap, interpolation='nearest')
    plt.title("Procedurally Generated World Map")
    plt.colorbar(label="Terrain Type (0: Water, 1: Land, 2: Mountains)")
    plt.show()

    print("\n--- Explanation of Key Concepts ---")
    print("Perlin Noise: A gradient noise function that creates natural-looking patterns.")
    print("SCALE: Controls the 'zoom' of the noise. Higher values mean more detail/smaller features.")
    print("OCTAVES: Multiple layers of noise summed together. More octaves add finer details and complexity.")
    print("PERSISTENCE: Determines how much influence each subsequent octave has. Lower persistence = smoother features.")
    print("LACUNARITY: Controls the frequency increase of each octave. Higher lacunarity = finer details.")
    print("SEED: Used to generate reproducible maps. If None, a new random map is created.")
    print("\nTo experiment:")
    print("- Change the SEED value (e.g., to 42) to see a reproducible map.")
    print("- Adjust SCALE, OCTAVES, PERSISTENCE, and LACUNARITY to alter the map's appearance.")
    print("- Modify the WATER_THRESHOLD and LAND_THRESHOLD in map_noise_to_terrain() to change terrain distribution.")