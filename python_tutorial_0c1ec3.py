# Paint with Sound: Mapping Audio Frequencies to Brush Strokes

# Learning Objective:
# This tutorial will teach you how to create a simple visualizer that
# "paints" on a graphical window by mapping the frequencies of an audio
# input to the size and color of brush strokes. This will demonstrate
# fundamental concepts of:
# 1. Audio processing (basic frequency detection).
# 2. Real-time graphics rendering.
# 3. Mapping one data type (frequency) to another (visual properties).
# 4. Using external libraries for audio and graphics.

# We'll use the following libraries:
# - `pygame`: A popular library for creating games and multimedia applications.
#   It's excellent for handling graphics, input, and sound.
# - `numpy`: A fundamental package for scientific computing with Python.
#   We'll use it for numerical operations, especially for analyzing audio data.
# - `sounddevice`: A library for playing and recording audio with NumPy.
#   It allows us to capture audio input from your microphone.

# Install these libraries if you haven't already:
# pip install pygame numpy sounddevice

import pygame
import numpy as np
import sounddevice as sd
import sys

# --- Configuration Constants ---
# These are values we can easily change to modify the behavior of our program.

# Graphics Window Dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60  # Frames per second: how often the screen updates.

# Audio Input Settings
SAMPLE_RATE = 44100  # Samples per second. Standard for audio.
BUFFER_SIZE = 1024  # Number of audio samples to process at once. Larger = more detail, but higher latency.
DEVICE = None  # Use default microphone. Can specify a device index if you have multiple.

# Visual Mapping Parameters
# These control how audio frequencies affect the brush.

# Frequency Range Mapping:
# We'll focus on a specific range of human-audible frequencies.
# Lower frequencies will affect one aspect, higher frequencies another.
MIN_FREQ = 100  # Lowest frequency we'll consider for mapping (e.g., bass sounds)
MAX_FREQ = 5000 # Highest frequency we'll consider (e.g., high-pitched sounds)

# Brush Size Mapping:
# How frequency influences the size of the brush stroke.
MIN_BRUSH_SIZE = 5  # Smallest brush size
MAX_BRUSH_SIZE = 50 # Largest brush size

# Brush Color Mapping:
# How frequency influences the color. We'll map frequency to Hue (H) in HSL color space.
# Hue ranges from 0 to 360 degrees.
MIN_HUE = 0    # Reddish hues
MAX_HUE = 240  # Bluish hues

# --- Global Variables ---
# These variables will be shared and updated across different parts of the program.

# This will store the current audio data buffer.
audio_data_buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)

# This is a list to store our brush strokes. Each stroke will be a dictionary
# containing its position, size, and color.
brush_strokes = []

# --- Audio Processing Function ---
# This function is called by `sounddevice` whenever new audio data is available.
def audio_callback(indata, frames, time, status):
    """
    This function is the heart of our audio input.
    It receives new audio data (`indata`) and updates our global `audio_data_buffer`.
    """
    if status:
        print(status, file=sys.stderr) # Print any errors or warnings from sounddevice.

    # `indata` is a NumPy array containing the audio samples.
    # For stereo, it might have shape (frames, 2). We'll just take the first channel.
    # If it's mono, it will have shape (frames,).
    global audio_data_buffer
    audio_data_buffer = indata[:, 0] if indata.ndim > 1 else indata

# --- Frequency Analysis Function ---
# This function takes audio data and finds the dominant frequencies.
def get_dominant_frequencies(audio_samples, sample_rate, buffer_size):
    """
    Analyzes a buffer of audio samples to find dominant frequencies using FFT.
    FFT (Fast Fourier Transform) converts a time-domain signal into its frequency components.
    """
    # Apply a window function to reduce spectral leakage.
    # Hann window is a good general-purpose choice.
    window = np.hanning(len(audio_samples))
    windowed_samples = audio_samples * window

    # Perform the Fast Fourier Transform (FFT).
    # The result `fft_result` contains complex numbers representing amplitude and phase
    # for each frequency bin.
    fft_result = np.fft.fft(windowed_samples)

    # Calculate the frequencies corresponding to each bin in the FFT result.
    # `np.fft.fftfreq` gives us the frequency bins, considering the sample rate and buffer size.
    frequencies = np.fft.fftfreq(buffer_size, 1.0 / sample_rate)

    # We're only interested in positive frequencies (the first half of the FFT result).
    # The FFT is symmetric for real-valued input.
    positive_freq_mask = frequencies >= 0
    positive_frequencies = frequencies[positive_freq_mask]
    fft_magnitude = np.abs(fft_result[positive_freq_mask])

    # Filter out frequencies outside our defined range to focus our mapping.
    freq_range_mask = (positive_frequencies >= MIN_FREQ) & (positive_frequencies <= MAX_FREQ)
    filtered_frequencies = positive_frequencies[freq_range_mask]
    filtered_magnitudes = fft_magnitude[freq_range_mask]

    # Find the index of the frequency with the maximum magnitude within our range.
    if len(filtered_magnitudes) == 0:
        return None # No frequencies found in the desired range.

    dominant_freq_index = np.argmax(filtered_magnitudes)
    dominant_frequency = filtered_frequencies[dominant_freq_index]

    return dominant_frequency

# --- Visual Mapping Functions ---
# These functions translate frequency data into visual properties.

def map_freq_to_brush_size(frequency):
    """
    Maps a given frequency to a brush size within the defined MIN_BRUSH_SIZE and MAX_BRUSH_SIZE.
    We'll use a linear mapping: lower frequencies get smaller brushes, higher frequencies get larger.
    """
    if frequency is None:
        return MIN_BRUSH_SIZE # Default size if no frequency is detected.

    # Normalize the frequency to a 0-1 range based on our MIN_FREQ and MAX_FREQ.
    normalized_freq = (frequency - MIN_FREQ) / (MAX_FREQ - MIN_FREQ)

    # Clamp the normalized frequency to ensure it stays within 0 and 1.
    # This prevents out-of-bounds values if a frequency is slightly outside our range.
    normalized_freq = max(0, min(1, normalized_freq))

    # Map the normalized frequency to the brush size range.
    brush_size = MIN_BRUSH_SIZE + normalized_freq * (MAX_BRUSH_SIZE - MIN_BRUSH_SIZE)
    return int(brush_size) # Brush size needs to be an integer.

def map_freq_to_brush_color(frequency):
    """
    Maps a given frequency to a color (specifically, the Hue component).
    We use HSL (Hue, Saturation, Lightness) color model. Hue determines the color itself (red, green, blue, etc.).
    Lower frequencies will map to one end of the hue spectrum (e.g., red), and higher frequencies to the other (e.g., blue).
    """
    if frequency is None:
        return (255, 255, 255) # Default to white if no frequency is detected.

    # Normalize the frequency to a 0-1 range.
    normalized_freq = (frequency - MIN_FREQ) / (MAX_FREQ - MIN_FREQ)
    normalized_freq = max(0, min(1, normalized_freq))

    # Map the normalized frequency to the hue range.
    hue = MIN_HUE + normalized_freq * (MAX_HUE - MIN_HUE)

    # Convert HSL (with full saturation and lightness for simplicity) to RGB.
    # Pygame works with RGB tuples (0-255 for each color channel).
    # A common way to convert HSL to RGB is using a helper function or library.
    # For simplicity, we'll use a basic conversion logic here.
    # This conversion can be a bit complex, so we'll use a simplified approximation.

    # A simple approach for a visual mapping:
    # We can directly map to R, G, B components too, but Hue gives a smoother color transition.
    # Let's directly map to RGB for a simpler demonstration.
    # Low freq -> Red (255, 0, 0)
    # Mid freq -> Green (0, 255, 0)
    # High freq -> Blue (0, 0, 255)

    # We can divide the frequency range into segments for R, G, B.
    # Let's use the normalized frequency (0-1) to blend between R, G, B.
    # Example:
    # 0.0 - 0.33 -> Reddish
    # 0.33 - 0.66 -> Greenish
    # 0.66 - 1.0 -> Blueish

    r, g, b = 0, 0, 0
    if normalized_freq < 0.33:
        # Transition from Red to Yellow
        r = 255
        g = int(255 * (normalized_freq / 0.33))
        b = 0
    elif normalized_freq < 0.66:
        # Transition from Yellow to Cyan
        r = int(255 * (1 - (normalized_freq - 0.33) / 0.33))
        g = 255
        b = 0
    else:
        # Transition from Cyan to Blue
        r = 0
        g = int(255 * (1 - (normalized_freq - 0.66) / 0.33))
        b = 255

    # Ensure RGB values are within the valid 0-255 range.
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))

    return (r, g, b)

# --- Pygame Initialization ---
def initialize_pygame():
    """Initializes Pygame and sets up the display window."""
    pygame.init()  # Initialize all Pygame modules.
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT)) # Create the window.
    pygame.display.set_caption("Paint with Sound") # Set the window title.
    clock = pygame.time.Clock()  # Create a clock object to control frame rate.
    return screen, clock

# --- Main Application Logic ---
def run_paint_with_sound():
    """Sets up audio stream and runs the main Pygame loop."""

    # --- Set up the Audio Stream ---
    # We create an audio input stream using `sounddevice`.
    # `sd.InputStream` will continuously call our `audio_callback` function
    # whenever new audio data is ready.
    try:
        stream = sd.InputStream(
            device=DEVICE,
            channels=1,  # We only need one channel (mono).
            samplerate=SAMPLE_RATE,
            blocksize=BUFFER_SIZE,
            callback=audio_callback,
            dtype='float32' # Data type for audio samples.
        )
        stream.start() # Start the audio stream.
        print("Audio stream started. Speak into your microphone!")
    except Exception as e:
        print(f"Error starting audio stream: {e}")
        print("Please ensure your microphone is connected and working.")
        print("You might need to select a specific device if you have multiple.")
        pygame.quit() # Clean up Pygame resources.
        sys.exit() # Exit the program.

    # --- Initialize Pygame Graphics ---
    screen, clock = initialize_pygame()
    running = True

    # --- Main Pygame Loop ---
    while running:
        # --- Event Handling ---
        # We process events (like closing the window, key presses, etc.).
        for event in pygame.event.get():
            if event.type == pygame.QUIT: # If the user clicks the close button.
                running = False # Set running to False to exit the loop.

        # --- Audio Analysis and Visual Mapping ---
        # Get the latest audio data from the buffer.
        current_audio_samples = audio_data_buffer.copy() # Make a copy to avoid race conditions.

        # Analyze the audio data to find the dominant frequency.
        dominant_freq = get_dominant_frequencies(
            current_audio_samples,
            SAMPLE_RATE,
            BUFFER_SIZE
        )

        # Map the dominant frequency to visual properties (size and color).
        brush_size = map_freq_to_brush_size(dominant_freq)
        brush_color = map_freq_to_brush_color(dominant_freq)

        # --- Create a New Brush Stroke ---
        # If we detected a sound (dominant_freq is not None), add a new stroke.
        if dominant_freq is not None:
            # Get a random position for the new brush stroke within the window.
            pos_x = np.random.randint(0, SCREEN_WIDTH)
            pos_y = np.random.randint(0, SCREEN_HEIGHT)

            # Store the stroke's properties.
            new_stroke = {
                'pos': (pos_x, pos_y),
                'size': brush_size,
                'color': brush_color,
                'lifetime': 60 # How many frames this stroke will last (e.g., 1 second if FPS=60)
            }
            brush_strokes.append(new_stroke)

        # --- Drawing ---
        # 1. Clear the screen each frame to avoid drawing over old strokes continuously.
        #    We fill it with black.
        screen.fill((0, 0, 0))

        # 2. Draw existing brush strokes.
        #    We iterate through our `brush_strokes` list.
        for stroke in brush_strokes[:]: # Use a slice [:] to iterate over a copy, allowing modification.
            # Draw a circle representing the brush stroke.
            # `pygame.draw.circle(surface, color, center_position, radius)`
            # The radius is half the size.
            pygame.draw.circle(screen, stroke['color'], stroke['pos'], stroke['size'] // 2)

            # Decrease the stroke's lifetime.
            stroke['lifetime'] -= 1

            # If the stroke has expired, remove it from the list.
            if stroke['lifetime'] <= 0:
                brush_strokes.remove(stroke)

        # --- Update the Display ---
        # `pygame.display.flip()` updates the entire screen to show what we've drawn.
        pygame.display.flip()

        # --- Frame Rate Control ---
        # `clock.tick(FPS)` pauses the loop to ensure it runs at most `FPS` times per second.
        # This makes the animation smooth and consistent.
        clock.tick(FPS)

    # --- Cleanup ---
    # When the loop finishes, we stop the audio stream and quit Pygame.
    stream.stop()
    stream.close()
    pygame.quit()
    print("Program finished. Audio stream stopped.")

# --- Example Usage ---
if __name__ == "__main__":
    # This block ensures `run_paint_with_sound()` is only called when the script
    # is executed directly (not when imported as a module).
    run_paint_with_sound()