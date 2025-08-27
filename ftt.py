import numpy as np
import matplotlib.pyplot as plt

# 🌌 Synthetic SFIT Scalar Field Generation
def generate_scalar_field(N=512, decay_zone=(100, 300), clamp_zone=(350, 450)):
    x = np.linspace(0, 2 * np.pi, N)
    y = np.linspace(0, 2 * np.pi, N)
    X, Y = np.meshgrid(x, y)
    
    # Core scalar field: oscillatory + localized decay + curvature clamping
    field = np.sin(5 * X) * np.cos(7 * Y)
    field[decay_zone[0]:decay_zone[1], :] *= np.exp(-((X[decay_zone[0]:decay_zone[1], :] - np.pi)**2 + 
                                                       (Y[decay_zone[0]:decay_zone[1], :] - np.pi)**2))
    field[clamp_zone[0]:clamp_zone[1], :] += 0.3 * np.sin(20 * X[clamp_zone[0]:clamp_zone[1], :])
    
    return field

# ⚛️ FFT Magnitude Map
def plot_fft_magnitude(field):
    fft_result = np.fft.fftshift(np.fft.fft2(field))
    magnitude = np.abs(fft_result)

    plt.figure(figsize=(8, 6))
    plt.imshow(magnitude, extent=(-1, 1, -1, 1), cmap='plasma')
    plt.title("SFIT-Inspired FFT Magnitude Map")
    plt.xlabel("kx")
    plt.ylabel("ky")
    plt.colorbar(label="Magnitude")
    plt.tight_layout()
    plt.show()

# 🌀 Execution
scalar_field = generate_scalar_field()
plot_fft_magnitude(scalar_field)
