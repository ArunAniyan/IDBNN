import os
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageTk

# --------------------------Noise Injection Module ----------------------------
# This module introduces different kinds of noise in images
# author: Ninan Sajeeth Philip, April 14 2025
# ---------------------------------------------------------------------------------------


class NoiseInjectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Noise Injection Module")
        self.root.geometry("1000x800")

        # Variables
        self.input_folder = tk.StringVar()
        self.output_folder = tk.StringVar()
        self.noise_types = [
            "Gaussian",
            "Salt & Pepper",
            "Poisson",
            "Speckle",
            "Rayleigh",
            "Exponential",
            "Uniform",
        ]

        # Initialize dictionaries
        self.slider_widgets = {}  # Stores slider widgets
        self.noise_values = {}  # Stores current noise values
        self.scale_vars = {}  # Stores scale type variables

        self.preview_image = None
        self.processed_image = None
        self.tk_images = []  # To prevent garbage collection

        # Create UI
        self.create_widgets()

        # Initialize default values
        for noise in self.noise_types:
            self.noise_values[noise] = 0.0

    def create_widgets(self):
        # Frame for folder selection
        folder_frame = ttk.LabelFrame(self.root, text="Folder Selection", padding=10)
        folder_frame.pack(fill=tk.X, padx=10, pady=5)

        # Input folder
        ttk.Label(folder_frame, text="Input Folder:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(folder_frame, textvariable=self.input_folder, width=50).grid(
            row=0, column=1, padx=5
        )
        ttk.Button(folder_frame, text="Browse", command=self.browse_input).grid(
            row=0, column=2
        )

        # Output folder
        ttk.Label(folder_frame, text="Output Folder:").grid(
            row=1, column=0, sticky=tk.W
        )
        ttk.Entry(folder_frame, textvariable=self.output_folder, width=50).grid(
            row=1, column=1, padx=5
        )
        ttk.Button(folder_frame, text="Browse", command=self.browse_output).grid(
            row=1, column=2
        )

        # Noise controls frame
        noise_frame = ttk.LabelFrame(self.root, text="Noise Controls", padding=10)
        noise_frame.pack(fill=tk.X, padx=10, pady=5)

        # Configure grid weights to make sliders expand
        noise_frame.grid_columnconfigure(3, weight=1)

        # Create sliders for each noise type
        for i, noise in enumerate(self.noise_types):
            # Scale type selection
            scale_var = tk.StringVar(value="Linear")
            self.scale_vars[noise] = scale_var
            ttk.Radiobutton(
                noise_frame,
                text="Linear",
                variable=scale_var,
                value="Linear",
                command=lambda n=noise: self.update_slider_scale(n),
            ).grid(row=i, column=0, padx=5, sticky=tk.W)
            ttk.Radiobutton(
                noise_frame,
                text="Log",
                variable=scale_var,
                value="Log",
                command=lambda n=noise: self.update_slider_scale(n),
            ).grid(row=i, column=1, padx=5, sticky=tk.W)

            # Noise slider
            ttk.Label(noise_frame, text=f"{noise} Noise:").grid(
                row=i, column=2, sticky=tk.W
            )
            slider = ttk.Scale(
                noise_frame,
                from_=0,
                to=1,
                orient=tk.HORIZONTAL,
                command=lambda val, n=noise: self.on_slider_change(val, n),
            )
            slider.grid(row=i, column=3, sticky=tk.EW, padx=5)
            self.slider_widgets[noise] = slider  # Store the slider widget

            # Value display
            value_label = ttk.Label(noise_frame, text="0.00")
            value_label.grid(row=i, column=4, padx=5)
            setattr(
                self,
                f"{noise.lower().replace(' ', '_').replace('&', '')}_value",
                value_label,
            )

        # Preview frame
        preview_frame = ttk.LabelFrame(self.root, text="Preview", padding=10)
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Image display using Tkinter
        self.image_frame = ttk.Frame(preview_frame)
        self.image_frame.pack(fill=tk.BOTH, expand=True)

        self.original_label = ttk.Label(self.image_frame)
        self.original_label.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

        self.processed_label = ttk.Label(self.image_frame)
        self.processed_label.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        # Noise gauge
        gauge_frame = ttk.LabelFrame(self.root, text="Total Noise Level", padding=10)
        gauge_frame.pack(fill=tk.X, padx=10, pady=5)

        self.gauge_canvas = tk.Canvas(gauge_frame, height=30, bg="white")
        self.gauge_canvas.pack(fill=tk.X)
        self.gauge_level = 0
        self.update_gauge()

        # Process button
        button_frame = ttk.Frame(self.root)
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(
            button_frame, text="Load Sample Image", command=self.load_sample_image
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            button_frame, text="Process All Images", command=self.process_all_images
        ).pack(side=tk.RIGHT, padx=5)

    def browse_input(self):
        folder = filedialog.askdirectory()
        if folder:
            self.input_folder.set(folder)
            if not self.output_folder.get():
                # Set default output folder
                output_path = Path(folder).parent / f"{Path(folder).name}_noisy"
                self.output_folder.set(str(output_path))

    def browse_output(self):
        folder = filedialog.askdirectory()
        if folder:
            self.output_folder.set(folder)

    def update_slider_scale(self, noise_type):
        scale_type = self.scale_vars[noise_type].get()
        slider = self.slider_widgets[noise_type]

        if scale_type == "Log":
            slider.config(from_=-3, to=0)  # Log10 scale from 0.001 to 1
        else:
            slider.config(from_=0, to=1)  # Linear scale from 0 to 1

    def on_slider_change(self, val, noise_type):
        try:
            value = float(val)
            scale_type = self.scale_vars[noise_type].get()

            if scale_type == "Log":
                actual_value = 10**value
            else:
                actual_value = value

            # Update the displayed value
            value_label = getattr(
                self, f"{noise_type.lower().replace(' ', '_').replace('&', '')}_value"
            )
            value_label.config(text=f"{actual_value:.4f}")

            # Store the actual value
            self.noise_values[noise_type] = actual_value

            # Update preview if we have an image loaded
            if self.preview_image is not None:
                self.update_preview()
                self.update_gauge()
        except ValueError:
            pass

    def load_sample_image(self):
        input_folder = self.input_folder.get()
        if not input_folder or not os.path.isdir(input_folder):
            messagebox.showerror("Error", "Please select a valid input folder first")
            return

        # Find first image in the folder
        image_path = None
        for root, _, files in os.walk(input_folder):
            for file in files:
                if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                    image_path = os.path.join(root, file)
                    break
            if image_path:
                break

        if not image_path:
            messagebox.showerror("Error", "No images found in the input folder")
            return

        # Load and display the image
        self.preview_image = Image.open(image_path).convert("RGB")
        self.update_preview()

    def update_preview(self):
        if self.preview_image is None:
            return

        # Convert PIL Image to tensor
        transform = transforms.ToTensor()
        img_tensor = transform(self.preview_image).unsqueeze(0)

        # Apply noise
        noisy_tensor = self.apply_noise(img_tensor)
        self.processed_image = noisy_tensor.squeeze(0)

        # Convert back to PIL Image
        noisy_image = transforms.ToPILImage()(self.processed_image)

        # Get available display size
        window_width = self.image_frame.winfo_width()
        window_height = self.image_frame.winfo_height()

        # Calculate max size for each image (half of available width, full height)
        max_width = max(1, (window_width // 2) - 20)  # Subtract padding
        max_height = max(1, window_height - 20)

        # Resize images while maintaining aspect ratio
        def resize_image(img):
            img_ratio = img.width / img.height
            frame_ratio = max_width / max_height

            if img_ratio > frame_ratio:
                # Image is wider than frame
                new_width = max_width
                new_height = int(max_width / img_ratio)
            else:
                # Image is taller than frame
                new_height = max_height
                new_width = int(max_height * img_ratio)

            return img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        original_display = resize_image(self.preview_image.copy())
        noisy_display = resize_image(noisy_image.copy())

        # Convert to Tkinter PhotoImage
        self.tk_images = []  # Clear previous images to prevent garbage collection
        original_tk = ImageTk.PhotoImage(original_display)
        noisy_tk = ImageTk.PhotoImage(noisy_display)

        self.tk_images.extend([original_tk, noisy_tk])

        # Display images
        self.original_label.config(image=original_tk)
        self.processed_label.config(image=noisy_tk)

    def apply_noise(self, image_tensor):
        # Clone the tensor to avoid modifying the original
        noisy_image = image_tensor.clone()

        # Apply each noise type based on its slider value
        total_noise_level = 0

        # Gaussian noise
        if self.noise_values["Gaussian"] > 0:
            std = self.noise_values["Gaussian"]
            noise = torch.randn_like(noisy_image) * std
            noisy_image += noise
            total_noise_level += std

        # Salt & Pepper noise
        if self.noise_values["Salt & Pepper"] > 0:
            amount = self.noise_values["Salt & Pepper"]
            salt = torch.rand_like(noisy_image) < (amount / 2)
            pepper = torch.rand_like(noisy_image) < (amount / 2)
            noisy_image[salt] = 1.0
            noisy_image[pepper] = 0.0
            total_noise_level += amount

        # Poisson noise
        if self.noise_values["Poisson"] > 0:
            lam = self.noise_values["Poisson"] * 10
            noisy_image = torch.poisson(noisy_image * lam) / lam
            total_noise_level += lam / 10

        # Speckle noise
        if self.noise_values["Speckle"] > 0:
            speckle = self.noise_values["Speckle"]
            noise = torch.randn_like(noisy_image) * speckle
            noisy_image = noisy_image + noisy_image * noise
            total_noise_level += speckle

        # Rayleigh noise
        if self.noise_values["Rayleigh"] > 0:
            scale = self.noise_values["Rayleigh"]
            noise = torch.from_numpy(
                np.random.rayleigh(scale, size=noisy_image.shape)
            ).float()
            noisy_image += noise
            total_noise_level += scale

        # Exponential noise
        if self.noise_values["Exponential"] > 0:
            scale = self.noise_values["Exponential"]
            noise = torch.from_numpy(
                np.random.exponential(scale, size=noisy_image.shape)
            ).float()
            noisy_image += noise
            total_noise_level += scale

        # Uniform noise
        if self.noise_values["Uniform"] > 0:
            scale = self.noise_values["Uniform"]
            noise = torch.rand_like(noisy_image) * scale - (scale / 2)
            noisy_image += noise
            total_noise_level += scale / 2

        # Clip to valid range
        noisy_image = torch.clamp(noisy_image, 0, 1)

        # Update total noise level for gauge
        self.gauge_level = min(total_noise_level, 1.0)

        return noisy_image

    def update_gauge(self):
        self.gauge_canvas.delete("all")
        width = self.gauge_canvas.winfo_width()
        height = self.gauge_canvas.winfo_height()

        # Calculate fill width
        fill_width = int(width * self.gauge_level)

        # Determine color (green to red)
        r = int(min(255, 255 * self.gauge_level * 2))
        g = int(min(255, 255 * (1 - self.gauge_level * 2)))
        color = f"#{r:02x}{g:02x}00"

        # Draw gauge
        self.gauge_canvas.create_rectangle(
            0, 0, fill_width, height, fill=color, outline=""
        )
        self.gauge_canvas.create_rectangle(0, 0, width, height, outline="black")

        # Add text
        self.gauge_canvas.create_text(
            width // 2,
            height // 2,
            text=f"Total Noise: {self.gauge_level:.2f}",
            fill="black" if self.gauge_level < 0.5 else "white",
        )

    def process_all_images(self):
        input_folder = self.input_folder.get()
        output_folder = self.output_folder.get()

        if not input_folder or not os.path.isdir(input_folder):
            messagebox.showerror("Error", "Please select a valid input folder")
            return

        if not output_folder:
            messagebox.showerror("Error", "Please select an output folder")
            return

        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Process all images
        processed_count = 0
        for root, dirs, files in os.walk(input_folder):
            # Create corresponding subdirectories in output folder
            rel_path = os.path.relpath(root, input_folder)
            output_root = os.path.join(output_folder, rel_path)
            os.makedirs(output_root, exist_ok=True)

            for file in files:
                if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                    try:
                        # Load image
                        input_path = os.path.join(root, file)
                        output_path = os.path.join(output_root, file)

                        img = Image.open(input_path).convert("RGB")

                        # Convert to tensor and apply noise
                        transform = transforms.ToTensor()
                        img_tensor = transform(img).unsqueeze(0)
                        noisy_tensor = self.apply_noise(img_tensor)
                        noisy_image = transforms.ToPILImage()(noisy_tensor.squeeze(0))

                        # Save with original metadata
                        noisy_image.save(
                            output_path, quality=95, exif=img.info.get("exif", b"")
                        )

                        processed_count += 1
                    except Exception as e:
                        print(f"Error processing {file}: {str(e)}")

        messagebox.showinfo(
            "Processing Complete", f"Successfully processed {processed_count} images"
        )


if __name__ == "__main__":
    root = tk.Tk()
    app = NoiseInjectionApp(root)
    root.mainloop()
