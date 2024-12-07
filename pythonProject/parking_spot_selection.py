# Ultralytics YOLO 🚀, AGPL-3.0 license

import json

import cv2
import numpy as np
import pynput

from ultralytics.solutions.solutions import LOGGER, BaseSolution, check_requirements
from ultralytics.utils.plotting import Annotator

mouse_ctr = pynput.mouse.Controller()


class ParkingPtsSelection:
    """
    A class for selecting and managing parking zone points on images using a Tkinter-based UI.

    This class provides functionality to upload an image, select points to define parking zones, and save the
    selected points to a JSON file. It uses Tkinter for the graphical user interface.

    Attributes:
        tk (module): The Tkinter module for GUI operations.
        filedialog (module): Tkinter's filedialog module for file selection operations.
        messagebox (module): Tkinter's messagebox module for displaying message boxes.
        master (tk.Tk): The main Tkinter window.
        canvas (tk.Canvas): The canvas widget for displaying the image and drawing bounding boxes.
        image (PIL.Image.Image): The uploaded image.
        canvas_image (ImageTk.PhotoImage): The image displayed on the canvas.
        rg_data (List[List[Tuple[int, int]]]): List of bounding boxes, each defined by 4 points.
        current_box (List[Tuple[int, int]]): Temporary storage for the points of the current bounding box.
        imgw (int): Original width of the uploaded image.
        imgh (int): Original height of the uploaded image.
        canvas_max_width (int): Maximum width of the canvas.
        canvas_max_height (int): Maximum height of the canvas.

    Methods:
        setup_ui: Sets up the Tkinter UI components.
        initialize_properties: Initializes the necessary properties.
        upload_image: Uploads an image, resizes it to fit the canvas, and displays it.
        on_canvas_click: Handles mouse clicks to add points for bounding boxes.
        draw_box: Draws a bounding box on the canvas.
        remove_last_bounding_box: Removes the last bounding box and redraws the canvas.
        redraw_canvas: Redraws the canvas with the image and all bounding boxes.
        save_to_json: Saves the bounding boxes to a JSON file.

    Examples:
        >>> parking_selector = ParkingPtsSelection()
        >>> # Use the GUI to upload an image, select parking zones, and save the data
    """

    def __init__(self):
        """Initializes the ParkingPtsSelection class, setting up UI and properties for parking zone point selection."""
        check_requirements("tkinter")
        import tkinter as tk
        from tkinter import filedialog, messagebox

        self.tk, self.filedialog, self.messagebox = tk, filedialog, messagebox
        self.setup_ui()
        self.initialize_properties()
        self.master.mainloop()

    def setup_ui(self):
        """Sets up the Tkinter UI components for the parking zone points selection interface."""
        self.master = self.tk.Tk()
        self.master.title("Ultralytics Parking Zones Points Selector")
        self.master.resizable(True, True)

        # Canvas for image display
        self.canvas = self.tk.Canvas(self.master, bg="white")
        self.canvas.pack(side=self.tk.BOTTOM)

        # Button frame with buttons
        button_frame = self.tk.Frame(self.master)
        button_frame.pack(side=self.tk.TOP)

        self.ctrl_down = False

        for text, cmd in [
            ("Upload Image", self.upload_image),
            ("Remove Last BBox", self.remove_last_bounding_box),
            ("Save", self.save_to_json),
            ("Load Boxes", self.load_boxes),
        ]:
            self.tk.Button(button_frame, text=text, command=cmd).pack(side=self.tk.LEFT)

    def initialize_properties(self):
        """Initialize properties for image, canvas, bounding boxes, and dimensions."""
        self.image = self.canvas_image = None
        self.rg_data, self.current_box, self.support_lines, self.help_lines, self.current_line = [], [], [], [], []
        self.box_lines = []
        self.imgw = self.imgh = 0
        self.canvas_max_width, self.canvas_max_height = 5000, 1280

    def upload_image(self):
        """Uploads and displays an image on the canvas, resizing it to fit within specified dimensions."""
        from PIL import Image, ImageTk  # scope because ImageTk requires tkinter package

        self.image = Image.open(self.filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg"),
                                                                           ("Image Files", "*.jpeg"),
                                                                           ("Image Files", "*.png")
                                                                           ]))
        if not self.image:
            return

        self.imgw, self.imgh = self.image.size
        aspect_ratio = self.imgw / self.imgh
        canvas_width = (
            min(self.canvas_max_width, self.imgw) if aspect_ratio > 1 else int(self.canvas_max_height * aspect_ratio)
        )
        canvas_height = (
            min(self.canvas_max_height, self.imgh) if aspect_ratio <= 1 else int(canvas_width / aspect_ratio)
        )

        print(f"Canvas size: {canvas_width}, {canvas_height}")
        self.scale_w = self.imgw / canvas_width
        self.scale_h = self.imgh / canvas_height

        self.canvas.config(width=canvas_width, height=canvas_height)
        self.canvas_image = ImageTk.PhotoImage(self.image.resize((canvas_width, canvas_height), Image.LANCZOS))
        self.canvas.create_image(0, 0, anchor=self.tk.NW, image=self.canvas_image)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Button-3>", self.on_canvas_click)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.master.bind("<KeyPress>", self.on_key_press)
        self.master.bind("<KeyRelease>", self.on_key_release)

        self.rg_data.clear(), self.current_box.clear()
        self.support_lines.clear(), self.current_line.clear(), self.help_lines.clear()

    def load_boxes(self):
        with open(self.filedialog.askopenfilename(filetypes=[("Box File", "*.json")])) as f:
            data = json.load(f)

        for dbox in data:
            box = [(x / self.scale_w, y / self.scale_h) for x, y in dbox["points"]]
            self.rg_data.append(box.copy())
            self.draw_box(box)

    def clean(self):
        for line in self.help_lines:
            self.canvas.delete(line)
        self.help_lines.clear()

    def on_canvas_click(self, event):
        """Handles mouse clicks to add points for bounding boxes on the canvas."""
        if event.num == 1:  # Left button
            self.current_box.append((event.x, event.y))
            #self.canvas.create_oval(event.x - 3, event.y - 3, event.x + 3, event.y + 3, fill="red")
            if len(self.current_box) == 4:
                self.rg_data.append(self.current_box.copy())
                self.draw_box(self.current_box)
                self.current_box.clear()
                self.clean()
            elif len(self.current_box) > 1:
                for i in range(1, len(self.current_box)):
                    self.help_lines.insert(0, self.canvas.create_line(self.current_box[i-1], self.current_box[i],
                                                                   fill="blue", width=2))

        elif event.num == 3:  # Right button
            self.current_line.append((event.x, event.y))
            #self.canvas.create_oval(event.x - 3, event.y - 3, event.x + 3, event.y + 3, fill="red")
            if len(self.current_line) == 2:
                self.support_lines.append(self.current_line.copy())
                self.canvas.create_line(self.current_line[0], self.current_line[1], fill="green", width=2)
                self.current_line.clear()
                self.clean()

    def on_mouse_move(self, event):
        """Handle mouse movement for snapping or visualization."""
        x, y = event.x, event.y

        if self.ctrl_down and self.support_lines:
            # Snap to the closest helper line
            distance, snapped_line = self.find_closest_helper_line((x, y))
            if snapped_line and distance > 1:
                snapped_x, snapped_y = self.snap_to_line(snapped_line, (x, y))

                # Get canvas position on the screen
                canvas_x = self.canvas.winfo_rootx()
                canvas_y = self.canvas.winfo_rooty()

                # Convert canvas coordinates to screen coordinates
                x = canvas_x + snapped_x
                y = canvas_y + snapped_y

                mouse_ctr.position = (x, y)

        if len(self.help_lines) > 0:
            self.canvas.delete(self.help_lines.pop())

        if len(self.current_line) == 1:
            self.help_lines.append(self.canvas.create_line(self.current_line[0], (x, y), dash=(4, 2), fill="lime", width=1))
        elif len(self.current_box) > 0:
            self.help_lines.append(
                self.canvas.create_line(self.current_box[-1], (x, y), dash=(4, 2), fill="lime", width=1))

    def on_key_press(self, event):
        """Handle key press events (e.g., Ctrl for snapping)."""
        if event.keysym == "Control_L" or event.keysym == "Control_R":
            self.ctrl_down = True
        elif event.keysym == "BackSpace":
            self.remove_last_bounding_box()

    def on_key_release(self, event):
        """Handle key release events."""
        if event.keysym == "Control_L" or event.keysym == "Control_R":
            self.ctrl_down = False

    # Function to find the closest helper line
    def find_closest_helper_line(self, point):
        if len(self.support_lines) == 0:
            return None, None
        distances = []
        for line in self.support_lines:
            dist = self.point_to_line_distance(line, point)
            distances.append((dist, line))
        distances.sort()
        return distances[0] if distances[0][0] < 20 else None  # Threshold for snapping

    # Function to calculate the distance from a point to a line
    @staticmethod
    def point_to_line_distance(line, point):
        x1, y1 = line[0]
        x2, y2 = line[1]
        px, py = point
        norm = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        distance = abs((px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)) / norm
        return distance

    # Function to snap a point to a line
    @staticmethod
    def snap_to_line(line, point):
        x1, y1 = line[0]
        x2, y2 = line[1]
        px, py = point
        dx, dy = x2 - x1, y2 - y1
        t = ((px - x1) * dx + (py - y1) * dy) / (dx ** 2 + dy ** 2)
        t = max(0, min(1, t))  # Clamp t to [0, 1]
        snapped_x = int(x1 + t * dx)
        snapped_y = int(y1 + t * dy)
        return snapped_x, snapped_y

    def draw_box(self, box):
        """Draws a bounding box on the canvas using the provided coordinates."""
        box_line = []
        for i in range(4):
            box_line.append(self.canvas.create_line(box[i], box[(i + 1) % 4], fill="blue", width=2))
        self.box_lines.append(box_line)

    def remove_last_bounding_box(self):
        """Removes the last bounding box from the list and redraws the canvas."""
        if not self.rg_data:
            self.messagebox.showwarning("Warning", "No bounding boxes to remove.")
            return
        self.rg_data.pop()

        for box_line in self.box_lines.pop():
            self.canvas.delete(box_line)

    def redraw_canvas(self):
        """Redraws the canvas with the image and all bounding boxes."""
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=self.tk.NW, image=self.canvas_image)
        for box in self.rg_data:
            self.draw_box(box)

    def save_to_json(self):
        """Saves the selected parking zone points to a JSON file with scaled coordinates."""
        data = [{"points": [(int(x * self.scale_w), int(y * self.scale_h)) for x, y in box]} for box in self.rg_data]
        with open("bounding_boxes.json", "w") as f:
            json.dump(data, f, indent=4)
        self.messagebox.showinfo("Success", "Bounding boxes saved to parking_slots.json")


class ParkingManagement(BaseSolution):
    """
    Manages parking occupancy and availability using YOLO model for real-time monitoring and visualization.

    This class extends BaseSolution to provide functionality for parking lot management, including detection of
    occupied spaces, visualization of parking regions, and display of occupancy statistics.

    Attributes:
        json_file (str): Path to the JSON file containing parking region details.
        json (List[Dict]): Loaded JSON data containing parking region information.
        pr_info (Dict[str, int]): Dictionary storing parking information (Occupancy and Available spaces).
        arc (Tuple[int, int, int]): RGB color tuple for available region visualization.
        occ (Tuple[int, int, int]): RGB color tuple for occupied region visualization.
        dc (Tuple[int, int, int]): RGB color tuple for centroid visualization of detected objects.

    Methods:
        process_data: Processes model data for parking lot management and visualization.

    Examples:
        >>> from ultralytics.solutions import ParkingManagement
        >>> parking_manager = ParkingManagement(model="yolov8n.pt", json_file="parking_regions.json")
        >>> print(f"Occupied spaces: {parking_manager.pr_info['Occupancy']}")
        >>> print(f"Available spaces: {parking_manager.pr_info['Available']}")
    """

    def __init__(self, **kwargs):
        """Initializes the parking management system with a YOLO model and visualization settings."""
        super().__init__(**kwargs)

        self.json_file = self.CFG["json_file"]  # Load JSON data
        if self.json_file is None:
            LOGGER.warning("❌ json_file argument missing. Parking region details required.")
            raise ValueError("❌ Json file path can not be empty")

        with open(self.json_file) as f:
            self.json = json.load(f)

        self.pr_info = {"Occupancy": 0, "Available": 0}  # dictionary for parking information

        self.arc = (0, 0, 255)  # available region color
        self.occ = (0, 255, 0)  # occupied region color
        self.dc = (255, 0, 189)  # centroid color for each box

    def process_data(self, im0):
        """
        Processes the model data for parking lot management.

        This function analyzes the input image, extracts tracks, and determines the occupancy status of parking
        regions defined in the JSON file. It annotates the image with occupied and available parking spots,
        and updates the parking information.

        Args:
            im0 (np.ndarray): The input inference image.

        Examples:
            >>> parking_manager = ParkingManagement(json_file="parking_regions.json")
            >>> image = cv2.imread("parking_lot.jpg")
            >>> parking_manager.process_data(image)
        """
        self.extract_tracks(im0)  # extract tracks from im0
        es, fs = len(self.json), 0  # empty slots, filled slots
        annotator = Annotator(im0, self.line_width)  # init annotator

        for region in self.json:
            # Convert points to a NumPy array with the correct dtype and reshape properly
            pts_array = np.array(region["points"], dtype=np.int32).reshape((-1, 1, 2))
            rg_occupied = False  # occupied region initialization
            for box, cls in zip(self.boxes, self.clss):
                xc, yc = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
                dist = cv2.pointPolygonTest(pts_array, (xc, yc), False)
                if dist >= 0:
                    # cv2.circle(im0, (xc, yc), radius=self.line_width * 4, color=self.dc, thickness=-1)
                    annotator.display_objects_labels(
                        im0, self.model.names[int(cls)], (104, 31, 17), (255, 255, 255), xc, yc, 10
                    )
                    rg_occupied = True
                    break
            fs, es = (fs + 1, es - 1) if rg_occupied else (fs, es)
            # Plotting regions
            cv2.polylines(im0, [pts_array], isClosed=True, color=self.occ if rg_occupied else self.arc, thickness=2)

        self.pr_info["Occupancy"], self.pr_info["Available"] = fs, es

        annotator.display_analytics(im0, self.pr_info, (104, 31, 17), (255, 255, 255), 10)
        self.display_output(im0)  # display output with base class function
        return im0  # return output image for more usage


if __name__ == "__main__":
    ParkingPtsSelection()